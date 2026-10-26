#!/usr/bin/env python3
"""
cover_colours.py - Extract dominant colour and colour palette percentages
from comic book cover images and export them as JSON.

Usage:
    python cover_colours.py covers/ -o xmen_colours.json
    python cover_colours.py covers/ -k 8 --ignore-neutrals
    python cover_colours.py cover1.jpg cover2.jpg -o out.json

Requirements:
    pip install pillow numpy

How it works:
    1. Each image is downscaled (speed + noise reduction).
    2. Pixels are converted to CIE Lab space, so "distance" between colours
       roughly matches how humans see differences.
    3. K-means clustering groups pixels into K colours.
    4. Clusters that look nearly identical are merged.
    5. Each cluster becomes a palette entry with its share of the cover (%).
       The biggest one is the "dominant" colour.
"""

import argparse
import colorsys
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# Colour space helpers
# ---------------------------------------------------------------------------

# sRGB D65 -> XYZ matrix and reference white
_M = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_WHITE = np.array([0.95047, 1.0, 1.08883])
_M_INV = np.linalg.inv(_M)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (N,3) floats in 0-255 -> (N,3) Lab."""
    c = rgb / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ _M.T / _WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """(N,3) Lab -> (N,3) RGB floats clipped to 0-255."""
    fy = (lab[:, 0] + 16) / 116
    fx = fy + lab[:, 1] / 500
    fz = fy - lab[:, 2] / 200
    f = np.stack([fx, fy, fz], axis=1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * _WHITE
    c = xyz @ _M_INV.T
    c = np.clip(c, 0, 1)
    c = np.where(c > 0.0031308, 1.055 * c ** (1 / 2.4) - 0.055, 12.92 * c)
    return np.clip(c * 255, 0, 255)


def to_hex(rgb) -> str:
    r, g, b = (int(round(v)) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def describe(rgb) -> dict:
    """Colour in several formats, handy for later visualisation work."""
    r, g, b = (int(round(v)) for v in rgb)
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    return {
        "hex": to_hex((r, g, b)),
        "rgb": [r, g, b],
        "hsl": [round(h * 360, 1), round(s * 100, 1), round(l * 100, 1)],
    }


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def kmeans(data: np.ndarray, k: int, iters: int = 30, seed: int = 0):
    """Small k-means (k-means++ init) so no scikit-learn dependency needed."""
    rng = np.random.default_rng(seed)
    n = len(data)
    k = min(k, n)

    # k-means++ initialisation
    centres = [data[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(
            ((data[:, None, :] - np.array(centres)[None, :, :]) ** 2).sum(-1),
            axis=1,
        )
        total = d2.sum()
        if total == 0:
            break
        centres.append(data[rng.choice(n, p=d2 / total)])
    centres = np.array(centres)

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dists = ((data[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for i in range(len(centres)):
            members = data[labels == i]
            if len(members):
                centres[i] = members.mean(axis=0)
    return centres, labels


def merge_similar(centres, counts, threshold: float):
    """Merge clusters closer than `threshold` (Lab delta-E) into the bigger one."""
    order = np.argsort(-counts)
    merged_c, merged_n = [], []
    for idx in order:
        c, n = centres[idx], counts[idx]
        for j, mc in enumerate(merged_c):
            if np.linalg.norm(c - mc) < threshold:
                total = merged_n[j] + n
                merged_c[j] = (mc * merged_n[j] + c * n) / total
                merged_n[j] = total
                break
        else:
            merged_c.append(c.astype(float))
            merged_n.append(float(n))
    return np.array(merged_c), np.array(merged_n)


# ---------------------------------------------------------------------------
# Image analysis
# ---------------------------------------------------------------------------

def analyse_image(
    path: Path,
    k: int = 8,
    size: int = 200,
    merge_threshold: float = 12.0,
    ignore_neutrals: bool = False,
    min_percentage: float = 0.5,
) -> dict:
    with Image.open(path) as im:
        original_size = im.size
        im = im.convert("RGBA")
        im.thumbnail((size, size), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.float64)

    # Drop transparent pixels
    pixels = arr[arr[..., 3] > 128][:, :3]

    if ignore_neutrals:
        # Ignore near-black ink, near-white logos/borders and flat greys
        mx, mn = pixels.max(axis=1), pixels.min(axis=1)
        chroma = mx - mn
        keep = ~((mx < 40) | (mn > 225) | (chroma < 15))
        if keep.sum() > 50:  # don't strip everything from B&W covers
            pixels = pixels[keep]

    if len(pixels) == 0:
        raise ValueError("no usable pixels")

    lab = rgb_to_lab(pixels)
    centres, labels = kmeans(lab, k)
    counts = np.bincount(labels, minlength=len(centres)).astype(float)
    nonzero = counts > 0
    centres, counts = centres[nonzero], counts[nonzero]

    centres, counts = merge_similar(centres, counts, merge_threshold)
    order = np.argsort(-counts)
    centres, counts = centres[order], counts[order]

    percentages = counts / counts.sum() * 100
    rgb_centres = lab_to_rgb(centres)

    palette = []
    for rgb, pct in zip(rgb_centres, percentages):
        if pct < min_percentage:
            continue
        entry = describe(rgb)
        entry["percentage"] = round(float(pct), 2)
        palette.append(entry)

    # Re-normalise so the kept entries sum to 100
    total = sum(p["percentage"] for p in palette) or 1
    for p in palette:
        p["percentage"] = round(p["percentage"] / total * 100, 2)

    return {
        "filename": path.name,
        "path": str(path),
        "width": original_size[0],
        "height": original_size[1],
        "dominant": palette[0],
        "palette": palette,
    }


def find_images(inputs):
    files = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files += sorted(
                f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS
            )
        elif p.is_file():
            files.append(p)
        else:
            print(f"warning: {item} not found", file=sys.stderr)
    return files


def main():
    ap = argparse.ArgumentParser(description="Extract colour palettes from comic covers.")
    ap.add_argument("inputs", nargs="+", help="image files and/or folders")
    ap.add_argument("-o", "--output", default="cover_colours.json")
    ap.add_argument("-k", "--colors", type=int, default=8,
                    help="number of clusters to start with (default 8)")
    ap.add_argument("--size", type=int, default=200,
                    help="longest side in px that images are shrunk to (default 200)")
    ap.add_argument("--merge", type=float, default=12.0,
                    help="merge colours closer than this Lab distance (default 12; 0 disables)")
    ap.add_argument("--ignore-neutrals", action="store_true",
                    help="ignore near-black, near-white and grey pixels when picking colours")
    ap.add_argument("--min-percentage", type=float, default=0.5,
                    help="drop palette entries below this %% (default 0.5)")
    args = ap.parse_args()

    files = find_images(args.inputs)
    if not files:
        sys.exit("No images found.")

    covers, errors = [], []
    for i, f in enumerate(files, 1):
        try:
            covers.append(analyse_image(
                f, k=args.colors, size=args.size,
                merge_threshold=args.merge,
                ignore_neutrals=args.ignore_neutrals,
                min_percentage=args.min_percentage,
            ))
            print(f"[{i}/{len(files)}] {f.name}  ->  {covers[-1]['dominant']['hex']}")
        except Exception as e:  # keep going on bad files
            errors.append({"filename": f.name, "error": str(e)})
            print(f"[{i}/{len(files)}] {f.name}  FAILED: {e}", file=sys.stderr)

    result = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": {
            "colors": args.colors,
            "size": args.size,
            "merge_threshold": args.merge,
            "ignore_neutrals": args.ignore_neutrals,
            "min_percentage": args.min_percentage,
        },
        "count": len(covers),
        "covers": covers,
        "errors": errors,
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\nWrote {len(covers)} covers to {args.output}")


if __name__ == "__main__":
    main()
