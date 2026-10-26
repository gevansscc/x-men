#!/usr/bin/env python3
"""
Download images referenced by a webpage into a local directory.

Usage:
    python download_images.py "https://example.com/page" --output images

Install dependencies:
    python -m pip install requests beautifulsoup4
"""

import argparse
import hashlib
import mimetypes
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup
from typing import Optional


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".avif"}


def safe_filename(name: str) -> str:
    """Make a filename safe for common desktop operating systems."""
    name = unquote(name).strip()
    name = re.sub(r"[\\/:*?\"<>|\x00-\x1f]", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:180] or "image"


def get_image_urls(page_url: str, session: requests.Session) -> list:
    response = session.get(page_url, timeout=25)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    urls = []

    # Standard image elements, including lazy-loaded image attributes.
    for img in soup.find_all("img"):
        for attr in ("src", "data-src", "data-lazy-src", "data-original"):
            value = img.get(attr)
            if value:
                urls.append(urljoin(response.url, value.strip()))

        # Responsive image sources.
        for attr in ("srcset", "data-srcset"):
            value = img.get(attr)
            if value:
                for candidate in value.split(","):
                    candidate = candidate.strip().split()
                    if candidate:
                        urls.append(urljoin(response.url, candidate[0]))

    # Images declared as CSS backgrounds in inline style attributes.
    for element in soup.find_all(style=True):
        style = element.get("style", "")
        for match in re.findall(r"url\((?:['\"])?(.*?)(?:['\"])?\)", style, flags=re.I):
            if match and not match.startswith("data:"):
                urls.append(urljoin(response.url, match.strip()))

    # Open Graph / social preview image.
    for meta in soup.find_all("meta"):
        if meta.get("property") in ("og:image", "og:image:url") or meta.get("name") == "twitter:image":
            value = meta.get("content")
            if value:
                urls.append(urljoin(response.url, value.strip()))

    # Remove duplicates while preserving order; ignore embedded data URLs.
    return list(dict.fromkeys(u for u in urls if u and not u.startswith("data:")))


def filename_for(url, content_type, index):
    path_name = Path(urlparse(url).path).name
    path_name = safe_filename(path_name)

    suffix = Path(path_name).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        guessed = mimetypes.guess_extension((content_type or "").split(";")[0].strip())
        suffix = guessed if guessed in IMAGE_EXTENSIONS else ".img"
        stem = Path(path_name).stem or f"image_{index:03d}"
        path_name = stem + suffix

    if path_name in ("", ".", suffix):
        path_name = f"image_{index:03d}{suffix}"

    return path_name


def download_images(page_url: str, output_dir: Path, delay: float = 0.2) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; ImageDownloader/1.0)"
    })

    image_urls = get_image_urls(page_url, session)
    print(f"Found {len(image_urls)} possible image URL(s).")
    seen_hashes = set()
    downloaded = 0
    failed = 0

    for index, image_url in enumerate(image_urls, start=1):
        try:
            response = session.get(image_url, timeout=30, stream=True)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            if content_type and not content_type.startswith("image/"):
                print(f"Skipping non-image response: {image_url}")
                continue

            content = response.content
            if not content:
                print(f"Skipping empty response: {image_url}")
                continue

            digest = hashlib.sha256(content).hexdigest()
            if digest in seen_hashes:
                print(f"Skipping duplicate image: {image_url}")
                continue
            seen_hashes.add(digest)

            filename = filename_for(image_url, content_type, index)
            destination = output_dir / filename

            # Avoid overwriting a different file with the same name.
            if destination.exists():
                stem, suffix = destination.stem, destination.suffix
                destination = output_dir / f"{stem}_{digest[:8]}{suffix}"

            destination.write_bytes(content)
            downloaded += 1
            print(f"Saved: {destination}")
        except requests.RequestException as exc:
            failed += 1
            print(f"Failed: {image_url} ({exc})")
        except OSError as exc:
            failed += 1
            print(f"Could not save image: {image_url} ({exc})")

        if delay > 0:
            time.sleep(delay)

    print(f"\nFinished. Downloaded {downloaded} image(s); failed {failed} request(s).")
    print(f"Output directory: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download images from a webpage.")
    parser.add_argument("url", help="URL of the webpage to scan")
    parser.add_argument("--output", default="downloaded_images", help="Destination directory")
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds to wait between image requests")
    args = parser.parse_args()

    if urlparse(args.url).scheme not in ("http", "https"):
        parser.error("URL must start with http:// or https://")
    if args.delay < 0:
        parser.error("--delay cannot be negative")

    download_images(args.url, Path(args.output).expanduser(), args.delay)


if __name__ == "__main__":
    main()
