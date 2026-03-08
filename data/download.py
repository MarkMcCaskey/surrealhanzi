#!/usr/bin/env python3
"""Download Make Me a Hanzi data files."""

import os
import urllib.request
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "graphics.txt": "https://raw.githubusercontent.com/skishore/makemeahanzi/master/graphics.txt",
    "dictionary.txt": "https://raw.githubusercontent.com/skishore/makemeahanzi/master/dictionary.txt",
    "ids_babelstone.txt": "https://www.babelstone.co.uk/CJK/IDS.TXT",
}


def download(filename: str, url: str) -> None:
    dest = os.path.join(DATA_DIR, filename)
    if os.path.exists(dest):
        print(f"  {filename} already exists, skipping")
        return
    print(f"  Downloading {filename}...")
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"  {filename}: {size_mb:.1f} MB")


def main() -> None:
    print("Downloading Make Me a Hanzi data files...")
    for filename, url in FILES.items():
        download(filename, url)
    print("Done.")


if __name__ == "__main__":
    main()
