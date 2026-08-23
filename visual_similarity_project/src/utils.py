"""
utils.py
Shared helper functions used across the pipeline.
"""

import os
import re
from pathlib import Path
from typing import List


def list_images(image_dir: str) -> List[str]:
    """Return a sorted list of absolute paths to all images in a directory."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = [
        str(p.resolve())
        for p in Path(image_dir).iterdir()
        if p.suffix.lower() in exts
    ]
    return sorted(paths)


def get_category(filename: str) -> str:
    """
    Extract the product category from a filename like:
        'backpack_10_1771054475342.jpg' -> 'backpack'
    Falls back to 'unknown' if the pattern doesn't match.
    """
    name = os.path.basename(filename)
    match = re.match(r"^([a-zA-Z]+)_", name)
    return match.group(1).lower() if match else "unknown"
