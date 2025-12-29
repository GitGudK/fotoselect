"""
Simple file-based cache for photo dates.
Stores dates in a JSON file to avoid re-reading EXIF on every filter operation.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import subprocess

from PIL import Image


CACHE_FILENAME = ".photo_dates.json"


def get_cache_path(folder: Path) -> Path:
    """Get the path to the date cache file for a folder."""
    return folder / CACHE_FILENAME


def load_cache(folder: Path) -> Dict[str, Optional[str]]:
    """Load the date cache from disk."""
    cache_path = get_cache_path(folder)
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_cache(folder: Path, cache: Dict[str, Optional[str]]):
    """Save the date cache to disk."""
    cache_path = get_cache_path(folder)
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache, f)
    except Exception:
        pass


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse a date string from the cache."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
    except ValueError:
        return None


def extract_date_pil(image_path: str) -> Optional[str]:
    """Extract date from image using PIL."""
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if exif:
            for tag_id in [36867, 306]:  # DateTimeOriginal, DateTime
                if tag_id in exif:
                    return exif[tag_id]
    except Exception:
        pass
    return None


def extract_dates_batch_exiftool(paths: List[str]) -> Dict[str, Optional[str]]:
    """Extract dates from multiple files using exiftool in batch mode."""
    if not paths:
        return {}

    results = {}
    batch_size = 500  # Process 500 files at a time

    for batch_start in range(0, len(paths), batch_size):
        batch = paths[batch_start:batch_start + batch_size]

        try:
            cmd = ['exiftool', '-json', '-DateTimeOriginal', '-d', '%Y:%m:%d %H:%M:%S'] + batch
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                for item in data:
                    file_path = item.get('SourceFile', '')
                    date_str = item.get('DateTimeOriginal', '')
                    results[file_path] = date_str if date_str else None
        except Exception:
            # Mark batch as processed with no dates
            for path in batch:
                if path not in results:
                    results[path] = None

    return results


# Module-level cache to avoid reloading JSON repeatedly
_loaded_caches: Dict[str, Dict[str, Optional[str]]] = {}


def get_photo_dates(
    folder: Path,
    image_paths: List[str],
    progress_callback: Optional[callable] = None
) -> Dict[str, Optional[datetime]]:
    """
    Get dates for a list of photos, using cache when available.

    Args:
        folder: The folder containing the photos (for cache location)
        image_paths: List of image file paths
        progress_callback: Optional callback(current, total, phase)

    Returns:
        Dict mapping image path to datetime (or None if no date)
    """
    # Use module-level cache to avoid reloading JSON file repeatedly
    folder_key = str(folder)
    if folder_key not in _loaded_caches:
        _loaded_caches[folder_key] = load_cache(folder)
    cache = _loaded_caches[folder_key]

    results = {}
    needs_extraction = []
    total = len(image_paths)

    # Check cache first - this should be fast now
    for path in image_paths:
        filename = os.path.basename(path)

        if filename in cache:
            # Use cached value
            results[path] = parse_date(cache[filename])
        else:
            needs_extraction.append(path)

    if not needs_extraction:
        return results

    # Extract dates for uncached files
    # First pass: PIL (fast for JPEGs with EXIF)
    still_needs_exiftool = []
    cache_modified = False

    for i, path in enumerate(needs_extraction):
        date_str = extract_date_pil(path)
        filename = os.path.basename(path)

        if date_str:
            cache[filename] = date_str
            results[path] = parse_date(date_str)
            cache_modified = True
        else:
            still_needs_exiftool.append(path)

        if progress_callback and (i + 1) % 500 == 0:
            progress_callback(len(image_paths) - len(needs_extraction) + i + 1, total, 'filtering')

    # Second pass: exiftool batch for remaining files
    if still_needs_exiftool:
        exif_results = extract_dates_batch_exiftool(still_needs_exiftool)

        for path, date_str in exif_results.items():
            filename = os.path.basename(path)
            cache[filename] = date_str
            results[path] = parse_date(date_str)
            cache_modified = True

    # Only save if we added new entries
    if cache_modified:
        save_cache(folder, cache)

    # Fill in any missing paths
    for path in image_paths:
        if path not in results:
            results[path] = None

    return results


def invalidate_cache(folder: Path = None):
    """Invalidate the in-memory cache for a folder, or all folders if None."""
    global _loaded_caches
    if folder is None:
        _loaded_caches = {}
    else:
        folder_key = str(folder)
        if folder_key in _loaded_caches:
            del _loaded_caches[folder_key]


def rebuild_cache(folder: Path, progress_callback: Optional[callable] = None) -> int:
    """
    Rebuild the date cache for all images in a folder.

    Returns the number of images with dates.
    """
    from dataset import find_images

    images = find_images(folder)
    image_paths = [str(p) for p in images]

    # Clear existing cache
    cache_path = get_cache_path(folder)
    if cache_path.exists():
        cache_path.unlink()

    # Build new cache
    dates = get_photo_dates(folder, image_paths, progress_callback)

    return sum(1 for d in dates.values() if d is not None)
