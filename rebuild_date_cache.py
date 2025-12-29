#!/usr/bin/env python3
"""
Rebuild the date cache using osxphotos metadata instead of EXIF.

This is faster than re-exporting all photos because it reads dates directly
from the Photos library database and matches them to exported files by UUID.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

try:
    import osxphotos
except ImportError:
    print("osxphotos is required. Install with: pip install osxphotos")
    sys.exit(1)


def rebuild_cache_from_photos_db(folder: Path, progress_callback=None) -> dict:
    """
    Rebuild the date cache by matching exported files to Photos library metadata.

    Exported files are named by UUID (first 8 chars), so we can match them
    to the Photos database to get the original date.

    Args:
        folder: Folder containing exported photos
        progress_callback: Optional callback(current, total, message)

    Returns:
        Dict with stats: {total, matched, unmatched}
    """
    cache_path = folder / ".photo_dates.json"

    # Get all jpg files in the folder
    jpg_files = list(folder.glob("*.jpg"))
    total = len(jpg_files)

    if total == 0:
        print(f"No jpg files found in {folder}")
        return {"total": 0, "matched": 0, "unmatched": 0}

    print(f"Found {total} photos in {folder}")
    print("Loading Photos library...")

    # Load Photos database
    photosdb = osxphotos.PhotosDB()
    photos = photosdb.photos(images=True, movies=False)

    # Build UUID lookup (first 8 chars -> date)
    print(f"Building UUID lookup from {len(photos)} photos...")
    uuid_to_date = {}
    for photo in photos:
        if photo.date:
            # Store as EXIF format string
            date_str = photo.date.strftime('%Y:%m:%d %H:%M:%S')
            # Use first 8 chars of UUID (matches our export naming)
            uuid_prefix = photo.uuid[:8].upper()
            uuid_to_date[uuid_prefix] = date_str

    print(f"Found dates for {len(uuid_to_date)} unique UUID prefixes")

    # Match exported files to Photos metadata
    cache = {}
    matched = 0
    unmatched = 0

    for i, jpg_file in enumerate(jpg_files):
        # Filename is UUID prefix (e.g., "A1B2C3D4.jpg")
        uuid_prefix = jpg_file.stem.upper()

        if uuid_prefix in uuid_to_date:
            cache[jpg_file.name] = uuid_to_date[uuid_prefix]
            matched += 1
        else:
            cache[jpg_file.name] = None
            unmatched += 1

        if progress_callback and (i + 1) % 1000 == 0:
            progress_callback(i + 1, total, f"Matching: {i+1}/{total}")

    # Save cache
    print(f"Saving cache to {cache_path}")
    with open(cache_path, 'w') as f:
        json.dump(cache, f)

    # Invalidate in-memory cache so next access reloads from disk
    try:
        from date_cache import invalidate_cache
        invalidate_cache(folder)
    except ImportError:
        pass

    print(f"\nResults:")
    print(f"  Total files: {total}")
    print(f"  Matched: {matched} ({100*matched/total:.1f}%)")
    print(f"  Unmatched: {unmatched} ({100*unmatched/total:.1f}%)")

    # Show date distribution
    dates_by_year = {}
    for date_str in cache.values():
        if date_str:
            try:
                year = int(date_str[:4])
                dates_by_year[year] = dates_by_year.get(year, 0) + 1
            except:
                pass

    if dates_by_year:
        print("\nPhotos by year:")
        for year in sorted(dates_by_year.keys()):
            print(f"  {year}: {dates_by_year[year]}")

    # Count July 2025
    july_2025 = sum(1 for d in cache.values() if d and d.startswith('2025:07'))
    print(f"\nJuly 2025 photos: {july_2025}")

    return {"total": total, "matched": matched, "unmatched": unmatched}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild date cache from Photos library")
    parser.add_argument("--folder", type=str, default="photos/raw",
                       help="Folder to rebuild cache for (default: photos/raw)")

    args = parser.parse_args()
    folder = Path(args.folder)

    if not folder.exists():
        print(f"Folder not found: {folder}")
        sys.exit(1)

    rebuild_cache_from_photos_db(folder)


if __name__ == "__main__":
    main()
