#!/usr/bin/env python3
"""
Download photos from iCloud via osxphotos CLI and downsample for training.

Uses osxphotos CLI export with --download-missing to fetch photos from iCloud,
then downsamples them to the target size.

Usage:
    python download_icloud.py                    # Download all photos
    python download_icloud.py --favorites-only   # Only favorites
    python download_icloud.py --max-photos 100   # Limit to 100 photos
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

# Register HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("Warning: pillow-heif not installed, HEIC files may fail")

# Get the path to osxphotos in the same environment as this script
OSXPHOTOS_PATH = Path(sys.executable).parent / "osxphotos"

# Tracking file for downsampled photos
TRACKING_FILE = ".downsampled.json"


def load_tracking(output_dir: Path) -> set:
    """Load set of already-downsampled filenames."""
    tracking_path = output_dir / TRACKING_FILE
    if tracking_path.exists():
        try:
            with open(tracking_path) as f:
                data = json.load(f)
                return set(data.get("downsampled", []))
        except Exception:
            pass
    return set()


def save_tracking(output_dir: Path, downsampled: set):
    """Save set of downsampled filenames."""
    tracking_path = output_dir / TRACKING_FILE
    with open(tracking_path, 'w') as f:
        json.dump({"downsampled": list(downsampled)}, f)


def downsample_image(image_path: Path, output_path: Path, max_size: int = 512, quality: int = 85) -> bool:
    """Downsample an image to a maximum dimension while preserving aspect ratio."""
    try:
        img = Image.open(image_path)

        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')

        width, height = img.size
        if width > height:
            if width > max_size:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_width, new_height = width, height
        else:
            if height > max_size:
                new_height = max_size
                new_width = int(width * (max_size / height))
            else:
                new_width, new_height = width, height

        if (new_width, new_height) != (width, height):
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        img.save(output_path, format='JPEG', quality=quality, optimize=True)
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


def export_and_downsample(
    output_dir: Path,
    max_size: int,
    quality: int,
    favorites_only: bool = False,
    max_photos: int = None,
    timeout: int = 300
) -> tuple[int, int, int]:
    """
    Export photos using osxphotos CLI and downsample them.

    Returns (success, skipped, failed) counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tracking of already-downsampled files
    already_downsampled = load_tracking(output_dir)

    # Build osxphotos export command
    cmd = [
        str(OSXPHOTOS_PATH), "export",
        str(output_dir),
        "--download-missing",  # Download from iCloud if needed
        "--skip-edited",
        "--skip-live",
        "--skip-raw",
        "--skip-bursts",
        "--update",  # Only export new/changed photos
        # Note: Don't use --cleanup as it removes our downsampled files
    ]

    if favorites_only:
        cmd.append("--favorite")

    if max_photos:
        cmd.extend(["--limit", str(max_photos)])

    print(f"Running: {' '.join(cmd[:5])}...")
    print()

    try:
        # Run osxphotos export
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show progress
            timeout=timeout * (max_photos or 1000),  # Scale timeout with photo count
        )

        if result.returncode != 0:
            print(f"osxphotos export failed with code {result.returncode}")
            return 0, 0, 1

    except subprocess.TimeoutExpired:
        print("Export timed out")
        return 0, 0, 1
    except Exception as e:
        print(f"Export error: {e}")
        return 0, 0, 1

    # Now downsample all exported images
    print()
    print("Downsampling exported photos...")

    success = 0
    skipped = 0
    failed = 0

    image_extensions = {'.jpg', '.jpeg', '.heic', '.png', '.tiff', '.dng', '.gif', '.bmp'}

    for f in sorted(output_dir.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() not in image_extensions:
            continue
        if f.name.startswith('.'):
            continue

        # Check if already downsampled using tracking file
        output_name = f.stem + '.jpg'
        if output_name in already_downsampled:
            skipped += 1
            continue

        # Create temp file for downsampled version
        temp_path = f.with_suffix('.tmp.jpg')

        if downsample_image(f, temp_path, max_size, quality):
            # Replace original with downsampled version
            f.unlink()
            final_path = f.with_suffix('.jpg')
            temp_path.rename(final_path)

            # Track this file as downsampled
            already_downsampled.add(final_path.name)

            success += 1
            print(f"  Downsampled: {f.name}")
        else:
            if temp_path.exists():
                temp_path.unlink()
            failed += 1

    # Save tracking file
    save_tracking(output_dir, already_downsampled)

    return success, skipped, failed


def main():
    parser = argparse.ArgumentParser(
        description="Download photos from iCloud and downsample for training"
    )
    parser.add_argument("--raw-dir", type=str, default="photos/raw",
                       help="Directory for raw photos (default: photos/raw)")
    parser.add_argument("--curated-dir", type=str, default="photos/curated",
                       help="Directory for favorited photos (default: photos/curated)")
    parser.add_argument("--favorites-only", action="store_true",
                       help="Only download favorited photos to curated dir")
    parser.add_argument("--max-size", type=int, default=512,
                       help="Maximum image dimension (default: 512)")
    parser.add_argument("--quality", type=int, default=85,
                       help="JPEG quality 1-100 (default: 85)")
    parser.add_argument("--max-photos", type=int, default=None,
                       help="Maximum number of photos to process")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Base timeout in seconds (default: 300)")
    parser.add_argument("--reset", action="store_true",
                       help="Reset tracking and re-downsample all photos")

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    curated_dir = Path(args.curated_dir)

    # Reset tracking if requested
    if args.reset:
        for d in [raw_dir, curated_dir]:
            tracking_file = d / TRACKING_FILE
            if tracking_file.exists():
                tracking_file.unlink()
                print(f"Reset tracking for {d}")

    if args.favorites_only:
        print(f"Exporting favorites to {curated_dir}")
        print("=" * 50)

        success, skipped, failed = export_and_downsample(
            output_dir=curated_dir,
            max_size=args.max_size,
            quality=args.quality,
            favorites_only=True,
            max_photos=args.max_photos,
            timeout=args.timeout
        )

        print()
        print(f"Done! Downsampled: {success}, Skipped: {skipped}, Failed: {failed}")
    else:
        # Phase 1: All photos to raw
        print(f"Phase 1: Exporting all photos to {raw_dir}")
        print("=" * 50)

        raw_success, raw_skipped, raw_failed = export_and_downsample(
            output_dir=raw_dir,
            max_size=args.max_size,
            quality=args.quality,
            favorites_only=False,
            max_photos=args.max_photos,
            timeout=args.timeout
        )

        print()
        print(f"Raw: Downsampled: {raw_success}, Skipped: {raw_skipped}, Failed: {raw_failed}")

        # Phase 2: Favorites to curated
        print()
        print(f"Phase 2: Exporting favorites to {curated_dir}")
        print("=" * 50)

        curated_success, curated_skipped, curated_failed = export_and_downsample(
            output_dir=curated_dir,
            max_size=args.max_size,
            quality=args.quality,
            favorites_only=True,
            max_photos=None,  # All favorites
            timeout=args.timeout
        )

        print()
        print(f"Curated: Downsampled: {curated_success}, Skipped: {curated_skipped}, Failed: {curated_failed}")


if __name__ == "__main__":
    main()
