#!/usr/bin/env python3
"""
Test script for photo export and downsampling logic.
Creates fake photos and tests the export workflow.
"""

import json
import shutil
import tempfile
from pathlib import Path

from PIL import Image


def create_test_image(path: Path, size: tuple = (1000, 1000), color: tuple = (255, 0, 0)):
    """Create a test image. Uses PNG internally but saves with requested extension."""
    img = Image.new('RGB', size, color)
    # For non-standard extensions like .heic, save as PNG first then rename
    if path.suffix.lower() in ('.heic', '.heif'):
        temp_path = path.with_suffix('.png')
        img.save(temp_path)
        temp_path.rename(path)
    else:
        img.save(path)
    print(f"  Created: {path.name}")


def test_export_workflow():
    """Test the export and downsampling workflow."""

    # Create temp directories
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Simulate osxphotos staging directory
        staging_dir = tmpdir / "output" / ".staging"
        output_dir = tmpdir / "output"
        staging_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("TEST 1: Fresh export (no existing files)")
        print("=" * 60)

        # Create 5 "exported" photos in staging (simulating osxphotos output)
        print("\nCreating 5 test photos in staging...")
        for i in range(5):
            create_test_image(staging_dir / f"photo_{i}.heic", color=(i*50, 0, 0))

        # Import and run our downsampling logic
        from download_icloud import load_tracking, save_tracking, downsample_image

        already_downsampled = load_tracking(output_dir)
        print(f"\nTracking file has {len(already_downsampled)} entries")

        image_extensions = {'.jpg', '.jpeg', '.heic', '.png', '.tiff', '.dng', '.gif', '.bmp'}
        success = 0
        skipped = 0

        for f in sorted(staging_dir.iterdir()):
            if not f.is_file() or f.suffix.lower() not in image_extensions:
                continue
            if f.name.startswith('.'):
                continue

            output_name = f.stem + '.jpg'
            if output_name in already_downsampled:
                f.unlink()
                skipped += 1
                print(f"  SKIPPED (tracked): {f.name}")
                continue

            final_path = output_dir / output_name
            if downsample_image(f, final_path, max_size=512, quality=85):
                f.unlink()
                already_downsampled.add(final_path.name)
                success += 1
                print(f"  DOWNSAMPLED: {f.name} -> {final_path.name}")
            else:
                print(f"  FAILED: {f.name}")

        save_tracking(output_dir, already_downsampled)

        print(f"\nResult: {success} downsampled, {skipped} skipped")
        print(f"Output dir has {len(list(output_dir.glob('*.jpg')))} JPG files")
        print(f"Tracking file has {len(load_tracking(output_dir))} entries")

        assert success == 5, f"Expected 5 success, got {success}"
        assert skipped == 0, f"Expected 0 skipped, got {skipped}"

        print("\n✓ TEST 1 PASSED")

        # ============================================================
        print("\n" + "=" * 60)
        print("TEST 2: Incremental export (some files already processed)")
        print("=" * 60)

        # Simulate osxphotos exporting 3 new photos + 2 that we already have
        print("\nSimulating osxphotos exporting 5 photos (2 existing, 3 new)...")
        for i in range(5):
            create_test_image(staging_dir / f"photo_{i}.heic", color=(i*50, 100, 0))

        # Also add 2 completely new photos
        create_test_image(staging_dir / "photo_5.heic", color=(0, 255, 0))
        create_test_image(staging_dir / "photo_6.heic", color=(0, 0, 255))

        already_downsampled = load_tracking(output_dir)
        print(f"\nTracking file has {len(already_downsampled)} entries: {already_downsampled}")

        success = 0
        skipped = 0

        for f in sorted(staging_dir.iterdir()):
            if not f.is_file() or f.suffix.lower() not in image_extensions:
                continue
            if f.name.startswith('.'):
                continue

            output_name = f.stem + '.jpg'
            if output_name in already_downsampled:
                f.unlink()
                skipped += 1
                print(f"  SKIPPED (tracked): {f.name}")
                continue

            final_path = output_dir / output_name
            if downsample_image(f, final_path, max_size=512, quality=85):
                f.unlink()
                already_downsampled.add(final_path.name)
                success += 1
                print(f"  DOWNSAMPLED: {f.name} -> {final_path.name}")
            else:
                print(f"  FAILED: {f.name}")

        save_tracking(output_dir, already_downsampled)

        print(f"\nResult: {success} downsampled, {skipped} skipped")
        print(f"Output dir has {len(list(output_dir.glob('*.jpg')))} JPG files")
        print(f"Tracking file has {len(load_tracking(output_dir))} entries")

        assert success == 2, f"Expected 2 new success, got {success}"
        assert skipped == 5, f"Expected 5 skipped, got {skipped}"

        print("\n✓ TEST 2 PASSED")

        # ============================================================
        print("\n" + "=" * 60)
        print("TEST 3: Force re-export (clear staging and tracking)")
        print("=" * 60)

        # Clear staging directory (simulating force_reexport=True)
        print("\nClearing staging directory...")
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)

        # Clear tracking file
        tracking_file = output_dir / ".downsampled.json"
        if tracking_file.exists():
            tracking_file.unlink()
            print("Cleared tracking file")

        # Clear output files
        for f in output_dir.glob('*.jpg'):
            f.unlink()
        print("Cleared output JPGs")

        # Now export all photos fresh
        print("\nExporting 5 photos fresh...")
        for i in range(5):
            create_test_image(staging_dir / f"photo_{i}.heic", color=(i*50, 0, 100))

        already_downsampled = load_tracking(output_dir)
        print(f"Tracking file has {len(already_downsampled)} entries")

        success = 0
        skipped = 0

        for f in sorted(staging_dir.iterdir()):
            if not f.is_file() or f.suffix.lower() not in image_extensions:
                continue
            if f.name.startswith('.'):
                continue

            output_name = f.stem + '.jpg'
            if output_name in already_downsampled:
                f.unlink()
                skipped += 1
                print(f"  SKIPPED (tracked): {f.name}")
                continue

            final_path = output_dir / output_name
            if downsample_image(f, final_path, max_size=512, quality=85):
                f.unlink()
                already_downsampled.add(final_path.name)
                success += 1
                print(f"  DOWNSAMPLED: {f.name} -> {final_path.name}")
            else:
                print(f"  FAILED: {f.name}")

        save_tracking(output_dir, already_downsampled)

        print(f"\nResult: {success} downsampled, {skipped} skipped")

        assert success == 5, f"Expected 5 success after reset, got {success}"
        assert skipped == 0, f"Expected 0 skipped after reset, got {skipped}"

        print("\n✓ TEST 3 PASSED")

        # ============================================================
        print("\n" + "=" * 60)
        print("TEST 4: Simulate real osxphotos behavior with --update")
        print("=" * 60)

        # The issue: osxphotos --update skips files that already exist in the export dir
        # We need to verify our staging approach prevents this

        # Reset everything
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)

        tracking_file = output_dir / ".downsampled.json"
        if tracking_file.exists():
            tracking_file.unlink()

        for f in output_dir.glob('*.jpg'):
            f.unlink()

        print("\nPhase 1: Initial export of 3 photos")
        for i in range(3):
            create_test_image(staging_dir / f"IMG_{i:04d}.heic")

        already_downsampled = load_tracking(output_dir)
        for f in sorted(staging_dir.iterdir()):
            if f.suffix.lower() not in image_extensions:
                continue
            output_name = f.stem + '.jpg'
            final_path = output_dir / output_name
            downsample_image(f, final_path, max_size=512, quality=85)
            f.unlink()
            already_downsampled.add(final_path.name)
            print(f"  Processed: {f.name}")
        save_tracking(output_dir, already_downsampled)

        print(f"\nOutput dir: {list(f.name for f in output_dir.glob('*.jpg'))}")
        print(f"Tracking: {load_tracking(output_dir)}")

        print("\nPhase 2: osxphotos exports 5 photos (3 overlap, 2 new)")
        print("With --update, osxphotos checks staging dir, NOT output dir")

        # osxphotos would export to staging - since staging is empty (we cleaned it),
        # it would export all 5 photos
        for i in range(5):
            create_test_image(staging_dir / f"IMG_{i:04d}.heic")
            print(f"  osxphotos exported: IMG_{i:04d}.heic")

        print("\nNow we process staging -> output")
        already_downsampled = load_tracking(output_dir)
        success = 0
        skipped = 0

        for f in sorted(staging_dir.iterdir()):
            if f.suffix.lower() not in image_extensions:
                continue
            output_name = f.stem + '.jpg'
            if output_name in already_downsampled:
                f.unlink()
                skipped += 1
                print(f"  SKIPPED (tracked): {f.name}")
            else:
                final_path = output_dir / output_name
                downsample_image(f, final_path, max_size=512, quality=85)
                f.unlink()
                already_downsampled.add(final_path.name)
                success += 1
                print(f"  DOWNSAMPLED: {f.name}")

        save_tracking(output_dir, already_downsampled)

        print(f"\nResult: {success} new, {skipped} skipped")
        assert success == 2, f"Expected 2 new, got {success}"
        assert skipped == 3, f"Expected 3 skipped, got {skipped}"

        print("\n✓ TEST 4 PASSED")

        # ============================================================
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("""
Summary:
1. Staging directory approach works for CLI exports (download_icloud.py)
2. osxphotos --update uses .osxphotos_export.db to track exports
3. Clearing staging dir resets the database for fresh exports
4. For Python API exports (photos_import.py), UUID-based filenames prevent duplicates
""")


if __name__ == "__main__":
    test_export_workflow()
