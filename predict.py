"""
Inference module for predicting photo curation on new images.
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta
from collections import defaultdict
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import imagehash

from model import load_model, PhotoCurationCNN
from dataset import InferenceDataset


class PhotoCurator:
    """Class for running inference on photos to predict curation worthiness."""

    def __init__(
        self,
        checkpoint_path: str,
        backbone: str = 'resnet50',
        device: Optional[torch.device] = None,
        threshold: float = 0.5,
        top_n: Optional[int] = None,
        top_percent: Optional[float] = None,
        deduplicate: bool = False,
        similarity_threshold: float = 0.75,
        time_grouping: Optional[str] = None,
        photos_per_group: int = 1,
        last_n_days: Optional[int] = None
    ):
        """
        Initialize the curator.

        Args:
            checkpoint_path: Path to trained model checkpoint
            backbone: Model backbone architecture
            device: Device to run inference on
            threshold: Probability threshold for curation (default 0.5)
            top_n: If set, select exactly this many top-scoring photos
            top_percent: If set, select top X% of photos (0-100)
            deduplicate: If True, remove similar photos from selection
            similarity_threshold: Perceptual hash similarity threshold (0-1, default 0.75)
                Lower values = more aggressive deduplication
            time_grouping: If set, group photos by time period ('Year', 'Month', or 'Last N Days')
            photos_per_group: Number of best photos to select from each time group
            last_n_days: For 'Last N Days' mode, the number of days to look back
        """
        self.threshold = threshold
        self.top_n = top_n
        self.top_percent = top_percent
        self.deduplicate = deduplicate
        self.similarity_threshold = similarity_threshold
        self.time_grouping = time_grouping
        self.photos_per_group = photos_per_group
        self.last_n_days = last_n_days

        # Device setup
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load model
        self.model = load_model(checkpoint_path, backbone, self.device)
        print(f"Loaded model from {checkpoint_path}")

    def get_photo_date(self, image_path: str, use_mtime_fallback: bool = True) -> Optional[datetime]:
        """
        Extract date from photo EXIF data.

        Args:
            image_path: Path to the image file
            use_mtime_fallback: If True, fall back to file modification time when no EXIF date.
                               Set to False when filtering by date range to avoid incorrect dates.

        Returns datetime or None if date cannot be determined.
        """
        # Method 1: Try PIL _getexif()
        try:
            img = Image.open(image_path)
            exif = img._getexif()
            if exif:
                # Try DateTimeOriginal (36867) first, then DateTime (306)
                for tag_id in [36867, 306]:
                    if tag_id in exif:
                        date_str = exif[tag_id]
                        try:
                            return datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                        except ValueError:
                            pass
        except Exception:
            pass

        # Fall back to file modification time only if requested
        if use_mtime_fallback:
            try:
                mtime = os.path.getmtime(image_path)
                return datetime.fromtimestamp(mtime)
            except Exception:
                pass

        return None

    def get_photo_dates_batch(self, folder: Path, image_paths: List[str], progress_callback: Optional[callable] = None) -> Dict[str, Optional[datetime]]:
        """
        Extract dates from multiple photos using cached date database.

        Args:
            folder: The folder containing images (for cache location)
            image_paths: List of image file paths
            progress_callback: Optional callback(current, total, phase)

        Returns:
            Dict mapping image path to datetime (or None if no date)
        """
        from date_cache import get_photo_dates
        return get_photo_dates(folder, image_paths, progress_callback)

    def get_time_group_key(self, dt: datetime) -> str:
        """Get the grouping key for a datetime based on time_grouping setting."""
        if self.time_grouping == "Year":
            return str(dt.year)
        elif self.time_grouping == "Month":
            return f"{dt.year}-{dt.month:02d}"
        else:  # Last N Days - all photos in range go to same group
            return "recent"

    def select_by_time_groups(
        self,
        results: List[Tuple[str, float]],
        progress_callback: Optional[callable] = None,
        folder: Optional[Path] = None
    ) -> List[Tuple[str, float, bool]]:
        """
        Select best photos from each time group.

        Args:
            results: List of (path, score) tuples, sorted by score descending
            progress_callback: Optional callback for progress updates
            folder: Folder containing images (for date cache)

        Returns:
            List of (path, score, should_curate) tuples
        """
        cutoff_date = None

        if self.time_grouping == "Last N Days" and self.last_n_days:
            cutoff_date = datetime.now() - timedelta(days=self.last_n_days)

        print(f"\nGrouping photos by {self.time_grouping}...")

        # Get all image paths and determine folder
        image_paths = [path for path, score in results]
        if folder is None and image_paths:
            folder = Path(image_paths[0]).parent

        # Use batch date extraction from cache
        date_map = self.get_photo_dates_batch(folder, image_paths, progress_callback)

        # Debug: count how many dates we got
        dates_found = sum(1 for dt in date_map.values() if dt is not None)
        print(f"  Date cache returned {dates_found}/{len(image_paths)} photos with dates")

        # Build photo_dates dict with cutoff filter
        photo_dates = {}
        for path in image_paths:
            dt = date_map.get(path)
            if dt:
                # For "Last N Days", filter out photos outside the range
                if cutoff_date and dt < cutoff_date:
                    photo_dates[path] = None  # Mark as outside range
                else:
                    photo_dates[path] = dt
            else:
                photo_dates[path] = None

        # Group photos by time period
        groups = defaultdict(list)
        no_date_photos = []

        for path, score in results:
            dt = photo_dates.get(path)
            if dt is None:
                no_date_photos.append((path, score))
            else:
                key = self.get_time_group_key(dt)
                groups[key].append((path, score, dt))

        # Select best from each group (already sorted by score)
        selected_paths = set()

        for key in sorted(groups.keys(), reverse=True):  # Most recent first
            group_photos = groups[key]
            # Sort by score within group
            group_photos.sort(key=lambda x: x[1], reverse=True)

            # Take top N from this group (applying threshold if set)
            selected_from_group = 0
            for path, score, dt in group_photos[:self.photos_per_group]:
                # Apply threshold filter if threshold is not default (0.5)
                if self.threshold > 0 and score < self.threshold:
                    continue
                selected_paths.add(path)
                selected_from_group += 1

            if len(group_photos) > 0:
                print(f"  {key}: {len(group_photos)} photos, selected {selected_from_group}")

        print(f"Total selected: {len(selected_paths)} photos from {len(groups)} time groups")
        if no_date_photos:
            print(f"  ({len(no_date_photos)} photos had no date info)")

        # Print year distribution summary
        if groups:
            year_counts = defaultdict(int)
            for key in groups.keys():
                year = key[:4] if len(key) >= 4 else key
                year_counts[year] += len(groups[key])
            print("\nPhotos by year:")
            for year in sorted(year_counts.keys(), reverse=True):
                print(f"  {year}: {year_counts[year]:,} photos")

        # Build final results
        final_results = []
        for path, score in results:
            should_curate = path in selected_paths
            final_results.append((path, score, should_curate))

        return final_results

    def sample_photos_by_time_group(
        self,
        image_paths: List[str],
        folder: Path,
        samples_per_group: int = 50,
        progress_callback: Optional[callable] = None
    ) -> List[str]:
        """
        Sample photos from each time group to reduce the number to score.

        Instead of scoring all 55k photos, sample N photos per time group
        (year/month) and only score those. This dramatically speeds up
        time-based selection.

        Args:
            image_paths: List of all image paths
            folder: Folder containing images (for date cache)
            samples_per_group: How many photos to sample per time group
            progress_callback: Optional callback for progress

        Returns:
            Sampled list of image paths
        """
        import random
        from collections import defaultdict

        print(f"\nPre-sampling photos by {self.time_grouping} for faster scoring...")

        # Get dates for all photos from cache
        date_map = self.get_photo_dates_batch(folder, image_paths, progress_callback)

        # Group photos by time period
        groups = defaultdict(list)
        no_date_photos = []

        cutoff_date = None
        if self.time_grouping == "Last N Days" and self.last_n_days:
            cutoff_date = datetime.now() - timedelta(days=self.last_n_days)

        for path in image_paths:
            dt = date_map.get(path)
            if dt is None:
                no_date_photos.append(path)
            elif cutoff_date and dt < cutoff_date:
                continue  # Outside "Last N Days" range
            else:
                key = self.get_time_group_key(dt)
                groups[key].append(path)

        # Sample from each group
        sampled = []
        random.seed(42)  # For reproducibility

        for key in sorted(groups.keys(), reverse=True):
            group_photos = groups[key]
            # Sample up to samples_per_group photos from this group
            if len(group_photos) <= samples_per_group:
                sampled.extend(group_photos)
            else:
                sampled.extend(random.sample(group_photos, samples_per_group))
            print(f"  {key}: {len(group_photos)} photos -> sampled {min(len(group_photos), samples_per_group)}")

        print(f"Total sampled: {len(sampled)} photos from {len(groups)} time groups")
        print(f"  (Reduced from {len(image_paths)} photos, {len(no_date_photos)} had no date)")

        return sampled

    def filter_photos_by_pool(
        self,
        image_paths: List[str],
        folder: Optional[Path] = None,
        max_photos: Optional[int] = None,
        percentage: Optional[float] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        progress_callback: Optional[callable] = None
    ) -> List[str]:
        """
        Filter photos to create a pool for curation.

        Args:
            image_paths: List of image file paths
            folder: Folder containing images (for date cache)
            max_photos: Maximum number of photos to include
            percentage: Percentage of photos to include (0-100)
            date_from: Only include photos taken on or after this date
            date_to: Only include photos taken on or before this date
            progress_callback: Optional callback for progress updates

        Returns:
            Filtered list of image paths
        """
        import random

        filtered = list(image_paths)

        # Apply date filter first (most restrictive typically)
        # Use batch date extraction for efficiency
        if date_from or date_to:
            # Determine folder from first image path if not provided
            if folder is None and filtered:
                folder = Path(filtered[0]).parent

            print(f"Extracting dates from {len(filtered)} photos...")
            date_map = self.get_photo_dates_batch(folder, filtered, progress_callback)

            date_filtered = []
            for path in filtered:
                dt = date_map.get(path)
                if dt:
                    if date_from and dt < date_from:
                        continue
                    if date_to and dt > date_to:
                        continue
                    date_filtered.append(path)
                # Photos without readable dates are excluded when date filtering is active

            print(f"Date filter: {len(filtered)} -> {len(date_filtered)} photos")
            filtered = date_filtered

        # Apply percentage filter (random sampling)
        if percentage is not None and 0 < percentage < 100:
            num_to_keep = max(1, int(len(filtered) * percentage / 100))
            random.seed(42)  # For reproducibility
            filtered = random.sample(filtered, min(num_to_keep, len(filtered)))
            print(f"Percentage filter ({percentage}%): keeping {len(filtered)} photos")

        # Apply max_photos limit
        if max_photos is not None and max_photos < len(filtered):
            random.seed(42)
            filtered = random.sample(filtered, max_photos)
            print(f"Max photos filter: limiting to {len(filtered)} photos")

        return filtered

    @torch.no_grad()
    def predict_folder(
        self,
        input_folder: str,
        batch_size: int = 16,
        image_size: int = 224,
        progress_callback: Optional[callable] = None,
        max_photos: Optional[int] = None,
        percentage: Optional[float] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        image_paths: Optional[List[str]] = None
    ) -> List[Tuple[str, float, bool]]:
        """
        Predict curation scores for all images in a folder.

        Args:
            input_folder: Path to folder containing images
            batch_size: Batch size for inference
            image_size: Image size for processing
            progress_callback: Optional callback(current, total, phase) for progress updates
                phase is 'scoring', 'filtering', 'features', 'dates', or 'dedup'
            max_photos: Maximum number of photos to consider (photo pool filter)
            percentage: Percentage of photos to consider (photo pool filter, 0-100)
            date_from: Only consider photos taken on or after this date
            date_to: Only consider photos taken on or before this date
            image_paths: Pre-filtered list of image paths to use (bypasses pool filtering)

        Returns:
            List of (image_path, score, should_curate) tuples
        """
        from dataset import find_images

        # If pre-filtered paths provided, use them directly
        if image_paths is not None:
            print(f"Using pre-filtered photo pool: {len(image_paths)} photos")
            dataset = InferenceDataset(input_folder, image_size=image_size, image_paths=image_paths)
        else:
            # Check if we need to filter the photo pool
            has_pool_filter = any([
                max_photos is not None,
                percentage is not None,
                date_from is not None,
                date_to is not None
            ])

            if has_pool_filter:
                # Get all images first, then filter
                all_images = find_images(Path(input_folder))
                all_image_paths = [str(p) for p in all_images]

                print(f"\nFiltering photo pool from {len(all_image_paths)} photos...")
                filtered_paths = self.filter_photos_by_pool(
                    all_image_paths,
                    folder=Path(input_folder),
                    max_photos=max_photos,
                    percentage=percentage,
                    date_from=date_from,
                    date_to=date_to,
                    progress_callback=progress_callback
                )
                print(f"Photo pool: {len(filtered_paths)} photos")

                dataset = InferenceDataset(input_folder, image_size=image_size, image_paths=filtered_paths)
            else:
                dataset = InferenceDataset(input_folder, image_size=image_size)

        if len(dataset) == 0:
            print("No images found in folder (or all filtered out)")
            return []

        # Check score cache for already-scored photos
        from score_cache import get_cached_scores, update_scores

        all_paths = [str(p) for p in dataset.images]
        all_filenames = [os.path.basename(p) for p in all_paths]
        cached = get_cached_scores(Path(input_folder), all_filenames)

        # Separate cached and uncached photos
        cached_results = []
        needs_scoring = []

        for path, filename in zip(all_paths, all_filenames):
            if cached.get(filename) is not None:
                cached_results.append((path, cached[filename]))
            else:
                needs_scoring.append(path)

        print(f"Score cache: {len(cached_results)} cached, {len(needs_scoring)} need scoring")

        results = list(cached_results)

        # Only score uncached photos
        if needs_scoring:
            # Create a dataset with only uncached photos
            uncached_dataset = InferenceDataset(input_folder, image_size=image_size, image_paths=needs_scoring)
            loader = DataLoader(
                uncached_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            new_scores = {}
            total_batches = len(loader)
            processed = 0

            for images, paths in tqdm(loader, desc='Scoring'):
                images = images.to(self.device)
                scores = self.model.predict_proba(images)

                for path, score in zip(paths, scores.cpu().numpy()):
                    score_val = float(score)
                    results.append((path, score_val))
                    new_scores[os.path.basename(path)] = score_val

                processed += 1
                if progress_callback:
                    progress_callback(processed, total_batches, 'scoring')

            # Save new scores to cache
            if new_scores:
                update_scores(Path(input_folder), new_scores)
                print(f"Cached {len(new_scores)} new scores")
        elif progress_callback:
            # All from cache - still call progress to update UI
            progress_callback(1, 1, 'scoring')

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Determine which photos to curate based on selection mode
        if self.time_grouping is not None:
            # Time-based selection: pick best from each time group
            final_results = self.select_by_time_groups(results, progress_callback, folder=Path(input_folder))
        else:
            # Standard selection modes
            total = len(results)
            if self.top_n is not None:
                # Select exactly top_n photos
                num_to_select = min(self.top_n, total)
            elif self.top_percent is not None:
                # Select top X% of photos
                num_to_select = max(1, int(total * self.top_percent / 100))
            else:
                # Use threshold mode
                num_to_select = None

            # Add should_curate flag
            final_results = []
            for i, (path, score) in enumerate(results):
                if num_to_select is not None:
                    should_curate = i < num_to_select
                else:
                    should_curate = score >= self.threshold
                final_results.append((path, score, bool(should_curate)))

        # Apply deduplication if enabled
        if self.deduplicate:
            # Get all image paths for hashing
            all_paths = [p for p, _, _ in final_results]
            hashes = self.compute_perceptual_hashes(all_paths, progress_callback, folder=Path(input_folder))
            final_results = self.deduplicate_selection(final_results, hashes, progress_callback)

        return final_results

    def curate_folder(
        self,
        input_folder: str,
        output_folder: str,
        batch_size: int = 16,
        image_size: int = 224,
        copy_files: bool = True,
        create_rejected_folder: bool = False,
        results: Optional[List[Tuple[str, float, bool]]] = None
    ) -> Tuple[int, int]:
        """
        Automatically curate photos from input folder to output folder.

        Args:
            input_folder: Path to folder containing images to curate
            output_folder: Path to folder where curated images will be saved
            batch_size: Batch size for inference
            image_size: Image size for processing
            copy_files: If True, copy files; if False, move files
            create_rejected_folder: If True, also create a rejected folder
            results: Pre-computed results from predict_folder (optional)

        Returns:
            Tuple of (num_curated, num_rejected)
        """
        if results is None:
            results = self.predict_folder(input_folder, batch_size, image_size)

        if not results:
            return 0, 0

        # Create output directories
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        rejected_path = None
        if create_rejected_folder:
            rejected_path = output_path.parent / f"{output_path.name}_rejected"
            rejected_path.mkdir(parents=True, exist_ok=True)

        num_curated = 0
        num_rejected = 0

        for img_path, score, should_curate in tqdm(results, desc='Organizing'):
            src_path = Path(img_path)

            if should_curate:
                dst_path = output_path / src_path.name
                if copy_files:
                    shutil.copy2(src_path, dst_path)
                else:
                    shutil.move(src_path, dst_path)
                num_curated += 1
            else:
                num_rejected += 1
                if rejected_path:
                    dst_path = rejected_path / src_path.name
                    if copy_files:
                        shutil.copy2(src_path, dst_path)
                    else:
                        shutil.move(src_path, dst_path)

        print(f"\nCuration complete!")
        print(f"Curated: {num_curated} images -> {output_folder}")
        print(f"Rejected: {num_rejected} images")
        if rejected_path:
            print(f"Rejected folder: {rejected_path}")

        return num_curated, num_rejected

    def export_results(
        self,
        results: List[Tuple[str, float, bool]],
        output_file: str
    ):
        """Export prediction results to JSON file."""
        output = []
        for path, score, should_curate in results:
            output.append({
                'path': path,
                'score': score,
                'curated': should_curate
            })

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Results exported to {output_file}")

    def compute_perceptual_hashes(
        self,
        image_paths: List[str],
        progress_callback: Optional[callable] = None,
        folder: Optional[Path] = None
    ) -> Dict[str, imagehash.ImageHash]:
        """
        Compute perceptual hashes for images using multiple hash types.

        Uses a combination of pHash (perceptual hash) and dHash (difference hash)
        for robust similarity detection. Results are cached to avoid recomputation.

        Args:
            image_paths: List of image file paths
            progress_callback: Optional callback(current, total, phase) for progress updates
            folder: Folder containing images (for cache location)

        Returns:
            Dictionary mapping image paths to perceptual hash objects
        """
        from hash_cache import get_cached_hashes, update_hashes

        # Determine folder from first image if not provided
        if folder is None and image_paths:
            folder = Path(image_paths[0]).parent

        # Check cache for already-computed hashes
        all_filenames = [os.path.basename(p) for p in image_paths]
        cached = get_cached_hashes(folder, all_filenames) if folder else {}

        hashes = {}
        needs_hashing = []

        # Separate cached and uncached
        for path in image_paths:
            filename = os.path.basename(path)
            cached_hash = cached.get(filename)
            if cached_hash is not None:
                # Convert stored hex string back to ImageHash
                hashes[path] = imagehash.hex_to_hash(cached_hash)
            else:
                needs_hashing.append(path)

        print(f"Hash cache: {len(hashes)} cached, {len(needs_hashing)} need hashing")

        if not needs_hashing:
            if progress_callback:
                progress_callback(1, 1, 'features')
            return hashes

        # Compute hashes for uncached photos
        total = len(needs_hashing)
        new_hashes = {}

        for i, path in enumerate(tqdm(needs_hashing, desc='Computing hashes')):
            try:
                img = Image.open(path)
                # Use pHash which is good for detecting similar images
                phash = imagehash.phash(img, hash_size=16)
                hashes[path] = phash
                # Store as hex string for JSON serialization
                new_hashes[os.path.basename(path)] = str(phash)
            except Exception as e:
                print(f"  Error hashing {Path(path).name}: {e}")

            if progress_callback:
                progress_callback(i + 1, total, 'features')

        # Save new hashes to cache
        if new_hashes and folder:
            update_hashes(folder, new_hashes)
            print(f"Cached {len(new_hashes)} new hashes")

        return hashes

    def compute_hash_distance(self, hash1: imagehash.ImageHash, hash2: imagehash.ImageHash) -> int:
        """
        Compute Hamming distance between two perceptual hashes.

        Returns:
            Hamming distance (0 = identical, higher = more different)
            For 16x16 pHash, max distance is 256.
        """
        return hash1 - hash2

    def deduplicate_selection(
        self,
        results: List[Tuple[str, float, bool]],
        hashes: Dict[str, imagehash.ImageHash],
        progress_callback: Optional[callable] = None
    ) -> List[Tuple[str, float, bool]]:
        """
        Remove similar photos from the selection and replace with alternatives.

        Uses perceptual hashing to detect visually similar images.
        For each pair of similar selected photos, keeps the higher-scoring one
        and replaces the other with the next best non-similar alternative.

        Args:
            results: List of (path, score, should_curate) tuples, sorted by score
            hashes: Dictionary of path -> perceptual hash
            progress_callback: Optional callback(current, total, phase) for progress updates

        Returns:
            Updated results with deduplicated selection
        """
        # Split into selected and candidates
        selected = [(p, s, c) for p, s, c in results if c]
        candidates = [(p, s, c) for p, s, c in results if not c]

        if len(selected) <= 1:
            return results

        # Convert similarity_threshold (0-1) to hash distance threshold
        # For 16x16 pHash (256 bits), we use a more aggressive formula
        # Lower similarity_threshold = more aggressive dedup (higher distance threshold)
        # similarity 0.70 -> distance ~100, similarity 0.85 -> distance ~50, similarity 0.95 -> distance ~20
        # Scale factor of 320 makes the threshold more aggressive for catching similar-looking photos
        max_distance = int((1 - self.similarity_threshold) * 320)
        max_distance = max(10, min(max_distance, 120))  # Clamp between 10 and 120

        print(f"\nDeduplicating {len(selected)} selected photos (max hash distance: {max_distance})...")
        total_to_check = len(selected)

        # Track which photos are in the final selection
        final_selected = []
        removed_count = 0

        for i, (path, score, _) in enumerate(selected):
            if progress_callback:
                progress_callback(i + 1, total_to_check, 'dedup')

            if path not in hashes:
                final_selected.append((path, score, True))
                continue

            current_hash = hashes[path]

            # Check similarity with already-selected photos
            is_duplicate = False
            for prev_path, _, _ in final_selected:
                if prev_path in hashes:
                    distance = self.compute_hash_distance(current_hash, hashes[prev_path])
                    if distance <= max_distance:
                        is_duplicate = True
                        print(f"  Duplicate: {Path(path).name} similar to {Path(prev_path).name} (distance: {distance})")
                        break

            if not is_duplicate:
                final_selected.append((path, score, True))
            else:
                removed_count += 1
                # Only try to find a replacement if NOT using time-based grouping
                # (time-based grouping has strict per-period counts that shouldn't be exceeded)
                if self.time_grouping is None:
                    for j, (cand_path, cand_score, _) in enumerate(candidates):
                        if cand_path not in hashes:
                            continue

                        cand_hash = hashes[cand_path]

                        # Check if candidate is similar to any already-selected photo
                        cand_is_dup = False
                        for prev_path, _, _ in final_selected:
                            if prev_path in hashes:
                                distance = self.compute_hash_distance(cand_hash, hashes[prev_path])
                                if distance <= max_distance:
                                    cand_is_dup = True
                                    break

                        if not cand_is_dup:
                            # Found a good replacement
                            print(f"  Replaced with: {Path(cand_path).name} (score: {cand_score:.3f})")
                            final_selected.append((cand_path, cand_score, True))
                            # Remove from candidates
                            candidates = candidates[:j] + candidates[j+1:]
                            break

        print(f"Deduplication complete: removed {removed_count} duplicates, final selection: {len(final_selected)}")

        # Rebuild full results list
        selected_paths = {p for p, _, _ in final_selected}
        new_results = []

        # Add selected photos first (sorted by score)
        for path, score, curate in sorted(final_selected, key=lambda x: x[1], reverse=True):
            new_results.append((path, score, True))

        # Add remaining photos as not-curated
        for path, score, _ in results:
            if path not in selected_paths:
                new_results.append((path, score, False))

        return new_results


def predict_photos(
    checkpoint_path: str,
    input_folder: str,
    output_folder: Optional[str] = None,
    threshold: float = 0.5,
    top_n: Optional[int] = None,
    top_percent: Optional[float] = None,
    backbone: str = 'resnet50',
    batch_size: int = 16,
    copy_files: bool = True,
    export_json: Optional[str] = None,
    deduplicate: bool = False,
    similarity_threshold: float = 0.75
) -> List[Tuple[str, float, bool]]:
    """
    Main prediction function.

    Args:
        checkpoint_path: Path to trained model checkpoint
        input_folder: Path to folder containing images
        output_folder: If provided, curated images will be copied/moved here
        threshold: Probability threshold for curation
        top_n: If set, select exactly this many top-scoring photos
        top_percent: If set, select top X% of photos (0-100)
        backbone: Model backbone architecture
        batch_size: Batch size for inference
        copy_files: If True, copy files; if False, move files
        export_json: If provided, export results to this JSON file
        deduplicate: If True, remove similar photos from selection
        similarity_threshold: Cosine similarity threshold (0-1) for detecting duplicates

    Returns:
        List of (image_path, score, should_curate) tuples
    """
    curator = PhotoCurator(
        checkpoint_path=checkpoint_path,
        backbone=backbone,
        threshold=threshold,
        top_n=top_n,
        top_percent=top_percent,
        deduplicate=deduplicate,
        similarity_threshold=similarity_threshold
    )

    results = curator.predict_folder(input_folder, batch_size)

    # Print summary
    curated = sum(1 for _, _, c in results if c)
    rejected = len(results) - curated

    print(f"\n{'='*50}")
    print(f"Total images: {len(results)}")
    print(f"Recommended for curation: {curated} ({curated/len(results)*100:.1f}%)")
    print(f"Recommended for rejection: {rejected} ({rejected/len(results)*100:.1f}%)")
    print(f"{'='*50}")

    # Print top 10 curated
    print("\nTop 10 recommended photos:")
    for path, score, _ in results[:10]:
        print(f"  {score:.3f}: {Path(path).name}")

    # Export to JSON if requested
    if export_json:
        curator.export_results(results, export_json)

    # Curate to output folder if requested
    if output_folder:
        curator.curate_folder(
            input_folder=input_folder,
            output_folder=output_folder,
            copy_files=copy_files
        )

    return results
