"""
Inference module for predicting photo curation on new images.
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict
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
        similarity_threshold: float = 0.75
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
        """
        self.threshold = threshold
        self.top_n = top_n
        self.top_percent = top_percent
        self.deduplicate = deduplicate
        self.similarity_threshold = similarity_threshold

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

    @torch.no_grad()
    def predict_folder(
        self,
        input_folder: str,
        batch_size: int = 16,
        image_size: int = 224,
        progress_callback: Optional[callable] = None
    ) -> List[Tuple[str, float, bool]]:
        """
        Predict curation scores for all images in a folder.

        Args:
            input_folder: Path to folder containing images
            batch_size: Batch size for inference
            image_size: Image size for processing
            progress_callback: Optional callback(current, total, phase) for progress updates
                phase is 'scoring', 'features', or 'dedup'

        Returns:
            List of (image_path, score, should_curate) tuples
        """
        dataset = InferenceDataset(input_folder, image_size=image_size)

        if len(dataset) == 0:
            print("No images found in folder")
            return []

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        results = []
        total_batches = len(loader)
        processed = 0

        for images, paths in tqdm(loader, desc='Predicting'):
            images = images.to(self.device)
            scores = self.model.predict_proba(images)

            for path, score in zip(paths, scores.cpu().numpy()):
                results.append((path, float(score)))

            processed += 1
            if progress_callback:
                progress_callback(processed, total_batches, 'scoring')

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Determine which photos to curate based on selection mode
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
            hashes = self.compute_perceptual_hashes(all_paths, progress_callback)
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
        progress_callback: Optional[callable] = None
    ) -> Dict[str, imagehash.ImageHash]:
        """
        Compute perceptual hashes for images using multiple hash types.

        Uses a combination of pHash (perceptual hash) and dHash (difference hash)
        for robust similarity detection.

        Args:
            image_paths: List of image file paths
            progress_callback: Optional callback(current, total, phase) for progress updates

        Returns:
            Dictionary mapping image paths to perceptual hash objects
        """
        hashes = {}
        total = len(image_paths)

        for i, path in enumerate(tqdm(image_paths, desc='Computing hashes')):
            try:
                img = Image.open(path)
                # Use pHash which is good for detecting similar images
                phash = imagehash.phash(img, hash_size=16)
                hashes[path] = phash
            except Exception as e:
                print(f"  Error hashing {Path(path).name}: {e}")

            if progress_callback:
                progress_callback(i + 1, total, 'features')

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
                # Try to find a replacement from candidates
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
