"""
Inference module for predicting photo curation on new images.
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import load_model, PhotoCurationCNN
from dataset import InferenceDataset


class PhotoCurator:
    """Class for running inference on photos to predict curation worthiness."""

    def __init__(
        self,
        checkpoint_path: str,
        backbone: str = 'resnet50',
        device: Optional[torch.device] = None,
        threshold: float = 0.5
    ):
        """
        Initialize the curator.

        Args:
            checkpoint_path: Path to trained model checkpoint
            backbone: Model backbone architecture
            device: Device to run inference on
            threshold: Probability threshold for curation (default 0.5)
        """
        self.threshold = threshold

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
        image_size: int = 224
    ) -> List[Tuple[str, float, bool]]:
        """
        Predict curation scores for all images in a folder.

        Args:
            input_folder: Path to folder containing images
            batch_size: Batch size for inference
            image_size: Image size for processing

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

        for images, paths in tqdm(loader, desc='Predicting'):
            images = images.to(self.device)
            scores = self.model.predict_proba(images)

            for path, score in zip(paths, scores.cpu().numpy()):
                should_curate = score >= self.threshold
                results.append((path, float(score), bool(should_curate)))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def curate_folder(
        self,
        input_folder: str,
        output_folder: str,
        batch_size: int = 16,
        image_size: int = 224,
        copy_files: bool = True,
        create_rejected_folder: bool = False
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

        Returns:
            Tuple of (num_curated, num_rejected)
        """
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


def predict_photos(
    checkpoint_path: str,
    input_folder: str,
    output_folder: Optional[str] = None,
    threshold: float = 0.5,
    backbone: str = 'resnet50',
    batch_size: int = 16,
    copy_files: bool = True,
    export_json: Optional[str] = None
) -> List[Tuple[str, float, bool]]:
    """
    Main prediction function.

    Args:
        checkpoint_path: Path to trained model checkpoint
        input_folder: Path to folder containing images
        output_folder: If provided, curated images will be copied/moved here
        threshold: Probability threshold for curation
        backbone: Model backbone architecture
        batch_size: Batch size for inference
        copy_files: If True, copy files; if False, move files
        export_json: If provided, export results to this JSON file

    Returns:
        List of (image_path, score, should_curate) tuples
    """
    curator = PhotoCurator(
        checkpoint_path=checkpoint_path,
        backbone=backbone,
        threshold=threshold
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
