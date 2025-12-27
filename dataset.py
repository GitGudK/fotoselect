"""
Dataset module for loading and preprocessing photos for curation training.
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional
import hashlib

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Supported image extensions
RAW_EXTENSIONS = {'.nef', '.cr2', '.cr3', '.arw', '.raf', '.orf', '.rw2', '.dng'}
STANDARD_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'}
ALL_EXTENSIONS = RAW_EXTENSIONS | STANDARD_EXTENSIONS


def get_image_hash(filepath: Path) -> str:
    """Generate a hash from image filename (without extension) for matching."""
    return filepath.stem.lower()


def load_image(filepath: Path) -> Optional[Image.Image]:
    """Load an image file, handling both RAW and standard formats."""
    ext = filepath.suffix.lower()

    try:
        if ext in RAW_EXTENSIONS:
            import rawpy
            with rawpy.imread(str(filepath)) as raw:
                rgb = raw.postprocess()
            return Image.fromarray(rgb)
        else:
            return Image.open(filepath).convert('RGB')
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None


def find_images(folder: Path) -> List[Path]:
    """Find all image files in a folder."""
    images = []
    for ext in ALL_EXTENSIONS:
        images.extend(folder.glob(f'*{ext}'))
        images.extend(folder.glob(f'*{ext.upper()}'))
    # Remove duplicates (can occur on case-insensitive filesystems)
    seen = set()
    unique_images = []
    for img in images:
        # Use resolved path for deduplication
        key = str(img.resolve())
        if key not in seen:
            seen.add(key)
            unique_images.append(img)
    return sorted(unique_images)


class PhotoCurationDataset(Dataset):
    """
    Dataset for photo curation training.

    Labels photos as:
    - 1 (curated): photos that appear in the curated folder
    - 0 (rejected): photos only in raw folder, not in curated
    """

    def __init__(
        self,
        raw_folder: str,
        curated_folder: str,
        transform: Optional[transforms.Compose] = None,
        image_size: int = 224
    ):
        self.raw_folder = Path(raw_folder)
        self.curated_folder = Path(curated_folder)
        self.image_size = image_size

        # Default transform for training
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

        # Build dataset
        self.samples = self._build_samples()
        print(f"Dataset: {len(self.samples)} images "
              f"({sum(1 for _, l in self.samples if l == 1)} curated, "
              f"{sum(1 for _, l in self.samples if l == 0)} rejected)")

    def _build_samples(self) -> List[Tuple[Path, int]]:
        """Build list of (image_path, label) tuples."""
        samples = []

        # Get all images from both folders
        raw_images = find_images(self.raw_folder)
        curated_images = find_images(self.curated_folder)

        # Create hash set of curated images for quick lookup
        curated_hashes = {get_image_hash(img) for img in curated_images}

        # Label raw images based on whether they're in curated set
        for img_path in raw_images:
            img_hash = get_image_hash(img_path)
            label = 1 if img_hash in curated_hashes else 0
            samples.append((img_path, label))

        # Also include curated images that might not be in raw folder
        raw_hashes = {get_image_hash(img) for img in raw_images}
        for img_path in curated_images:
            img_hash = get_image_hash(img_path)
            if img_hash not in raw_hashes:
                samples.append((img_path, 1))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        image = load_image(img_path)
        if image is None:
            # Return a black image if loading fails
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


class InferenceDataset(Dataset):
    """Dataset for inference on new images."""

    def __init__(
        self,
        folder: str,
        image_size: int = 224
    ):
        self.folder = Path(folder)
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.images = find_images(self.folder)
        print(f"Found {len(self.images)} images for inference")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path = self.images[idx]

        image = load_image(img_path)
        if image is None:
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))

        image = self.transform(image)

        return image, str(img_path)


def create_data_loaders(
    raw_folder: str,
    curated_folder: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    image_size: int = 224,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    from sklearn.model_selection import train_test_split

    # Create full dataset
    full_dataset = PhotoCurationDataset(
        raw_folder=raw_folder,
        curated_folder=curated_folder,
        image_size=image_size
    )

    # Split into train/val
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.samples[i][1] for i in indices]

    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_split,
        stratify=labels,
        random_state=42
    )

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

    return train_loader, val_loader
