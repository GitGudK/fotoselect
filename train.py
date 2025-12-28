"""
Training pipeline for photo curation model.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import PhotoCurationCNN, create_model
from dataset import create_data_loaders


class Trainer:
    """Training class for photo curation model."""

    def __init__(
        self,
        model: PhotoCurationCNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
        output_dir: str = 'checkpoints'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
        self.model.to(self.device)

        # Loss function with class weighting for imbalanced data
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

    def train_epoch(
        self,
        epoch: int = 1,
        total_epochs: int = 1,
        progress_callback=None
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        total_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar, 1):
            images = images.to(self.device)
            labels = labels.float().to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Statistics
            total_loss += loss.item() * images.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })

            # Call progress callback for batch-level updates
            if progress_callback:
                progress_callback(
                    epoch=epoch,
                    total_epochs=total_epochs,
                    batch=batch_idx,
                    total_batches=total_batches,
                    phase='train',
                    batch_loss=loss.item(),
                    batch_acc=correct/total
                )

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(
        self,
        epoch: int = 1,
        total_epochs: int = 1,
        progress_callback=None
    ) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        total_batches = len(self.val_loader)

        for batch_idx, (images, labels) in enumerate(tqdm(self.val_loader, desc='Validating'), 1):
            images = images.to(self.device)
            labels = labels.float().to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Call progress callback for batch-level updates
            if progress_callback:
                progress_callback(
                    epoch=epoch,
                    total_epochs=total_epochs,
                    batch=batch_idx,
                    total_batches=total_batches,
                    phase='val',
                    batch_loss=loss.item(),
                    batch_acc=correct/total
                )

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'backbone': self.model.backbone_name,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }

        # Save latest checkpoint
        latest_path = self.output_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_acc: {self.best_val_acc:.4f}")

    def train(
        self,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        progress_callback=None
    ) -> Dict:
        """Full training loop.

        Args:
            epochs: Number of training epochs
            early_stopping_patience: Epochs to wait before early stopping
            progress_callback: Optional callback function with kwargs:
                - epoch: current epoch (1-indexed)
                - total_epochs: total epochs
                - batch: current batch in epoch (1-indexed)
                - total_batches: total batches per epoch
                - phase: 'train' or 'val'
                - train_loss, val_loss, train_acc, val_acc: metrics (available after each phase)
        """
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Model parameters: {self.model.get_num_params():,} trainable\n")

        patience_counter = 0
        total_train_batches = len(self.train_loader)
        total_val_batches = len(self.val_loader)

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 40)

            # Train with batch-level progress
            train_loss, train_acc = self.train_epoch(
                epoch=epoch,
                total_epochs=epochs,
                progress_callback=progress_callback
            )

            # Validate with batch-level progress
            val_loss, val_acc = self.validate(
                epoch=epoch,
                total_epochs=epochs,
                progress_callback=progress_callback
            )

            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")

            # Check for improvement
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Call progress callback with epoch-end summary
            if progress_callback:
                progress_callback(
                    epoch=epoch,
                    total_epochs=epochs,
                    batch=None,  # Indicates epoch end
                    total_batches=None,
                    phase='epoch_end',
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_acc=train_acc,
                    val_acc=val_acc
                )

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

        print(f"\nTraining complete!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")

        # Save training history
        history_path = self.output_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        return self.history


def train_model(
    raw_folder: str,
    curated_folder: str,
    output_dir: str = 'checkpoints',
    backbone: str = 'resnet50',
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    image_size: int = 224,
    freeze_backbone: bool = False,
    early_stopping_patience: int = 10,
    progress_callback=None
) -> Dict:
    """
    Main training function.

    Args:
        raw_folder: Path to folder containing all raw photos
        curated_folder: Path to folder containing curated/selected photos
        output_dir: Directory to save checkpoints
        backbone: Model backbone (resnet18, resnet50, efficientnet_b0, etc.)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        image_size: Image size for training
        freeze_backbone: Whether to freeze backbone weights
        early_stopping_patience: Epochs to wait before early stopping
        progress_callback: Optional callback function for progress updates

    Returns:
        Training history dictionary
    """
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        raw_folder=raw_folder,
        curated_folder=curated_folder,
        batch_size=batch_size,
        image_size=image_size
    )

    # Create model
    model = create_model(
        backbone=backbone,
        pretrained=True,
        freeze_backbone=freeze_backbone
    )

    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        output_dir=output_dir
    )

    history = trainer.train(
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        progress_callback=progress_callback
    )

    return history
