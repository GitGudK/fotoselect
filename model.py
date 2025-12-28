"""
CNN Model for photo curation prediction.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class PhotoCurationCNN(nn.Module):
    """
    CNN model for binary classification of photos (curate vs reject).

    Uses transfer learning with a pretrained backbone and custom classifier head.
    """

    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()

        self.backbone_name = backbone

        # Load pretrained backbone
        if backbone == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        elif backbone == 'efficientnet_b2':
            weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b2(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        elif backbone == 'mobilenet_v3':
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_large(weights=weights)
            num_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(128, 1)
        )

        # Initialize classifier weights
        self._init_classifier()

    def _init_classifier(self):
        """Initialize classifier weights with Xavier initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits.squeeze(-1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings from the backbone (before classifier)."""
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability of being curated (class 1)."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning only the classifier."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for full training."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Get number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_model(
    backbone: str = 'resnet50',
    pretrained: bool = True,
    dropout: float = 0.5,
    freeze_backbone: bool = False
) -> PhotoCurationCNN:
    """Factory function to create a model."""
    model = PhotoCurationCNN(
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout
    )

    if freeze_backbone:
        model.freeze_backbone()

    return model


def load_model(
    checkpoint_path: str,
    backbone: str = 'resnet50',
    device: Optional[torch.device] = None
) -> PhotoCurationCNN:
    """Load a trained model from checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get backbone from checkpoint if available
    if 'backbone' in checkpoint:
        backbone = checkpoint['backbone']

    model = PhotoCurationCNN(backbone=backbone, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model
