# FotoSelect

AI-powered photo curation tool that learns your preferences and automatically selects your best photos.

## How It Works

1. **Train**: The model learns from your past curation decisions by comparing a folder of raw photos against a folder of photos you've selected as "keepers"
2. **Predict**: Apply your learned preferences to new batches of photos automatically

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

Place your photos in the following folders:
- `photos/raw/` - All your original photos
- `photos/curated/` - Only the photos you selected as good (filenames should match those in raw)

Then train:

```bash
python main.py train --raw-folder ./photos/raw --curated-folder ./photos/curated
```

**Training options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--backbone` | resnet50 | Model architecture (resnet18, resnet50, efficientnet_b0, efficientnet_b2, mobilenet_v3) |
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 32 | Batch size |
| `--learning-rate` | 0.0001 | Initial learning rate |
| `--freeze-backbone` | false | Only train classifier (faster, for small datasets) |
| `--patience` | 10 | Early stopping patience |

### Prediction

Place new photos in `photos/input/`, then run:

```bash
python main.py predict \
  --input-folder ./photos/input \
  --checkpoint ./checkpoints/best.pt \
  --output-folder ./photos/output
```

**Prediction options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--threshold` | 0.5 | Curation threshold (higher = more selective) |
| `--move` | false | Move files instead of copying |
| `--export-json` | - | Export detailed results to JSON |

## Supported Formats

**Standard:** JPG, JPEG, PNG, TIFF, BMP, WebP

**RAW:** NEF (Nikon), CR2/CR3 (Canon), ARW (Sony), RAF (Fuji), ORF (Olympus), RW2 (Panasonic), DNG

## Project Structure

```
fotoselect/
├── main.py           # CLI entry point
├── model.py          # CNN model with transfer learning
├── train.py          # Training pipeline
├── predict.py        # Inference module
├── dataset.py        # Data loading
├── requirements.txt
├── checkpoints/      # Saved models (created after training)
└── photos/
    ├── raw/          # Training: all photos
    ├── curated/      # Training: selected photos
    ├── input/        # Prediction: new photos
    └── output/       # Prediction: auto-curated results
```

## Tips

- **More data = better results**: Aim for at least 100+ photos in your training set
- **Balanced classes**: Try to have a reasonable mix of curated vs rejected photos
- **Consistent style**: The model learns your specific preferences, so be consistent in what you select
- **Adjust threshold**: Use higher threshold (0.7-0.8) for stricter curation, lower (0.3-0.4) for more permissive
