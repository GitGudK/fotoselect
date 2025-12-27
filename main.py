#!/usr/bin/env python3
"""
FotoSelect - AI-powered photo curation tool.

Train a CNN model on your curated photos and use it to automatically
select the best photos from new batches.
"""

import argparse
import sys
from pathlib import Path


def cmd_train(args):
    """Train a new curation model."""
    from train import train_model

    print(f"\n{'='*60}")
    print("FotoSelect - Training Mode")
    print(f"{'='*60}")
    print(f"Raw photos folder: {args.raw_folder}")
    print(f"Curated photos folder: {args.curated_folder}")
    print(f"Model backbone: {args.backbone}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")

    history = train_model(
        raw_folder=args.raw_folder,
        curated_folder=args.curated_folder,
        output_dir=args.output_dir,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        freeze_backbone=args.freeze_backbone,
        early_stopping_patience=args.patience
    )

    print(f"\nModel saved to: {args.output_dir}/best.pt")
    print(f"Training history saved to: {args.output_dir}/history.json")


def cmd_predict(args):
    """Run prediction on new photos."""
    from predict import predict_photos

    print(f"\n{'='*60}")
    print("FotoSelect - Prediction Mode")
    print(f"{'='*60}")
    print(f"Input folder: {args.input_folder}")
    print(f"Model checkpoint: {args.checkpoint}")
    print(f"Threshold: {args.threshold}")
    if args.output_folder:
        print(f"Output folder: {args.output_folder}")
    print(f"{'='*60}\n")

    results = predict_photos(
        checkpoint_path=args.checkpoint,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        threshold=args.threshold,
        backbone=args.backbone,
        batch_size=args.batch_size,
        copy_files=not args.move,
        export_json=args.export_json
    )


def main():
    parser = argparse.ArgumentParser(
        description='FotoSelect - AI-powered photo curation tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train a model:
    python main.py train --raw-folder ./all_photos --curated-folder ./selected_photos

  Predict on new photos:
    python main.py predict --input-folder ./new_photos --checkpoint ./checkpoints/best.pt

  Predict and copy curated photos:
    python main.py predict --input-folder ./new_photos --checkpoint ./checkpoints/best.pt \\
                           --output-folder ./auto_curated --threshold 0.7
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train a new curation model')
    train_parser.add_argument(
        '--raw-folder', '-r',
        required=True,
        help='Path to folder containing all raw/unfiltered photos'
    )
    train_parser.add_argument(
        '--curated-folder', '-c',
        required=True,
        help='Path to folder containing curated/selected photos'
    )
    train_parser.add_argument(
        '--output-dir', '-o',
        default='checkpoints',
        help='Directory to save model checkpoints (default: checkpoints)'
    )
    train_parser.add_argument(
        '--backbone', '-b',
        choices=['resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b2', 'mobilenet_v3'],
        default='resnet50',
        help='Model backbone architecture (default: resnet50)'
    )
    train_parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    train_parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=1e-4,
        help='Initial learning rate (default: 0.0001)'
    )
    train_parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Image size for training (default: 224)'
    )
    train_parser.add_argument(
        '--freeze-backbone',
        action='store_true',
        help='Freeze backbone weights (only train classifier)'
    )
    train_parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience (default: 10)'
    )
    train_parser.set_defaults(func=cmd_train)

    # Predict subcommand
    predict_parser = subparsers.add_parser('predict', help='Predict curation on new photos')
    predict_parser.add_argument(
        '--input-folder', '-i',
        required=True,
        help='Path to folder containing photos to curate'
    )
    predict_parser.add_argument(
        '--checkpoint', '-m',
        required=True,
        help='Path to trained model checkpoint'
    )
    predict_parser.add_argument(
        '--output-folder', '-o',
        help='Optional: copy/move curated photos to this folder'
    )
    predict_parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.5,
        help='Probability threshold for curation (default: 0.5)'
    )
    predict_parser.add_argument(
        '--backbone', '-b',
        choices=['resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b2', 'mobilenet_v3'],
        default='resnet50',
        help='Model backbone architecture (default: resnet50)'
    )
    predict_parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for inference (default: 16)'
    )
    predict_parser.add_argument(
        '--move',
        action='store_true',
        help='Move files instead of copying (use with caution)'
    )
    predict_parser.add_argument(
        '--export-json',
        help='Export results to JSON file'
    )
    predict_parser.set_defaults(func=cmd_predict)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Run the appropriate command
    args.func(args)


if __name__ == '__main__':
    main()
