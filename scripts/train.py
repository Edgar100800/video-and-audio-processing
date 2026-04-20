"""Training script for video summarization.

Usage:
    uv run python scripts/train.py --features_dir data/features --epochs 50
"""

import argparse
import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.visual_ann import VisualANN
from src.models.audio_ann import AudioANN
from src.models.fusion_network import FusionScorer
from src.data_loader.dataset import VideoSummarizationDataset, collate_fn
from src.config import (
    DEVICE, VISUAL_DIM, AUDIO_DIM, BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, EPOCHS, EARLY_STOPPING_PATIENCE,
    CHECKPOINTS_DIR, RESULTS_DIR
)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        batch_loss = 0.0

        for sample in batch:
            visual = sample['visual_features'].to(device)    # (N, 4096)
            audio = sample['audio_features'].to(device)      # (N, 296)
            gt = sample['gt_scores'].to(device)              # (N,)

            # Forward pass
            v_scores, a_scores, av_scores = model(visual, audio)

            # Squeeze to (N,)
            av_scores = av_scores.squeeze()

            # Compute loss
            loss = criterion(av_scores, gt)
            batch_loss += loss

        # Backward pass (average over batch)
        batch_loss = batch_loss / len(batch)
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate_epoch(model, dataloader, criterion, device):
    """Run one validation epoch."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            batch_loss = 0.0

            for sample in batch:
                visual = sample['visual_features'].to(device)
                audio = sample['audio_features'].to(device)
                gt = sample['gt_scores'].to(device)

                v_scores, a_scores, av_scores = model(visual, audio)
                av_scores = av_scores.squeeze()

                loss = criterion(av_scores, gt)
                batch_loss += loss

            batch_loss = batch_loss / len(batch)
            total_loss += batch_loss.item()
            n_batches += 1

    return total_loss / n_batches


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train video summarization model")
    parser.add_argument("--features_dir", default="data/features", help="Directory with h5py features")
    parser.add_argument("--train_split", help="Text file with train video IDs")
    parser.add_argument("--val_split", help="Text file with val video IDs")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--checkpoint", help="Path to checkpoint to resume from")
    parser.add_argument("--output_dir", default=str(CHECKPOINTS_DIR), help="Checkpoint output directory")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    train_dataset = VideoSummarizationDataset(
        split_file=args.train_split,
        features_dir=args.features_dir
    )
    val_dataset = VideoSummarizationDataset(
        split_file=args.val_split,
        features_dir=args.features_dir
    )

    if len(train_dataset) == 0:
        print("ERROR: No training samples found!")
        print(f"Features directory: {args.features_dir}")
        print("Make sure you've run extract_features.py first.")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"Train: {len(train_dataset)} videos, Val: {len(val_dataset)} videos")

    # Initialize model
    print(f"Initializing model on {DEVICE}...")
    visual_model = VisualANN(input_dim=VISUAL_DIM)
    audio_model = AudioANN(input_dim=AUDIO_DIM)
    model = FusionScorer(visual_model, audio_model).to(DEVICE)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0

    # Resume from checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('loss', float('inf'))

    # Training loop
    print("\nStarting training...")
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate_epoch(model, val_loader, criterion, DEVICE)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                output_dir / "best_model.pth"
            )
        else:
            patience_counter += 1

        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            output_dir / "latest_model.pth"
        )

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # Save training history
    with open(RESULTS_DIR / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
