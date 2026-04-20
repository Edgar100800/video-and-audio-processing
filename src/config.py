"""Training configuration."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories
for d in [FEATURES_DIR, CHECKPOINTS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Video Processing
TARGET_FPS = 2.0
AUDIO_SR = 16000

# Feature Dimensions
VISUAL_DIM = 4096  # ResNet50 (2048) + InceptionV3 (2048)
AUDIO_DIM = 296    # MFCC (40) + Mel (128) + VGGish (128)

# Model Architecture
VISUAL_HIDDEN = [1024, 128]
AUDIO_HIDDEN = [2048, 1024, 2048, 1024]
DROPOUT_VISUAL = 0.2
DROPOUT_AUDIO = 0.3

# Training
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Loss
LOSS_FN = "mse"  # MSE between predicted scores and ground truth

# Evaluation
SUMMARY_RATIO = 0.15  # Target summary length as ratio of original

# Device
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Config loaded. Device: {DEVICE}")
