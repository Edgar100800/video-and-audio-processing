# Dynamic Video Summarizer

A PyTorch-based video summarization system that extracts key shots from videos using deep visual and audio features.

## Overview

This project implements a pipeline for dynamic video summarization based on Gamal El-Nagar et al. (2024). The system:

1. Extracts frames (2 FPS) and audio (16kHz) from input videos
2. Segments videos into shots using KTS (Kernel Temporal Segmentation)
3. Extracts deep features: ResNet50 + InceptionV3 for visual, VGGish + MFCC + Mel-Spectrogram for audio
4. Predicts frame importance scores using visual and audio ANNs
5. Selects shots above a mean threshold and renders the summary video

## Installation

Requires Python 3.10+ and `uv`:

```bash
# Clone the repository
git clone <repository-url>
cd video-and-audio-processing

# Create virtual environment and install dependencies
uv sync
```

Dependencies are managed in `pyproject.toml` and include: PyTorch, OpenCV, MoviePy, Librosa, NumPy, h5py.

## Running the Summarization Pipeline

Process a single video and generate a summary:

```bash
uv run python main.py /path/to/video.mp4 --output summary.mp4
```

Options:
- `--output`, `-o`: Output video path (default: `output_summary.mp4`)
- `--fps`: Target frames per second for extraction (default: 2.0)
- `--sr`: Audio sample rate in Hz (default: 16000)

Example:
```bash
uv run python main.py input_video.mp4 --output my_summary.mp4 --fps 2.0
```

Note: The model runs with random weights by default. For production use, train the model first (see Training section below).

## Training Pipeline

To train the model on your own dataset (e.g., TVSum, SumMe):

### 1. Prepare Your Data

Place videos in `data/raw/videos/`:
```
data/raw/videos/
  video_01.mp4
  video_02.mp4
  ...
```

Ensure each video has ground truth frame-level importance scores stored in the corresponding `.h5` file after feature extraction (see step 2).

### 2. Extract Features

Extract and cache deep features for all videos. This is a one-time step per dataset:

```bash
uv run python scripts/extract_features.py \
    --input_dir data/raw/videos \
    --output_dir data/features
```

This creates `data/features/{video_id}.h5` files containing:
- `visual_features`: (N_frames, 4096) — ResNet50 + InceptionV3
- `audio_features`: (N_frames, 296) — VGGish + MFCC + Mel-Spectrogram
- `gt_scores`: (N_frames,) — ground truth importance scores (initialize to zeros if unavailable)
- `n_frames`, `fps`, `duration`: metadata

### 3. Create Dataset Splits

Create text files listing video IDs for each split (one ID per line):

```bash
# data/train_split.txt
video_01
video_02
video_03
...

# data/val_split.txt
video_10

# data/test_split.txt
video_11
```

### 4. Train the Model

```bash
uv run python scripts/train.py \
    --features_dir data/features \
    --train_split data/train_split.txt \
    --val_split data/val_split.txt \
    --epochs 50 \
    --lr 0.0001 \
    --batch_size 4
```

Training outputs:
- `checkpoints/best_model.pth` — best validation loss
- `checkpoints/latest_model.pth` — last epoch
- `results/training_history.json` — loss curves

Resume from a checkpoint:
```bash
uv run python scripts/train.py \
    --checkpoint checkpoints/latest_model.pth \
    --epochs 50
```

### 5. Evaluate the Model

```bash
uv run python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --split data/test_split.txt \
    --features_dir data/features \
    --summary_ratio 0.15
```

Metrics computed:
- **Precision**: Fraction of selected frames that are in ground truth summary
- **Recall**: Fraction of ground truth summary frames that are selected
- **F-Score**: Harmonic mean of precision and recall

Results are saved to `results/evaluation_results.json`.

## Project Structure

```
video-and-audio-processing/
├── main.py                          # CLI entry point for summarization
├── pyproject.toml                   # UV project configuration
│
├── src/
│   ├── config.py                    # Centralized configuration
│   ├── data_loader/
│   │   ├── video_processor.py       # Frame/audio extraction
│   │   ├── kts_segmentation.py      # Shot boundary detection
│   │   └── dataset.py               # PyTorch Dataset for h5py features
│   ├── feature_extraction/
│   │   ├── visual_extractor.py      # ResNet50 + InceptionV3
│   │   ├── audio_extractor.py       # VGGish + MFCC + Mel-Spectrogram
│   │   └── __init__.py              # Feature extraction pipeline
│   ├── models/
│   │   ├── visual_ann.py            # Visual importance predictor
│   │   ├── audio_ann.py             # Audio importance predictor
│   │   └── fusion_network.py        # Score fusion (AV = (V+A)/2)
│   └── summarization/
│       └── video_summarizer.py      # Shot selection and video rendering
│
├── scripts/
│   ├── extract_features.py          # Batch feature extraction to h5py
│   ├── train.py                     # Training loop with MSE loss
│   └── evaluate.py                  # F-score, Precision, Recall metrics
│
├── data/
│   ├── raw/                         # Input videos
│   ├── features/                    # Pre-extracted h5py features
│   └── processed/                   # Output summaries
│
├── checkpoints/                     # Saved model weights
└── results/                         # Training history and evaluation metrics
```

## Architecture Details

### Visual Feature Extraction
- ResNet50 (2048D) + InceptionV3 (2048D)
- Early fusion by concatenation: 4096D vector per frame

### Audio Feature Extraction
- VGGish embeddings (128D)
- MFCCs (40D)
- Mel-Spectrogram (128D)
- Fusion: 296D vector per frame

### Score Prediction
- Visual ANN: 4096 -> 1024 -> 128 -> 1
- Audio ANN: 296 -> 2048 -> 1024 -> 2048 -> 1024 -> 1
- Fusion: AV_Score = (V_Score + A_Score) / 2

### Shot Selection
- Shot score = mean of frame scores within shot
- Threshold = mean of all shot scores
- Select shots where score >= threshold

## Configuration

Edit `src/config.py` to adjust hyperparameters:

```python
TARGET_FPS = 2.0          # Frame extraction rate
AUDIO_SR = 16000          # Audio sample rate
BATCH_SIZE = 4            # Training batch size
LEARNING_RATE = 1e-4      # Adam learning rate
EPOCHS = 50               # Maximum training epochs
EARLY_STOPPING_PATIENCE = 10  # Epochs before stopping
SUMMARY_RATIO = 0.15      # Target summary length ratio
```

## Requirements

- Python 3.10+
- uv (Python package manager)
- See `pyproject.toml` for full dependency list

## License

[Your license here]
