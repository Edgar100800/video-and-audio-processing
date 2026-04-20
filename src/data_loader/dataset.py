"""PyTorch Dataset for video summarization features."""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import FEATURES_DIR, VISUAL_DIM, AUDIO_DIM


class VideoSummarizationDataset(Dataset):
    """Dataset that loads pre-extracted features from h5py files."""

    def __init__(self, split_file=None, features_dir=None, video_ids=None):
        """
        Args:
            split_file: Path to text file with video IDs (one per line)
            features_dir: Directory containing h5py feature files
            video_ids: List of video IDs (alternative to split_file)
        """
        self.features_dir = features_dir or FEATURES_DIR
        self.video_ids = []

        if split_file and os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.video_ids = [line.strip() for line in f if line.strip()]
        elif video_ids:
            self.video_ids = video_ids
        else:
            # Auto-discover all h5 files
            self.video_ids = [
                f.replace('.h5', '')
                for f in os.listdir(self.features_dir)
                if f.endswith('.h5')
            ]

        self.valid_samples = []
        self._validate_samples()

        print(f"Dataset loaded: {len(self.valid_samples)} valid videos")

    def _validate_samples(self):
        """Check which samples have valid data."""
        for vid in self.video_ids:
            path = os.path.join(self.features_dir, f"{vid}.h5")
            if not os.path.exists(path):
                continue

            try:
                with h5py.File(path, 'r') as f:
                    if 'visual_features' not in f:
                        continue
                    if 'gt_scores' not in f:
                        continue
                    n_frames = f['visual_features'].shape[0]
                    if n_frames == 0:
                        continue
                    self.valid_samples.append(vid)
            except Exception as e:
                print(f"Warning: Could not load {vid}: {e}")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        """Returns a single video's features and ground truth."""
        video_id = self.valid_samples[idx]
        path = os.path.join(self.features_dir, f"{video_id}.h5")

        with h5py.File(path, 'r') as f:
            visual = torch.tensor(f['visual_features'][:], dtype=torch.float32)
            audio = torch.tensor(f['audio_features'][:], dtype=torch.float32)
            gt_scores = torch.tensor(f['gt_scores'][:], dtype=torch.float32)

            # Ensure gt_scores is 1D
            if gt_scores.dim() > 1:
                gt_scores = gt_scores.squeeze()

        return {
            'video_id': video_id,
            'visual_features': visual,      # (N_frames, 4096)
            'audio_features': audio,        # (N_frames, 296)
            'gt_scores': gt_scores,         # (N_frames,)
        }


def collate_fn(batch):
    """
    Collate function for variable-length sequences.
    Returns a list of dicts (no padding) since videos have different lengths.
    """
    return batch
