"""Evaluation script for video summarization.

Computes F-score, Precision, and Recall against ground truth summaries.

Usage:
    uv run python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --features_dir data/features --split data/test_split.txt
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.visual_ann import VisualANN
from src.models.audio_ann import AudioANN
from src.models.fusion_network import FusionScorer
from src.data_loader.dataset import VideoSummarizationDataset, collate_fn
from src.config import DEVICE, VISUAL_DIM, AUDIO_DIM, RESULTS_DIR


def knapsack_selection(scores, max_length):
    """
    Select frames using 0/1 knapsack to match target summary length.
    This is the standard approach for video summarization evaluation.
    """
    n = len(scores)
    # Dynamic programming approach
    dp = np.zeros((n + 1, max_length + 1))

    for i in range(1, n + 1):
        for w in range(max_length + 1):
            if i <= w:
                dp[i, w] = max(dp[i - 1, w], dp[i - 1, w - 1] + scores[i - 1])
            else:
                dp[i, w] = dp[i - 1, w]

    # Backtrack to find selected frames
    selected = np.zeros(n, dtype=bool)
    w = max_length
    for i in range(n, 0, -1):
        if dp[i, w] != dp[i - 1, w]:
            selected[i - 1] = True
            w -= 1

    return selected


def evaluate_sample(pred_scores, gt_scores, gt_summary=None, summary_ratio=0.15):
    """
    Evaluate a single video.

    Args:
        pred_scores: Predicted frame importance scores (N,)
        gt_scores: Ground truth frame importance scores (N,)
        gt_summary: Binary ground truth summary (N,) - if available
        summary_ratio: Target summary length ratio

    Returns:
        dict with precision, recall, f_score
    """
    n_frames = len(pred_scores)
    max_length = int(n_frames * summary_ratio)

    # Method 1: Select top-k frames by predicted score
    selected_pred = np.zeros(n_frames, dtype=bool)
    top_indices = np.argsort(pred_scores)[-max_length:]
    selected_pred[top_indices] = True

    # Method 2: Use ground truth summary if available
    if gt_summary is not None:
        selected_gt = gt_summary.astype(bool)
    else:
        # Use top-k from ground truth scores as proxy
        selected_gt = np.zeros(n_frames, dtype=bool)
        top_gt_indices = np.argsort(gt_scores)[-max_length:]
        selected_gt[top_gt_indices] = True

    # Compute metrics
    true_positives = np.sum(selected_pred & selected_gt)
    false_positives = np.sum(selected_pred & ~selected_gt)
    false_negatives = np.sum(~selected_pred & selected_gt)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f_score': f_score,
        'selected_frames': int(np.sum(selected_pred)),
        'total_frames': n_frames
    }


def evaluate_model(model, dataloader, device, summary_ratio=0.15):
    """Evaluate model on a dataset."""
    model.eval()
    all_results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            for sample in batch:
                visual = sample['visual_features'].to(device)
                audio = sample['audio_features'].to(device)
                gt_scores = sample['gt_scores'].cpu().numpy()
                video_id = sample['video_id']

                # Forward pass
                _, _, av_scores = model(visual, audio)
                pred_scores = av_scores.squeeze().cpu().numpy()

                # Evaluate
                result = evaluate_sample(pred_scores, gt_scores, summary_ratio=summary_ratio)
                result['video_id'] = video_id
                all_results.append(result)

    # Aggregate metrics
    avg_precision = np.mean([r['precision'] for r in all_results])
    avg_recall = np.mean([r['recall'] for r in all_results])
    avg_f_score = np.mean([r['f_score'] for r in all_results])

    return {
        'average': {
            'precision': avg_precision,
            'recall': avg_recall,
            'f_score': avg_f_score,
        },
        'per_video': all_results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate video summarization model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--features_dir", default="data/features", help="Directory with h5py features")
    parser.add_argument("--split", required=True, help="Text file with test video IDs")
    parser.add_argument("--summary_ratio", type=float, default=0.15, help="Target summary ratio")
    parser.add_argument("--output", default=str(RESULTS_DIR / "evaluation_results.json"), help="Output JSON path")

    args = parser.parse_args()

    # Load dataset
    print("Loading test dataset...")
    test_dataset = VideoSummarizationDataset(
        split_file=args.split,
        features_dir=args.features_dir
    )

    if len(test_dataset) == 0:
        print("ERROR: No test samples found!")
        sys.exit(1)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"Test set: {len(test_dataset)} videos")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    visual_model = VisualANN(input_dim=VISUAL_DIM)
    audio_model = AudioANN(input_dim=AUDIO_DIM)
    model = FusionScorer(visual_model, audio_model).to(DEVICE)

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    print("\nEvaluating...")
    results = evaluate_model(model, test_loader, DEVICE, summary_ratio=args.summary_ratio)

    # Print results
    print(f"\n{'='*50}")
    print("Evaluation Results")
    print(f"{'='*50}")
    print(f"Precision: {results['average']['precision']:.4f}")
    print(f"Recall:    {results['average']['recall']:.4f}")
    print(f"F-Score:   {results['average']['f_score']:.4f}")
    print(f"{'='*50}")

    # Save results
    import json
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
