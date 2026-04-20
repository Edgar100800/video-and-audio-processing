"""Batch feature extraction script.

Extracts visual and audio features from videos and saves to h5py files.
Usage:
    uv run python scripts/extract_features.py --input_dir data/raw/videos --output_dir data/features
"""

import argparse
import os
import sys
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader.video_processor import VideoProcessor
from src.data_loader.kts_segmentation import KTSSegmentor
from src.feature_extraction import FeatureExtractionPipeline
from src.config import TARGET_FPS, AUDIO_SR, FEATURES_DIR


def extract_video_features(video_path, output_path, target_fps=TARGET_FPS, audio_sr=AUDIO_SR):
    """Extract features from a single video and save to h5py."""
    video_id = Path(video_path).stem

    # Step 1: Extract frames and audio
    processor = VideoProcessor(target_fps=target_fps)
    frames = processor.extract_frames(video_path)

    if len(frames) == 0:
        print(f"Failed to extract frames from {video_path}")
        return False

    audio_path = f"/tmp/{video_id}_audio.wav"
    audio = processor.extract_audio(video_path, audio_path)

    # Step 2: Segment into shots (for structure, but we save frame-level)
    # We use dummy features for KTS since it's simulated
    segmentor = KTSSegmentor()
    dummy_features = np.zeros((len(frames), 4096))
    boundaries = segmentor.segment_into_shots(dummy_features)
    shots = segmentor.split_frames_and_audio(frames, audio, boundaries, fps=target_fps, sample_rate=audio_sr)

    # Step 3: Extract deep features
    pipeline = FeatureExtractionPipeline()
    processed_shots = pipeline.process_shots(shots)

    # Step 4: Concatenate all frames
    all_visual = np.concatenate([s['visual_features'] for s in processed_shots], axis=0)
    all_audio = np.concatenate([s['audio_features'] for s in processed_shots], axis=0)

    # Save to h5py
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('video_id', data=video_id)
        f.create_dataset('visual_features', data=all_visual, compression='gzip')
        f.create_dataset('audio_features', data=all_audio, compression='gzip')
        f.create_dataset('n_frames', data=len(frames))
        f.create_dataset('fps', data=target_fps)
        f.create_dataset('duration', data=len(frames) / target_fps)
        # Placeholder for gt_scores - will be filled when annotations are available
        f.create_dataset('gt_scores', data=np.zeros(len(frames), dtype=np.float32))

    # Cleanup
    if os.path.exists(audio_path):
        os.remove(audio_path)

    print(f"Saved features to {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Extract features from videos")
    parser.add_argument("--input_dir", "-i", required=True, help="Directory containing video files")
    parser.add_argument("--output_dir", "-o", default=str(FEATURES_DIR), help="Output directory for h5py files")
    parser.add_argument("--fps", type=float, default=TARGET_FPS, help="Target FPS")
    parser.add_argument("--sr", type=int, default=AUDIO_SR, help="Audio sample rate")
    parser.add_argument("--ext", default="mp4", help="Video file extension")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(input_dir.glob(f"*.{args.ext}"))
    if not video_files:
        print(f"No .{args.ext} files found in {input_dir}")
        return

    print(f"Found {len(video_files)} videos to process")

    for video_path in tqdm(video_files, desc="Extracting features"):
        output_path = output_dir / f"{video_path.stem}.h5"
        if output_path.exists():
            print(f"Skipping {video_path.name} (already extracted)")
            continue

        try:
            extract_video_features(
                str(video_path),
                str(output_path),
                target_fps=args.fps,
                audio_sr=args.sr
            )
        except Exception as e:
            print(f"Error processing {video_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print("\nFeature extraction complete!")


if __name__ == "__main__":
    main()
