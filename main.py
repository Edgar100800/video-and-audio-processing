import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader.video_processor import VideoProcessor
from src.data_loader.kts_segmentation import KTSSegmentor
from src.feature_extraction import FeatureExtractionPipeline
from src.models.fusion_network import ScorePredictor
from src.summarization.video_summarizer import DynamicVideoSummarizer


def main():
    parser = argparse.ArgumentParser(description="Dynamic Video Summarizer")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--output", "-o", default="output_summary.mp4", help="Output video path")
    parser.add_argument("--fps", type=float, default=2.0, help="Target FPS for frames")
    parser.add_argument("--sr", type=int, default=16000, help="Audio sample rate")

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"Processing: {os.path.basename(args.video_path)}")
    print(f"{'='*50}\n")

    print("=" * 30)
    print("STEP 1: Extract frames & audio")
    print("=" * 30)
    processor = VideoProcessor(target_fps=args.fps)

    base_name = os.path.splitext(os.path.basename(args.video_path))[0]
    frames = processor.extract_frames(args.video_path)
    if len(frames) == 0:
        print("Failed to extract frames")
        sys.exit(1)

    audio_path = f"/tmp/{base_name}_audio.wav"
    audio = processor.extract_audio(args.video_path, audio_path)

    print("\n" + "=" * 30)
    print("STEP 2: Segment into shots (KTS)")
    print("=" * 30)

    import numpy as np
    # KTS needs frame count; using dummy features since segmentation is simulated
    sample_features = np.zeros((len(frames), 4096))

    segmentor = KTSSegmentor()
    boundaries = segmentor.segment_into_shots(sample_features)
    shots = segmentor.split_frames_and_audio(frames, audio, boundaries, fps=args.fps, sample_rate=args.sr)

    print("\n" + "=" * 30)
    print("STEP 3: Extract deep features")
    print("=" * 30)
    pipeline = FeatureExtractionPipeline()
    processed_shots = pipeline.process_shots(shots)

    print("\n" + "=" * 30)
    print("STEP 4: Predict importance scores")
    print("=" * 30)
    scorer = ScorePredictor()
    scored_shots = scorer.predict_shot_scores(processed_shots)

    print("\n" + "=" * 30)
    print("STEP 5: Build summary video")
    print("=" * 30)
    summarizer = DynamicVideoSummarizer(target_fps=args.fps, audio_sr=args.sr)
    selected = summarizer.select_key_shots(scored_shots)
    summarizer.build_summary_video(selected, args.output)

    if os.path.exists(audio_path):
        os.remove(audio_path)

    print("\nDone!")


if __name__ == "__main__":
    main()