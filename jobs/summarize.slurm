#!/bin/bash
#SBATCH --job-name=vasp-summary
#SBATCH --account=tesis
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=jobs/logs/summary_%j.out
#SBATCH --error=jobs/logs/summary_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sebastian.quispe@khipu.utec.edu.pe

# Video Summarization Job
# This job generates a summary video from a trained model.
# GPU is recommended for feature extraction speed.

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Started at: $(date)"
echo "=========================================="

# Change to project directory
cd "$SLURM_SUBMIT_DIR" || exit 1

mkdir -p jobs/logs

# Example: summarize a single video using a trained checkpoint
# Modify VIDEO_PATH and OUTPUT as needed
VIDEO_PATH="data/raw/videos/example.mp4"
OUTPUT="results/summary_example.mp4"

# Run using local venv (no internet needed)
.venv/bin/python main.py "$VIDEO_PATH" --output "$OUTPUT"

echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
