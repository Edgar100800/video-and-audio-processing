#!/bin/bash
#SBATCH --job-name=vasp-extract
#SBATCH --account=tesis
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=jobs/logs/extract_%j.out
#SBATCH --error=jobs/logs/extract_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sebastian.quispe@khipu.utec.edu.pe

# Feature Extraction Job
# This job extracts visual and audio features from videos and saves them to h5py files.
# It is CPU-intensive but does not require GPU.

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Started at: $(date)"
echo "=========================================="

# Change to project directory
cd "$SLURM_SUBMIT_DIR" || exit 1

# Create logs directory
mkdir -p jobs/logs

# Use portable Python from project directory (compute nodes lack system Python 3.11)
PYTHON="$SLURM_SUBMIT_DIR/.venv/bin/python"

# Run feature extraction using local venv (no internet needed)
$PYTHON scripts/extract_features.py \
    --input_dir data/raw/videos \
    --output_dir data/features \
    --fps 2.0 \
    --sr 16000

echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
