#!/bin/bash
#SBATCH --job-name=vasp-extract
#SBATCH --account=tesis
#SBATCH --partition=a-tesis
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
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

# Load modules or activate environment
# Option 1: If using uv installed in the project
export PATH="$HOME/.cargo/bin:$PATH"

# Option 2: If using a conda environment
# module load miniconda
# conda activate video-summarizer

# Change to project directory
cd "$SLURM_SUBMIT_DIR" || exit 1

# Create logs directory
mkdir -p jobs/logs

# Run feature extraction
# Adjust --input_dir and --output_dir as needed
uv run python scripts/extract_features.py \
    --input_dir data/raw/videos \
    --output_dir data/features \
    --fps 2.0 \
    --sr 16000

echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
