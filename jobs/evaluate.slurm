#!/bin/bash
#SBATCH --job-name=vasp-eval
#SBATCH --account=tesis
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=jobs/logs/eval_%j.out
#SBATCH --error=jobs/logs/eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sebastian.quispe@khipu.utec.edu.pe

# Evaluation Job
# This job evaluates a trained model on the test set.
# Can run on CPU, but GPU will be faster for inference.

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Started at: $(date)"
echo "=========================================="

# Change to project directory
cd "$SLURM_SUBMIT_DIR" || exit 1

# Create output directories
mkdir -p results jobs/logs

# Run evaluation using local venv (no internet needed)
.venv/bin/python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --split data/test_split.txt \
    --features_dir data/features \
    --summary_ratio 0.15 \
    --output results/evaluation_results.json

echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
