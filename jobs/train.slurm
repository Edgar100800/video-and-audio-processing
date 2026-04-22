#!/bin/bash
#SBATCH --job-name=vasp-train
#SBATCH --account=tesis
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=jobs/logs/train_%j.out
#SBATCH --error=jobs/logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sebastian.quispe@khipu.utec.edu.pe

# Training Job
# This job trains the visual and audio ANNs.
# Requires GPU for reasonable training speed.

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started at: $(date)"
echo "=========================================="

# Verify GPU is available
nvidia-smi

# Load modules or activate environment
export PATH="$HOME/.cargo/bin:$PATH"

# Change to project directory
cd "$SLURM_SUBMIT_DIR" || exit 1

# Create output directories
mkdir -p checkpoints results jobs/logs

# Run training
uv run python scripts/train.py \
    --features_dir data/features \
    --train_split data/train_split.txt \
    --val_split data/val_split.txt \
    --epochs 50 \
    --lr 0.0001 \
    --batch_size 4 \
    --output_dir checkpoints

echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
