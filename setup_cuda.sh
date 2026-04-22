#!/bin/bash
# Fix PyTorch CUDA installation on Khipu
# Run on login node or interactive GPU session

echo "Checking CUDA version on GPU node..."

# Get CUDA version from nvidia-smi (need to run on GPU node)
CUDA_VERSION=$(srun --partition=gpu --gres=gpu:1 --time=00:05:00 nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)

echo "GPU driver version: $CUDA_VERSION"

# Common CUDA versions on clusters: 11.8, 12.1, 12.4
# We'll install for CUDA 12.1 which is widely supported
# If that doesn't work, try 11.8

echo "Installing PyTorch with CUDA support..."

# Uninstall CPU-only torch if present
.venv/bin/python -m pip uninstall -y torch torchvision torchaudio

# Install CUDA-enabled PyTorch
.venv/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is detected
echo "Verifying GPU support..."
srun --partition=gpu --gres=gpu:1 --time=00:05:00 .venv/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo "Done!"
