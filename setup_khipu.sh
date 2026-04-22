#!/bin/bash
# Setup script to run ONCE on Khipu login node
# This creates the uv venv and installs all dependencies

echo "Setting up uv environment on Khipu login node..."

# Ensure uv is available
export PATH="$HOME/.cargo/bin:$PATH"
which uv || (echo "uv not found. Install it first:" && echo "curl -LsSf https://astral.sh/uv/install.sh | sh" && exit 1)

# Create venv in project directory (cached locally)
uv venv .venv --python 3.11

# Install dependencies (downloads happen on login node with internet)
uv sync

# Verify installation
.venv/bin/python -c "import torch; print('PyTorch:', torch.__version__)"
.venv/bin/python -c "import cv2; print('OpenCV ok')"
.venv/bin/python -c "import librosa; print('Librosa ok')"
.venv/bin/python -c "import moviepy; print('MoviePy ok')"
.venv/bin/python -c "import h5py; print('h5py ok')"

echo "Setup complete! You can now submit SLURM jobs."
