#!/bin/bash
# Setup portable Python for Khipu compute nodes
# Run once on login node

echo "Downloading portable Python 3.11..."
mkdir -p .python
cd .python || exit 1

# Download portable Python (standalone, no system deps)
wget -c https://github.com/indygreg/python-build-standalone/releases/download/20240107/cpython-3.11.7+20240107-x86_64-unknown-linux-gnu-install_only.tar.gz -O python.tar.gz

echo "Extracting..."
tar -xzf python.tar.gz

# Clean up
rm python.tar.gz

echo "Portable Python installed at $(pwd)/python/bin/python3.11"
cd ..

# Create venv using portable Python
.python/python/bin/python3.11 -m venv .venv

# Install deps
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install opencv-python librosa resampy numpy h5py moviepy tqdm scipy pillow

echo "Setup complete!"
echo "Test: .venv/bin/python --version"
.venv/bin/python --version
