#!/bin/bash
# Pre-download model weights on login node using wget
# Run once before submitting SLURM jobs

echo "Pre-downloading model weights with wget..."

# Create cache directory
mkdir -p ~/.cache/torch/hub/checkpoints

cd ~/.cache/torch/hub/checkpoints

# Download ResNet50 weights
echo "Downloading ResNet50..."
wget -c --no-check-certificate https://download.pytorch.org/models/resnet50-0676ba61.pth

# Download InceptionV3 weights
echo "Downloading InceptionV3..."
wget -c --no-check-certificate https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth

# Download VGGish weights (from GitHub)
echo "Downloading VGGish..."
wget -c --no-check-certificate https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth
wget -c --no-check-certificate https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish_pca_params-970ea276.pth

echo "All models downloaded to ~/.cache/torch/hub/checkpoints/"
ls -lh ~/.cache/torch/hub/checkpoints/
