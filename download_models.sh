#!/bin/bash
# Pre-download model weights on login node (where internet works)
# Run once before submitting SLURM jobs

echo "Pre-downloading model weights..."

.venv/bin/python << 'PYEOF'
import torch
import torchvision.models as models
import torch.nn as nn

print("Downloading ResNet50...")
resnet = models.resnet50(weights='IMAGENET1K_V1')
print("ResNet50 downloaded")

print("Downloading InceptionV3...")
inception = models.inception_v3(weights='IMAGENET1K_V1')
print("InceptionV3 downloaded")

print("Downloading VGGish...")
try:
    vggish = torch.hub.load('harritaylor/torchvggish', 'vggish', verbose=False)
    print("VGGish downloaded")
except Exception as e:
    print(f"VGGish download failed: {e}")
    print("Will retry during extraction")

print("\nAll models cached to ~/.cache/torch/")
print("Cache location:", torch.hub.get_dir())
PYEOF

echo "Done! Model weights are now cached."
