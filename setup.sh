#!/usr/bin/env bash
#
# Set up TRELLIS.2 for Apple Silicon.
# Creates a venv, installs dependencies, clones the repo, and applies patches.
#

set -euo pipefail
cd "$(dirname "$0")"

echo "=== TRELLIS.2 for Apple Silicon — Setup ==="
echo

# Create venv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    if command -v uv &>/dev/null; then
        uv venv .venv --python python3.11
    else
        python3 -m venv .venv
    fi
fi

source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
if command -v uv &>/dev/null; then
    uv pip install torch torchvision torchaudio
    uv pip install transformers accelerate huggingface_hub safetensors
    uv pip install pillow numpy trimesh scipy tqdm
else
    pip install torch torchvision torchaudio
    pip install transformers accelerate huggingface_hub safetensors
    pip install pillow numpy trimesh scipy tqdm
fi

# Clone TRELLIS.2
if [ ! -d "TRELLIS.2" ]; then
    echo "Cloning TRELLIS.2..."
    git clone --depth 1 https://github.com/microsoft/TRELLIS.git TRELLIS.2
fi

# Create stub packages
echo "Installing stub packages..."
python3 backends/stubs.py stubs

# Copy mesh extraction into stubs
cp backends/mesh_extract.py stubs/o_voxel/convert.py

# Apply source patches
echo "Applying MPS compatibility patches..."
python3 patches/mps_compat.py

echo
echo "=== Setup complete ==="
echo "Activate the environment:  source .venv/bin/activate"
echo "Generate a 3D model:       python generate.py path/to/image.png"
