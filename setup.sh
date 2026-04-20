#!/usr/bin/env bash
#
# Set up TRELLIS.2 for Apple Silicon.
# Creates a venv, installs dependencies, clones the repo, and applies patches.
#

set -euo pipefail
cd "$(dirname "$0")"

echo "=== TRELLIS.2 for Apple Silicon — Setup ==="
echo

# Check Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "Warning: This project requires Apple Silicon (M1 or later)."
fi

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
DEPS="torch torchvision torchaudio transformers accelerate huggingface_hub safetensors pillow numpy trimesh scipy tqdm easydict kornia timm imageio opencv-python-headless xatlas fast-simplification"
if command -v uv &>/dev/null; then
    PIP="uv pip install"
else
    PIP="pip install"
fi
$PIP $DEPS
$PIP git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Optional Metal acceleration for texture baking.
# Requires Xcode Metal Toolchain:
#     xcodebuild -downloadComponent MetalToolchain
# Without these, we fall back to a pure-Python KDTree-based texture baker.
#
# --no-build-isolation is critical: these packages need torch at build time,
# and uv's default isolated build env has no torch installed.
if [ "${SKIP_METAL:-0}" != "1" ]; then
    echo
    echo "Installing Metal backends for texture baking (set SKIP_METAL=1 to skip)..."
    PIP_NB="$PIP --no-build-isolation"
    # Build deps required by the Metal packages' setup.py
    $PIP setuptools wheel pybind11
    $PIP_NB git+https://github.com/pedronaugusto/mtlbvh.git      || echo "  mtlbvh install failed — continuing without Metal BVH"
    $PIP_NB git+https://github.com/pedronaugusto/mtldiffrast.git || echo "  mtldiffrast install failed — continuing without Metal rasterizer"
    $PIP_NB git+https://github.com/pedronaugusto/mtlmesh.git     || echo "  mtlmesh install failed — continuing without Metal mesh ops"
    # mtlgemm provides flex_gemm.ops.grid_sample. The Metal baker in
    # o_voxel.postprocess prefers this over a torch.nn.functional.grid_sample
    # fallback, and the flex_gemm sparse sampling produces noticeably cleaner
    # texture baking (no concentric ring artifacts on curved surfaces).
    $PIP_NB git+https://github.com/pedronaugusto/mtlgemm.git     || echo "  mtlgemm install failed — baker will use a lower-quality torch.nn.functional.grid_sample fallback"
    # Pedro Naugusto's o_voxel CPU fork — exposes o_voxel.postprocess.to_glb
    # which wraps the Metal stack. Install last so its deps already present.
    $PIP_NB "git+https://github.com/pedronaugusto/trellis2-apple.git#subdirectory=o-voxel" \
        || echo "  o_voxel (Apple fork) install failed — falling back to KDTree baker"
fi

# Clone TRELLIS.2
if [ ! -d "TRELLIS.2" ]; then
    echo "Cloning TRELLIS.2..."
    git clone --depth 1 https://github.com/microsoft/TRELLIS.2.git TRELLIS.2
fi

# Apply source patches (this also installs stubs and backends)
echo "Applying MPS compatibility patches..."
python3 patches/mps_compat.py

# Check HuggingFace auth
echo
if python3 -c "from huggingface_hub import get_token; assert get_token()" 2>/dev/null; then
    echo "HuggingFace auth: OK"
else
    echo "WARNING: Not logged into HuggingFace."
    echo "Some model weights require authentication. Run:"
    echo "  hf auth login"
    echo ""
    echo "You also need to request access to these gated models:"
    echo "  https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m"
    echo "  https://huggingface.co/briaai/RMBG-2.0"
fi

echo
echo "=== Setup complete ==="
echo "Activate the environment:  source .venv/bin/activate"
echo "Generate a 3D model:       python generate.py path/to/image.png"
