# TRELLIS.2 for Apple Silicon

Run [TRELLIS.2](https://github.com/microsoft/TRELLIS) image-to-3D generation natively on Mac.

This is a port of Microsoft's TRELLIS.2 — a state-of-the-art image-to-3D model — from CUDA-only to Apple Silicon via PyTorch MPS. No NVIDIA GPU required.

## Results

Generates **400K+ vertex meshes with baked PBR textures** from single images in **~3.5 minutes on M4 Pro**.

Output is a GLB with base-color, metallic, and roughness textures — ready for use in 3D applications.

### Example

**Input** &rarr; **Generated 3D mesh (424K vertices, 858K triangles)**

<p>
<img src="assets/shoe_input.png" width="200">
<img src="assets/shoe_front.png" width="280">
<img src="assets/shoe_side.png" width="280">
</p>

## Requirements

- macOS on Apple Silicon (M1 or later)
- Python 3.11+
- 24GB+ unified memory recommended (the 4B model is large)
- ~15GB disk space for model weights (downloaded on first run)

## Quick Start

```bash
# Clone this repo
git clone https://github.com/shivampkumar/trellis-mac.git
cd trellis-mac

# (Recommended) Download the Xcode Metal Toolchain so setup can build the
# Metal-accelerated texture baker. Without this, setup falls back to a pure
# Python KDTree baker (slower, slightly lower quality).
xcodebuild -downloadComponent MetalToolchain

# Log into HuggingFace (needed for gated model weights)
hf auth login

# Request access to these gated models (usually instant approval):
#   https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
#   https://huggingface.co/briaai/RMBG-2.0

# Run setup (creates venv, installs deps, clones & patches TRELLIS.2,
# builds Metal backends if the toolchain is available)
bash setup.sh

# Activate the environment
source .venv/bin/activate

# Generate a 3D model from an image
python generate.py path/to/image.png
```

To skip the Metal build (for example on older hardware or to speed up setup):

```bash
SKIP_METAL=1 bash setup.sh
```

Output files are saved to the current directory (or use `--output` to specify a path).

## Usage

```bash
# Basic usage
python generate.py photo.png

# With options
python generate.py photo.png --seed 123 --output my_model --pipeline-type 512

# All options
python generate.py --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--seed` | 42 | Random seed for generation |
| `--output` | `output_3d` | Output filename (without extension) |
| `--pipeline-type` | `512` | Pipeline resolution: `512`, `1024`, `1024_cascade` |
| `--texture-size` | `1024` | PBR texture resolution: `512`, `1024`, `2048` |
| `--no-texture` | — | Skip texture baking, export geometry only |

## What Was Ported

TRELLIS.2 depends on several CUDA-only libraries. This port replaces each of them with a backend that runs on Apple Silicon:

| Original (CUDA) | Replacement | Purpose |
|---|---|---|
| `flex_gemm` | `backends/conv_none.py` | Sparse 3D convolution via gather-scatter (pure PyTorch) |
| `o_voxel._C` hashmap | `backends/mesh_extract.py` | Mesh extraction from dual voxel grid (pure Python) |
| `flash_attn` | PyTorch SDPA | Scaled dot-product attention for sparse transformers |
| `cumesh` | Skipped during decode | Called on meshes large enough to crash the Metal port; replaced with `fast_simplification` before baking |
| `nvdiffrast` | `mtldiffrast` (Metal) with pure-Python fallback | Differentiable rasterization for texture baking |

Additionally, all hardcoded `.cuda()` calls throughout the codebase were patched to use the active device instead.

### Technical Details

**Sparse 3D Convolution** (`backends/conv_none.py`): Implements submanifold sparse convolution by building a spatial hash of active voxels, gathering neighbor features for each kernel position, applying weights via matrix multiplication, and scatter-adding results back. Neighbor maps are cached per-tensor to avoid redundant computation.

**Mesh Extraction** (`backends/mesh_extract.py`): Reimplements `flexible_dual_grid_to_mesh` using Python dictionaries instead of CUDA hashmap operations. Builds a coordinate-to-index lookup table, finds connected voxels for each edge, and triangulates quads using normal alignment heuristics.

**Attention** (patched `full_attn.py`): Adds an SDPA backend to the sparse attention module. Pads variable-length sequences into batches, runs `torch.nn.functional.scaled_dot_product_attention`, then unpads results.

**Texture Baking**: By default we use the Metal stack released by [@pedronaugusto](https://github.com/pedronaugusto) — [`mtldiffrast`](https://github.com/pedronaugusto/mtldiffrast), [`mtlbvh`](https://github.com/pedronaugusto/mtlbvh), [`mtlmesh`](https://github.com/pedronaugusto/mtlmesh), and his CPU fork of [`o_voxel`](https://github.com/pedronaugusto/trellis2-apple) — which exposes `o_voxel.postprocess.to_glb`. We pre-simplify the decoder mesh to ~200K faces with `fast_simplification` before handing it to the Metal BVH (the BVH builder is unstable on 800K+ face inputs). If the Metal toolchain is unavailable, we fall back to `backends/texture_baker.py`: xatlas UV unwrap, then a scipy cKDTree + inverse-distance weighting over the sparse voxel grid at native 512 resolution.

## Performance

Benchmarks on M4 Pro (24GB), pipeline type `512`, Metal texture baker enabled:

| Stage | Time |
|-------|------|
| Model loading | ~45s |
| Image preprocessing | ~5s |
| Sparse structure sampling | ~15s |
| Shape SLat sampling | ~90s |
| Texture SLat sampling | ~50s |
| Mesh decoding | ~30s |
| Texture bake (Metal, 1024²) | ~15s |
| **Total** | **~3.7 min** |

Memory usage peaks at around 18GB unified memory during generation.

With `SKIP_METAL=1` (pure-Python KDTree baker), the texture bake adds ~20s instead of ~15s, and coverage near UV chart boundaries is slightly softer.

## Limitations

- **Hole filling disabled**: Decode-time hole filling requires `cumesh`. The Metal port segfaults on decoder-sized meshes, so we skip this step. Output meshes may have small holes.
- **Slower than CUDA**: The pure-PyTorch sparse convolution is ~10× slower than the CUDA `flex_gemm` kernel. This is the main inference bottleneck.
- **Pre-simplified before texture bake**: The mesh is decimated from ~800K to ~200K faces before Metal BVH construction to avoid builder instability. If you need the full-resolution mesh, export it via the OBJ output (which is written before simplification).
- **No training support**: Inference only.

## License

The porting code in this repository (backends, patches, scripts) is released under the MIT License.

Upstream model weights are subject to their own licenses:
- **TRELLIS.2**: [MIT License](https://github.com/microsoft/TRELLIS.2/blob/main/LICENSE)
- **DINOv3**: [Meta custom license](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/blob/main/LICENSE.md) (gated, review before commercial use)
- **RMBG-2.0**: [CC BY-NC 4.0](https://huggingface.co/briaai/RMBG-2.0) (non-commercial; commercial use requires a license from BRIA)

## Credits

- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) by Microsoft Research — the original model and codebase
- [DINOv3](https://github.com/facebookresearch/dinov3) by Meta — image feature extraction
- [RMBG-2.0](https://github.com/Bria-AI/RMBG-2.0) by BRIA AI — background removal
- [@pedronaugusto](https://github.com/pedronaugusto) — `mtldiffrast`, `mtlbvh`, `mtlmesh`, and the CPU fork of `o_voxel` that together provide the Metal texture-baking path used by this repo
