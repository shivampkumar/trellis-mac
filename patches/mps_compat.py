"""
Apply all MPS compatibility patches to a fresh TRELLIS.2 clone.

Modifies source files in-place to replace CUDA-only code paths with
device-agnostic alternatives that work on Apple Silicon (MPS).

Run once after cloning TRELLIS.2:
    python patches/mps_compat.py
"""

import os
import shutil

TRELLIS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "TRELLIS.2")
BACKENDS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backends")


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)
    print(f"  Patched: {os.path.relpath(path, TRELLIS_ROOT)}")


def patch_sparse_config():
    """Add 'sdpa' and 'naive' to the allowed attention backends."""
    path = os.path.join(TRELLIS_ROOT, "trellis2/modules/sparse/config.py")
    src = read_file(path)

    if "'sdpa'" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    src = src.replace(
        "env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3']",
        "env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive']",
    )
    write_file(path, src)


def patch_sparse_attention():
    """Add SDPA backend to the sparse attention dispatch."""
    path = os.path.join(TRELLIS_ROOT, "trellis2/modules/sparse/attention/full_attn.py")
    src = read_file(path)

    if "'sdpa'" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    sdpa_block = """\
    elif config.ATTN in ('sdpa', 'naive'):
        from torch.nn.functional import scaled_dot_product_attention as sdpa_fn
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        H = q.shape[-2]
        max_q = max(q_seqlen)
        max_kv = max(kv_seqlen) if kv_seqlen is not None else max_q
        B = len(q_seqlen)
        C_q = q.shape[-1]
        C_v = v.shape[-1]
        q_padded = torch.zeros(B, max_q, H, C_q, device=device, dtype=q.dtype)
        k_padded = torch.zeros(B, max_kv, H, C_q, device=device, dtype=k.dtype)
        v_padded = torch.zeros(B, max_kv, H, C_v, device=device, dtype=v.dtype)
        q_offset = 0
        kv_offset = 0
        for b in range(B):
            ql = q_seqlen[b]
            kvl = kv_seqlen[b] if kv_seqlen is not None else ql
            q_padded[b, :ql] = q[q_offset:q_offset+ql]
            k_padded[b, :kvl] = k[kv_offset:kv_offset+kvl]
            v_padded[b, :kvl] = v[kv_offset:kv_offset+kvl]
            q_offset += ql
            kv_offset += kvl
        q_padded = q_padded.permute(0, 2, 1, 3)
        k_padded = k_padded.permute(0, 2, 1, 3)
        v_padded = v_padded.permute(0, 2, 1, 3)
        out_padded = sdpa_fn(q_padded, k_padded, v_padded)
        out_padded = out_padded.permute(0, 2, 1, 3)
        out_list = []
        for b in range(B):
            ql = q_seqlen[b]
            out_list.append(out_padded[b, :ql])
        out = torch.cat(out_list, dim=0)
"""

    src = src.replace(
        "    else:\n        raise ValueError(f\"Unknown attention module: {config.ATTN}\")",
        sdpa_block + "    else:\n        raise ValueError(f\"Unknown attention module: {config.ATTN}\")",
    )
    write_file(path, src)


def patch_image_feature_extractor():
    """Add device property and replace .cuda() calls with device-aware code."""
    path = os.path.join(TRELLIS_ROOT, "trellis2/modules/image_feature_extractor.py")
    src = read_file(path)

    if "def device(self)" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    # DinoV2FeatureExtractor: add device property, fix cuda()
    src = src.replace(
        "    def to(self, device):\n"
        "        self.model.to(device)\n"
        "\n"
        "    def cuda(self):\n"
        "        self.model.cuda()\n"
        "\n"
        "    def cpu(self):\n"
        "        self.model.cpu()\n"
        "    \n"
        "    @torch.no_grad()\n"
        "    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:\n"
        '        """\n'
        "        Extract features from the image.",

        "    @property\n"
        "    def device(self):\n"
        "        return next(self.model.parameters()).device\n"
        "\n"
        "    def to(self, device):\n"
        "        self.model.to(device)\n"
        "\n"
        "    def cuda(self):\n"
        "        self.model.to(self.device)\n"
        "\n"
        "    def cpu(self):\n"
        "        self.model.cpu()\n"
        "\n"
        "    @torch.no_grad()\n"
        "    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:\n"
        '        """\n'
        "        Extract features from the image.",
        1,  # only first occurrence
    )

    # Fix hardcoded .cuda() in both extractors
    src = src.replace(
        "            image = torch.stack(image).cuda()",
        "            image = torch.stack(image).to(self.device)",
    )
    src = src.replace(
        "        image = self.transform(image).cuda()",
        "        image = self.transform(image).to(self.device)",
    )

    # DinoV3FeatureExtractor: add device property, fix cuda()
    # The second class has the same to/cuda/cpu pattern
    src = src.replace(
        "    def to(self, device):\n"
        "        self.model.to(device)\n"
        "\n"
        "    def cuda(self):\n"
        "        self.model.cuda()\n"
        "\n"
        "    def cpu(self):\n"
        "        self.model.cpu()",

        "    @property\n"
        "    def device(self):\n"
        "        return next(self.model.parameters()).device\n"
        "\n"
        "    def to(self, device):\n"
        "        self.model.to(device)\n"
        "\n"
        "    def cuda(self):\n"
        "        self.model.to(self.device)\n"
        "\n"
        "    def cpu(self):\n"
        "        self.model.cpu()",
    )

    # Fix DINOv3 model.layer -> model.model.layer (HuggingFace structure)
    src = src.replace(
        "        for i, layer_module in enumerate(self.model.layer):",
        "        layers = self.model.model.layer if hasattr(self.model, 'model') and hasattr(self.model.model, 'layer') else self.model.layer\n"
        "        for i, layer_module in enumerate(layers):",
    )

    write_file(path, src)


def patch_birefnet():
    """Add device property and fix hardcoded .cuda()/.to('cuda') calls."""
    path = os.path.join(TRELLIS_ROOT, "trellis2/pipelines/rembg/BiRefNet.py")
    src = read_file(path)

    if "def device(self)" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    # Replace to/cuda/cpu block
    src = src.replace(
        "    def to(self, device: str):\n"
        "        self.model.to(device)\n"
        "\n"
        "    def cuda(self):\n"
        "        self.model.cuda()\n"
        "\n"
        "    def cpu(self):\n"
        "        self.model.cpu()",

        "    @property\n"
        "    def device(self):\n"
        "        return next(self.model.parameters()).device\n"
        "\n"
        "    def to(self, device):\n"
        "        self.model.to(device)\n"
        "        return self\n"
        "\n"
        "    def cuda(self):\n"
        "        self.model.to(self.device)\n"
        "\n"
        "    def cpu(self):\n"
        "        self.model.cpu()",
    )

    # Fix hardcoded .to("cuda") in __call__
    src = src.replace(
        '.unsqueeze(0).to("cuda")',
        ".unsqueeze(0).to(self.device)",
    )

    write_file(path, src)


def patch_mesh_base():
    """Guard cumesh/flex_gemm imports and unconditionally skip in-place mesh
    ops. TRELLIS.2 calls fill_holes/remove_faces/simplify during decode on the
    full 400K-vertex mesh; the Metal port of cumesh segfaults on inputs that
    large, so we skip these decode-time ops entirely. Post-decode mesh
    simplification happens later via fast_simplification before texture bake.
    """
    path = os.path.join(TRELLIS_ROOT, "trellis2/representations/mesh/base.py")
    src = read_file(path)

    if "except (ImportError, RuntimeError)" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    # Guard imports — cumesh/flex_gemm may or may not be present
    src = src.replace(
        "import cumesh\n"
        "from flex_gemm.ops.grid_sample import grid_sample_3d",
        "try:\n"
        "    import cumesh\n"
        "except (ImportError, RuntimeError):\n"
        "    cumesh = None\n"
        "try:\n"
        "    from flex_gemm.ops.grid_sample import grid_sample_3d\n"
        "except (ImportError, RuntimeError):\n"
        '    def grid_sample_3d(*args, **kwargs):\n'
        '        raise RuntimeError("flex_gemm requires CUDA")',
    )

    # Unconditionally return from fill_holes (Metal cumesh segfaults on large meshes)
    src = src.replace(
        "    def fill_holes(self, max_hole_perimeter=3e-2):\n"
        "        vertices = self.vertices.cuda()\n"
        "        faces = self.faces.cuda()",
        "    def fill_holes(self, max_hole_perimeter=3e-2):\n"
        "        return  # Skip — Metal cumesh segfaults on large decode meshes\n"
        "        vertices = self.vertices.to(self.device)\n"
        "        faces = self.faces.to(self.device)",
    )

    # Unconditionally return from remove_faces
    src = src.replace(
        "    def remove_faces(self, face_mask: torch.Tensor):\n"
        "        vertices = self.vertices.cuda()\n"
        "        faces = self.faces.cuda()",
        "    def remove_faces(self, face_mask: torch.Tensor):\n"
        "        return\n"
        "        vertices = self.vertices.to(self.device)\n"
        "        faces = self.faces.to(self.device)",
    )

    # Unconditionally return from simplify
    src = src.replace(
        "    def simplify(self, target=1000000, verbose: bool=False, options: dict={}):\n"
        "        vertices = self.vertices.cuda()\n"
        "        faces = self.faces.cuda()",
        "    def simplify(self, target=1000000, verbose: bool=False, options: dict={}):\n"
        "        return\n"
        "        vertices = self.vertices.to(self.device)\n"
        "        faces = self.faces.to(self.device)",
    )

    write_file(path, src)


def patch_fdg_vae():
    """Force our pure-Python flexible_dual_grid_to_mesh over any installed
    o_voxel. The Metal-port o_voxel.convert segfaults on decoder output even
    when it imports cleanly, so we always prefer our stub implementation.
    """
    path = os.path.join(TRELLIS_ROOT, "trellis2/models/sc_vaes/fdg_vae.py")
    src = read_file(path)

    if "o_voxel_override_convert" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    src = src.replace(
        "from o_voxel.convert import flexible_dual_grid_to_mesh\n",
        "# Force pure-Python mesh extraction — real o_voxel.convert (CUDA or Metal port)\n"
        "# segfaults on decoder output. Import our stub version explicitly.\n"
        "# stubs/ is appended (not prepended) so a pip-installed o_voxel still wins\n"
        "# for other submodules like o_voxel.postprocess.\n"
        "import sys, os\n"
        "_stubs = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'stubs')\n"
        "if _stubs not in sys.path:\n"
        "    sys.path.append(_stubs)\n"
        "try:\n"
        "    from o_voxel_override_convert import flexible_dual_grid_to_mesh\n"
        "except ImportError:\n"
        "    try:\n"
        "        from o_voxel.convert import flexible_dual_grid_to_mesh\n"
        "    except (ImportError, RuntimeError):\n"
        "        def flexible_dual_grid_to_mesh(*args, **kwargs):\n"
        '            raise RuntimeError("flexible_dual_grid_to_mesh unavailable")\n',
    )
    write_file(path, src)


def patch_pipeline():
    """Guard torch.cuda.empty_cache() call."""
    path = os.path.join(TRELLIS_ROOT, "trellis2/pipelines/trellis2_image_to_3d.py")
    src = read_file(path)

    if "if torch.cuda.is_available():" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    src = src.replace(
        "        torch.cuda.empty_cache()\n",
        "        if torch.cuda.is_available():\n"
        "            torch.cuda.empty_cache()\n",
    )
    write_file(path, src)


def patch_pipeline_base():
    """Fix hardcoded cuda device in Pipeline.cuda()."""
    path = os.path.join(TRELLIS_ROOT, "trellis2/pipelines/base.py")
    src = read_file(path)

    if "torch.backends.mps.is_available()" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    src = src.replace(
        '        self.to(torch.device("cuda"))',
        '        self.to(torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda"))',
    )
    write_file(path, src)


def install_conv_backend():
    """Copy the pure-PyTorch sparse convolution backend into place."""
    src = os.path.join(BACKENDS_DIR, "conv_none.py")
    dst = os.path.join(TRELLIS_ROOT, "trellis2/modules/sparse/conv/conv_none.py")

    if os.path.exists(dst):
        print(f"  Already installed: trellis2/modules/sparse/conv/conv_none.py")
        return

    shutil.copy2(src, dst)
    print(f"  Installed: trellis2/modules/sparse/conv/conv_none.py")


def install_mesh_extract():
    """Copy the pure-Python mesh extraction into the o_voxel stub and also as
    a flat override module. The flat module takes precedence over any
    Metal/CUDA o_voxel package that might be installed alongside us.
    """
    stubs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stubs")
    ovoxel_dir = os.path.join(stubs_dir, "o_voxel")
    os.makedirs(ovoxel_dir, exist_ok=True)

    src = os.path.join(BACKENDS_DIR, "mesh_extract.py")

    # Flat override module — loaded before real o_voxel by fdg_vae patch
    flat_dst = os.path.join(stubs_dir, "o_voxel_override_convert.py")
    shutil.copy2(src, flat_dst)
    print(f"  Installed: stubs/o_voxel_override_convert.py")

    # Also the stub package for environments without any o_voxel install
    dst = os.path.join(ovoxel_dir, "convert.py")
    shutil.copy2(src, dst)
    print(f"  Installed: stubs/o_voxel/convert.py")

    # __init__.py
    with open(os.path.join(ovoxel_dir, "__init__.py"), "w") as f:
        f.write("pass\n")

    # io.py stub
    with open(os.path.join(ovoxel_dir, "io.py"), "w") as f:
        f.write('def read(*args, **kwargs):\n    raise RuntimeError("o_voxel.io requires CUDA")\n\n')
        f.write('def write(*args, **kwargs):\n    raise RuntimeError("o_voxel.io requires CUDA")\n\n')
        f.write('def read_vxz(*args, **kwargs):\n    raise RuntimeError("o_voxel.io requires CUDA")\n')

    # rasterize.py stub
    with open(os.path.join(ovoxel_dir, "rasterize.py"), "w") as f:
        f.write('class VoxelRenderer:\n    def __init__(self, *args, **kwargs):\n')
        f.write('        raise RuntimeError("o_voxel.rasterize requires CUDA")\n')


def install_stubs():
    """Create all stub modules for CUDA-only libraries."""
    stubs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stubs")

    from backends.stubs import install_stubs as _install
    _install(stubs_dir)
    print(f"  Installed stub packages in stubs/")


def main():
    print("Applying MPS compatibility patches to TRELLIS.2...")
    print(f"  TRELLIS root: {TRELLIS_ROOT}")
    print()

    if not os.path.isdir(TRELLIS_ROOT):
        print(f"Error: TRELLIS.2 not found at {TRELLIS_ROOT}")
        print("Run setup.sh first to clone the repository.")
        return False

    patch_sparse_config()
    patch_sparse_attention()
    patch_image_feature_extractor()
    patch_birefnet()
    patch_mesh_base()
    patch_fdg_vae()
    patch_pipeline()
    patch_pipeline_base()
    install_conv_backend()
    install_mesh_extract()

    print()
    print("All patches applied.")
    return True


if __name__ == "__main__":
    main()
