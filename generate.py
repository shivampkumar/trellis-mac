"""
Generate a 3D mesh from a single image using TRELLIS.2 on Apple Silicon.
"""

import sys
import os

# Set up backends before any TRELLIS imports. Use setdefault so the caller
# can override from the environment (e.g. SPARSE_CONV_BACKEND=flex_gemm).
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Add paths. stubs/ is appended (not prepended) so a pip-installed o_voxel
# wins over our package stub — the flat override module o_voxel_override_convert
# is still discoverable either way because it doesn't collide with any package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "TRELLIS.2"))
sys.path.append(os.path.join(os.path.dirname(__file__), "stubs"))

import argparse
import time
import torch
import numpy as np
from PIL import Image as PILImage


def main():
    parser = argparse.ArgumentParser(description="Generate 3D mesh from an image using TRELLIS.2")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", default="output_3d", help="Output filename without extension (default: output_3d)")
    parser.add_argument(
        "--pipeline-type", default="512",
        choices=["512", "1024", "1024_cascade"],
        help="Pipeline resolution (default: 512)",
    )
    parser.add_argument(
        "--texture-size", type=int, default=1024,
        choices=[512, 1024, 2048],
        help="Texture resolution for PBR baking (default: 1024)",
    )
    parser.add_argument(
        "--no-texture", action="store_true",
        help="Skip texture baking, export geometry only",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: {args.image} not found")
        sys.exit(1)

    print("=" * 60)
    print("TRELLIS.2 on Apple Silicon")
    print("=" * 60)

    # Load pipeline
    print("\nLoading pipeline...")
    t0 = time.time()
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    print(f"Loaded in {time.time() - t0:.0f}s")

    # Move to MPS
    pipeline.to(torch.device("mps"))
    print("Device: MPS")

    # Load image
    img = PILImage.open(args.image)
    print(f"Input: {args.image} ({img.size[0]}x{img.size[1]})")

    # Generate
    print(f"\nGenerating 3D model (pipeline={args.pipeline_type}, seed={args.seed})...")
    t0 = time.time()

    outputs = pipeline.run(img, seed=args.seed, pipeline_type=args.pipeline_type)
    t_gen = time.time() - t0

    mesh_out = outputs[0] if isinstance(outputs, list) else outputs

    verts = mesh_out.vertices.cpu().numpy()
    faces = mesh_out.faces.cpu().numpy()
    print(f"\nMesh: {verts.shape[0]:,} vertices, {faces.shape[0]:,} triangles")
    print(f"Generation time: {t_gen:.1f}s")

    # Check for voxel texture data
    has_voxels = hasattr(mesh_out, "attrs") and mesh_out.attrs is not None
    tex_size = args.texture_size

    if has_voxels and not args.no_texture:
        # Try Metal-accelerated bake via o_voxel + mtldiffrast if available.
        # Catch AttributeError too: our stubs/o_voxel/ stub has no .postprocess
        # submodule, so a shadowing stub package trips getattr, not import.
        try:
            import o_voxel.postprocess
            backend = getattr(o_voxel.postprocess, '_BACKEND', None)
            has_dr = getattr(o_voxel.postprocess, '_HAS_DR', False)
            use_metal = backend == 'metal' and has_dr
            if use_metal and not getattr(o_voxel.postprocess, '_HAS_FLEX_GEMM', False):
                # o_voxel's _grid_sample_3d fallback returns [B*C, M] but the
                # bake consumes it as [M, C]. Patch it to transpose before the
                # reshape. We avoid installing flex_gemm itself because its
                # import slows the diffusion hot path ~10x on MPS.
                import torch.nn.functional as _F_gs
                def _gs3d_fix(feats, coords, shape, grid, mode='trilinear'):
                    B, C = shape[0], shape[1]
                    D, H, W = shape[2], shape[3], shape[4]
                    device = feats.device
                    dense_vol = torch.zeros(B, C, D, H, W, dtype=feats.dtype, device=device)
                    batch_idx = coords[:, 0].long()
                    cx = coords[:, 1].long(); cy = coords[:, 2].long(); cz = coords[:, 3].long()
                    dense_vol[batch_idx, :, cx, cy, cz] = feats
                    grid_norm = torch.stack([
                        grid[..., 2] / (W - 1) * 2 - 1,
                        grid[..., 1] / (H - 1) * 2 - 1,
                        grid[..., 0] / (D - 1) * 2 - 1,
                    ], dim=-1).reshape(B, 1, 1, -1, 3)
                    sampled = _F_gs.grid_sample(
                        dense_vol, grid_norm, mode='bilinear',
                        align_corners=True, padding_mode='border',
                    )
                    M = grid.shape[1]
                    return sampled.reshape(B, C, M).permute(0, 2, 1).reshape(B * M, C)
                o_voxel.postprocess._grid_sample_3d = _gs3d_fix
        except (ImportError, AttributeError):
            use_metal = False

        glb_path = f"{args.output}.glb"
        t_bake = time.time()

        if use_metal:
            print(f"\nBaking PBR textures via Metal ({tex_size}x{tex_size})...")
            import o_voxel

            # Pre-simplify mesh to avoid mtlbvh crash on large meshes.
            # Target ~200K faces — keeps detail, avoids Metal BVH issues.
            import fast_simplification
            verts_np = mesh_out.vertices.cpu().numpy()
            faces_np = mesh_out.faces.cpu().numpy()
            target_faces = min(200000, len(faces_np))
            if len(faces_np) > target_faces:
                ratio = 1.0 - (target_faces / len(faces_np))
                print(f"  Simplifying mesh: {len(faces_np):,} -> ~{target_faces:,} faces")
                simp_verts, simp_faces = fast_simplification.simplify(verts_np, faces_np, ratio)
                simp_verts_t = torch.from_numpy(simp_verts).float().to(mesh_out.vertices.device)
                simp_faces_t = torch.from_numpy(simp_faces.astype('int32')).to(mesh_out.faces.device)
            else:
                simp_verts_t = mesh_out.vertices
                simp_faces_t = mesh_out.faces

            # Move all mesh tensors to CPU — o_voxel.to_glb mixes device-neutral
            # AABB tensor with mesh tensors; keep everything on CPU to avoid mismatch.
            glb = o_voxel.postprocess.to_glb(
                vertices=simp_verts_t.cpu(),
                faces=simp_faces_t.cpu(),
                attr_volume=mesh_out.attrs.cpu(),
                coords=mesh_out.coords.cpu(),
                attr_layout=mesh_out.layout,
                voxel_size=mesh_out.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=target_faces,
                texture_size=tex_size,
                verbose=True,
            )
            glb.export(glb_path)
            print(f"  Saved: {glb_path}")
        else:
            print(f"\nBaking PBR textures via KDTree ({tex_size}x{tex_size})...")
            from backends.texture_baker import uv_unwrap, bake_texture, export_glb_with_texture

            voxel_coords = mesh_out.coords.cpu().float()
            voxel_attrs = mesh_out.attrs.cpu().float()
            origin = mesh_out.origin.cpu().float()
            vs = mesh_out.voxel_size

            print("  UV unwrapping with xatlas...")
            new_verts, new_faces, uvs, vmapping = uv_unwrap(verts, faces)
            print(f"  UV unwrap: {len(verts):,} -> {len(new_verts):,} vertices")

            base_color_img, mr_img, mask = bake_texture(
                new_verts, new_faces, uvs,
                voxel_coords.numpy(), voxel_attrs.numpy(),
                origin.numpy(), vs,
                texture_size=tex_size,
            )

            PILImage.fromarray(base_color_img).save(f"{args.output}_basecolor.png")
            export_glb_with_texture(new_verts, new_faces, uvs, base_color_img, mr_img, glb_path)
            print(f"  Saved: {glb_path}")

        t_bake_total = time.time() - t_bake
        print(f"  Bake time: {t_bake_total:.0f}s")
    else:
        # Fallback: vertex colors only
        print("\nExporting with vertex colors (use --texture-size for PBR textures)...")
        import trimesh
        tm = trimesh.Trimesh(vertices=verts, faces=faces)
        glb_path = f"{args.output}.glb"
        tm.export(glb_path)
        print(f"Saved: {glb_path}")

    # Also save OBJ
    obj_path = f"{args.output}.obj"
    with open(obj_path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"Saved: {obj_path}")

    print(f"\nTotal time: {t_gen:.1f}s generation + baking")


if __name__ == "__main__":
    main()
