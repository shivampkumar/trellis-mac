"""
Generate a 3D mesh from a single image using TRELLIS.2 on Apple Silicon.
"""

import sys
import os

# Set up backends before any TRELLIS imports
os.environ["ATTN_BACKEND"] = "sdpa"
os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
os.environ["SPARSE_CONV_BACKEND"] = "none"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stubs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "TRELLIS.2"))

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
        print(f"\nBaking PBR textures ({tex_size}x{tex_size})...")
        t_bake = time.time()

        from backends.texture_baker import uv_unwrap, bake_texture, export_glb_with_texture

        voxel_coords = mesh_out.coords.cpu().float()
        voxel_attrs = mesh_out.attrs.cpu().float()
        origin = mesh_out.origin.cpu().float()
        vs = mesh_out.voxel_size

        # UV unwrap
        print("  UV unwrapping with xatlas...")
        new_verts, new_faces, uvs, vmapping = uv_unwrap(verts, faces)
        print(f"  UV unwrap: {len(verts):,} -> {len(new_verts):,} vertices ({len(uvs):,} UVs)")

        # Bake texture
        base_color_img, mr_img, mask = bake_texture(
            new_verts, new_faces, uvs,
            voxel_coords.numpy(), voxel_attrs.numpy(),
            origin.numpy(), vs,
            texture_size=tex_size,
        )

        # Save texture images
        PILImage.fromarray(base_color_img).save(f"{args.output}_basecolor.png")
        print(f"  Saved: {args.output}_basecolor.png")

        # Export GLB with proper textures
        glb_path = f"{args.output}.glb"
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
