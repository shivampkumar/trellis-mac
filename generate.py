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
from PIL import Image


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
    img = Image.open(args.image)
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
    print(f"Time: {t_gen:.1f}s")

    # Extract vertex colors from voxel attributes
    has_color = False
    colors = None

    # Try pre-computed vertex attrs first
    if hasattr(mesh_out, "vertex_attrs") and mesh_out.vertex_attrs is not None:
        colors = mesh_out.vertex_attrs[:, :3].cpu().numpy()
        colors = np.clip(colors, 0, 1)
        has_color = True

    # Otherwise, sample from voxel grid using nearest-neighbor
    if not has_color and hasattr(mesh_out, "attrs") and mesh_out.attrs is not None:
        print("Sampling vertex colors from voxel attributes...")
        coords = mesh_out.coords.cpu().float()       # [N_voxels, 3]
        attrs = mesh_out.attrs.cpu().float()          # [N_voxels, C]
        origin = mesh_out.origin.cpu().float()        # [3]
        vs = mesh_out.voxel_size

        # Convert mesh vertices to voxel grid coordinates
        verts_t = torch.from_numpy(verts).float()
        grid_pos = (verts_t - origin) / vs            # [N_verts, 3]

        # Nearest-neighbor lookup: for each vertex, find closest voxel
        # Build a spatial hash for fast lookup
        coord_map = {}
        for i in range(coords.shape[0]):
            key = (int(coords[i, 0].item()), int(coords[i, 1].item()), int(coords[i, 2].item()))
            coord_map[key] = i

        vertex_colors = np.zeros((len(verts), 3), dtype=np.float32)
        matched = 0
        for vi in range(len(verts)):
            # Round to nearest voxel
            gx, gy, gz = int(round(grid_pos[vi, 0].item())), int(round(grid_pos[vi, 1].item())), int(round(grid_pos[vi, 2].item()))
            # Search in a small neighborhood
            best_idx = None
            for dx in range(3):
                for dy in range(3):
                    for dz in range(3):
                        key = (gx + dx - 1, gy + dy - 1, gz + dz - 1)
                        if key in coord_map:
                            best_idx = coord_map[key]
                            break
                    if best_idx is not None:
                        break
                if best_idx is not None:
                    break
            if best_idx is not None:
                a = attrs[best_idx]
                # Layout: base_color=0:3, metallic=3, roughness=4, alpha=5
                rgb = a[0:3].numpy()
                # Sigmoid if values are in logit space
                if rgb.max() > 1.5 or rgb.min() < -0.5:
                    rgb = 1.0 / (1.0 + np.exp(-rgb))
                vertex_colors[vi] = np.clip(rgb, 0, 1)
                matched += 1

        if matched > len(verts) * 0.1:
            colors = vertex_colors
            has_color = True
            print(f"  Matched {matched:,}/{len(verts):,} vertices ({100*matched/len(verts):.0f}%)")
        else:
            print(f"  Low match rate ({matched}/{len(verts)}), skipping vertex colors")

    # Save OBJ
    obj_path = f"{args.output}.obj"
    with open(obj_path, "w") as f:
        for i, v in enumerate(verts):
            if has_color:
                r, g, b = colors[i]
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r:.4f} {g:.4f} {b:.4f}\n")
            else:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"Saved: {obj_path}")

    # Save GLB
    try:
        import trimesh

        if has_color:
            vertex_colors = np.concatenate(
                [(colors * 255).astype(np.uint8), np.full((len(colors), 1), 255, dtype=np.uint8)],
                axis=1,
            )
            tm = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vertex_colors)
        else:
            tm = trimesh.Trimesh(vertices=verts, faces=faces)
        glb_path = f"{args.output}.glb"
        tm.export(glb_path)
        print(f"Saved: {glb_path}")
    except ImportError:
        print("Install trimesh for GLB export: pip install trimesh")
    except Exception as e:
        print(f"GLB export failed: {e}")

    print(f"\nDone in {t_gen:.1f}s.")


if __name__ == "__main__":
    main()
