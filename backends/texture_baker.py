"""
UV unwrap + texture baking for TRELLIS.2 meshes on Apple Silicon.

Replaces nvdiffrast (CUDA-only) with:
  - xatlas for UV unwrapping (C++ library, no GPU needed)
  - Pure-numpy software rasterizer for baking voxel attributes into texture maps

Produces GLB files with proper PBR textures (base color, metallic, roughness).
"""

import numpy as np
import torch


def uv_unwrap(vertices, faces):
    """
    Compute UV coordinates for a mesh using xatlas.

    Returns:
        new_vertices: Remapped vertices (may have more than input due to seams)
        new_faces: Triangle indices into new_vertices
        uvs: UV coordinates per new vertex, in [0, 1]
        vmapping: Maps new vertex index -> original vertex index
    """
    import xatlas

    v = np.ascontiguousarray(vertices.astype(np.float32))
    f = np.ascontiguousarray(faces.astype(np.uint32))

    vmapping, indices, uvs = xatlas.parametrize(v, f)

    new_vertices = v[vmapping]
    new_faces = indices.reshape(-1, 3)

    return new_vertices, new_faces, uvs, vmapping


def bake_texture(vertices, faces, uvs, voxel_coords, voxel_attrs, origin, voxel_size, texture_size=2048):
    """
    Bake voxel attributes into a UV-mapped texture image.

    For each texel in the output image:
      1. Find which triangle covers this UV position
      2. Compute barycentric coordinates
      3. Interpolate 3D position from triangle vertices
      4. Sample voxel grid at that 3D position (nearest neighbor)
      5. Write the sampled attributes to the texel

    Args:
        vertices: [N, 3] mesh vertices
        faces: [F, 3] triangle indices
        uvs: [N, 2] UV coordinates in [0, 1]
        voxel_coords: [V, 3] voxel grid coordinates
        voxel_attrs: [V, C] voxel attributes (base_color=0:3, metallic=3, roughness=4, alpha=5)
        origin: [3] voxel grid origin
        voxel_size: float
        texture_size: output texture resolution

    Returns:
        base_color: [H, W, 3] uint8 RGB texture
        metallic_roughness: [H, W, 3] uint8 (R=0, G=roughness, B=metallic) for glTF
    """
    H = W = texture_size

    # Build voxel spatial hash for fast lookup
    coord_to_idx = {}
    coords_np = voxel_coords.numpy() if isinstance(voxel_coords, torch.Tensor) else voxel_coords
    attrs_np = voxel_attrs.numpy() if isinstance(voxel_attrs, torch.Tensor) else voxel_attrs
    origin_np = origin.numpy() if isinstance(origin, torch.Tensor) else np.array(origin)

    for i in range(len(coords_np)):
        key = (int(coords_np[i, 0]), int(coords_np[i, 1]), int(coords_np[i, 2]))
        coord_to_idx[key] = i

    # Rasterize triangles in UV space
    base_color = np.zeros((H, W, 3), dtype=np.float32)
    metallic = np.zeros((H, W), dtype=np.float32)
    roughness = np.ones((H, W), dtype=np.float32)  # default roughness = 1
    mask = np.zeros((H, W), dtype=bool)

    # Process each triangle
    n_faces = len(faces)
    print(f"  Baking {n_faces:,} triangles into {texture_size}x{texture_size} texture...")

    for fi in range(n_faces):
        if fi % 50000 == 0 and fi > 0:
            print(f"    {fi:,}/{n_faces:,} ({100*fi/n_faces:.0f}%)")

        i0, i1, i2 = faces[fi]

        # UV coords of this triangle's vertices, scaled to pixel space
        uv0 = uvs[i0] * np.array([W - 1, H - 1])
        uv1 = uvs[i1] * np.array([W - 1, H - 1])
        uv2 = uvs[i2] * np.array([W - 1, H - 1])

        # 3D positions
        p0 = vertices[i0]
        p1 = vertices[i1]
        p2 = vertices[i2]

        # Bounding box in UV pixel space
        min_x = max(int(np.floor(min(uv0[0], uv1[0], uv2[0]))), 0)
        max_x = min(int(np.ceil(max(uv0[0], uv1[0], uv2[0]))), W - 1)
        min_y = max(int(np.floor(min(uv0[1], uv1[1], uv2[1]))), 0)
        max_y = min(int(np.ceil(max(uv0[1], uv1[1], uv2[1]))), H - 1)

        if max_x < min_x or max_y < min_y:
            continue

        # Precompute barycentric denominator
        d00 = uv1[0] - uv0[0]
        d01 = uv2[0] - uv0[0]
        d10 = uv1[1] - uv0[1]
        d11 = uv2[1] - uv0[1]
        denom = d00 * d11 - d01 * d10
        if abs(denom) < 1e-10:
            continue
        inv_denom = 1.0 / denom

        # Scan pixels in bounding box
        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                # Barycentric coordinates
                dx = px - uv0[0]
                dy = py - uv0[1]
                u = (dx * d11 - d01 * dy) * inv_denom
                v = (d00 * dy - dx * d10) * inv_denom
                w = 1.0 - u - v

                if u < -0.001 or v < -0.001 or w < -0.001:
                    continue

                # Interpolate 3D position
                pos = w * p0 + u * p1 + v * p2

                # Convert to voxel grid coords
                grid_pos = (pos - origin_np) / voxel_size

                # Nearest-neighbor voxel lookup
                gx = int(round(grid_pos[0]))
                gy = int(round(grid_pos[1]))
                gz = int(round(grid_pos[2]))

                # Search small neighborhood
                best_idx = None
                for ddx in range(-1, 2):
                    for ddy in range(-1, 2):
                        for ddz in range(-1, 2):
                            key = (gx + ddx, gy + ddy, gz + ddz)
                            if key in coord_to_idx:
                                best_idx = coord_to_idx[key]
                                break
                        if best_idx is not None:
                            break
                    if best_idx is not None:
                        break

                if best_idx is not None:
                    a = attrs_np[best_idx]
                    # Attrs are already in [0,1] for color, just clip
                    base_color[py, px] = np.clip(a[0:3], 0, 1)
                    if len(a) > 3:
                        metallic[py, px] = np.clip(float(a[3]), 0, 1)
                    if len(a) > 4:
                        roughness[py, px] = np.clip(float(a[4]), 0, 1)
                    mask[py, px] = True

    # Aggressive hole-filling: iteratively dilate colored pixels into empty neighbors
    from scipy.ndimage import binary_dilation, uniform_filter
    print(f"  Filling texture holes...")
    for iteration in range(8):
        dilated = binary_dilation(mask, iterations=1)
        unfilled = dilated & ~mask
        if not unfilled.any():
            break
        for c in range(3):
            channel = base_color[:, :, c]
            blurred = uniform_filter(channel, size=3)
            channel[unfilled] = blurred[unfilled]
        mask = dilated

    # Gamma correction: linear -> sRGB for correct display
    # The voxel attrs are in linear color space; displays expect sRGB
    base_color = np.power(np.clip(base_color, 0, 1), 1.0 / 2.2)

    # Convert to uint8
    base_color_img = (base_color * 255).astype(np.uint8)

    # glTF metallic-roughness: R=0, G=roughness, B=metallic
    mr_img = np.zeros((H, W, 3), dtype=np.uint8)
    mr_img[:, :, 1] = (roughness * 255).astype(np.uint8)
    mr_img[:, :, 2] = (metallic * 255).astype(np.uint8)

    coverage = mask.sum() / (H * W) * 100
    print(f"  Texture coverage: {coverage:.1f}%")

    return base_color_img, mr_img, mask


def export_glb_with_texture(vertices, faces, uvs, base_color_img, mr_img=None, output_path="output.glb"):
    """
    Export mesh with UV-mapped PBR textures as GLB.
    """
    import trimesh
    from PIL import Image

    # Create trimesh with UV coordinates
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Create material with texture
    base_color_pil = Image.fromarray(base_color_img)

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=base_color_pil,
        metallicFactor=0.0,
        roughnessFactor=0.8,
    )

    if mr_img is not None:
        mr_pil = Image.fromarray(mr_img)
        material.metallicRoughnessTexture = mr_pil

    # Apply texture visual
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uvs,
        material=material,
    )

    mesh.export(output_path)
    return output_path
