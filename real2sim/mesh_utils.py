"""Mesh conversion helpers for SAM3D output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def choose_sam3d_mesh(
    raw: trimesh.Trimesh | trimesh.Scene,
    processed: trimesh.Trimesh | trimesh.Scene,
    scale: np.ndarray,
) -> trimesh.Trimesh:
    try:
        mesh = to_trimesh(processed)
        if mesh.vertices.size:
            return mesh
    except Exception:
        pass

    mesh = to_trimesh(raw)
    mesh.apply_scale(np.asarray(scale, dtype=float))
    return mesh


def process_sam3d_mesh(
    raw: trimesh.Trimesh | trimesh.Scene,
    scale: np.ndarray,
    target_faces: int | None = 10000,
) -> trimesh.Trimesh:
    """Match the SAM3D raw-mesh processing used by Foundation Pose callers."""

    mesh = to_trimesh(raw)
    mesh.apply_scale(np.asarray(scale, dtype=float))
    return simplify_mesh(mesh, target_faces)


def export_color_cluster_visual_meshes(
    mesh: trimesh.Trimesh,
    output_dir: Path,
    safe_name: str,
    *,
    max_clusters: int,
) -> list[dict[str, object]]:
    """Approximate vertex-colored meshes with a small set of colored OBJ parts."""

    max_clusters = int(max_clusters)
    if max_clusters <= 0 or len(mesh.faces) == 0:
        return []

    colors = _vertex_colors(mesh)
    if colors is None:
        return []
    rgba = np.asarray(colors, dtype=np.float64)
    if rgba.shape[1] < 3:
        return []
    if rgba.max(initial=0.0) > 1.0:
        rgba[:, : min(4, rgba.shape[1])] /= 255.0
    if rgba.shape[1] == 3:
        alpha = np.ones((rgba.shape[0], 1), dtype=np.float64)
        rgba = np.concatenate([rgba, alpha], axis=1)

    face_rgba = rgba[np.asarray(mesh.faces, dtype=np.int64)].mean(axis=1)
    face_rgb = face_rgba[:, :3]
    bin_ids = _quantized_color_bins(face_rgb)
    unique_bins, counts = np.unique(bin_ids, return_counts=True)
    order = np.argsort(counts)[::-1]
    selected_bins = unique_bins[order[: min(max_clusters, len(unique_bins))]]
    palette = np.asarray(
        [face_rgba[bin_ids == bin_id].mean(axis=0) for bin_id in selected_bins],
        dtype=np.float64,
    )
    distances = np.linalg.norm(face_rgb[:, None, :] - palette[None, :, :3], axis=2)
    assignments = np.argmin(distances, axis=1)

    specs: list[dict[str, object]] = []
    for cluster_index in range(len(palette)):
        face_indices = np.flatnonzero(assignments == cluster_index)
        if face_indices.size == 0:
            continue
        part = mesh.submesh([face_indices], append=True, repair=False)
        if not isinstance(part, trimesh.Trimesh) or len(part.faces) == 0:
            continue
        path = output_dir / f"{safe_name}_visual_{cluster_index:02d}.obj"
        part.export(path)
        color = np.clip(palette[cluster_index], 0.0, 1.0)
        specs.append(
            {
                "file": f"meshes/{path.name}",
                "rgba": [float(v) for v in color[:4]],
            }
        )
    return specs


def to_trimesh(mesh_like: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(mesh_like, trimesh.Trimesh):
        return mesh_like.copy()
    if isinstance(mesh_like, trimesh.Scene):
        try:
            dumped = mesh_like.dump(concatenate=True)
            if isinstance(dumped, trimesh.Trimesh):
                return dumped.copy()
        except TypeError:
            pass
        geometries = [geom.copy() for geom in mesh_like.geometry.values()]
        if not geometries:
            raise ValueError("scene has no geometries")
        return trimesh.util.concatenate(geometries)
    raise TypeError(f"unsupported mesh type {type(mesh_like).__name__}")


def simplify_mesh(mesh: trimesh.Trimesh, target_faces: int | None) -> trimesh.Trimesh:
    if target_faces is None or len(mesh.faces) <= target_faces:
        return mesh

    source_vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    source_colors = _vertex_colors(mesh)
    for method_name in ("simplify_quadric_decimation", "simplify_quadratic_decimation"):
        method = getattr(mesh, method_name, None)
        if method is None:
            continue
        try:
            if method_name == "simplify_quadric_decimation":
                simplified = method(face_count=int(target_faces))
            else:
                simplified = method(int(target_faces))
            if isinstance(simplified, trimesh.Trimesh) and len(simplified.faces) > 0:
                _transfer_vertex_colors(source_vertices, source_colors, simplified)
                return simplified
        except Exception:
            continue
    try:
        simplified = _simplify_mesh_open3d(mesh, int(target_faces))
        if len(simplified.faces) > 0:
            _transfer_vertex_colors(source_vertices, source_colors, simplified)
            return simplified
    except Exception:
        pass
    return mesh


def _vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    colors = getattr(mesh.visual, "vertex_colors", None)
    if colors is None or len(colors) != len(mesh.vertices):
        return None
    return np.asarray(colors).copy()


def _quantized_color_bins(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    quantized = np.floor(rgb * 5.0).astype(np.int64)
    quantized = np.clip(quantized, 0, 5)
    return quantized[:, 0] * 36 + quantized[:, 1] * 6 + quantized[:, 2]


def _transfer_vertex_colors(
    source_vertices: np.ndarray,
    source_colors: np.ndarray | None,
    simplified: trimesh.Trimesh,
) -> None:
    if source_colors is None or len(source_colors) != len(source_vertices):
        return
    if len(simplified.vertices) == 0:
        return
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(source_vertices)
        _, indices = tree.query(np.asarray(simplified.vertices), k=1)
        simplified.visual = trimesh.visual.ColorVisuals(
            simplified,
            vertex_colors=source_colors[indices],
        )
    except Exception:
        return


def _simplify_mesh_open3d(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    import open3d as o3d

    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    colors = getattr(mesh.visual, "vertex_colors", None)
    if colors is not None and len(colors) == len(mesh.vertices):
        color_arr = np.asarray(colors, dtype=np.float64)
        if color_arr.shape[1] >= 3:
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(color_arr[:, :3] / 255.0)

    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh.remove_non_manifold_edges()
    o3d_mesh = o3d_mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_faces
    )
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_unreferenced_vertices()

    simplified = trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices, dtype=np.float64),
        faces=np.asarray(o3d_mesh.triangles, dtype=np.int64),
        process=False,
    )
    if o3d_mesh.has_vertex_colors():
        vertex_colors = np.clip(np.asarray(o3d_mesh.vertex_colors) * 255.0, 0, 255)
        simplified.visual = trimesh.visual.ColorVisuals(
            simplified,
            vertex_colors=vertex_colors.astype(np.uint8),
        )
    return simplified
