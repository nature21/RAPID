"""Affine resizing of a rigid target in its collision geometry's local axes."""

from __future__ import annotations

import mujoco
import numpy as np


def _rotation(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def _quaternion(matrix: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, matrix.reshape(9))
    return quaternion


def _resize_shape(
    element: mujoco.MjsGeom | mujoco.MjsSite,
    kind: int,
    size: np.ndarray,
    rotation: np.ndarray,
    transform: np.ndarray,
) -> None:
    """Keep analytic shapes exact; reject affine shapes they cannot represent."""
    axes = transform @ rotation
    if kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        radii = np.full(3, size[0]) if kind == mujoco.mjtGeom.mjGEOM_SPHERE else size
        axes, radii, _ = np.linalg.svd(axes @ np.diag(radii))
        if np.linalg.det(axes) < 0:
            axes[:, -1] *= -1
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE and np.allclose(radii, radii[0]):
            element.size = (radii[0], 0, 0)
        else:
            element.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
            element.size = radii
    else:
        factors = np.linalg.norm(axes, axis=0)
        axes = axes / factors
        if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-10):
            raise ValueError(f"anisotropic scaling would shear {element.name!r}; use a mesh")
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            element.size = size * factors
        elif kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            circular = np.isclose(factors[0], factors[1])
            round_caps = kind != mujoco.mjtGeom.mjGEOM_CAPSULE or np.allclose(factors, factors[0])
            if not circular or not round_caps:
                raise ValueError(f"anisotropic scaling of {element.name!r} requires a mesh")
            element.size = (size[0] * factors[0], size[1] * factors[2], size[2])
        else:
            raise ValueError(f"unsupported target shape for resizing: {element.name!r} ({kind})")
    element.quat = _quaternion(axes)
    element.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
    element.fromto = np.full(6, np.nan)


def _resize_mesh(
    scene: mujoco.MjSpec,
    geom: mujoco.MjsGeom,
    model: mujoco.MjModel,
    gid: int,
    transform: np.ndarray,
) -> None:
    """Copy compiled vertices and attributes so shared assets remain untouched."""
    mid = int(model.geom_dataid[gid])
    rotation = _rotation(model.geom_quat[gid])
    local_transform = rotation.T @ transform @ rotation
    mesh = scene.add_mesh(name=f"{geom.name}_resized")
    start, count = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    mesh.uservert = (model.mesh_vert[start:start + count] @ local_transform.T).reshape(-1)
    start, count = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
    mesh.userface = model.mesh_face[start:start + count].reshape(-1)
    faces = slice(start, start + count)
    start, count = int(model.mesh_normaladr[mid]), int(model.mesh_normalnum[mid])
    if count:
        normals = model.mesh_normal[start:start + count] @ np.linalg.inv(local_transform)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
        mesh.usernormal = normals.reshape(-1)
        mesh.userfacenormal = model.mesh_facenormal[faces].reshape(-1)
    start, count = int(model.mesh_texcoordadr[mid]), int(model.mesh_texcoordnum[mid])
    if count:
        mesh.usertexcoord = model.mesh_texcoord[start:start + count].reshape(-1)
        mesh.userfacetexcoord = model.mesh_facetexcoord[faces].reshape(-1)
    geom.meshname = mesh.name
    geom.quat = model.geom_quat[gid]
    geom.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT


def scale_target(
    scene: mujoco.MjSpec,
    target_name: str,
    reference_geom: str,
    factors: tuple[float, float, float],
    mass_scale: float,
) -> None:
    """Scale around the body origin, along the first bound collision geom axes.

    Transform mass moments as well as geometry. Analytic boxes remain boxes
    when their axes align with the scaling frame; meshes allow arbitrary axes.
    """
    if factors == (1.0, 1.0, 1.0) and mass_scale == 1.0:
        return
    body = scene.body(target_name)
    if body is None:
        raise KeyError(f"missing target body: {target_name}")
    if list(body.bodies):
        raise ValueError("target resizing requires a single rigid body")
    elements = [*body.geoms, *body.sites]
    if any(element.frame is not None for element in elements):
        raise ValueError("target resizing requires body-local geoms and sites")
    model = scene.compile()
    bid = model.body(target_name).id
    reference = model.geom(reference_geom).id
    if model.geom_bodyid[reference] != bid:
        raise ValueError("resizing reference geom must belong to the target")
    frame = _rotation(model.geom_quat[reference])
    transform = frame @ np.diag(factors) @ frame.T
    if factors != (1.0, 1.0, 1.0):
        for geom in body.geoms:
            gid = model.geom(geom.name).id
            geom.pos = transform @ model.geom_pos[gid]
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                _resize_mesh(scene, geom, model, gid, transform)
            else:
                _resize_shape(geom, model.geom_type[gid], model.geom_size[gid],
                              _rotation(model.geom_quat[gid]), transform)
        for site in body.sites:
            sid = model.site(site.name).id
            site.pos = transform @ model.site_pos[sid]
            _resize_shape(site, model.site_type[sid], model.site_size[sid],
                          _rotation(model.site_quat[sid]), transform)

    # C = integral(r r^T dm); I = trace(C) identity - C, about the COM.
    inertial_frame = _rotation(model.body_iquat[bid])
    inertia = inertial_frame @ np.diag(model.body_inertia[bid]) @ inertial_frame.T
    second_moment = np.trace(inertia) / 2 * np.eye(3) - inertia
    density_volume = float(np.prod(factors)) * mass_scale
    second_moment = density_volume * transform @ second_moment @ transform.T
    inertia = np.trace(second_moment) * np.eye(3) - second_moment
    body.explicitinertial = True
    body.mass = model.body_mass[bid] * density_volume
    body.ipos = transform @ model.body_ipos[bid]
    moments, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    body.fullinertia = np.full(6, np.nan)
    body.inertia = moments
    body.iquat = _quaternion(axes)
