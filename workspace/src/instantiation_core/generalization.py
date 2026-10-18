"""Deterministic multidimensional generalization for generated test scenes."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

from .contracts import BindingReport, SceneBinding
from .json_values import to_json_value
from .scene_scaling import scale_target


DEFAULT_GENERALIZATION_SEED = 7


def _tuple3(value: Sequence[float], field_name: str) -> tuple[float, float, float]:
    result = tuple(float(item) for item in value)
    if len(result) != 3:
        raise ValueError(f"{field_name} must contain exactly three values")
    return result


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _freeze_json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    # Round-trip validation also prevents mutable or non-serializable values from
    # leaking into an immutable generated-scene specification.
    normalized = json.loads(json.dumps(dict(value), sort_keys=True))
    return _freeze_json_value(normalized)


@dataclass(frozen=True)
class SceneVariantSpec:
    """One reproducible scene plus its complete primitive invocation goal.

    ``size_scale_xyz`` acts along the first target collision geom's local axes.
    """

    base_scene_module: str
    variant_index: int
    seed: int
    target_body: str
    target_free_joint: str
    target_collision_geoms: tuple[str, ...]
    support_geom: str
    size_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    target_position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_yaw: float | None = None
    mass_scale: float = 1.0
    target_friction_scale: float = 1.0
    support_friction_scale: float = 1.0
    fixture_body: str | None = None
    fixture_contact_geoms: tuple[str, ...] = ()
    fixture_position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fixture_friction_scale: float = 1.0
    goal_parameters: Mapping[str, Any] = field(default_factory=dict)
    varied_goal_parameters: tuple[str, ...] = ()
    yaw_coupled_bodies: tuple[str, ...] = ()
    allow_independent_fixture_yaw: bool = False

    def __post_init__(self) -> None:
        if not self.base_scene_module.strip():
            raise ValueError("base_scene_module cannot be empty")
        if self.variant_index < 0:
            raise ValueError("variant_index cannot be negative")
        for field_name in (
            "target_body",
            "target_free_joint",
            "support_geom",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} cannot be empty")
        if not self.target_collision_geoms:
            raise ValueError("target_collision_geoms cannot be empty")
        for field_name in (
            "mass_scale",
            "target_friction_scale",
            "support_friction_scale",
            "fixture_friction_scale",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
        scales = _tuple3(self.size_scale_xyz, "size_scale_xyz")
        if any(not math.isfinite(value) or value <= 0.0 for value in scales):
            raise ValueError("size_scale_xyz must be finite and positive")
        object.__setattr__(self, "size_scale_xyz", scales)
        object.__setattr__(
            self,
            "target_collision_geoms",
            tuple(str(item) for item in self.target_collision_geoms),
        )
        coupled_bodies = tuple(str(item) for item in self.yaw_coupled_bodies)
        if any(not item.strip() for item in coupled_bodies):
            raise ValueError("yaw_coupled_bodies cannot contain empty names")
        if len(coupled_bodies) != len(set(coupled_bodies)):
            raise ValueError("yaw_coupled_bodies cannot contain duplicate names")
        if self.target_body in coupled_bodies:
            raise ValueError(
                "yaw_coupled_bodies names companion bodies; target_body is implicit"
            )
        if coupled_bodies and self.target_yaw is None:
            raise ValueError("yaw_coupled_bodies requires target_yaw")
        if self.allow_independent_fixture_yaw and (
            self.target_yaw is None or self.fixture_body is None
        ):
            raise ValueError(
                "allow_independent_fixture_yaw requires target_yaw and fixture_body"
            )
        if (
            self.target_yaw is not None
            and self.fixture_body is not None
            and self.fixture_body not in coupled_bodies
            and not self.allow_independent_fixture_yaw
        ):
            raise ValueError(
                "target_yaw with fixture_body must couple the fixture in "
                "yaw_coupled_bodies; set allow_independent_fixture_yaw=True "
                "only when relative target-to-fixture yaw is intentional"
            )
        object.__setattr__(self, "yaw_coupled_bodies", coupled_bodies)
        object.__setattr__(
            self,
            "fixture_contact_geoms",
            tuple(str(item) for item in self.fixture_contact_geoms),
        )
        object.__setattr__(
            self,
            "target_position_offset",
            _tuple3(self.target_position_offset, "target_position_offset"),
        )
        object.__setattr__(
            self,
            "fixture_position_offset",
            _tuple3(self.fixture_position_offset, "fixture_position_offset"),
        )
        object.__setattr__(
            self,
            "goal_parameters",
            _freeze_json_mapping(self.goal_parameters),
        )
        varied_goal_parameters = tuple(
            str(item) for item in self.varied_goal_parameters
        )
        if any(not item for item in varied_goal_parameters):
            raise ValueError("varied_goal_parameters cannot contain empty names")
        if len(varied_goal_parameters) != len(set(varied_goal_parameters)):
            raise ValueError("varied_goal_parameters cannot contain duplicate names")
        unknown_varied = sorted(
            set(varied_goal_parameters) - set(self.goal_parameters)
        )
        if unknown_varied:
            raise ValueError(
                "varied_goal_parameters must name goal_parameters entries: "
                f"{unknown_varied}"
            )
        object.__setattr__(
            self,
            "varied_goal_parameters",
            varied_goal_parameters,
        )
        if self.target_yaw is not None:
            normalized_yaw = (
                (float(self.target_yaw) + math.pi) % (2.0 * math.pi)
            ) - math.pi
            object.__setattr__(self, "target_yaw", normalized_yaw)

    @property
    def changed_dimensions(self) -> tuple[str, ...]:
        """Return declared dimensions that differ from the assigned baseline."""

        changed: list[str] = []
        if any(not math.isclose(value, 1.0) for value in self.size_scale_xyz):
            changed.append("object_size")
        if np.linalg.norm(self.target_position_offset) > 1e-12 or self.target_yaw is not None:
            changed.append("object_pose")
        if not math.isclose(self.mass_scale, 1.0):
            changed.append("object_mass")
        if not math.isclose(self.target_friction_scale, 1.0) or not math.isclose(
            self.support_friction_scale, 1.0
        ):
            changed.append("contact_friction")
        if self.fixture_body is not None and (
            np.linalg.norm(self.fixture_position_offset) > 1e-12
            or not math.isclose(self.fixture_friction_scale, 1.0)
            or self.fixture_body in self.yaw_coupled_bodies
        ):
            changed.append("fixture")
        if self.varied_goal_parameters:
            changed.append("goal")
        return tuple(changed)

    @property
    def changed_physical_dimensions(self) -> tuple[str, ...]:
        return tuple(item for item in self.changed_dimensions if item != "goal")

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_scene_module": self.base_scene_module,
            "variant_index": self.variant_index,
            "seed": self.seed,
            "target_body": self.target_body,
            "target_free_joint": self.target_free_joint,
            "target_collision_geoms": list(self.target_collision_geoms),
            "support_geom": self.support_geom,
            "size_scale_xyz": list(self.size_scale_xyz),
            "target_position_offset": list(self.target_position_offset),
            "target_yaw": self.target_yaw,
            "yaw_coupled_bodies": list(self.yaw_coupled_bodies),
            "allow_independent_fixture_yaw": self.allow_independent_fixture_yaw,
            "mass_scale": self.mass_scale,
            "target_friction_scale": self.target_friction_scale,
            "support_friction_scale": self.support_friction_scale,
            "fixture_body": self.fixture_body,
            "fixture_contact_geoms": list(self.fixture_contact_geoms),
            "fixture_position_offset": list(self.fixture_position_offset),
            "fixture_friction_scale": self.fixture_friction_scale,
            "goal_parameters": to_json_value(self.goal_parameters),
            "varied_goal_parameters": list(self.varied_goal_parameters),
            "changed_dimensions": list(self.changed_dimensions),
            "changed_physical_dimensions": list(self.changed_physical_dimensions),
        }


@dataclass(frozen=True)
class GeneralizationSample:
    """Scene-independent balanced sample used to author a concrete variant."""

    variant_index: int
    size_scale_xyz: tuple[float, float, float]
    translation_uv: tuple[float, float]
    target_yaw: float
    goal_scale: float
    goal_lateral_fraction: float
    goal_yaw: float
    mass_scale: float
    friction_scale: float
    fixture_offset_fraction: float


def _stratified_unit_values(
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    order = rng.permutation(count)
    within_bin = rng.uniform(0.2, 0.8, size=count)
    return (order + within_bin) / count


def _scale(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return lower + values * (upper - lower)


def balanced_generalization_samples(
    count: int,
    *,
    seed: int = DEFAULT_GENERALIZATION_SEED,
) -> tuple[GeneralizationSample, ...]:
    """Return deterministic joint samples with stratified marginal coverage."""

    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    dimensions = [_stratified_unit_values(count, rng) for _ in range(10)]
    size = _scale(np.column_stack(dimensions[:3]), 0.8, 1.2)
    tx = _scale(dimensions[3], -1.0, 1.0)
    ty = _scale(dimensions[4], -1.0, 1.0)
    goal_scale = _scale(dimensions[5], 0.7, 1.3)
    goal_lateral = _scale(dimensions[6], -1.0, 1.0)
    mass = _scale(dimensions[7], 0.5, 2.0)
    friction = _scale(dimensions[8], 0.5, 1.5)
    fixture = _scale(dimensions[9], -1.0, 1.0)

    # Midpoint phases include a near-opposite pose for small suites while
    # retaining uniform circular coverage as the suite grows.
    yaw = np.asarray(
        [
            (
                (
                    2.0 * math.pi * index / count
                    + math.pi / count
                    + math.pi
                )
                % (2.0 * math.pi)
            )
            - math.pi
            for index in range(count)
        ]
    )
    yaw = yaw[rng.permutation(count)]
    goal_yaw = np.roll(yaw, 1)

    return tuple(
        GeneralizationSample(
            variant_index=index + 1,
            size_scale_xyz=tuple(float(value) for value in size[index]),
            translation_uv=(float(tx[index]), float(ty[index])),
            target_yaw=float(yaw[index]),
            goal_scale=float(goal_scale[index]),
            goal_lateral_fraction=float(goal_lateral[index]),
            goal_yaw=float(goal_yaw[index]),
            mass_scale=float(mass[index]),
            friction_scale=float(friction[index]),
            fixture_offset_fraction=float(fixture[index]),
        )
        for index in range(count)
    )


def _base_scene_module(spec: SceneVariantSpec) -> Any:
    module = importlib.import_module(spec.base_scene_module)
    if not callable(getattr(module, "make_spec", None)):
        raise AttributeError(
            f"{spec.base_scene_module!r} does not expose make_spec()"
        )
    return module


def _scale_target(specification: SceneVariantSpec, scene: mujoco.MjSpec) -> None:
    scale_target(
        scene,
        specification.target_body,
        specification.target_collision_geoms[0],
        specification.size_scale_xyz,
        specification.mass_scale,
    )


def _scale_friction(
    scene: mujoco.MjSpec,
    geom_names: Sequence[str],
    scale: float,
) -> None:
    for name in geom_names:
        geom = scene.geom(name)
        if geom is None:
            raise KeyError(f"missing friction geom: {name}")
        geom.friction = np.asarray(geom.friction) * scale


def _rotate_yaw_coupled_bodies(
    specification: SceneVariantSpec,
    scene: mujoco.MjSpec,
) -> None:
    """Rotate task-layout companions with the target about its initial origin."""

    if specification.target_yaw is None or not specification.yaw_coupled_bodies:
        return
    target = scene.body(specification.target_body)
    if target is None:
        raise KeyError(f"missing target body: {specification.target_body}")
    if target.frame is not None:
        raise ValueError(
            "yaw-coupled target must be a direct child of its parent, "
            "not frame-attached"
        )
    anchor = np.asarray(target.pos, dtype=float)
    yaw = float(specification.target_yaw)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=float,
    )
    yaw_quaternion = _yaw_quaternion(yaw)
    for body_name in specification.yaw_coupled_bodies:
        body = scene.body(body_name)
        if body is None:
            raise KeyError(f"missing yaw-coupled body: {body_name}")
        if body.parent is not target.parent or body.frame is not None:
            raise ValueError(
                f"yaw-coupled body {body_name!r} must be a direct sibling of "
                f"target body {specification.target_body!r}"
            )
        body.pos = anchor + rotation @ (np.asarray(body.pos, dtype=float) - anchor)
        rotated_quaternion = np.empty(4, dtype=float)
        mujoco.mju_mulQuat(
            rotated_quaternion,
            yaw_quaternion,
            np.asarray(body.quat, dtype=float),
        )
        body.quat = _normalize_quaternion(rotated_quaternion)


def make_variant_spec(specification: SceneVariantSpec) -> mujoco.MjSpec:
    """Build a fresh, physically coherent derived MuJoCo specification."""

    scene = _base_scene_module(specification).make_spec()
    _scale_target(specification, scene)
    _scale_friction(
        scene,
        specification.target_collision_geoms,
        specification.target_friction_scale,
    )
    _scale_friction(
        scene,
        (specification.support_geom,),
        specification.support_friction_scale,
    )
    _rotate_yaw_coupled_bodies(specification, scene)
    if specification.fixture_body is not None:
        fixture = scene.body(specification.fixture_body)
        if fixture is None:
            raise KeyError(f"missing fixture body: {specification.fixture_body}")
        fixture.pos = np.asarray(fixture.pos) + np.asarray(
            specification.fixture_position_offset
        )
    _scale_friction(
        scene,
        specification.fixture_contact_geoms,
        specification.fixture_friction_scale,
    )
    return scene


def make_variant_model(specification: SceneVariantSpec) -> mujoco.MjModel:
    return make_variant_spec(specification).compile()


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.asarray(
        [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)],
        dtype=float,
    )


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise ValueError("quaternion cannot have zero norm")
    return quaternion / norm


def _support_plane(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    normal = rotation[:, 2].copy()
    point = data.geom_xpos[geom_id].copy()
    if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
        point += normal * float(model.geom_size[geom_id, 2])
    return point, normal


def _geom_minimum_projection(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    normal: np.ndarray,
) -> float:
    center = data.geom_xpos[geom_id]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
        projected_radius = float(
            np.sum(np.abs(rotation.T @ normal) * model.geom_size[geom_id])
        )
    else:
        projected_radius = float(model.geom_rbound[geom_id])
    return float(np.dot(center, normal) - projected_radius)


def make_variant_data(
    specification: SceneVariantSpec,
    model: mujoco.MjModel | None = None,
) -> mujoco.MjData:
    """Reset a derived model, apply target pose, and align it to its support."""

    model = make_variant_model(specification) if model is None else model
    base = _base_scene_module(specification)
    base_make_data = getattr(base, "make_data", None)
    if callable(base_make_data):
        data = base_make_data(model)
    else:
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

    joint_id = model.joint(specification.target_free_joint).id
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("target_free_joint must name a free joint")
    address = int(model.jnt_qposadr[joint_id])
    data.qpos[address : address + 3] += np.asarray(
        specification.target_position_offset
    )
    if specification.target_yaw is not None:
        current = data.qpos[address + 3 : address + 7].copy()
        yaw_quat = _yaw_quaternion(specification.target_yaw)
        composed = np.empty(4, dtype=float)
        mujoco.mju_mulQuat(composed, yaw_quat, current)
        data.qpos[address + 3 : address + 7] = _normalize_quaternion(composed)

    mujoco.mj_forward(model, data)
    support_alignment_needed = (
        any(not math.isclose(value, 1.0) for value in specification.size_scale_xyz)
        or np.linalg.norm(specification.target_position_offset) > 1e-12
    )
    # A world-z yaw preserves every target point's support-normal projection,
    # so re-snapping it would only erase small baseline target/fixture offsets.
    if not support_alignment_needed:
        return data
    support_id = model.geom(specification.support_geom).id
    support_point, support_normal = _support_plane(model, data, support_id)
    minimum = min(
        _geom_minimum_projection(
            model,
            data,
            model.geom(name).id,
            support_normal,
        )
        for name in specification.target_collision_geoms
    )
    support_height = float(np.dot(support_point, support_normal))
    data.qpos[address : address + 3] += support_normal * (
        support_height - minimum
    )
    mujoco.mj_forward(model, data)
    return data


_MODEL_FINGERPRINT_ARRAYS = (
    "body_pos",
    "body_quat",
    "body_mass",
    "body_inertia",
    "body_ipos",
    "body_iquat",
    "geom_type",
    "geom_size",
    "geom_pos",
    "geom_quat",
    "geom_friction",
    "geom_contype",
    "geom_conaffinity",
    "mesh_vert",
    "jnt_type",
    "jnt_axis",
    "actuator_trntype",
    "actuator_trnid",
    "actuator_gear",
)


def _hash_array(digest: Any, name: str, value: Any) -> None:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating):
        array = np.round(array.astype(np.float64), decimals=12)
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def runtime_scene_fingerprint(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    goal_parameters: Mapping[str, Any] | None = None,
) -> str:
    """Hash compiled physics, initialized state, and optional task goal."""

    digest = hashlib.sha256()
    digest.update(bytes(model.names[: int(model.nnames)]))
    for name in _MODEL_FINGERPRINT_ARRAYS:
        _hash_array(digest, name, getattr(model, name))
    for name in ("qpos", "qvel", "ctrl"):
        _hash_array(digest, name, getattr(data, name))
    if goal_parameters is not None:
        digest.update(
            json.dumps(
                to_json_value(goal_parameters),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    return digest.hexdigest()


def validate_variant_initial_state(
    specification: SceneVariantSpec,
    *,
    settle_steps: int = 50,
    maximum_initial_penetration: float = 0.005,
    maximum_settle_drift: float = 0.02,
    maximum_settle_rotation: float = math.radians(5.0),
    maximum_settle_linear_speed: float = 0.05,
    maximum_settle_angular_speed: float = 0.5,
) -> list[str]:
    """Return physical diagnostics for a generated scene before optimization.

    Imported reconstruction resets can begin slightly above their support and
    undergo a one-time gravity/contact relaxation.  Use one window for that
    relaxation, then judge continued motion over a second equal window.  This
    rejects genuinely unstable scenes without treating a converged landing as
    persistent drift.
    """

    findings: list[str] = []
    try:
        model = make_variant_model(specification)
        data = make_variant_data(specification, model)
    except Exception as exc:
        return [f"could not construct generated scene: {exc}"]

    target_id = model.body(specification.target_body).id
    if (
        not np.isfinite(model.body_mass[target_id])
        or model.body_mass[target_id] <= 0
    ):
        findings.append("target mass must be finite and positive")
    if np.any(~np.isfinite(model.body_inertia[target_id])) or np.any(
        model.body_inertia[target_id] <= 0
    ):
        findings.append("target inertia must be finite and positive")
    friction_ids = [
        model.geom(name).id
        for name in (
            *specification.target_collision_geoms,
            specification.support_geom,
            *specification.fixture_contact_geoms,
        )
    ]
    if any(
        np.any(~np.isfinite(model.geom_friction[geom_id]))
        or np.any(model.geom_friction[geom_id] < 0)
        for geom_id in friction_ids
    ):
        findings.append("contact friction must be finite and nonnegative")

    maximum_penetration = max(
        (-float(data.contact[index].dist) for index in range(data.ncon)),
        default=0.0,
    )
    if maximum_penetration > maximum_initial_penetration:
        findings.append(
            f"initial penetration {maximum_penetration:.6g} exceeds "
            f"{maximum_initial_penetration:.6g}"
        )

    joint_id = model.joint(specification.target_free_joint).id
    qpos_address = int(model.jnt_qposadr[joint_id])
    dof_address = int(model.jnt_dofadr[joint_id])
    try:
        # Allow a reconstructed reset to land on its support before measuring
        # whether it continues to translate, rotate, or carry residual speed.
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
        settled_position = data.qpos[qpos_address : qpos_address + 3].copy()
        settled_quaternion = data.qpos[
            qpos_address + 3 : qpos_address + 7
        ].copy()
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
    except Exception as exc:
        findings.append(f"settling rollout failed: {exc}")
        return findings

    drift = float(
        np.linalg.norm(
            data.qpos[qpos_address : qpos_address + 3] - settled_position
        )
    )
    if not np.isfinite(drift) or drift > maximum_settle_drift:
        findings.append(
            "target post-relaxation drift "
            f"{drift:.6g} exceeds {maximum_settle_drift:.6g}"
        )

    final_quaternion = data.qpos[qpos_address + 3 : qpos_address + 7]
    quaternion_alignment = abs(
        float(np.dot(settled_quaternion, final_quaternion))
    )
    rotation = 2.0 * math.acos(float(np.clip(quaternion_alignment, 0.0, 1.0)))
    if not np.isfinite(rotation) or rotation > maximum_settle_rotation:
        findings.append(
            "target post-relaxation rotation "
            f"{rotation:.6g} exceeds {maximum_settle_rotation:.6g}"
        )

    linear_speed = float(
        np.linalg.norm(data.qvel[dof_address : dof_address + 3])
    )
    if (
        not np.isfinite(linear_speed)
        or linear_speed > maximum_settle_linear_speed
    ):
        findings.append(
            "target terminal settle linear speed "
            f"{linear_speed:.6g} exceeds {maximum_settle_linear_speed:.6g}"
        )
    angular_speed = float(
        np.linalg.norm(data.qvel[dof_address + 3 : dof_address + 6])
    )
    if (
        not np.isfinite(angular_speed)
        or angular_speed > maximum_settle_angular_speed
    ):
        findings.append(
            "target terminal settle angular speed "
            f"{angular_speed:.6g} exceeds {maximum_settle_angular_speed:.6g}"
        )
    return findings


def generalization_metadata(
    binding: SceneBinding,
) -> dict[str, Any] | None:
    """Return authoritative variant metadata for one generated binding."""

    module = importlib.import_module(binding.scene_module)
    specification = getattr(module, "VARIANT_SPEC", None)
    if not isinstance(specification, SceneVariantSpec):
        return None
    specified_goal = to_json_value(specification.goal_parameters)
    bound_goal = to_json_value(binding.parameters)
    if specified_goal != bound_goal:
        raise ValueError(
            "binding parameters must exactly match VARIANT_SPEC.goal_parameters"
        )
    model = module.make_model()
    make_data = getattr(module, "make_data", None)
    data = make_data(model) if callable(make_data) else mujoco.MjData(model)
    physical = runtime_scene_fingerprint(model, data)
    combined = runtime_scene_fingerprint(
        model,
        data,
        goal_parameters=binding.parameters,
    )
    return {
        "specification": specification.to_dict(),
        "physical_fingerprint": physical,
        "runtime_fingerprint": combined,
    }


def attach_generalization_metadata(
    report: BindingReport,
    binding: SceneBinding,
) -> BindingReport:
    """Attach generated-scene provenance to binding reports and metrics."""

    metadata = generalization_metadata(binding)
    if metadata is None:
        return report
    combined = dict(report.metadata)
    combined["generalization"] = metadata
    return replace(report, metadata=combined)


__all__ = [
    "DEFAULT_GENERALIZATION_SEED",
    "GeneralizationSample",
    "SceneVariantSpec",
    "attach_generalization_metadata",
    "balanced_generalization_samples",
    "generalization_metadata",
    "make_variant_data",
    "make_variant_model",
    "make_variant_spec",
    "runtime_scene_fingerprint",
    "validate_variant_initial_state",
]
