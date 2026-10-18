
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import inspect
from typing import Any, Callable, Union

import mujoco
import numpy as np
from scipy.interpolate import interp1d

from primitive_core.common import (
    CompileResult,
    GLOBAL_MAX_OPTIMIZATION_BUDGET,
    PrimitiveCompilerConfig,
    ToolTrajectory,
    resolve_parallel_rollouts,
)
from primitive_core.fr3_osc import fr3_controller_state_from_observation
from primitive_core.fr3_safety import (
    FR3SafetySpec,
    audit_fr3_trajectory,
    enforce_pose_trajectory_timing,
    minimum_pose_trajectory_duration,
)
from primitive_core.policy import (
    GripperIntent, GripperOperation, NominalPlan, validate_constant_rotation,
)
from primitive_core.profiling import timed_execution
from primitive_core.result_retention import retain_execution_result
from judo.optimizers.cem import CrossEntropyMethod, CrossEntropyMethodConfig
from judo.optimizers.mppi import MPPI, MPPIConfig
from judo.utils.mj_rollout_backend import MJRolloutBackend
from judo.utils.rollout_backend import RolloutBackend

from primitive_core.rollout_control import (
    RolloutControlAdapter,
    RolloutTelemetrySpec,
)


CostFn = Callable[..., np.ndarray]
PredicateFn = Callable[..., np.ndarray]
InitPoseClampFn = Callable[[np.ndarray], np.ndarray]

Optimizer = Union[CrossEntropyMethod, MPPI]

_MJWARP_MIN_FRICTION = 0.01


def _parallel_rollout_count(cfg: PrimitiveCompilerConfig) -> int:
    """Return the number of MuJoCo rollout threads to use at one time."""

    return resolve_parallel_rollouts(
        cfg.num_rollouts,
        cfg.max_parallel_rollouts,
    )


def _rollout_in_chunks(
    backend: RolloutBackend,
    x0_batch: np.ndarray,
    controls: np.ndarray,
    chunk_size: int,
    *,
    return_sensors: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Evaluate rollouts in fixed-size chunks to cap CPU thread fan-out."""

    states_chunks: list[np.ndarray] = []
    sensor_chunks: list[np.ndarray] = []
    for start in range(0, controls.shape[0], chunk_size):
        end = min(start + chunk_size, controls.shape[0])
        batch_size = end - start
        x0_chunk = x0_batch[start:end]
        controls_chunk = controls[start:end]
        if batch_size < chunk_size:
            pad = chunk_size - batch_size
            x0_chunk = np.concatenate(
                [x0_chunk, np.repeat(x0_chunk[-1:], pad, axis=0)],
                axis=0,
            )
            controls_chunk = np.concatenate(
                [controls_chunk, np.repeat(controls_chunk[-1:], pad, axis=0)],
                axis=0,
            )
        states_chunk, sensors_chunk, _policy = backend.rollout(
            x0_chunk,
            controls_chunk,
        )
        states_chunks.append(states_chunk[:batch_size])
        sensor_chunks.append(sensors_chunk[:batch_size])
    states = np.concatenate(states_chunks, axis=0)
    if not return_sensors:
        return states
    return states, np.concatenate(sensor_chunks, axis=0)


def _load_mjwarp() -> tuple[type[RolloutBackend], Any]:
    """Load the optional GPU backend and Warp module on demand."""

    try:
        import warp as wp
        from judo.utils.mjwarp_rollout_backend import MJWarpRolloutBackend
    except ImportError as exc:
        raise RuntimeError(
            "The MJWarp rollout backend requires the optional 'mujoco_warp' "
            "package. Install it in the rapid environment with "
            "`conda run -n rapid python -m pip install "
            "'mujoco_warp>=3.6,<3.7' 'warp-lang==1.12.1'`."
        ) from exc
    return MJWarpRolloutBackend, wp


def _make_rollout_backend(
    model: mujoco.MjModel,
    cfg: PrimitiveCompilerConfig,
    rollout_control: RolloutControlAdapter | None = None,
    initial_controller_state: np.ndarray | None = None,
    rollout_telemetry: RolloutTelemetrySpec | None = None,
    gripper_operations: np.ndarray | None = None,
) -> tuple[RolloutBackend, int]:
    """Create the requested rollout backend and return its batch size."""

    name = cfg.rollout_backend.lower()
    if rollout_control is not None:
        if name != "mujoco":
            raise ValueError(
                "controller-aware rollouts currently require "
                "rollout_backend='mujoco'; "
                f"got {cfg.rollout_backend!r}"
            )
        parallel_rollouts = _parallel_rollout_count(cfg)
        if rollout_control.name != "fr3_cartesian_impedance_osc":
            raise ValueError(
                "Native controller rollouts currently support FR3 OSC only; "
                f"got {rollout_control.name!r}"
            )
        from primitive_core.native_fr3_osc import NativeFR3OSCRolloutBackend

        backend: RolloutBackend = NativeFR3OSCRolloutBackend(
            model=model,
            num_threads=parallel_rollouts,
            adapter=rollout_control,
            initial_controller_state=initial_controller_state,
            telemetry=rollout_telemetry,
            gripper_operations=gripper_operations,
        )
        return backend, parallel_rollouts
    if rollout_telemetry is not None:
        raise ValueError(
            "rollout telemetry currently requires a controller-aware native "
            "FR3 OSC rollout"
        )
    if initial_controller_state is not None:
        raise ValueError(
            "initial_controller_state requires a controller-aware rollout"
        )
    if name == "mujoco":
        parallel_rollouts = _parallel_rollout_count(cfg)
        return (
            MJRolloutBackend(model=model, num_threads=parallel_rollouts),
            parallel_rollouts,
        )
    if name != "mjwarp":
        raise ValueError(
            f"Unknown rollout backend '{cfg.rollout_backend}'. "
            "Expected one of: 'mujoco', 'mjwarp'."
        )
    if cfg.num_rollouts < 1:
        raise ValueError("num_rollouts must be positive")

    backend_cls, wp = _load_mjwarp()
    if not wp.is_cuda_available():
        raise RuntimeError(
            f"MJWarp rollout device '{cfg.rollout_device}' was requested, "
            "but NVIDIA Warp reports that CUDA is unavailable."
        )

    # Judo applies these compatibility fixups before constructing its MJWarp
    # backend. Work on a private model copy so CPU playback and cost code keep
    # the primitive's original model parameters.
    warp_model = deepcopy(model)
    warp_model.opt.density = 0
    np.maximum(
        warp_model.geom_friction,
        _MJWARP_MIN_FRICTION,
        out=warp_model.geom_friction,
    )

    # Each optimizer candidate may carry a different initial pose. Judo's
    # MJWarp backend accepts one initial state per "problem" and repeats it for
    # num_threads, so represent candidates as N problems x 1 rollout. The GPU
    # still receives and advances all N worlds in a single batch.
    backend = backend_cls(
        model=warp_model,
        num_threads=1,
        num_problems=cfg.num_rollouts,
        device=cfg.rollout_device,
    )
    return backend, cfg.num_rollouts


def _make_spline(times: np.ndarray, knots: np.ndarray) -> interp1d:
    """Linear interpolation over spline knots with constant-fill extrapolation."""
    fill = (knots[..., 0, :], knots[..., -1, :])
    return interp1d(
        times,
        knots,
        axis=-2,
        kind="linear",
        copy=False,
        fill_value=fill,  # type: ignore[arg-type]
        bounds_error=False,
    )


def _minimum_jerk_controls(
    times: np.ndarray,
    knots: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """Interpolate batched Cartesian knots with zero endpoint velocity."""

    knot_times = np.asarray(times, dtype=float)
    values = np.asarray(knots, dtype=float)
    query = np.asarray(query_times, dtype=float)
    segment = np.searchsorted(knot_times, query, side="right") - 1
    segment = np.clip(segment, 0, knot_times.shape[0] - 2)
    start_times = knot_times[segment]
    spans = knot_times[segment + 1] - start_times
    fractions = np.clip((query - start_times) / spans, 0.0, 1.0)
    blend = fractions**3 * (
        10.0 + fractions * (-15.0 + 6.0 * fractions)
    )
    start = np.take(values, segment, axis=-2)
    end = np.take(values, segment + 1, axis=-2)
    blend_shape = (1,) * (values.ndim - 2) + (query.shape[0], 1)
    weight = blend.reshape(blend_shape)
    delta = end - start
    delta[..., 3:6] = (delta[..., 3:6] + np.pi) % (2.0 * np.pi) - np.pi
    output = start + weight * delta
    output[..., 3:6] = (
        (output[..., 3:6] + np.pi) % (2.0 * np.pi) - np.pi
    )
    return output


def _interpolate_controls(
    times: np.ndarray,
    knots: np.ndarray,
    query_times: np.ndarray,
    *,
    minimum_jerk: bool,
) -> np.ndarray:
    if minimum_jerk:
        return _minimum_jerk_controls(times, knots, query_times)
    return np.asarray(_make_spline(times, knots)(query_times))


def _build_optimizer(
    cfg: PrimitiveCompilerConfig, nu: int, sigma_init: np.ndarray,
) -> Optimizer:
    """Build a judo optimizer configured from *cfg*.

    The back-end is chosen by ``cfg.optimizer``:

    * ``"cem"``: :class:`CrossEntropyMethod`. The primitive's per-node
      ``sigma_init`` is installed as the initial sigma and subsequently
      re-fit from elite knots every iteration.
    * ``"mppi"``: :class:`MPPI`. MPPI uses a single scalar sigma, so we
      take the mean of the primitive's ``sigma_init`` as a reasonable
      default.
    """
    name = cfg.optimizer.lower()
    if name == "cem":
        if cfg.num_elites < 1:
            raise ValueError("num_elites must be positive for CEM")
        opt_cfg = CrossEntropyMethodConfig(
            num_rollouts=cfg.num_rollouts,
            num_nodes=cfg.num_nodes,
            num_elites=min(cfg.num_elites, cfg.num_rollouts),
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
        )
        return CrossEntropyMethod(opt_cfg, nu=nu)
    if name == "mppi":
        opt_cfg = MPPIConfig(
            num_rollouts=cfg.num_rollouts,
            num_nodes=cfg.num_nodes,
            sigma=float(np.clip(
                float(np.mean(sigma_init)), cfg.sigma_min, cfg.sigma_max,
            )),
            temperature=cfg.mppi_temperature,
        )
        return MPPI(opt_cfg, nu=nu)
    raise ValueError(
        f"Unknown optimizer '{cfg.optimizer}'. Expected one of: 'cem', 'mppi'."
    )


@dataclass
class PrimitiveCompileSpec:
    """Everything a primitive has to give the driver.

    ``initial_pose_slots`` holds pairs ``(state_index, knot_channel)``:
    when ``optimize_initial_pose`` is true, the driver will copy the
    first knot's value in channel ``knot_channel`` into the initial
    state's ``state_index`` before rolling out. This lets the optimiser
    choose selected tool start coordinates.
    """

    model: mujoco.MjModel
    initial_state: np.ndarray  # (nq + nv,)
    nominal_knots: np.ndarray  # (num_nodes, nu)
    sigma_init: np.ndarray  # (num_nodes, nu); may already include per-node boosts
    cost_fn: CostFn
    initial_pose_slots: tuple[tuple[int, int], ...] = ()
    init_pose_clamp_fn: InitPoseClampFn | None = None
    action_ctrlrange: np.ndarray | None = None
    rollout_control: RolloutControlAdapter | None = None
    initial_controller_state: np.ndarray | None = None
    rollout_telemetry: RolloutTelemetrySpec | None = None
    rollout_safety: FR3SafetySpec | None = None
    knot_fractions: np.ndarray | None = None
    fixed_first_action: bool = False
    gripper_intent: GripperIntent | None = None
    gripper_operations: np.ndarray | None = None
    effect_predicate: PredicateFn | None = None
    constant_rotation_from_knot: int | None = None

    @classmethod
    def from_nominal_plan(
        cls,
        *,
        nominal_plan: NominalPlan,
        **kwargs: Any,
    ) -> "PrimitiveCompileSpec":
        """Construct the internal optimizer spec from learned policy output."""

        return cls(
            nominal_knots=nominal_plan.knots.copy(),
            sigma_init=nominal_plan.sigma.copy(),
            fixed_first_action=nominal_plan.fixed_first_action,
            constant_rotation_from_knot=nominal_plan.constant_rotation_from_knot,
            knot_fractions=nominal_plan.knot_fractions.copy(),
            gripper_intent=nominal_plan.gripper_intent,
            gripper_operations=(
                None
                if nominal_plan.gripper_operations is None
                else nominal_plan.gripper_operations.copy()
            ),
            **kwargs,
        )


def _couple_rotation(knots: np.ndarray, start: int | None) -> None:
    """Keep one sampled attitude throughout the suffix, including batch inputs."""
    if start is not None:
        knots[..., start:, 3:6] = knots[..., start:start + 1, 3:6]


def _fr3_gripper_diagnostics(
    model: mujoco.MjModel,
    adapter: RolloutControlAdapter,
    controls: np.ndarray,
    states: np.ndarray,
    safety_spec: FR3SafetySpec,
) -> dict[str, Any]:
    """Measure commanded and physical aperture behavior on the retained rollout.

    A held command does not imply a perfectly immobile physical finger: contact
    compliance and controller tracking can move the joint slightly.  Keep those
    physical measurements as diagnostics, but distinguish them from commanded
    aperture motion, which is the policy invariant enforced by the compiler.
    """

    commanded = np.asarray(controls[:, 6], dtype=float)
    actual = adapter.gripper_apertures(model, states)
    aperture_range = float(adapter.gripper_range[1] - adapter.gripper_range[0])
    command_tolerance = max(1e-12, 1e-9 * aperture_range)
    actual_motion_tolerance = max(1e-12, 1e-6 * aperture_range)
    commanded_step_motion = np.zeros_like(commanded)
    actual_step_motion = np.zeros_like(actual)
    if commanded.size > 1:
        commanded_step_motion[1:] = np.abs(np.diff(commanded))
    if actual.size > 1:
        actual_step_motion[1:] = np.abs(np.diff(actual))

    commanded_contact_motion_steps = 0
    actual_contact_motion_steps = 0
    maximum_commanded_contact_motion = 0.0
    maximum_actual_contact_motion = 0.0
    if safety_spec.allowed_contact_pairs:
        data = mujoco.MjData(model)
        for step, state in enumerate(np.asarray(states, dtype=float)):
            data.qpos[:] = state[: model.nq]
            data.qvel[:] = state[model.nq :]
            mujoco.mj_forward(model, data)
            target_contact = any(
                pair.matches(
                    int(data.contact[contact_index].geom1),
                    int(data.contact[contact_index].geom2),
                )
                for pair in safety_spec.allowed_contact_pairs
                for contact_index in range(data.ncon)
            )
            if not target_contact:
                continue
            if commanded_step_motion[step] > command_tolerance:
                commanded_contact_motion_steps += 1
                maximum_commanded_contact_motion = max(
                    maximum_commanded_contact_motion,
                    float(commanded_step_motion[step]),
                )
            if actual_step_motion[step] > actual_motion_tolerance:
                actual_contact_motion_steps += 1
                maximum_actual_contact_motion = max(
                    maximum_actual_contact_motion,
                    float(actual_step_motion[step]),
                )

    return {
        "commanded_aperture_range": [
            float(np.min(commanded)),
            float(np.max(commanded)),
        ],
        "actual_aperture_range": [float(np.min(actual)), float(np.max(actual))],
        "commanded_aperture_total_variation": float(
            np.sum(np.abs(np.diff(commanded)))
        ),
        "actual_aperture_total_variation": float(
            np.sum(np.abs(np.diff(actual)))
        ),
        "maximum_aperture_tracking_error": float(
            np.max(np.abs(commanded - actual))
        ),
        "commanded_aperture_motion_during_target_contact_steps": (
            commanded_contact_motion_steps
        ),
        "actual_aperture_motion_during_target_contact_steps": (
            actual_contact_motion_steps
        ),
        # Backward-compatible names now explicitly describe physical motion.
        "aperture_motion_during_target_contact_steps": actual_contact_motion_steps,
        "maximum_aperture_motion_during_target_contact": (
            maximum_actual_contact_motion
        ),
        "maximum_commanded_aperture_motion_during_target_contact": (
            maximum_commanded_contact_motion
        ),
        "maximum_actual_aperture_velocity_during_target_contact": (
            maximum_actual_contact_motion / float(model.opt.timestep)
        ),
    }


def _cost_accepts_sensors(cost_fn: CostFn) -> bool:
    """Return whether ``cost_fn`` declares a third positional argument."""

    signature = inspect.signature(cost_fn)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    ]
    has_varargs = any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    return has_varargs or len(positional) >= 3


def _evaluate_population_fn(
    function: Callable[..., np.ndarray],
    states: np.ndarray,
    controls: np.ndarray,
    sensors: np.ndarray,
) -> np.ndarray:
    if _cost_accepts_sensors(function):
        return np.asarray(function(states, controls, sensors))
    return np.asarray(function(states, controls))


@timed_execution("primitive_compile")
def compile_primitive(
    spec: PrimitiveCompileSpec,
    cfg: PrimitiveCompilerConfig,
    *,
    verbose: bool = True,
) -> tuple[CompileResult, np.ndarray]:
    """Run ``cfg.max_iters`` sampling iterations and return the best trajectory.

    Returns:
        (result, best_knots) – ``result`` contains the best rolled-out
        trajectory, per-step controls, and cost history; ``best_knots``
        is the ``(num_nodes, nu)`` knot array that produced it, which is
        useful for e.g. extracting the optimised initial pose.
    """
    model = spec.model
    if spec.nominal_knots.ndim != 2:
        raise ValueError("nominal_knots must be a two-dimensional array")
    action_dim = int(spec.nominal_knots.shape[1])
    if action_dim < 1:
        raise ValueError("optimizer action dimension must be positive")
    if cfg.num_nodes < 2:
        raise ValueError("num_nodes must be at least two")
    if cfg.num_rollouts < 1 or cfg.max_iters < 1:
        raise ValueError("num_rollouts and max_iters must be positive")
    if (
        cfg.num_rollouts > GLOBAL_MAX_OPTIMIZATION_BUDGET.num_rollouts
        or cfg.max_iters > GLOBAL_MAX_OPTIMIZATION_BUDGET.max_iters
    ):
        raise ValueError("optimizer compute exceeds GLOBAL_MAX_OPTIMIZATION_BUDGET")
    if cfg.min_iters < 1 or cfg.min_iters > cfg.max_iters:
        raise ValueError("min_iters must be between one and max_iters")
    if cfg.early_stop_patience < 0:
        raise ValueError("early_stop_patience cannot be negative")
    if (
        not np.isfinite(cfg.early_stop_cost_tolerance)
        or cfg.early_stop_cost_tolerance < 0.0
    ):
        raise ValueError("early_stop_cost_tolerance must be finite and nonnegative")
    if spec.knot_fractions is None:
        knot_fractions = np.linspace(0.0, 1.0, cfg.num_nodes, endpoint=True)
    else:
        knot_fractions = np.asarray(spec.knot_fractions, dtype=float).reshape(-1)
    if knot_fractions.shape != (cfg.num_nodes,):
        raise ValueError("knot_fractions must contain one value per optimizer node")
    if (
        not np.all(np.isfinite(knot_fractions))
        or not np.isclose(knot_fractions[0], 0.0)
        or not np.isclose(knot_fractions[-1], 1.0)
        or np.any(np.diff(knot_fractions) <= 0.0)
    ):
        raise ValueError(
            "knot_fractions must be finite, strictly increasing, and span [0, 1]"
        )
    requested_horizon = float(cfg.horizon)
    if not np.isfinite(requested_horizon) or requested_horizon <= 0.0:
        raise ValueError("horizon must be positive and finite")
    runtime_minimum_horizon = 0.0
    fr3_minimum_jerk = bool(
        spec.rollout_control is not None and spec.rollout_control.name == (
            "fr3_cartesian_impedance_osc"
        )
    )
    if spec.rollout_safety is not None and not fr3_minimum_jerk:
        raise ValueError(
            "FR3 retained trajectory auditing requires the native FR3 OSC"
        )
    if fr3_minimum_jerk:
        if spec.gripper_intent is None:
            raise ValueError(
                "every native FR3 plan must declare GripperIntent.HOLD or "
                "GripperIntent.ACTUATE"
            )
        if not isinstance(spec.gripper_intent, GripperIntent):
            raise TypeError("FR3 gripper_intent must be a GripperIntent")
        if spec.gripper_operations is None:
            raise ValueError(
                "every native FR3 plan must declare one gripper operation "
                "per nominal knot"
            )
        gripper_knot_operations = np.asarray(
            spec.gripper_operations,
            dtype=np.uint8,
        ).reshape(-1)
        if gripper_knot_operations.shape != (cfg.num_nodes,):
            raise ValueError(
                "gripper_operations must contain one operation per optimizer node"
            )
        if np.any(gripper_knot_operations > int(GripperOperation.GRASP)):
            raise ValueError("unknown gripper operation")
        if (
            spec.gripper_intent is GripperIntent.HOLD
            and not np.allclose(
                spec.nominal_knots[:, 6],
                spec.nominal_knots[0, 6],
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(
                "GripperIntent.HOLD requires constant nominal gripper knots"
            )
        if (
            spec.gripper_intent is GripperIntent.HOLD
            and np.any(gripper_knot_operations != int(GripperOperation.MOVE))
        ):
            raise ValueError("GripperIntent.HOLD permits only MOVE operations")
        grasp_intervals = (
            gripper_knot_operations[:-1] == int(GripperOperation.GRASP)
        )
        if np.any(np.diff(spec.nominal_knots[:, 6])[grasp_intervals] > 1e-12):
            raise ValueError(
                "GRASP may only accompany a non-increasing aperture"
            )
        if spec.rollout_safety is None:
            spec.rollout_safety = FR3SafetySpec.from_model(
                model,
                spec.rollout_control,
            )
        runtime_minimum_horizon = minimum_pose_trajectory_duration(
            spec.rollout_control,
            spec.nominal_knots,
            knot_fractions,
            spec.rollout_safety.limits,
        )
        cfg = replace(
            cfg,
            horizon=max(requested_horizon, runtime_minimum_horizon),
        )
    num_timesteps = int(np.ceil(cfg.horizon / model.opt.timestep))
    knot_times = knot_fractions * cfg.horizon
    rollout_times = np.arange(num_timesteps) * model.opt.timestep
    gripper_operations: np.ndarray | None = None
    if fr3_minimum_jerk:
        operation_indices = np.searchsorted(
            knot_times,
            rollout_times,
            side="right",
        ) - 1
        operation_indices = np.clip(operation_indices, 0, cfg.num_nodes - 1)
        gripper_operations = np.ascontiguousarray(
            gripper_knot_operations[operation_indices],
            dtype=np.uint8,
        )

    if spec.action_ctrlrange is not None:
        action_ctrlrange = np.asarray(spec.action_ctrlrange, dtype=float)
    elif spec.rollout_control is not None:
        action_ctrlrange = np.asarray(
            spec.rollout_control.action_ctrlrange,
            dtype=float,
        )
    else:
        action_ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=float)
    if action_ctrlrange.shape != (action_dim, 2):
        raise ValueError(
            "action_ctrlrange must have shape "
            f"({action_dim}, 2); got {action_ctrlrange.shape}"
        )
    if not np.all(np.isfinite(action_ctrlrange)):
        raise ValueError("optimizer action ranges must be finite")
    if np.any(action_ctrlrange[:, 0] >= action_ctrlrange[:, 1]):
        raise ValueError("every optimizer action range must be increasing")
    if (
        spec.rollout_control is not None
        and action_dim != spec.rollout_control.action_dim
    ):
        raise ValueError(
            f"rollout controller expects {spec.rollout_control.action_dim} actions; "
            f"nominal knots contain {action_dim}"
        )
    ctrl_lo = action_ctrlrange[:, 0]
    ctrl_hi = action_ctrlrange[:, 1]

    if spec.nominal_knots.shape != (cfg.num_nodes, action_dim):
        raise ValueError(
            "nominal_knots must have shape "
            f"({cfg.num_nodes}, {action_dim}); got {spec.nominal_knots.shape}"
        )
    if spec.sigma_init.shape != (cfg.num_nodes, action_dim):
        raise ValueError(
            "sigma_init must have shape "
            f"({cfg.num_nodes}, {action_dim}); got {spec.sigma_init.shape}"
        )
    if not np.all(np.isfinite(spec.nominal_knots)):
        raise ValueError("nominal_knots must be finite")
    if not np.all(np.isfinite(spec.sigma_init)) or np.any(spec.sigma_init < 0.0):
        raise ValueError("sigma_init must be finite and nonnegative")
    if np.any(spec.nominal_knots < ctrl_lo) or np.any(spec.nominal_knots > ctrl_hi):
        raise ValueError("nominal_knots exceed the optimizer action range")
    if not isinstance(spec.fixed_first_action, (bool, np.bool_)):
        raise TypeError("fixed_first_action must be boolean")
    if fr3_minimum_jerk and not spec.fixed_first_action:
        raise ValueError(
            "native FR3 optimization requires fixed_first_action=True so the "
            "optimizer cannot introduce an instantaneous initial target jump"
        )
    rotation_start = spec.constant_rotation_from_knot
    validate_constant_rotation(spec.nominal_knots, rotation_start)
    if rotation_start is not None and not fr3_minimum_jerk:
        raise ValueError("shared rotation requires the native FR3 Cartesian controller")
    optimizer_sigma_init = spec.sigma_init.copy()
    fixed_gripper_knots: np.ndarray | None = None
    if fr3_minimum_jerk:
        fixed_gripper_knots = spec.nominal_knots[:, 6].copy()
        optimizer_sigma_init[:, 6] = 0.0
    optimizer = _build_optimizer(
        cfg,
        nu=action_dim,
        sigma_init=optimizer_sigma_init,
    )

    # CEM accepts a per-node sigma; seed it with the primitive-provided
    # initial distribution so that e.g. the first-knot gets extra
    # exploration when the starting pose is being optimized.
    if isinstance(optimizer, CrossEntropyMethod):
        optimizer.sigma = np.clip(
            optimizer_sigma_init.copy(), cfg.sigma_min, cfg.sigma_max,
        )
        if fixed_gripper_knots is not None:
            optimizer.sigma[:, 6] = 0.0

    backend, rollout_batch_size = _make_rollout_backend(
        model,
        cfg,
        spec.rollout_control,
        spec.initial_controller_state,
        spec.rollout_telemetry,
        gripper_operations,
    )

    nominal_knots = spec.nominal_knots.copy()

    nominal_x0 = np.asarray(spec.initial_state, dtype=float).reshape(1, -1).copy()
    if cfg.optimize_initial_pose:
        for state_idx, knot_ch in spec.initial_pose_slots:
            nominal_x0[:, state_idx] = nominal_knots[0, knot_ch]
    nominal_controls = np.ascontiguousarray(
        np.clip(
            _interpolate_controls(
                knot_times,
                nominal_knots[None],
                rollout_times,
                minimum_jerk=fr3_minimum_jerk,
            )[0],
            ctrl_lo,
            ctrl_hi,
        ).astype(np.float64)
    )

    best_cost = np.inf
    best_knots = nominal_knots.copy()
    best_states: np.ndarray | None = None
    best_controls: np.ndarray | None = None
    best_sensors: np.ndarray | None = None
    cost_history: list[float] = []
    nominal_cost = np.inf
    nominal_success: bool | None = None
    retained_safety_audit = None
    retained_minimum_horizon = runtime_minimum_horizon
    timing_limited_candidate_segments = 0
    actual_iterations = 0
    best_predicate_success = False
    last_significant_improvement = -1

    try:
        nominal_states_batch, nominal_sensors_batch = _rollout_in_chunks(
            backend,
            nominal_x0,
            nominal_controls[None],
            rollout_batch_size,
            return_sensors=True,
        )
        nominal_costs = _evaluate_population_fn(
            spec.cost_fn,
            nominal_states_batch,
            nominal_controls[None],
            nominal_sensors_batch,
        ).reshape(-1)
        if nominal_costs.shape != (1,) or not np.all(np.isfinite(nominal_costs)):
            raise ValueError("cost_fn must return one finite value per rollout")
        nominal_cost = float(nominal_costs[0])
        best_cost = nominal_cost
        best_states = nominal_states_batch[0].copy()
        best_controls = nominal_controls.copy()
        best_sensors = nominal_sensors_batch[0].copy()
        if spec.effect_predicate is not None:
            nominal_predicate = _evaluate_population_fn(
                spec.effect_predicate,
                nominal_states_batch,
                nominal_controls[None],
                nominal_sensors_batch,
            ).reshape(-1)
            if nominal_predicate.shape != (1,):
                raise ValueError(
                    "effect_predicate must return one value per rollout"
                )
            nominal_success = bool(nominal_predicate[0])
            best_predicate_success = nominal_success

        for iteration in range(cfg.max_iters):
            actual_iterations = iteration + 1
            candidates = optimizer.sample_control_knots(nominal_knots)
            candidates = np.clip(candidates, ctrl_lo, ctrl_hi)
            if fixed_gripper_knots is not None:
                candidates[:, :, 6] = fixed_gripper_knots[None]

            if spec.init_pose_clamp_fn is not None:
                candidates[:, 0, :] = spec.init_pose_clamp_fn(candidates[:, 0, :])
            if spec.fixed_first_action:
                candidates[:, 0, :] = spec.nominal_knots[0]
            _couple_rotation(candidates, rotation_start)
            if fr3_minimum_jerk:
                candidates, limited_segments = enforce_pose_trajectory_timing(
                    spec.rollout_control,
                    candidates,
                    knot_fractions,
                    cfg.horizon,
                    spec.rollout_safety.limits,
                )
                if fixed_gripper_knots is not None:
                    candidates[:, :, 6] = fixed_gripper_knots[None]
                timing_limited_candidate_segments += limited_segments
                # The boundary attitude may have been shortened by timing
                # projection. Reuse that bounded attitude throughout the suffix;
                # subsequent angular displacement becomes zero.
                _couple_rotation(candidates, rotation_start)

            x0_batch = np.tile(spec.initial_state, (cfg.num_rollouts, 1))
            if cfg.optimize_initial_pose:
                for state_idx, knot_ch in spec.initial_pose_slots:
                    x0_batch[:, state_idx] = candidates[:, 0, knot_ch]

            sim_controls = np.ascontiguousarray(
                np.clip(
                    _interpolate_controls(
                        knot_times,
                        candidates,
                        rollout_times,
                        minimum_jerk=fr3_minimum_jerk,
                    ),
                    ctrl_lo,
                    ctrl_hi,
                ).astype(np.float64),
            )

            states, sensors = _rollout_in_chunks(
                backend,
                x0_batch,
                sim_controls,
                rollout_batch_size,
                return_sensors=True,
            )
            costs = _evaluate_population_fn(
                spec.cost_fn,
                states,
                sim_controls,
                sensors,
            ).reshape(-1)
            if costs.shape != (cfg.num_rollouts,) or not np.all(
                np.isfinite(costs)
            ):
                raise ValueError("cost_fn must return one finite value per rollout")
            candidate_success = np.zeros(cfg.num_rollouts, dtype=bool)
            if spec.effect_predicate is not None:
                candidate_predicate = _evaluate_population_fn(
                    spec.effect_predicate,
                    states,
                    sim_controls,
                    sensors,
                ).reshape(-1)
                if candidate_predicate.shape != (cfg.num_rollouts,):
                    raise ValueError(
                        "effect_predicate must return one value per rollout"
                    )
                candidate_success = candidate_predicate.astype(bool)

            iter_best = int(np.argmin(costs))
            previous_best = best_cost
            previous_predicate_success = best_predicate_success
            if costs[iter_best] < best_cost:
                best_cost = float(costs[iter_best])
                best_knots = candidates[iter_best].copy()
                best_states = states[iter_best].copy()
                best_controls = sim_controls[iter_best].copy()
                best_sensors = sensors[iter_best].copy()
                best_predicate_success = bool(candidate_success[iter_best])
            if previous_best - best_cost > cfg.early_stop_cost_tolerance:
                last_significant_improvement = iteration
            if best_predicate_success and not previous_predicate_success:
                last_significant_improvement = iteration

            cost_history.append(best_cost if np.isfinite(best_cost) else 1e12)

            # Both CEM and MPPI expect rewards (higher = better); costs are
            # minimised, so we pass the negation.
            nominal_knots = optimizer.update_nominal_knots(candidates, -costs)
            _couple_rotation(nominal_knots, rotation_start)
            if fixed_gripper_knots is not None:
                nominal_knots[:, 6] = fixed_gripper_knots
                if isinstance(optimizer, CrossEntropyMethod):
                    optimizer.sigma[:, 6] = 0.0

            should_stop = bool(
                cfg.early_stop_patience > 0
                and actual_iterations >= cfg.min_iters
                and best_predicate_success
                and iteration - last_significant_improvement
                >= cfg.early_stop_patience
            )

            if verbose and (
                iteration % 10 == 0 or iteration == cfg.max_iters - 1
            ):
                sigma = optimizer.sigma
                sigma_mean = (
                    float(np.mean(sigma))
                    if hasattr(sigma, "__len__")
                    else float(sigma)
                )
                print(
                    f"  iter {iteration:3d}  best_cost={best_cost:.6f}"
                    f"  sigma_mean={sigma_mean:.4f}"
                )
            if should_stop:
                break

        # Replay the selected result once before retained whole-arm auditing.
        retained_x0 = np.asarray(spec.initial_state, dtype=float).copy()
        if cfg.optimize_initial_pose:
            for state_idx, knot_ch in spec.initial_pose_slots:
                retained_x0[state_idx] = best_knots[0, knot_ch]
        replay_states, replay_sensors = _rollout_in_chunks(
            backend,
            retained_x0[None],
            best_controls[None],
            rollout_batch_size,
            return_sensors=True,
        )
        best_states = replay_states[0].copy()
        best_sensors = replay_sensors[0].copy()
        if spec.rollout_safety is not None:
            retained_safety_audit = audit_fr3_trajectory(
                model,
                best_states,
                spec.rollout_safety,
            )
            retained_minimum_horizon = minimum_pose_trajectory_duration(
                spec.rollout_control,
                best_knots,
                knot_fractions,
                spec.rollout_safety.limits,
            )
    except BaseException:
        close_backend = getattr(backend, "close", None)
        if callable(close_backend):
            close_backend()
        raise

    close_backend = getattr(backend, "close", None)
    if callable(close_backend):
        close_backend()

    assert (
        best_states is not None
        and best_controls is not None
        and best_sensors is not None
    )

    final_controller_state = None
    if spec.rollout_control is not None:
        observation_names = tuple(spec.rollout_control.observation_names)
        if spec.rollout_control.name == "fr3_cartesian_impedance_osc":
            final_controller_state = fr3_controller_state_from_observation(
                best_sensors[-1, -len(observation_names) :]
            )

    spline_best = _interpolate_controls(
        knot_times,
        best_knots[None],
        rollout_times,
        minimum_jerk=fr3_minimum_jerk,
    )[0]
    trajectory = ToolTrajectory(
        times=rollout_times,
        positions=spline_best,
        controls=best_controls,
        gripper_operations=gripper_operations,
    )
    result = CompileResult(
        trajectory=trajectory,
        metrics={
            "best_cost": best_cost,
            "nominal_cost": nominal_cost,
            "optimized_cost": best_cost,
            "optimizer_improvement": nominal_cost - best_cost,
            "nominal_success": nominal_success,
            "nominal_to_optimized_control_rms": float(
                np.sqrt(np.mean(np.square(best_controls - nominal_controls)))
            ),
            "nominal_to_optimized_control_max": float(
                np.max(np.abs(best_controls - nominal_controls))
            ),
            "num_iters": actual_iterations,
            "requested_max_iters": cfg.max_iters,
            "minimum_iterations": cfg.min_iters,
            "early_stop_patience": cfg.early_stop_patience,
            "early_stop_cost_tolerance": cfg.early_stop_cost_tolerance,
            "early_stopped": actual_iterations < cfg.max_iters,
            "early_stop_reason": (
                "effect_success_cost_plateau"
                if actual_iterations < cfg.max_iters
                else None
            ),
            "num_rollouts": cfg.num_rollouts,
            "num_elites": min(cfg.num_elites, cfg.num_rollouts),
            "elite_fraction": min(cfg.num_elites, cfg.num_rollouts)
            / cfg.num_rollouts,
            "max_parallel_rollouts": rollout_batch_size,
            "optimizer": cfg.optimizer.lower(),
            "rollout_backend": cfg.rollout_backend.lower(),
            "rollout_device": (
                cfg.rollout_device
                if cfg.rollout_backend.lower() == "mjwarp"
                else "cpu"
            ),
            "controller_rollout_backend": (
                getattr(backend, "implementation_name", "none")
                if spec.rollout_control is not None
                else "none"
            ),
            "controller_state_handoff": bool(
                spec.initial_controller_state is not None
            ),
            "action_dim": action_dim,
            "gripper_intent": (
                spec.gripper_intent.value
                if spec.gripper_intent is not None
                else None
            ),
            "gripper_optimizer_channel_fixed": bool(fr3_minimum_jerk),
            "requested_horizon": requested_horizon,
            "runtime_minimum_horizon": runtime_minimum_horizon,
            "retained_minimum_horizon": retained_minimum_horizon,
            "retained_timing_valid": bool(
                retained_minimum_horizon <= float(cfg.horizon) + 1e-12
            ),
            "timing_limited_candidate_segments": (
                timing_limited_candidate_segments
            ),
            "effective_horizon": float(cfg.horizon),
            "knot_fractions": knot_fractions.tolist(),
            "control_interpolation": (
                "minimum_jerk" if fr3_minimum_jerk else "linear"
            ),
            "whole_arm_audit_valid": (
                retained_safety_audit.valid
                if retained_safety_audit is not None
                else None
            ),
            "trajectory_audit_profile": (
                spec.rollout_safety.limits.version
                if spec.rollout_safety is not None
                else None
            ),
            "low_level_controller": (
                spec.rollout_control.name
                if spec.rollout_control is not None
                else "direct_actuator_control"
            ),
            "controller_observation_names": (
                list(spec.rollout_control.observation_names)
                if spec.rollout_control is not None
                else []
            ),
            "rollout_telemetry_names": (
                list(spec.rollout_telemetry.observation_names)
                if spec.rollout_telemetry is not None
                else []
            ),
        },
        best_states=best_states,
        cost_history=np.array(cost_history),
        controller_state=final_controller_state,
    )
    if retained_safety_audit is not None:
        result.metrics.update(
            {
                "minimum_environment_clearance": float(
                    retained_safety_audit.minimum_environment_clearance
                ),
                "minimum_enforced_environment_clearance": float(
                    retained_safety_audit.minimum_enforced_environment_clearance
                ),
                "minimum_declared_proximity_clearance": float(
                    retained_safety_audit.minimum_declared_proximity_clearance
                ),
                "self_collision_steps": retained_safety_audit.self_collision_steps,
                "permitted_fingertip_contact_steps": (
                    getattr(
                        retained_safety_audit,
                        "permitted_fingertip_contact_steps",
                        0,
                    )
                ),
                "maximum_closed_stop_penetration": (
                    getattr(
                        retained_safety_audit,
                        "maximum_closed_stop_penetration",
                        0.0,
                    )
                ),
                "maximum_finger_synchronization_error": (
                    getattr(
                        retained_safety_audit,
                        "maximum_finger_synchronization_error",
                        0.0,
                    )
                ),
                "unapproved_contact_steps": (
                    retained_safety_audit.unapproved_contact_steps
                ),
                "dominant_unapproved_contact_pair": (
                    list(retained_safety_audit.dominant_unapproved_contact_pair)
                    if retained_safety_audit.dominant_unapproved_contact_pair
                    is not None
                    else None
                ),
            }
        )
    retained_gripper_contact_valid = True
    if fr3_minimum_jerk:
        gripper_diagnostics = _fr3_gripper_diagnostics(
            model,
            spec.rollout_control,
            best_controls,
            best_states,
            spec.rollout_safety,
        )
        result.metrics.update(gripper_diagnostics)
        retained_gripper_contact_valid = bool(
            spec.gripper_intent is GripperIntent.ACTUATE
            or gripper_diagnostics[
                "commanded_aperture_motion_during_target_contact_steps"
            ]
            == 0
        )
        result.metrics["retained_gripper_contact_audit_valid"] = (
            retained_gripper_contact_valid
        )
    if spec.effect_predicate is not None:
        optimized_predicate = _evaluate_population_fn(
            spec.effect_predicate,
            best_states[None],
            best_controls[None],
            best_sensors[None],
        ).reshape(-1)
        if optimized_predicate.shape != (1,):
            raise ValueError("effect_predicate must return one value per rollout")
        result.metrics["optimized_success"] = bool(optimized_predicate[0])
        result.metrics["success"] = bool(optimized_predicate[0])
    retained_timing_valid = bool(result.metrics["retained_timing_valid"])
    retained_audit_valid = bool(
        retained_safety_audit is None or retained_safety_audit.valid
    )
    audit_advisories: list[str] = []
    if not retained_timing_valid:
        audit_advisories.append("retained_timing")
    if not retained_audit_valid:
        audit_advisories.append("whole_arm_clearance_or_contact")
    if not retained_gripper_contact_valid:
        audit_advisories.append("gripper_command_during_contact")
    result.metrics["audit_advisories"] = audit_advisories
    result.metrics["audit_advisory_triggered"] = bool(audit_advisories)
    result.metrics["constant_rotation_from_knot"] = (
        None if rotation_start is None else int(rotation_start)
    )
    retain_execution_result(result)
    return result, best_knots
