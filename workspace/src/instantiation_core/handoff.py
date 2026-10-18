"""Capture and replay actual strategy boundaries through the shared executor."""

from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np

from .qualification import source_fingerprint, diagnostic_execution
from .generalization import runtime_scene_fingerprint
from .json_values import to_json_value

_replay = ContextVar("handoff_replay", default=None)


class HandoffComplete(BaseException):
    """Internal control flow: stop a diagnostic after its selected phase."""

    def __init__(self, result):
        self.result = result


def current_replay():
    return _replay.get()


def _source_hash(program):
    src = Path(os.environ["RAPID_ACTIVE_SRC_DIR"])
    return source_fingerprint([src / "strategies" / program, src / "primitives"])


def _invocation_hash(model, binding):
    payload = {"roles": dict(binding.roles), "parameters": dict(binding.parameters),
               "goal": binding.goal_text,
               "physics": runtime_scene_fingerprint(model, mujoco.MjData(model))}
    return hashlib.sha256(json.dumps(to_json_value(payload), sort_keys=True).encode()).hexdigest()


def capture_boundary(binding, phase, state, controller, action, operation, *, model):
    """Keep immutable snapshots and a pointer to the latest executed boundary."""
    if not os.environ.get("RAPID_INVENTION_ATTEMPT_LOG") or current_replay() is not None:
        return
    directory = Path(os.environ["RAPID_WORKSPACE"]) / "agent_work/handoffs" / binding.program_name / binding.instance_name / phase
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {"program": binding.program_name, "instance": binding.instance_name,
                "phase": phase, "scene": binding.scene_module,
                "source_fingerprint": _source_hash(binding.program_name),
                "invocation_fingerprint": _invocation_hash(model, binding)}
    arrays = {"state": np.asarray(state), "controller": np.asarray([] if controller is None else controller),
              "action": np.asarray([] if action is None else action), "operation": np.asarray(int(operation))}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for key, value in arrays.items():
        digest.update(f"{key}:{value.shape}:{value.dtype}".encode())
        digest.update(value.tobytes())
    snapshot = directory / f"{digest.hexdigest()}.npz"
    if not snapshot.exists():
        np.savez_compressed(snapshot, **arrays, metadata=np.asarray(json.dumps(metadata)))
    pointer = directory / f"latest.{os.getpid()}.tmp"
    pointer.write_text(json.dumps({"snapshot": snapshot.name}) + "\n")
    pointer.replace(directory / "latest.json")


def restore_boundary(replay, model, binding):
    metadata, arrays = replay["metadata"], replay["arrays"]
    if (metadata["program"] != binding.program_name or metadata["instance"] != binding.instance_name
            or metadata["scene"] != binding.scene_module):
        raise ValueError("handoff checkpoint does not match the invocation")
    if metadata["invocation_fingerprint"] != _invocation_hash(model, binding):
        raise ValueError("handoff checkpoint physics or binding changed; rerun the development invocation")
    state = arrays["state"].copy()
    if state.shape != (model.nq + model.nv,):
        raise ValueError("handoff checkpoint model dimensions differ")
    delta = np.asarray(replay["perturbation"], dtype=float)
    if delta.shape != (model.nv,) or not np.all(np.isfinite(delta)):
        raise ValueError("handoff perturbation must be a finite model.nv tangent vector")
    mujoco.mj_integratePos(model, state[:model.nq], delta, 1.0)
    replay["visited"] = True
    return (state, arrays["controller"].copy() if arrays["controller"].size else None,
            arrays["action"].copy() if arrays["action"].size else None, int(arrays["operation"]))


def run_handoff_probe(manifest: Path, checkpoint: Path, perturbations: list[list[float]]) -> dict:
    """Replay transit and one next primitive from measured boundary states.

    Configuration perturbations are MuJoCo tangent vectors, not replacement
    poses. The exact learned dispatcher runs with two rollouts and one iteration;
    final effect success is reported separately from precontact reachability.
    """
    from .entrypoint import run_manifest_entrypoint
    if not perturbations:
        raise ValueError("handoff probe needs explicit perturbations")
    with np.load(checkpoint, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        arrays = {key: archive[key].copy() for key in ("state", "controller", "action", "operation")}
    if metadata["source_fingerprint"] != _source_hash(metadata["program"]):
        raise ValueError("handoff checkpoint is stale; rerun the development invocation")
    results = []
    for perturbation in perturbations:
        replay = {"metadata": metadata, "arrays": arrays, "perturbation": perturbation, "visited": False}
        token = _replay.set(replay)
        try:
            with diagnostic_execution() as arguments:
                run_manifest_entrypoint(manifest, arguments)
        except HandoffComplete as done:
            phases = done.result.metrics.get("phase_metrics", {})
            findings = []
            measured = {}
            for name, metrics in phases.items():
                transit = metrics.get("transit", metrics)
                valid = transit.get("transit_valid")
                converged = transit.get("transit_convergence_valid")
                measured[name] = {"transit_valid": valid, "transit_converged": converged,
                                  "effect_success": metrics.get("effect_success")}
                if valid is not True or converged is not True:
                    findings.append(f"{name}: precontact reachability was not established")
            if not replay["visited"] or not phases:
                findings.append("selected boundary did not execute")
            results.append({"findings": findings, "measurements": measured})
        except Exception as exc:
            results.append({"findings": [f"{type(exc).__name__}: {exc}"], "measurements": {"boundary_visited": replay["visited"]}})
        else:
            results.append({"findings": ["selected boundary did not execute"], "measurements": {}})
        finally:
            _replay.reset(token)
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "phase": metadata["phase"], "samples": results}
