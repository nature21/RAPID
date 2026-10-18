"""Run-specific evidence and prompts for scene instantiation."""

from __future__ import annotations

import json
import re
from pathlib import Path

from instantiation_core.scene import load_scene_module, write_scene_inventory

from .codex_exec import INSTANTIATIONS_DIR, PRIMITIVES_DIR, STRATEGIES_DIR, WORKSPACE


_SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

CANONICAL_INIT = '"""Canonical data-only RAPID scene binding."""\n'
CANONICAL_PARENT_INIT = ""
CANONICAL_MAIN = '''"""Generated RAPID manifest entry point. Do not edit."""

from pathlib import Path

from instantiation_core.entrypoint import run_manifest_entrypoint


def main(argv: list[str] | None = None) -> object:
    return run_manifest_entrypoint(Path(__file__).with_name("binding.json"), argv)


if __name__ == "__main__":
    main()
'''


def render_scene(
    scene_module: str,
    output: Path,
    *,
    azimuth: float | None,
) -> Path:
    """Load the rendering dependency only when scene evidence is requested."""

    from vistools.visualize_scene import render_scene as render

    return render(scene_module, output, azimuth=azimuth)


def validate_package_name(value: str, *, label: str) -> str:
    """Return a validated Python package component."""

    if not _SAFE_NAME.fullmatch(value):
        raise ValueError(
            f"{label} must match {_SAFE_NAME.pattern!r}; got {value!r}"
        )
    return value


def program_dir(kind: str, program: str) -> Path:
    if kind == "primitive":
        return PRIMITIVES_DIR / program
    if kind == "strategy":
        return STRATEGIES_DIR / program
    raise ValueError(f"unsupported instantiation kind: {kind!r}")


def instantiation_dir(kind: str, program: str, instance: str) -> Path:
    plural = "primitives" if kind == "primitive" else "strategies"
    return INSTANTIATIONS_DIR / plural / program / instance


def instantiation_module(kind: str, program: str, instance: str) -> str:
    plural = "primitives" if kind == "primitive" else "strategies"
    return f"instantiations.{plural}.{program}.{instance}"


def initial_binding_manifest(
    *,
    kind: str,
    program: str,
    instance: str,
    scene_module: str,
    goal: str,
) -> str:
    """Return the system-created manifest skeleton edited by a binding agent."""

    return json.dumps(
        {
            "schema_version": "rapid.binding/v1",
            "program_kind": kind,
            "program_name": program,
            "instance_name": instance,
            "scene_module": scene_module,
            "goal_text": goal,
            "roles": {},
            "inputs": {},
        },
        indent=2,
        sort_keys=True,
    ) + "\n"


def stage_canonical_instantiation(
    target: Path,
    *,
    kind: str,
    program: str,
    instance: str,
    scene_module: str,
    goal: str,
) -> Path:
    """Create the fixed entrypoint shell and agent-writable binding files."""

    target.parent.mkdir(parents=True, exist_ok=True)
    parent_init = target.parent / "__init__.py"
    if parent_init.exists():
        if parent_init.read_text(encoding="utf-8") != CANONICAL_PARENT_INIT:
            raise ValueError(
                f"program instantiation package is not canonical: {parent_init}"
            )
    else:
        parent_init.write_text(CANONICAL_PARENT_INIT, encoding="utf-8")
    target.mkdir(exist_ok=False)
    target.joinpath("__init__.py").write_text(CANONICAL_INIT, encoding="utf-8")
    target.joinpath("main.py").write_text(CANONICAL_MAIN, encoding="utf-8")
    manifest = target / "binding.json"
    manifest.write_text(
        initial_binding_manifest(
            kind=kind,
            program=program,
            instance=instance,
            scene_module=scene_module,
            goal=goal,
        ),
        encoding="utf-8",
    )
    return manifest


def prepare_scene_evidence(
    scene_module: str,
    run_dir: Path,
) -> tuple[Path, list[Path]]:
    """Write a model inventory and three initial-scene views."""

    evidence_dir = run_dir / "scene_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    inventory = write_scene_inventory(scene_module, evidence_dir / "inventory.json")

    module = load_scene_module(scene_module)
    default_camera = dict(getattr(module, "DEFAULT_CAMERA", {}) or {})
    base_azimuth = float(default_camera.get("camera_azimuth", 135.0))
    images: list[Path] = []
    for index, offset in enumerate((-60.0, 0.0, 60.0)):
        output = evidence_dir / f"initial_view_{index}.png"
        images.append(
            render_scene(
                scene_module,
                output,
                azimuth=base_azimuth + offset,
            )
        )
    return inventory, images


def build_instantiation_prompt(
    *,
    kind: str,
    program: str,
    instance: str,
    scene_module: str,
    goal: str,
    inventory: Path,
    images: list[Path],
    run_dir: Path,
) -> str:
    """Build the compact run-specific prompt for an instantiation skill."""

    skill = f"rapid-instantiate-{kind}"
    plural = "primitives" if kind == "primitive" else "strategies"
    package = instantiation_module(kind, program, instance)
    program_path = program_dir(kind, program).relative_to(WORKSPACE)
    output_path = instantiation_dir(kind, program, instance).relative_to(WORKSPACE)
    smoke_path = (run_dir / "smoke").relative_to(WORKSPACE)
    image_lines = "\n".join(
        f"  - `{path.relative_to(WORKSPACE)}`" for path in images
    )
    binding_clause = (
        "Populate every contract-declared primitive goal parameter from the "
        "assigned invocation. For a generated construction scene, "
        "VARIANT_SPEC.goal_parameters is authoritative and must exactly match "
        "binding inputs."
        if kind == "primitive"
        else
        "The strategy contract and manifest are role-only. Set binding inputs "
        "to an empty mapping; learned connecting constraints resolve primitive "
        "parameters during sequential execution."
    )
    return f"""
Use `${skill}` for this adaptive generalization binding.

## Selected learned program

- Kind: `{kind}`
- Name: `{program}`
- Read-only learned source: `{program_path}`

Read its binding contract, description, implementation, costs, feature
detectors, and imported primitive contracts.

Do not edit the selected learned program, imported primitives, or shared runtime.
Instantiation evaluates and binds frozen learned behavior; any generalization
defect must be preserved as evidence for a later invention run.

## Assigned scene

- Module: `{scene_module}`
- Deterministic MuJoCo inventory: `{inventory.relative_to(WORKSPACE)}`
- Initial scene views:
{image_lines}

Inspect both the XML/model inventory and the attached rendered views before
choosing role names.

## Goal reference

{goal}

Treat this text as binding-time reference metadata. If it is incompatible with
the selected program contract, stop with diagnostics.

{binding_clause}

Do not create a runnable partial package.

## Required output

- Package: `{output_path}`
- Import: `{package}`
- User-facing runtime command (execute once after installing a successful profile):

```bash
$RAPID_REPO_ROOT/scripts/rapid-python -m {package}.main
```

The launcher has already created canonical `__init__.py` and `main.py` files.
Keep both files unchanged. Edit
`{output_path}/binding.json` as strict JSON data containing semantic role names
and the inputs permitted for this program kind.
Do not create Python, waypoint, predicate, geometry, threshold, clearance,
timing, controller, or transit code. Do not add `binding.py` or `mapping.py`.
Run contract validation, then begin with one bounded smoke execution. If it
fails, inspect the retained trajectory and structured failure signature before
changing anything. Correct only genuine binding mistakes. You may rerun with a
corrected binding, but do not calibrate or retain a scene-specific optimization
budget.

Keep every exploratory smoke artifact under `{smoke_path}/`, including metrics,
recordings, logs, rendered reviews, and contact sheets. Use
`{smoke_path}/metrics.json` and `{smoke_path}/trajectory.mp4` when those
artifacts are produced. Never write smoke artifacts under `results/`.

For the first smoke, pass explicit, parser-supported CLI options that bound
every optimizer phase to one iteration, a small rollout count, and a short
horizon. Do not rely on environment variables alone to make a run bounded. The
canonical entry point applies launcher resource caps.

After the smoke confirms that binding and execution wiring are valid, execute
the user-facing command once without smoke caps, regardless of whether the
diagnostic smoke satisfied the task predicate. Explicit output arguments are
unnecessary for this evaluation: the entry point stores its metrics
and recording under `results_instantiation/{plural}/{program}/{instance}/`.
For FR3 OSC programs it also stores `cartesian_goals.json` there: 50 Hz
commanded end-effector goals, total gripper widths, explicit operations, and
embedded controller settings in the version-2 `play_cartesian_goals.py` JSON
format, not the achieved simulation poses. Do not infer or hand-author this
trajectory.
Report whether this learned-profile run succeeds. The launcher also preserves
the best completed smoke evidence in that results directory when the agent exits.

Binding validity is required; program predicate success is measured separately
and validator/audit disagreements are advisory to the workflow. Preserve useful
trajectories and disclose every retry and unresolved finding in
the final report.
""".strip()


__all__ = [
    "build_instantiation_prompt",
    "CANONICAL_INIT",
    "CANONICAL_MAIN",
    "CANONICAL_PARENT_INIT",
    "initial_binding_manifest",
    "instantiation_dir",
    "instantiation_module",
    "prepare_scene_evidence",
    "program_dir",
    "stage_canonical_instantiation",
    "validate_package_name",
]
