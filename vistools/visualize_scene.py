"""Render or interactively inspect a Python task scene module.

Examples:
    ./scripts/rapid-python -m vistools.visualize_scene tasks.invention_real2sim.demo_5_scene
    ./scripts/rapid-python -m vistools.visualize_scene tasks.invention_real2sim.demo_5_scene --viewer
    ./scripts/rapid-python -m vistools.visualize_scene tasks/invention_real2sim/demo_5_scene/scene.py
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any

from vistools.visualize_mujoco import (
    launch_model_viewer,
    render_model,
    _parse_vec3,
)


def render_scene(
    scene_ref: str | Path,
    output_path: str | Path | None = None,
    *,
    width: int = 1280,
    height: int = 720,
    azimuth: float | None = None,
    elevation: float | None = None,
    distance: float | None = None,
    lookat: tuple[float, float, float] | None = None,
    step_count: int = 0,
) -> Path:
    """Render a PNG preview of a scene module exposing ``make_model()``."""

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    module = _load_scene_module(scene_ref)
    model = _make_model(module)
    data = _make_data(module, model)
    camera = _camera_options(
        module,
        azimuth=azimuth,
        elevation=elevation,
        distance=distance,
        lookat=lookat,
    )
    output = Path(output_path) if output_path else _default_output_path(scene_ref, module)
    return render_model(
        model,
        output,
        data=data,
        width=width,
        height=height,
        step_count=step_count,
        **camera,
    )


def launch_scene_viewer(
    scene_ref: str | Path,
    *,
    azimuth: float | None = None,
    elevation: float | None = None,
    distance: float | None = None,
    lookat: tuple[float, float, float] | None = None,
) -> None:
    """Open MuJoCo's interactive viewer for a scene module."""

    module = _load_scene_module(scene_ref)
    model = _make_model(module)
    data = _make_data(module, model)
    camera = _camera_options(
        module,
        azimuth=azimuth,
        elevation=elevation,
        distance=distance,
        lookat=lookat,
    )
    launch_model_viewer(model, data=data, **camera)


def _load_scene_module(scene_ref: str | Path) -> ModuleType:
    text = str(scene_ref)
    path = Path(text).expanduser()
    if path.is_file():
        return _load_scene_file(path)
    return _load_module(_normalize_module_name(text))


def _load_scene_file(path: Path) -> ModuleType:
    module_name = f"_rapid_scene_view_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load scene file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_module_name(value: str) -> str:
    module = value.strip()
    if not module:
        raise ValueError("scene module cannot be empty")
    if module.endswith(".scene"):
        return module
    scene_module = f"{module}.scene"
    try:
        if importlib.util.find_spec(scene_module) is not None:
            return scene_module
    except (AttributeError, ImportError, ModuleNotFoundError):
        pass
    return module


def _load_module(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"could not import {module_name!r}; pass a module like "
            "'tasks.invention_real2sim.demo_5_scene' or a path to scene.py"
        ) from exc


def _make_model(module: ModuleType) -> Any:
    make_model = getattr(module, "make_model", None)
    if not callable(make_model):
        raise AttributeError(f"{module.__name__} does not expose callable make_model()")
    return make_model()


def _make_data(module: ModuleType, model: Any) -> Any | None:
    make_data = getattr(module, "make_data", None)
    if not callable(make_data):
        return None
    return make_data(model)


def _camera_options(
    module: ModuleType,
    *,
    azimuth: float | None,
    elevation: float | None,
    distance: float | None,
    lookat: tuple[float, float, float] | None,
) -> dict[str, Any]:
    default_camera = getattr(module, "DEFAULT_CAMERA", {}) or {}
    return {
        "azimuth": _coalesce_float(azimuth, default_camera.get("camera_azimuth"), 135.0),
        "elevation": _coalesce_float(
            elevation,
            default_camera.get("camera_elevation"),
            -25.0,
        ),
        "distance": _coalesce_float_or_none(
            distance,
            default_camera.get("camera_distance"),
        ),
        "lookat": lookat
        if lookat is not None
        else _coerce_vec3_or_none(default_camera.get("camera_lookat")),
    }


def _coalesce_float(*values: Any) -> float:
    for value in values:
        if value is not None:
            return float(value)
    raise ValueError("expected at least one non-None value")


def _coalesce_float_or_none(*values: Any) -> float | None:
    for value in values:
        if value is not None:
            return float(value)
    return None


def _coerce_vec3_or_none(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    parts = [float(part) for part in value]
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def _default_output_path(scene_ref: str | Path, module: ModuleType) -> Path:
    package = module.__name__.split(".")[-2] if module.__name__.endswith(".scene") else ""
    if not package or package.startswith("_rapid_scene_view_"):
        path = Path(str(scene_ref))
        package = path.parent.name if path.stem == "scene" else path.stem
    return Path("results") / "real2sim" / "previews" / f"{package}.png"


def build_arg_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "scene",
        help=(
            "Scene module or scene.py path. Module roots such as "
            "tasks.invention_real2sim.demo_5_scene are normalized to .scene."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="PNG output path. Defaults to results/real2sim/previews/<scene>.png.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--azimuth", type=float, default=None)
    parser.add_argument("--elevation", type=float, default=None)
    parser.add_argument("--distance", type=float, default=None)
    parser.add_argument("--lookat", type=_parse_vec3, default=None)
    parser.add_argument(
        "--step-count",
        type=int,
        default=0,
        help="Simulation steps to advance before rendering a PNG.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Launch the interactive MuJoCo viewer instead of writing a PNG.",
    )
    return parser


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    args = build_arg_parser(prog).parse_args(argv)
    if args.viewer:
        launch_scene_viewer(
            args.scene,
            azimuth=args.azimuth,
            elevation=args.elevation,
            distance=args.distance,
            lookat=args.lookat,
        )
    else:
        output_path = render_scene(
            args.scene,
            args.output,
            width=args.width,
            height=args.height,
            azimuth=args.azimuth,
            elevation=args.elevation,
            distance=args.distance,
            lookat=args.lookat,
            step_count=args.step_count,
        )
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
