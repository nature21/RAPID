"""Render or interactively inspect a MuJoCo XML scene.

Examples:
    conda run -n rapid python -m vistools.visualize_mujoco scene.xml
    conda run -n rapid python -m vistools.visualize_mujoco scene.xml --orbit-video scene_orbit.mp4
    conda run -n rapid python -m vistools.visualize_mujoco scene.xml --settle-video scene_settle.mp4
    conda run -n rapid python -m vistools.visualize_mujoco scene.xml --viewer
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


_HEADLIGHT_AMBIENT = (0.38, 0.38, 0.38)
_HEADLIGHT_DIFFUSE = (0.82, 0.82, 0.82)
_HEADLIGHT_SPECULAR = (0.35, 0.35, 0.35)
_DEFAULT_CAMERA_DISTANCE_SCALE = 2.8
_DEFAULT_ORBIT_DISTANCE_SCALE = 2.5
_DEFAULT_ORBIT_SETTLE_STEPS = 240


def render_xml(
    xml_path: str | Path,
    output_path: str | Path | None = None,
    *,
    width: int = 1280,
    height: int = 720,
    azimuth: float = 135.0,
    elevation: float = -25.0,
    distance: float | None = None,
    lookat: tuple[float, float, float] | None = None,
    step_count: int = 0,
    boost_headlight: bool = False,
) -> Path:
    """Render a PNG preview of a MuJoCo XML scene."""

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    xml_path = Path(xml_path)
    output_path = Path(output_path) if output_path else xml_path.with_suffix(".png")
    model = _load_model(xml_path)

    return render_model(
        model,
        output_path,
        width=width,
        height=height,
        azimuth=azimuth,
        elevation=elevation,
        distance=distance,
        lookat=lookat,
        step_count=step_count,
        boost_headlight=boost_headlight,
    )


def render_model(
    model: mujoco.MjModel,
    output_path: str | Path,
    *,
    data: mujoco.MjData | None = None,
    width: int = 1280,
    height: int = 720,
    azimuth: float = 135.0,
    elevation: float = -25.0,
    distance: float | None = None,
    lookat: tuple[float, float, float] | None = None,
    step_count: int = 0,
    boost_headlight: bool = False,
) -> Path:
    """Render a PNG preview of a compiled MuJoCo model."""

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import mujoco

    output_path = Path(output_path)
    if data is None:
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    for _ in range(max(0, step_count)):
        mujoco.mj_step(model, data)

    if boost_headlight:
        _boost_headlight(model)
    cam = _make_camera(model, data, azimuth, elevation, distance, lookat)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)

    renderer = mujoco.Renderer(model, width=width, height=height)
    try:
        renderer.update_scene(data, camera=cam)
        image = renderer.render()
    finally:
        renderer.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, image)
    return output_path


def render_orbit_video_xml(
    xml_path: str | Path,
    output_path: str | Path | None = None,
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration: float = 8.0,
    frames: int | None = None,
    start_azimuth: float = 135.0,
    elevation: float = -25.0,
    distance: float | None = None,
    lookat: tuple[float, float, float] | None = None,
    step_count: int = _DEFAULT_ORBIT_SETTLE_STEPS,
    turns: float = 1.0,
    clockwise: bool = False,
    distance_scale: float = _DEFAULT_ORBIT_DISTANCE_SCALE,
    boost_headlight: bool = False,
) -> Path:
    """Render a 360-degree orbit MP4 around a MuJoCo XML scene."""

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    xml_path = Path(xml_path)
    output = Path(output_path) if output_path else xml_path.with_name(
        f"{xml_path.stem}_orbit.mp4"
    )
    model = _load_model(xml_path)
    return render_orbit_video_model(
        model,
        output,
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        frames=frames,
        start_azimuth=start_azimuth,
        elevation=elevation,
        distance=distance,
        lookat=lookat,
        step_count=step_count,
        turns=turns,
        clockwise=clockwise,
        distance_scale=distance_scale,
        boost_headlight=boost_headlight,
    )


def render_orbit_video_model(
    model: mujoco.MjModel,
    output_path: str | Path,
    *,
    data: mujoco.MjData | None = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration: float = 8.0,
    frames: int | None = None,
    start_azimuth: float = 135.0,
    elevation: float = -25.0,
    distance: float | None = None,
    lookat: tuple[float, float, float] | None = None,
    step_count: int = _DEFAULT_ORBIT_SETTLE_STEPS,
    turns: float = 1.0,
    clockwise: bool = False,
    distance_scale: float = _DEFAULT_ORBIT_DISTANCE_SCALE,
    boost_headlight: bool = False,
) -> Path:
    """Render an orbit MP4 around a compiled MuJoCo model."""

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import mujoco

    fps = max(1, int(fps))
    total_frames = int(frames) if frames is not None else int(round(duration * fps))
    total_frames = max(2, total_frames)
    turns = float(turns)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if data is None:
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    for _ in range(max(0, step_count)):
        mujoco.mj_step(model, data)

    if boost_headlight:
        _boost_headlight(model)
    cam = _make_camera(
        model,
        data,
        start_azimuth,
        elevation,
        distance,
        lookat,
        distance_scale=distance_scale,
    )
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)

    direction = -1.0 if clockwise else 1.0
    renderer = mujoco.Renderer(model, width=width, height=height)
    try:
        with imageio.get_writer(
            output_path,
            fps=fps,
            quality=8,
            macro_block_size=1,
        ) as writer:
            for frame_idx in range(total_frames):
                phase = frame_idx / total_frames
                cam.azimuth = start_azimuth + direction * 360.0 * turns * phase
                renderer.update_scene(data, camera=cam)
                writer.append_data(renderer.render())
    finally:
        renderer.close()

    return output_path


def render_settle_video_xml(
    xml_path: str | Path,
    output_path: str | Path | None = None,
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration: float = 5.0,
    frames: int | None = None,
    start_azimuth: float = 135.0,
    elevation: float = -25.0,
    distance: float | None = None,
    lookat: tuple[float, float, float] | None = None,
    step_count: int = 0,
    distance_scale: float = _DEFAULT_ORBIT_DISTANCE_SCALE,
    boost_headlight: bool = False,
) -> Path:
    """Render a fixed-camera MP4 while the scene stabilizes from reset."""

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    xml_path = Path(xml_path)
    output = Path(output_path) if output_path else xml_path.with_name(
        f"{xml_path.stem}_settle.mp4"
    )
    model = _load_model(xml_path)
    return render_settle_video_model(
        model,
        output,
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        frames=frames,
        start_azimuth=start_azimuth,
        elevation=elevation,
        distance=distance,
        lookat=lookat,
        step_count=step_count,
        distance_scale=distance_scale,
        boost_headlight=boost_headlight,
    )


def render_settle_video_model(
    model: mujoco.MjModel,
    output_path: str | Path,
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration: float = 5.0,
    frames: int | None = None,
    start_azimuth: float = 135.0,
    elevation: float = -25.0,
    distance: float | None = None,
    lookat: tuple[float, float, float] | None = None,
    step_count: int = 0,
    distance_scale: float = _DEFAULT_ORBIT_DISTANCE_SCALE,
    boost_headlight: bool = False,
) -> Path:
    """Render a fixed-camera MP4 while MuJoCo advances from reset."""

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import mujoco

    fps = max(1, int(fps))
    total_frames = int(frames) if frames is not None else int(round(duration * fps))
    total_frames = max(2, total_frames)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    for _ in range(max(0, step_count)):
        mujoco.mj_step(model, data)

    if boost_headlight:
        _boost_headlight(model)
    cam = _make_camera(
        model,
        data,
        start_azimuth,
        elevation,
        distance,
        lookat,
        distance_scale=distance_scale,
    )
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)

    renderer = mujoco.Renderer(model, width=width, height=height)
    try:
        with imageio.get_writer(
            output_path,
            fps=fps,
            quality=8,
            macro_block_size=1,
        ) as writer:
            for frame_idx in range(total_frames):
                target_time = frame_idx / fps
                while data.time + 0.5 * model.opt.timestep < target_time:
                    mujoco.mj_step(model, data)
                renderer.update_scene(data, camera=cam)
                writer.append_data(renderer.render())
    finally:
        renderer.close()

    return output_path


def launch_viewer(
    xml_path: str | Path,
    *,
    azimuth: float = 135.0,
    elevation: float = -25.0,
    distance: float | None = None,
    lookat: tuple[float, float, float] | None = None,
    boost_headlight: bool = False,
) -> None:
    """Open MuJoCo's interactive viewer for a scene."""

    model = _load_model(xml_path)
    launch_model_viewer(
        model,
        azimuth=azimuth,
        elevation=elevation,
        distance=distance,
        lookat=lookat,
        boost_headlight=boost_headlight,
    )


def launch_model_viewer(
    model: mujoco.MjModel,
    *,
    data: mujoco.MjData | None = None,
    azimuth: float = 135.0,
    elevation: float = -25.0,
    distance: float | None = None,
    lookat: tuple[float, float, float] | None = None,
    boost_headlight: bool = False,
) -> None:
    """Open MuJoCo's interactive viewer for a compiled model."""

    import mujoco
    import mujoco.viewer

    if data is None:
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    if boost_headlight:
        _boost_headlight(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        cam = _make_camera(model, data, azimuth, elevation, distance, lookat)
        viewer.cam.type = cam.type
        viewer.cam.lookat[:] = cam.lookat
        viewer.cam.distance = cam.distance
        viewer.cam.azimuth = cam.azimuth
        viewer.cam.elevation = cam.elevation
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


def _load_model(xml_path: str | Path) -> mujoco.MjModel:
    import mujoco

    xml_path = Path(xml_path)
    if not xml_path.is_file():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
    return mujoco.MjModel.from_xml_path(str(xml_path))


def _boost_headlight(model: mujoco.MjModel) -> None:
    model.vis.headlight.ambient[:] = _HEADLIGHT_AMBIENT
    model.vis.headlight.diffuse[:] = _HEADLIGHT_DIFFUSE
    model.vis.headlight.specular[:] = _HEADLIGHT_SPECULAR


def _make_camera(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    azimuth: float,
    elevation: float,
    distance: float | None,
    lookat: tuple[float, float, float] | None,
    *,
    distance_scale: float = _DEFAULT_CAMERA_DISTANCE_SCALE,
) -> mujoco.MjvCamera:
    import mujoco

    center, radius = _scene_center_and_radius(model, data)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.asarray(lookat, dtype=float) if lookat is not None else center
    cam.distance = (
        float(distance)
        if distance is not None
        else max(0.18, radius * max(0.1, float(distance_scale)))
    )
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    return cam


def _scene_center_and_radius(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[np.ndarray, float]:
    import mujoco

    points = []

    for body_id in range(1, model.nbody):
        points.append(data.xpos[body_id].copy())

    for geom_id in range(model.ngeom):
        geom_type = model.geom_type[geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        pos = data.geom_xpos[geom_id]
        size = model.geom_size[geom_id]
        extent = float(np.max(size[:3])) if size.size else 0.0
        points.append(pos + extent)
        points.append(pos - extent)

    if not points:
        return np.zeros(3, dtype=float), 1.0

    pts = np.asarray(points, dtype=float)
    center = pts.mean(axis=0)
    radius = float(np.max(np.linalg.norm(pts - center, axis=1)))
    return center, max(radius, 0.1)


def _parse_vec3(value: str) -> tuple[float, float, float]:
    parts = [float(part) for part in value.replace(",", " ").split()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three values, e.g. '0 0 0'")
    return parts[0], parts[1], parts[2]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=Path, help="Path to a MuJoCo XML file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="PNG output path. Defaults to <xml-stem>.png next to the XML.",
    )
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--azimuth", type=float, default=90.0)
    parser.add_argument("--elevation", type=float, default=-45.0)
    parser.add_argument("--distance", type=float, default=None)
    parser.add_argument("--lookat", type=_parse_vec3, default=None)
    parser.add_argument(
        "--step-count",
        type=int,
        default=None,
        help=(
            "Simulation steps to advance before rendering. Defaults to 0 for "
            "PNG preview and settle video, and "
            f"{_DEFAULT_ORBIT_SETTLE_STEPS} for orbit video."
        ),
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Launch the interactive MuJoCo viewer instead of writing a PNG.",
    )
    parser.add_argument(
        "--orbit-video",
        type=Path,
        default=None,
        help=(
            "Write a 360-degree orbit MP4 instead of a PNG. If a path is given, "
            "it is used as the video output."
        ),
    )
    parser.add_argument(
        "--settle-video",
        type=Path,
        default=None,
        help=(
            "Write a fixed-camera MP4 from the orbit starting camera while the "
            "scene stabilizes from reset."
        ),
    )
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate.")
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Video duration in seconds.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Exact orbit video frame count; overrides --duration.",
    )
    parser.add_argument(
        "--turns",
        type=float,
        default=1.0,
        help="Number of full 360-degree camera orbits.",
    )
    parser.add_argument(
        "--clockwise",
        action="store_true",
        help="Orbit clockwise instead of counter-clockwise.",
    )
    parser.add_argument(
        "--orbit-distance-scale",
        type=float,
        default=_DEFAULT_ORBIT_DISTANCE_SCALE,
        help=(
            "Camera distance multiplier for orbit/settle video when --distance is not "
            f"set. Default: {_DEFAULT_ORBIT_DISTANCE_SCALE}."
        ),
    )
    parser.add_argument(
        "--boost-headlight",
        action="store_true",
        help="Override scene XML lighting with the old bright preview headlight.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.orbit_video is not None and args.settle_video is not None:
        raise SystemExit("--orbit-video and --settle-video cannot be used together")
    if args.viewer:
        if args.orbit_video is not None or args.settle_video is not None:
            raise SystemExit("--viewer cannot be combined with video output options")
        launch_viewer(
            args.xml,
            azimuth=args.azimuth,
            elevation=args.elevation,
            distance=args.distance,
            lookat=args.lookat,
            boost_headlight=args.boost_headlight,
        )
    elif args.settle_video is not None:
        step_count = 0 if args.step_count is None else args.step_count
        output_path = render_settle_video_xml(
            args.xml,
            args.settle_video,
            width=args.width,
            height=args.height,
            fps=args.fps,
            duration=args.duration,
            frames=args.frames,
            start_azimuth=args.azimuth,
            elevation=args.elevation,
            distance=args.distance,
            lookat=args.lookat,
            step_count=step_count,
            distance_scale=args.orbit_distance_scale,
            boost_headlight=args.boost_headlight,
        )
        print(output_path)
    elif args.orbit_video is not None:
        step_count = (
            _DEFAULT_ORBIT_SETTLE_STEPS if args.step_count is None else args.step_count
        )
        output_path = render_orbit_video_xml(
            args.xml,
            args.orbit_video,
            width=args.width,
            height=args.height,
            fps=args.fps,
            duration=args.duration,
            frames=args.frames,
            start_azimuth=args.azimuth,
            elevation=args.elevation,
            distance=args.distance,
            lookat=args.lookat,
            step_count=step_count,
            turns=args.turns,
            clockwise=args.clockwise,
            distance_scale=args.orbit_distance_scale,
            boost_headlight=args.boost_headlight,
        )
        print(output_path)
    else:
        step_count = 0 if args.step_count is None else args.step_count
        output_path = render_xml(
            args.xml,
            args.output,
            width=args.width,
            height=args.height,
            azimuth=args.azimuth,
            elevation=args.elevation,
            distance=args.distance,
            lookat=args.lookat,
            step_count=step_count,
            boost_headlight=args.boost_headlight,
        )
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
