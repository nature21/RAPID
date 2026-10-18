"""MP4 recording helper shared by every test scene's ``record_video``.

:func:`record_video` takes an :class:`mujoco.MjModel`, an
:class:`mujoco.MjData` and a state trajectory, and writes an offscreen
rendering to disk. Each test scene supplies its own default camera
parameters via :func:`scene_io.playback.make_playback`.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np

# Default offscreen headlight values that make videos look reasonable.
_RECORD_HEADLIGHT_AMBIENT = (0.38, 0.38, 0.38)
_RECORD_HEADLIGHT_DIFFUSE = (0.82, 0.82, 0.82)
_RECORD_HEADLIGHT_SPECULAR = (0.35, 0.35, 0.35)
_GPU_RENDER_QUEUE_ENV = "RAPID_GPU_RENDER_QUEUE"
_GPU_RENDER_PROTOCOL_VERSION = 1
_GPU_RENDER_REPLY_LIMIT = 1 << 20


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = float(raw)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _boost_headlight_for_recording(model: mujoco.MjModel) -> None:
    model.vis.headlight.ambient[:] = _RECORD_HEADLIGHT_AMBIENT
    model.vis.headlight.diffuse[:] = _RECORD_HEADLIGHT_DIFFUSE
    model.vis.headlight.specular[:] = _RECORD_HEADLIGHT_SPECULAR


def _read_broker_reply(reply_path: Path) -> dict[str, object]:
    if reply_path.stat().st_size > _GPU_RENDER_REPLY_LIMIT:
        raise RuntimeError("GPU render broker reply exceeds the size limit")
    payload = reply_path.read_text(encoding="utf-8")
    if not payload:
        raise RuntimeError("GPU render broker returned an empty reply")
    reply = json.loads(payload)
    if not isinstance(reply, dict):
        raise RuntimeError("GPU render broker reply must be a JSON object")
    return reply


def _gpu_render_queue() -> Path | None:
    """Resolve an explicit launcher queue or a live workspace broker."""

    if os.environ.get("RAPID_GPU_RENDER", "auto").strip().lower() == "off":
        return None
    queue_value = os.environ.get(_GPU_RENDER_QUEUE_ENV)
    if queue_value:
        return Path(queue_value).expanduser()
    workspace_value = os.environ.get("RAPID_WORKSPACE")
    if not workspace_value:
        return None
    workspace = Path(workspace_value).expanduser().resolve(strict=False)
    ready_path = workspace / "agent_work/gpu_render_ready.json"
    try:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        queue = Path(str(ready["queue"])).expanduser().resolve(strict=True)
        queue.relative_to(workspace)
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None
    return queue if queue.is_dir() else None


def _record_video_via_gpu_broker(
    model: mujoco.MjModel,
    states: np.ndarray,
    dt: float,
    output_path: str,
    *,
    width: int,
    height: int,
    fps: int,
    playback_speed: float,
    camera_lookat: tuple[float, float, float] | None,
    camera_distance: float | None,
    camera_azimuth: float | None,
    camera_elevation: float | None,
) -> bool:
    """Delegate to a verified host renderer when the launcher provides one."""

    request_root = _gpu_render_queue()
    if request_root is None:
        return False
    request_root.mkdir(parents=True, exist_ok=True)
    request_dir = Path(tempfile.mkdtemp(prefix="request_", dir=request_root))
    model_path = request_dir / "model.mjb"
    states_path = request_dir / "states.npy"
    pending_path = request_dir / "request.pending"
    request_path = request_dir / "request.json"
    reply_path = request_dir / "reply.json"
    try:
        mujoco.mj_saveModel(model, str(model_path))
        np.save(states_path, np.asarray(states), allow_pickle=False)
        output = Path(output_path).expanduser()
        if not output.is_absolute():
            output = Path.cwd() / output
        request = {
            "version": _GPU_RENDER_PROTOCOL_VERSION,
            "model_path": str(model_path),
            "states_path": str(states_path),
            "output_path": str(output.resolve(strict=False)),
            "dt": float(dt),
            "width": int(width),
            "height": int(height),
            "fps": int(fps),
            "playback_speed": float(playback_speed),
            "camera_lookat": list(camera_lookat) if camera_lookat is not None else None,
            "camera_distance": camera_distance,
            "camera_azimuth": camera_azimuth,
            "camera_elevation": camera_elevation,
        }
        pending_path.write_text(json.dumps(request), encoding="utf-8")
        pending_path.replace(request_path)
        deadline = time.monotonic() + 1800.0
        while not reply_path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for the GPU render broker")
            time.sleep(0.05)
        reply = _read_broker_reply(reply_path)
        if not reply.get("ok"):
            raise RuntimeError(
                "GPU render broker failed: " + str(reply.get("error", "unknown error"))
            )
        frame_count = int(reply["frames"])
        duration = float(reply["duration"])
        elapsed = float(reply["elapsed"])
        render_fps = reply.get("render_fps")
        rate = f", {float(render_fps):.1f} render fps" if render_fps is not None else ""
        print(
            f"Video saved to {output_path}  "
            f"({frame_count} frames, {duration:.1f}s at {fps}fps; "
            f"GPU broker {elapsed:.2f}s{rate})"
        )
        return True
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"GPU render broker communication failed: {exc}") from exc
    finally:
        model_path.unlink(missing_ok=True)
        states_path.unlink(missing_ok=True)
        pending_path.unlink(missing_ok=True)
        request_path.unlink(missing_ok=True)
        reply_path.unlink(missing_ok=True)
        try:
            request_dir.rmdir()
        except OSError:
            pass


def record_video(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    states: np.ndarray,
    dt: float,
    output_path: str,
    *,
    width: int = 960,
    height: int = 540,
    fps: int = 30,
    playback_speed: float = 1.0,
    camera_lookat: tuple[float, float, float] | None = None,
    camera_distance: float | None = None,
    camera_azimuth: float | None = None,
    camera_elevation: float | None = None,
    headlight_boost: bool = True,
) -> None:
    """Render a state trajectory offscreen and write to MP4."""
    import imageio

    width = _positive_int_env("RAPID_RECORD_WIDTH", width)
    height = _positive_int_env("RAPID_RECORD_HEIGHT", height)
    fps = _positive_int_env("RAPID_RECORD_FPS", fps)
    playback_speed = _positive_float_env(
        "RAPID_RECORD_PLAYBACK_SPEED", playback_speed
    )

    if headlight_boost:
        _boost_headlight_for_recording(model)

    if _record_video_via_gpu_broker(
        model,
        states,
        dt,
        output_path,
        width=width,
        height=height,
        fps=fps,
        playback_speed=playback_speed,
        camera_lookat=camera_lookat,
        camera_distance=camera_distance,
        camera_azimuth=camera_azimuth,
        camera_elevation=camera_elevation,
    ):
        return

    frame_stride = max(1, int(round(1.0 / (fps * dt) * playback_speed)))
    frame_indices = np.arange(0, len(states), frame_stride)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if camera_lookat is not None:
        cam.lookat[:] = camera_lookat
    if camera_distance is not None:
        cam.distance = camera_distance
    if camera_azimuth is not None:
        cam.azimuth = camera_azimuth
    if camera_elevation is not None:
        cam.elevation = camera_elevation

    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(model, width=width, height=height)

    writer = imageio.get_writer(output_path, fps=fps, quality=8)
    try:
        for i in frame_indices:
            state = states[i]
            nq = model.nq
            data.qpos[:] = state[:nq]
            data.qvel[:] = state[nq:]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()

    duration = len(frame_indices) / fps
    print(
        f"Video saved to {output_path}  "
        f"({len(frame_indices)} frames, {duration:.1f}s at {fps}fps)"
    )
