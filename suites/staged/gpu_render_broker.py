"""Trusted host-side MuJoCo video renderer for sandboxed RAPID agents.

Codex's Linux command sandbox intentionally exposes a synthetic ``/dev`` and
cannot open NVIDIA character devices read/write.  This broker stays outside
that nested sandbox, accepts only bounded render requests rooted in one suite
workspace, and returns the completed MP4 to the sandboxed caller.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TextIO


GPU_RENDER_MODE_ENV = "RAPID_GPU_RENDER"
GPU_RENDER_QUEUE_ENV = "RAPID_GPU_RENDER_QUEUE"
GPU_RENDER_MODES = ("auto", "required", "off")
PROTOCOL_VERSION = 1
MAX_REQUEST_BYTES = 1 << 20
MAX_RENDER_DIMENSION = 4096
MAX_RENDER_FPS = 240
MAX_STATE_COUNT = 2_000_000
MAX_RENDER_FRAMES = 100_000
MAX_MODEL_BYTES = 2 << 30
MAX_STATES_BYTES = 4 << 30
RENDER_OUTPUT_ROOT_NAMES = ("results", "results_instantiation", "agent_work")


class GpuRenderBrokerError(RuntimeError):
    """Raised when the host broker cannot start or satisfy a request."""


def parse_gpu_render_mode(value: str | None) -> str:
    """Return a normalized GPU-render policy or raise for invalid input."""

    mode = (value or "auto").strip().lower()
    if mode not in GPU_RENDER_MODES:
        choices = ", ".join(GPU_RENDER_MODES)
        raise ValueError(f"GPU render mode must be one of {choices}; got {value!r}")
    return mode


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return path == root
    return True


def _validate_render_output_path(output_path: Path, *, workspace: Path) -> None:
    """Require rendered videos to stay in one declared artifact directory."""

    writable_roots = tuple(workspace / name for name in RENDER_OUTPUT_ROOT_NAMES)
    if any(_is_under(output_path, root) for root in writable_roots):
        return
    rendered = ", ".join(f"workspace/{name}" for name in RENDER_OUTPUT_ROOT_NAMES)
    raise GpuRenderBrokerError(f"output_path must live under {rendered}")


def _workspace_path(
    raw: object,
    *,
    workspace: Path,
    description: str,
    must_exist: bool,
) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = workspace / path
    resolved = path.resolve(strict=must_exist)
    if not _is_under(resolved, workspace):
        raise GpuRenderBrokerError(
            f"{description} escapes the suite workspace: {resolved}"
        )
    return resolved


def _bounded_regular_file(path: Path, *, name: str, maximum: int) -> None:
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise GpuRenderBrokerError(f"{name} must be a regular file")
    if metadata.st_size > maximum:
        raise GpuRenderBrokerError(
            f"{name} exceeds the {maximum}-byte size limit"
        )


def _positive_int(value: object, *, name: str, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise GpuRenderBrokerError(f"{name} must be an integer") from exc
    if parsed < 1 or parsed > maximum:
        raise GpuRenderBrokerError(
            f"{name} must be between 1 and {maximum}; got {parsed}"
        )
    return parsed


def _positive_float(value: object, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GpuRenderBrokerError(f"{name} must be numeric") from exc
    if not (parsed > 0.0):
        raise GpuRenderBrokerError(f"{name} must be positive; got {parsed}")
    return parsed


def _optional_float(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise GpuRenderBrokerError(f"{name} must be numeric or null") from exc


def _camera_lookat(value: object) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise GpuRenderBrokerError("camera_lookat must contain exactly three values")
    try:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError) as exc:
        raise GpuRenderBrokerError("camera_lookat values must be numeric") from exc


def _graphics_backend_evidence() -> dict[str, object]:
    maps = Path("/proc/self/maps").read_text(encoding="utf-8", errors="replace")
    device_fds: set[str] = set()
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith("/dev/nvidia"):
            device_fds.add(target)
    return {
        "nvidia_device_fds": sorted(device_fds),
        "uses_swrast": "swrast_dri.so" in maps,
        "loads_nvidia_egl": "libEGL_nvidia" in maps,
    }


def probe_nvidia_egl() -> dict[str, object]:
    """Verify the NVIDIA context and the MP4 encoder used by recordings."""

    import imageio
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, width=64, height=64)
    try:
        renderer.update_scene(data)
        frame = renderer.render()
        evidence = _graphics_backend_evidence()
    finally:
        renderer.close()
    if (
        evidence["uses_swrast"]
        or not evidence["loads_nvidia_egl"]
        or not evidence["nvidia_device_fds"]
    ):
        raise GpuRenderBrokerError(
            "MuJoCo EGL did not acquire an NVIDIA device; "
            f"backend evidence: {evidence}"
        )
    with tempfile.TemporaryDirectory(prefix="rapid_gpu_render_probe_") as raw:
        probe_path = Path(raw) / "probe.mp4"
        writer = imageio.get_writer(probe_path, fps=1, quality=8)
        try:
            writer.append_data(frame)
        finally:
            writer.close()
        if not probe_path.is_file() or probe_path.stat().st_size == 0:
            raise GpuRenderBrokerError("the MP4 encoder produced no output")
        evidence["mp4_encoder"] = True
    return evidence


def _render_request(
    request: Mapping[str, object],
    *,
    workspace: Path,
    request_dir: Path,
) -> dict[str, object]:
    import imageio
    import mujoco
    import numpy as np

    if request.get("version") != PROTOCOL_VERSION:
        raise GpuRenderBrokerError(
            f"unsupported render protocol version: {request.get('version')!r}"
        )
    model_path = _workspace_path(
        request.get("model_path"),
        workspace=workspace,
        description="model_path",
        must_exist=True,
    )
    states_path = _workspace_path(
        request.get("states_path"),
        workspace=workspace,
        description="states_path",
        must_exist=True,
    )
    output_path = _workspace_path(
        request.get("output_path"),
        workspace=workspace,
        description="output_path",
        must_exist=False,
    )
    if model_path.suffix != ".mjb":
        raise GpuRenderBrokerError("model_path must use the .mjb extension")
    if states_path.suffix != ".npy":
        raise GpuRenderBrokerError("states_path must use the .npy extension")
    if output_path.suffix.lower() != ".mp4":
        raise GpuRenderBrokerError("output_path must use the .mp4 extension")
    expected_model = (request_dir / "model.mjb").resolve(strict=True)
    expected_states = (request_dir / "states.npy").resolve(strict=True)
    if model_path != expected_model or states_path != expected_states:
        raise GpuRenderBrokerError(
            "model_path and states_path must belong to the active render request"
        )
    _validate_render_output_path(output_path, workspace=workspace)
    _bounded_regular_file(model_path, name="model_path", maximum=MAX_MODEL_BYTES)
    _bounded_regular_file(states_path, name="states_path", maximum=MAX_STATES_BYTES)

    width = _positive_int(
        request.get("width"), name="width", maximum=MAX_RENDER_DIMENSION
    )
    height = _positive_int(
        request.get("height"), name="height", maximum=MAX_RENDER_DIMENSION
    )
    fps = _positive_int(request.get("fps"), name="fps", maximum=MAX_RENDER_FPS)
    dt = _positive_float(request.get("dt"), name="dt")
    playback_speed = _positive_float(
        request.get("playback_speed"), name="playback_speed"
    )

    model = mujoco.MjModel.from_binary_path(str(model_path))
    data = mujoco.MjData(model)
    states = np.load(states_path, mmap_mode="r", allow_pickle=False)
    if states.ndim != 2 or states.shape[1] != model.nq + model.nv:
        raise GpuRenderBrokerError(
            "states must have shape "
            f"(N, {model.nq + model.nv}); got {tuple(states.shape)}"
        )
    if states.shape[0] < 1 or states.shape[0] > MAX_STATE_COUNT:
        raise GpuRenderBrokerError(
            f"state count must be between 1 and {MAX_STATE_COUNT}; "
            f"got {states.shape[0]}"
        )

    frame_stride = max(1, int(round(1.0 / (fps * dt) * playback_speed)))
    frame_indices = range(0, len(states), frame_stride)
    if len(frame_indices) > MAX_RENDER_FRAMES:
        raise GpuRenderBrokerError(
            f"render request exceeds the {MAX_RENDER_FRAMES}-frame limit"
        )
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if lookat := _camera_lookat(request.get("camera_lookat")):
        cam.lookat[:] = lookat
    for attribute, request_name in (
        ("distance", "camera_distance"),
        ("azimuth", "camera_azimuth"),
        ("elevation", "camera_elevation"),
    ):
        value = _optional_float(request.get(request_name), name=request_name)
        if value is not None:
            setattr(cam, attribute, value)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(model, width=width, height=height)
    started = time.perf_counter()
    try:
        writer = imageio.get_writer(output_path, fps=fps, quality=8)
        try:
            for index in frame_indices:
                state = states[index]
                data.qpos[:] = state[: model.nq]
                data.qvel[:] = state[model.nq :]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam)
                writer.append_data(renderer.render())
        finally:
            writer.close()
    finally:
        renderer.close()

    elapsed = time.perf_counter() - started
    return {
        "ok": True,
        "frames": int(len(frame_indices)),
        "duration": float(len(frame_indices) / fps),
        "elapsed": elapsed,
        "render_fps": float(len(frame_indices) / elapsed) if elapsed > 0.0 else None,
        "output_path": str(output_path),
    }


def _read_request(request_path: Path) -> dict[str, object]:
    if request_path.stat().st_size > MAX_REQUEST_BYTES:
        raise GpuRenderBrokerError("render request exceeds the size limit")
    payload = request_path.read_text(encoding="utf-8")
    if not payload:
        raise GpuRenderBrokerError("empty render request")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise GpuRenderBrokerError("render request must be a JSON object")
    return value


def _write_reply(reply_path: Path, payload: Mapping[str, object]) -> None:
    pending_path = reply_path.with_suffix(".pending")
    pending_path.unlink(missing_ok=True)
    pending_path.write_text(
        json.dumps(dict(payload), sort_keys=True),
        encoding="utf-8",
    )
    pending_path.replace(reply_path)


def serve(*, workspace: Path, queue_path: Path, ready_path: Path) -> int:
    """Run the single-workspace broker until the parent requests shutdown."""

    workspace = workspace.expanduser().resolve()
    queue_path = queue_path.expanduser().resolve(strict=False)
    ready_path = ready_path.expanduser().resolve(strict=False)
    if not workspace.is_dir():
        raise GpuRenderBrokerError(f"workspace does not exist: {workspace}")
    for path, description in (
        (queue_path, "request queue"),
        (ready_path, "ready file"),
    ):
        if not _is_under(path.resolve(strict=False), workspace):
            raise GpuRenderBrokerError(f"{description} must live under {workspace}")

    queue_path.mkdir(parents=True, exist_ok=True)
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    ready_path.unlink(missing_ok=True)

    backend = probe_nvidia_egl()
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    try:
        ready_path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "queue": str(queue_path),
                    "backend": backend,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"GPU render broker ready: {queue_path}; backend={backend}",
            flush=True,
        )
        while not stopping:
            handled_request = False
            for candidate in sorted(queue_path.iterdir()):
                if not candidate.name.startswith("request_"):
                    continue
                try:
                    request_dir = candidate.resolve(strict=True)
                except (FileNotFoundError, OSError):
                    continue
                if request_dir.parent != queue_path or not request_dir.is_dir():
                    continue
                request_path = request_dir / "request.json"
                reply_path = request_dir / "reply.json"
                if not request_path.is_file() or reply_path.exists():
                    continue
                try:
                    request = _read_request(request_path)
                    response = _render_request(
                        request,
                        workspace=workspace,
                        request_dir=request_dir,
                    )
                except Exception as exc:
                    response = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                try:
                    _write_reply(reply_path, response)
                except OSError as exc:
                    print(
                        f"GPU render reply was abandoned for {request_dir}: {exc}",
                        flush=True,
                    )
                handled_request = True
            if not handled_request:
                time.sleep(0.05)
    finally:
        ready_path.unlink(missing_ok=True)
    return 0


@dataclass
class GpuRenderBrokerProcess:
    """Lifecycle handle for one trusted broker subprocess."""

    process: subprocess.Popen[str]
    queue_path: Path
    ready_path: Path
    log_path: Path
    log_handle: TextIO
    backend: Mapping[str, object]

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)
        self.log_handle.close()
        self.ready_path.unlink(missing_ok=True)
        try:
            self.queue_path.rmdir()
        except OSError:
            pass


def _nvidia_driver_visible(env: Mapping[str, str]) -> tuple[bool, str]:
    executable = shutil_which("nvidia-smi", env=env)
    if executable is None:
        return False, "nvidia-smi is not available on PATH"
    try:
        completed = subprocess.run(
            [executable, "-L"],
            env=dict(env),
            capture_output=True,
            text=True,
            timeout=8.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"could not execute nvidia-smi: {exc}"
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        return False, f"nvidia-smi failed: {detail or f'exit {completed.returncode}'}"
    if not completed.stdout.strip():
        return False, "nvidia-smi reported no GPUs"
    return True, completed.stdout.strip().replace("\n", "; ")


def shutil_which(command: str, *, env: Mapping[str, str]) -> str | None:
    """Resolve a command against an explicit environment PATH."""

    import shutil

    return shutil.which(command, path=env.get("PATH"))


def start_gpu_render_broker(
    *,
    workspace: Path,
    env: Mapping[str, str],
    mode: str,
    startup_timeout: float = 30.0,
) -> GpuRenderBrokerProcess | None:
    """Start a verified broker according to ``auto|required|off`` policy."""

    mode = parse_gpu_render_mode(mode)
    if mode == "off":
        return None
    visible, detail = _nvidia_driver_visible(env)
    if not visible:
        if mode == "required":
            raise GpuRenderBrokerError(detail)
        print(f"[rapid-gpu-render] unavailable: {detail}")
        return None

    workspace = workspace.expanduser().resolve()
    queue_root = workspace / "agent_work/gpu_render_requests"
    ready_path = workspace / "agent_work/gpu_render_ready.json"
    log_path = workspace / "agent_work/gpu_render_broker.log"
    ready_path.unlink(missing_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    queue_root.mkdir(parents=True, exist_ok=True)
    queue_path = Path(tempfile.mkdtemp(prefix="broker_", dir=queue_root))
    log_handle = log_path.open("a", encoding="utf-8")

    broker_env = dict(env)
    broker_env["MUJOCO_GL"] = "egl"
    broker_env["PYOPENGL_PLATFORM"] = "egl"
    command = [
        sys.executable,
        "-m",
        "suites.staged.gpu_render_broker",
        "--workspace",
        str(workspace),
        "--queue",
        str(queue_path),
        "--ready-file",
        str(ready_path),
    ]
    try:
        process = subprocess.Popen(
            command,
            cwd=workspace,
            env=broker_env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        log_handle.close()
        queue_path.rmdir()
        message = f"could not start GPU render broker: {exc}"
        if mode == "required":
            raise GpuRenderBrokerError(message) from exc
        print(f"[rapid-gpu-render] unavailable: {message}")
        return None
    deadline = time.monotonic() + startup_timeout
    error: str | None = None
    while time.monotonic() < deadline:
        if ready_path.is_file() and queue_path.is_dir():
            try:
                ready = json.loads(ready_path.read_text(encoding="utf-8"))
                backend = ready["backend"]
                if not isinstance(backend, dict):
                    raise ValueError("backend evidence is not an object")
            except (KeyError, json.JSONDecodeError, ValueError) as exc:
                error = f"invalid broker readiness file: {exc}"
                break
            print(
                "[rapid-gpu-render] ready: "
                f"{detail}; queue={queue_path}; backend={backend}"
            )
            return GpuRenderBrokerProcess(
                process=process,
                queue_path=queue_path,
                ready_path=ready_path,
                log_path=log_path,
                log_handle=log_handle,
                backend=backend,
            )
        if process.poll() is not None:
            error = f"broker exited with status {process.returncode}"
            break
        time.sleep(0.1)

    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
    log_handle.close()
    ready_path.unlink(missing_ok=True)
    try:
        queue_path.rmdir()
    except OSError:
        pass
    if error is None:
        error = f"broker did not become ready within {startup_timeout:.1f}s"
    try:
        log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:].strip()
    except OSError:
        log_tail = ""
    if log_tail:
        error = f"{error}; log: {log_tail}"
    if mode == "required":
        raise GpuRenderBrokerError(error)
    print(f"[rapid-gpu-render] unavailable: {error}")
    return None


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--ready-file", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return serve(
            workspace=Path(args.workspace),
            queue_path=Path(args.queue),
            ready_path=Path(args.ready_file),
        )
    except Exception as exc:
        print(f"GPU render broker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
