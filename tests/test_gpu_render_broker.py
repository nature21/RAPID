from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from scene_io.record import _gpu_render_queue, record_video
from suites.staged.codex_launcher import _parse_args
from suites.staged.gpu_render_broker import (
    GpuRenderBrokerError,
    _render_request,
    _validate_render_output_path,
    _workspace_path,
    parse_gpu_render_mode,
    start_gpu_render_broker,
)


class GpuRenderBrokerTests(unittest.TestCase):
    def test_recorder_discovers_live_workspace_broker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            queue = workspace / "agent_work/gpu_render_requests/broker_dev"
            queue.mkdir(parents=True)
            ready = workspace / "agent_work/gpu_render_ready.json"
            ready.write_text(json.dumps({"queue": str(queue)}), encoding="utf-8")

            with patch.dict(
                os.environ,
                {"RAPID_WORKSPACE": str(workspace)},
                clear=True,
            ):
                self.assertEqual(_gpu_render_queue(), queue)
            with patch.dict(
                os.environ,
                {
                    "RAPID_WORKSPACE": str(workspace),
                    "RAPID_GPU_RENDER": "off",
                },
                clear=True,
            ):
                self.assertIsNone(_gpu_render_queue())

    def test_gpu_render_modes_are_normalized_and_validated(self) -> None:
        self.assertEqual(parse_gpu_render_mode(None), "auto")
        self.assertEqual(parse_gpu_render_mode(" REQUIRED "), "required")
        self.assertEqual(parse_gpu_render_mode("off"), "off")
        with self.assertRaisesRegex(ValueError, "auto, required, off"):
            parse_gpu_render_mode("sometimes")

    def test_launcher_accepts_generic_gpu_render_policy(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            automatic = _parse_args(
                ["--workspace", "/tmp/workspace", "--", "agent/example.py"]
            )
        required = _parse_args(
            [
                "--workspace",
                "/tmp/workspace",
                "--gpu-render",
                "required",
                "--",
                "agent/example.py",
            ]
        )
        with patch.dict(os.environ, {"RAPID_GPU_RENDER": "off"}, clear=False):
            disabled = _parse_args(
                ["--workspace", "/tmp/workspace", "--", "agent/example.py"]
            )

        self.assertEqual(automatic.gpu_render, "auto")
        self.assertEqual(required.gpu_render, "required")
        self.assertEqual(disabled.gpu_render, "off")

    def test_workspace_path_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            inside = workspace / "inside.npy"
            inside.write_bytes(b"data")

            self.assertEqual(
                _workspace_path(
                    inside,
                    workspace=workspace,
                    description="input",
                    must_exist=True,
                ),
                inside,
            )
            with self.assertRaisesRegex(GpuRenderBrokerError, "escapes"):
                _workspace_path(
                    workspace.parent / "outside.npy",
                    workspace=workspace,
                    description="input",
                    must_exist=False,
                )

    def test_render_output_is_limited_to_artifact_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            request_dir = workspace / "agent_work/gpu_render_requests/request_test"
            request_dir.mkdir(parents=True)
            model_path = request_dir / "model.mjb"
            states_path = request_dir / "states.npy"
            model_path.write_bytes(b"placeholder")
            states_path.write_bytes(b"placeholder")
            request = {
                "version": 1,
                "model_path": str(model_path),
                "states_path": str(states_path),
                "output_path": str(workspace / "src/forbidden.mp4"),
            }

            with self.assertRaisesRegex(
                GpuRenderBrokerError,
                "workspace/results, workspace/results_instantiation, "
                "workspace/agent_work",
            ):
                _render_request(
                    request,
                    workspace=workspace,
                    request_dir=request_dir,
                )

    def test_render_output_accepts_deployment_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            output = workspace / "results_instantiation/strategies/s/i/trajectory.mp4"

            _validate_render_output_path(output, workspace=workspace)

    def test_off_mode_does_not_probe_or_start_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            broker = start_gpu_render_broker(
                workspace=Path(tmp),
                env={},
                mode="off",
            )
        self.assertIsNone(broker)

    def test_unavailable_driver_respects_auto_and_required_modes(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "suites.staged.gpu_render_broker._nvidia_driver_visible",
                return_value=(False, "no allocated GPU"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            workspace = Path(tmp)
            self.assertIsNone(
                start_gpu_render_broker(
                    workspace=workspace,
                    env={},
                    mode="auto",
                )
            )
            with self.assertRaisesRegex(GpuRenderBrokerError, "no allocated GPU"):
                start_gpu_render_broker(
                    workspace=workspace,
                    env={},
                    mode="required",
                )

    def test_record_video_delegates_to_launcher_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            (workspace / "agent_work").mkdir()
            (workspace / "results").mkdir()
            queue_path = workspace / "agent_work/gpu_render_requests"
            queue_path.mkdir()
            output_path = workspace / "results/delegated.mp4"
            received: list[dict[str, object]] = []
            server_errors: list[BaseException] = []

            def serve_once() -> None:
                try:
                    deadline = time.monotonic() + 5.0
                    request_path: Path | None = None
                    while request_path is None and time.monotonic() < deadline:
                        matches = list(queue_path.glob("request_*/request.json"))
                        if matches:
                            request_path = matches[0]
                        else:
                            time.sleep(0.01)
                    if request_path is None:
                        raise TimeoutError("request did not arrive")
                    request = json.loads(request_path.read_text(encoding="utf-8"))
                    received.append(request)
                    self.assertTrue(Path(request["model_path"]).is_file())
                    self.assertTrue(Path(request["states_path"]).is_file())
                    Path(request["output_path"]).write_bytes(b"fake-mp4")
                    (request_path.parent / "reply.json").write_text(
                        json.dumps(
                            {
                                "ok": True,
                                "frames": 3,
                                "duration": 0.1,
                                "elapsed": 0.01,
                                "render_fps": 300.0,
                            }
                        ),
                        encoding="utf-8",
                    )
                except BaseException as exc:  # pragma: no cover - surfaced below
                    server_errors.append(exc)

            thread = threading.Thread(target=serve_once, daemon=True)
            thread.start()
            model = mujoco.MjModel.from_xml_string("<mujoco/>")
            data = mujoco.MjData(model)
            states = np.zeros((3, model.nq + model.nv))
            broker_env = {
                "RAPID_WORKSPACE": str(workspace),
                "RAPID_GPU_RENDER_QUEUE": str(queue_path),
            }
            with (
                patch.dict(os.environ, broker_env, clear=False),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                record_video(
                    model,
                    data,
                    states,
                    1.0 / 30.0,
                    str(output_path),
                )
            thread.join(timeout=5.0)

            if server_errors:
                raise server_errors[0]
            self.assertFalse(thread.is_alive())
            self.assertEqual(output_path.read_bytes(), b"fake-mp4")
            self.assertEqual(received[0]["width"], 960)
            self.assertEqual(received[0]["height"], 540)
            self.assertEqual(
                list((workspace / "agent_work/gpu_render_requests").glob("*")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
