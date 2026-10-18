from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from agent import codex_exec


def _config_overrides(command: list[str]) -> list[str]:
    return [command[index + 1] for index, arg in enumerate(command) if arg == "-c"]


class _InputBuffer:
    def __init__(self) -> None:
        self.text = ""

    def write(self, value: str) -> None:
        self.text += value

    def close(self) -> None:
        pass


class _FakeProcess:
    def __init__(self, events: list[dict[str, object]], returncode: int) -> None:
        self.stdin = _InputBuffer()
        self.stdout = io.StringIO(
            "".join(json.dumps(event) + "\n" for event in events)
        )
        self.returncode = returncode

    def wait(self) -> int:
        return self.returncode


class CodexExecRetryTest(unittest.TestCase):
    def test_initial_and_resume_commands_enable_fast_mode(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = codex_exec.CodexExecConfig(
                model="test-model",
                reasoning_effort="high",
                max_turns=10,
                output_last_message=Path(temp_dir) / "final.md",
            )
            with (
                mock.patch.object(
                    codex_exec.shutil, "which", return_value="/usr/bin/codex"
                ),
                mock.patch.dict(
                    codex_exec.os.environ,
                    {"RAPID_CODEX_PERMISSION_PROFILE": "1"},
                ),
            ):
                initial = codex_exec._codex_exec_command(config)
                resumed = codex_exec._codex_exec_resume_command(
                    config,
                    "thread-123",
                    config.output_last_message,
                )

        for command in (initial, resumed):
            overrides = _config_overrides(command)
            self.assertIn('service_tier="fast"', overrides)
            self.assertIn("features.fast_mode=true", overrides)

    def test_capacity_failure_resumes_same_thread(self) -> None:
        failed = _FakeProcess(
            [
                {"type": "thread.started", "thread_id": "thread-123"},
                {
                    "type": "error",
                    "message": "Selected model is at capacity. Please try a different model.",
                },
                {"type": "turn.failed"},
            ],
            returncode=1,
        )
        succeeded = _FakeProcess(
            [
                {"type": "thread.started", "thread_id": "thread-123"},
                {"type": "turn.completed"},
            ],
            returncode=0,
        )

        with TemporaryDirectory() as temp_dir:
            config = codex_exec.CodexExecConfig(
                model="test-model",
                reasoning_effort="high",
                max_turns=10,
                output_last_message=Path(temp_dir) / "final.md",
                max_retries=5,
                retry_delay=5.0,
            )
            with (
                mock.patch.object(
                    codex_exec, "_codex_exec_command", return_value=["initial"]
                ) as initial_command,
                mock.patch.object(
                    codex_exec,
                    "_codex_exec_resume_command",
                    return_value=["resume"],
                ) as resume_command,
                mock.patch.object(
                    codex_exec.subprocess,
                    "Popen",
                    side_effect=[failed, succeeded],
                ) as popen,
                mock.patch.object(codex_exec.time, "sleep") as sleep,
            ):
                codex_exec.run_codex_exec("invent it", config)

        initial_command.assert_called_once()
        self.assertEqual(resume_command.call_args.args[1], "thread-123")
        self.assertEqual(popen.call_count, 2)
        sleep.assert_called_once_with(5.0)
        self.assertIn("invent it", failed.stdin.text)
        self.assertIn("Continue the invention task", succeeded.stdin.text)

    def test_non_capacity_failure_is_not_retried(self) -> None:
        failed = _FakeProcess(
            [
                {"type": "thread.started", "thread_id": "thread-456"},
                {"type": "error", "message": "invalid configuration"},
                {"type": "turn.failed"},
            ],
            returncode=2,
        )
        config = codex_exec.CodexExecConfig(
            model="test-model",
            reasoning_effort="high",
            max_turns=10,
            max_retries=5,
            retry_delay=0,
        )

        with (
            mock.patch.object(
                codex_exec, "_codex_exec_command", return_value=["initial"]
            ),
            mock.patch.object(codex_exec.subprocess, "Popen", return_value=failed),
            mock.patch.object(codex_exec.time, "sleep") as sleep,
            self.assertRaisesRegex(SystemExit, "2"),
        ):
            codex_exec.run_codex_exec("invent it", config)

        sleep.assert_not_called()

    def test_defaults_allow_recovery_after_prolonged_capacity_failure(self) -> None:
        failures = [
            _FakeProcess(
                [
                    {"type": "thread.started", "thread_id": "thread-123"},
                    {"type": "error", "message": "Selected model is at capacity."},
                ],
                returncode=1,
            )
            for _ in range(7)
        ]
        succeeded = _FakeProcess([{"type": "turn.completed"}], returncode=0)
        with TemporaryDirectory() as temp_dir:
            config = codex_exec.CodexExecConfig(
                model="test-model",
                reasoning_effort="high",
                max_turns=10,
                output_last_message=Path(temp_dir) / "final.md",
            )
            with (
                mock.patch.object(
                    codex_exec, "_codex_exec_command", return_value=["initial"]
                ),
                mock.patch.object(
                    codex_exec, "_codex_exec_resume_command", return_value=["resume"]
                ) as resume,
                mock.patch.object(
                    codex_exec.subprocess, "Popen", side_effect=[*failures, succeeded]
                ),
                mock.patch.object(codex_exec.time, "sleep") as sleep,
            ):
                thread_id = codex_exec.run_codex_exec("invent it", config)

        self.assertEqual(thread_id, "thread-123")
        self.assertEqual(
            sleep.call_args_list,
            [mock.call(delay) for delay in (5, 10, 20, 40, 80, 120, 120)],
        )
        self.assertEqual(resume.call_count, 7)
        for call in resume.call_args_list:
            self.assertEqual(call.args[1], "thread-123")

    def test_capacity_retries_stop_at_configured_limit(self) -> None:
        failures = [
            _FakeProcess(
                [{"type": "error", "message": "Selected model is at capacity."}],
                returncode=1,
            )
            for _ in range(3)
        ]
        with TemporaryDirectory() as temp_dir:
            config = codex_exec.CodexExecConfig(
                model="test-model",
                reasoning_effort="high",
                max_turns=10,
                output_last_message=Path(temp_dir) / "final.md",
                max_retries=2,
                retry_delay=17,
            )
            with (
                mock.patch.object(
                    codex_exec, "_codex_exec_command", return_value=["initial"]
                ) as initial,
                mock.patch.object(codex_exec, "_codex_exec_resume_command") as resume,
                mock.patch.object(
                    codex_exec.subprocess, "Popen", side_effect=failures
                ) as popen,
                mock.patch.object(codex_exec.time, "sleep") as sleep,
                self.assertRaisesRegex(SystemExit, "1"),
            ):
                codex_exec.run_codex_exec("invent it", config)

        self.assertEqual(popen.call_count, 3)
        self.assertEqual(initial.call_count, 3)
        resume.assert_not_called()
        self.assertEqual(sleep.call_args_list, [mock.call(17), mock.call(34)])

    def test_explicit_resume_uses_new_phase_prompt_and_environment(self) -> None:
        succeeded = _FakeProcess(
            [
                {"type": "thread.started", "thread_id": "thread-789"},
                {"type": "turn.completed"},
            ],
            returncode=0,
        )
        config = codex_exec.CodexExecConfig(
            model="test-model",
            reasoning_effort="high",
            max_turns=10,
            environment={"RAPID_INVENTION_ATTEMPT_LOG": "/tmp/attempts.jsonl"},
        )

        with (
            mock.patch.object(
                codex_exec,
                "_codex_exec_command",
                return_value=["initial"],
            ) as initial,
            mock.patch.object(
                codex_exec,
                "_codex_exec_resume_command",
                return_value=["resume"],
            ) as resume,
            mock.patch.object(
                codex_exec.subprocess,
                "Popen",
                return_value=succeeded,
            ) as popen,
        ):
            thread_id = codex_exec.run_codex_exec(
                "run held-out evaluation",
                config,
                resume_thread_id="thread-789",
            )

        self.assertEqual(thread_id, "thread-789")
        initial.assert_not_called()
        resume.assert_called_once()
        self.assertIn("run held-out evaluation", succeeded.stdin.text)
        self.assertNotIn("transient model-capacity", succeeded.stdin.text)
        self.assertEqual(
            popen.call_args.kwargs["env"]["RAPID_INVENTION_ATTEMPT_LOG"],
            "/tmp/attempts.jsonl",
        )


if __name__ == "__main__":
    unittest.main()
