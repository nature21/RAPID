from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from devtools.agent_session_viewer import (
    default_output_path,
    find_session_files,
    load_session,
    render_html,
    render_terminal,
    select_session,
)


class AgentSessionViewerTests(unittest.TestCase):
    def _write_session(self, path: Path) -> None:
        records = [
            {
                "timestamp": "2026-08-25T12:00:00.000Z",
                "type": "session_meta",
                "payload": {
                    "session_id": "session-test",
                    "cwd": "/workspace",
                    "cli_version": "test",
                },
            },
            {
                "timestamp": "2026-08-25T12:00:01.000Z",
                "type": "event_msg",
                "payload": {"type": "task_started", "turn_id": "turn-1", "model_context_window": 1000},
            },
            {
                "timestamp": "2026-08-25T12:00:02.000Z",
                "type": "turn_context",
                "payload": {"turn_id": "turn-1", "model": "test-model", "effort": "high"},
            },
            {
                "timestamp": "2026-08-25T12:00:02.500Z",
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "Inspect <unsafe> & report"},
            },
            {
                "timestamp": "2026-08-25T12:00:03.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Inspect <unsafe> & report"}],
                },
            },
            {
                "timestamp": "2026-08-25T12:00:04.000Z",
                "type": "event_msg",
                "payload": {"type": "agent_reasoning", "text": "Checking the files"},
            },
            {
                "timestamp": "2026-08-25T12:00:04.100Z",
                "type": "response_item",
                "payload": {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Checking the files\nMore detail"}],
                    "encrypted_content": "secret-state",
                },
            },
            {
                "timestamp": "2026-08-25T12:00:05.000Z",
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "name": "exec",
                    "call_id": "call-1",
                    "input": "ls -la",
                    "status": "completed",
                },
            },
            {
                "timestamp": "2026-08-25T12:00:05.100Z",
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "call_id": "call-1",
                    "output": [{"type": "input_text", "text": "file.txt"}],
                },
            },
            {
                "timestamp": "2026-08-25T12:00:06.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done"}],
                },
            },
            {
                "timestamp": "2026-08-25T12:00:07.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "task_complete",
                    "turn_id": "turn-1",
                    "duration_ms": 6000,
                    "time_to_first_token_ms": 300,
                },
            },
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    def test_resolves_short_run_and_selects_latest_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            older = root / "suite_runs/run_1/codex_home/sessions/2026/08/24/rollout-old.jsonl"
            newer = root / "suite_runs/run_1/codex_home/sessions/2026/08/25/rollout-new.jsonl"
            self._write_session(older)
            self._write_session(newer)
            older.touch()
            newer.touch()

            files = find_session_files("run_1", root=root)

            self.assertEqual({path.name for path in files}, {"rollout-old.jsonl", "rollout-new.jsonl"})
            self.assertEqual(select_session(files, "new"), newer.resolve())

    def test_renders_messages_reasoning_and_paired_tools_safely(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rollout-test.jsonl"
            self._write_session(path)

            rendered = render_html(load_session(path))

            self.assertIn("session-test", rendered)
            self.assertIn("test-model", rendered)
            self.assertIn("Inspect &lt;unsafe&gt; &amp; report", rendered)
            self.assertIn("Checking the files", rendered)
            self.assertIn("Encrypted continuation state is present", rendered)
            self.assertIn("ls -la", rendered)
            self.assertIn("file.txt", rendered)
            self.assertEqual(rendered.count("Unpaired tool output"), 0)
            self.assertIn('data-mirror="1"', rendered)
            self.assertNotIn("secret-state", rendered)

    def test_tolerates_a_partial_final_jsonl_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rollout-test.jsonl"
            self._write_session(path)
            with path.open("a", encoding="utf-8") as stream:
                stream.write('{"timestamp":"unfinished"')

            loaded = load_session(path)
            rendered = render_html(loaded)

            self.assertEqual(len(loaded.warnings), 1)
            self.assertIn("possibly still being written", rendered)

    def test_terminal_render_is_compact_and_hides_stream_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rollout-test.jsonl"
            self._write_session(path)

            rendered = render_terminal(load_session(path), color=False)

            self.assertIn("CODEX SESSION", rendered)
            self.assertIn("Inspect <unsafe> & report", rendered)
            self.assertEqual(rendered.count("Inspect <unsafe> & report"), 1)
            self.assertEqual(rendered.count("Checking the files"), 1)
            self.assertIn("TOOL · exec", rendered)
            self.assertIn("file.txt", rendered)
            self.assertNotIn("TURN CONTEXT", rendered)
            self.assertNotIn("secret-state", rendered)
            self.assertNotIn("\033[", rendered)

    def test_terminal_can_show_context_raw_json_and_color(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rollout-test.jsonl"
            self._write_session(path)

            rendered = render_terminal(
                load_session(path),
                color=True,
                show_system=True,
                show_context=True,
                show_stream_mirrors=True,
                show_raw=True,
                max_chars=20,
            )

            self.assertIn("TURN CONTEXT", rendered)
            self.assertIn("RAW", rendered)
            self.assertIn("characters omitted", rendered)
            self.assertIn("\033[", rendered)

    def test_terminal_follow_offset_keeps_late_tool_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rollout-test.jsonl"
            self._write_session(path)

            rendered = render_terminal(
                load_session(path),
                color=False,
                start_index=8,
                include_header=False,
            )

            self.assertIn("UPDATE", rendered)
            self.assertIn("TOOL OUTPUT", rendered)
            self.assertIn("file.txt", rendered)

    def test_default_report_stays_in_the_run_codex_home(self) -> None:
        session = Path("/runs/example/codex_home/sessions/2026/08/25/rollout-test.jsonl")

        self.assertEqual(
            default_output_path(session),
            Path("/runs/example/codex_home/session_reports/rollout-test.html"),
        )


if __name__ == "__main__":
    unittest.main()
