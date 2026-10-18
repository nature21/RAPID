"""Shared helpers for Codex exec invention launchers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(
    os.environ.get("RAPID_REPO_ROOT", Path(__file__).resolve().parents[1])
).resolve()
WORKSPACE = Path(os.environ.get("RAPID_WORKSPACE", REPO_ROOT / "workspace")).resolve()
SRC_DIR = Path(
    os.environ.get("RAPID_ACTIVE_SRC_DIR")
    or os.environ.get("RAPID_SRC_DIR", WORKSPACE / "src")
).resolve()
PRIMITIVES_DIR = SRC_DIR / "primitives"
STRATEGIES_DIR = SRC_DIR / "strategies"
INSTANTIATIONS_DIR = SRC_DIR / "instantiations"
AGENT_WORK_DIR = WORKSPACE / "agent_work"
INVENT_WORKSPACE_TEMPLATE_DIR = (
    Path(__file__).resolve().parent / "templates" / "invent_workspace"
)

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

DEFAULT_MODEL = "gpt-6-astra"
DEFAULT_MAX_RETRIES = 30
DEFAULT_RETRY_DELAY = 5.0
MAX_RETRY_DELAY = 120.0
REASONING_EFFORTS = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
}
LOG_MODE = os.environ.get("RAPID_AGENT_LOG", "compact").strip().lower()


@dataclass(frozen=True)
class CodexExecConfig:
    """Options for one Codex CLI noninteractive run."""

    model: str
    reasoning_effort: str
    max_turns: int
    local_images: Sequence[Path] = ()
    output_last_message: Path | None = None
    # Resume the same conversation after transient service-capacity failures.
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay: float = DEFAULT_RETRY_DELAY
    environment: Mapping[str, str] = field(default_factory=dict)


def stage_workspace_codex_guidance() -> list[Path]:
    """Copy the invent-agent AGENTS.md and SKILL.md bundle into WORKSPACE."""
    if not INVENT_WORKSPACE_TEMPLATE_DIR.is_dir():
        raise RuntimeError(
            f"missing Codex invent-agent templates: {INVENT_WORKSPACE_TEMPLATE_DIR}"
        )

    staged: list[Path] = []
    for src in sorted(INVENT_WORKSPACE_TEMPLATE_DIR.rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(INVENT_WORKSPACE_TEMPLATE_DIR)
        dst = WORKSPACE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.read_bytes() == src.read_bytes():
            staged.append(dst)
            continue
        shutil.copy2(src, dst)
        staged.append(dst)
    return staged


def copy_demo_sources(paths: Sequence[Path], run_dir: Path) -> list[Path]:
    """Copy original demo inputs under run_dir so the agent can resample them."""
    dest_dir = run_dir / "demo_sources"
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    used_names: set[str] = set()

    for index, path in enumerate(paths):
        suffix = path.suffix
        base = path.stem or f"demo_{index}"
        name = f"{base}{suffix}"
        if name in used_names:
            name = f"{base}_{index}{suffix}"
        used_names.add(name)
        dst = dest_dir / name
        if path.resolve() != dst.resolve():
            shutil.copy2(path, dst)
        copied.append(dst)
    return copied


def _probe_duration(video: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _extract_video_frames(video: Path, out_dir: Path, num_frames: int) -> list[Path]:
    """Sample evenly spaced keyframes from a demo video."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError(
            f"ffmpeg/ffprobe are required to decode {video.name!r} into keyframes."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    duration = _probe_duration(video)
    frames: list[Path] = []
    for index in range(num_frames):
        timestamp = duration * (index + 0.5) / num_frames
        out = out_dir / f"{video.stem}_frame{index:02d}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-y",
                str(out),
            ],
            check=True,
        )
        frames.append(out)
    return frames


def normalize_demos(
    paths: Sequence[Path],
    run_dir: Path,
    num_frames_per_video: int,
) -> list[Path]:
    """Return image-frame paths under the workspace for Codex to inspect."""
    frames_dir = run_dir / "demo_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    out: list[Path] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTS:
            dst = frames_dir / path.name
            shutil.copy2(path, dst)
            out.append(dst)
        elif suffix in VIDEO_EXTS:
            out.extend(
                _extract_video_frames(path, frames_dir / path.stem, num_frames_per_video)
            )
        else:
            raise ValueError(
                f"Unsupported demo file type {suffix!r} for {path}. "
                f"Images: {sorted(IMAGE_EXTS)}; videos: {sorted(VIDEO_EXTS)}."
            )
    return out


def _clip(text: object, limit: int = 700) -> str:
    value = str(text).strip()
    if len(value) <= limit:
        return value
    return value[:limit] + " ..."


def _is_verbose_log() -> bool:
    return LOG_MODE in {"debug", "verbose", "raw"}


def _one_line_command(command: object, limit: int = 180) -> str:
    text = " ".join(str(command).split())
    return _clip(text, limit)


def _output_stats(output: object) -> str:
    text = str(output)
    line_count = 0 if not text else text.count("\n") + (0 if text.endswith("\n") else 1)
    return f"{line_count} line(s), {len(text)} char(s)"


def _should_show_command_output(
    command: object,
    exit_code: object,
    output: object,
) -> bool:
    if not output:
        return False
    if _is_verbose_log():
        return True
    if exit_code not in {0, "0", None}:
        return True
    command_text = str(command)
    output_text = str(output)
    validation_markers = (
        "rapid-python",
        "python -m",
        "pytest",
        "SUCCESS",
        "Metrics",
        "success:",
        'success":',
    )
    return any(marker in command_text or marker in output_text for marker in validation_markers)


def _format_exec_item(item: dict[str, object]) -> str | None:
    item_type = str(item.get("type", ""))
    if item_type == "command_execution":
        command = item.get("command", "")
        return f"  [cmd] {_one_line_command(command)}"
    if item_type in {"mcp_tool_call", "dynamic_tool_call"}:
        tool = item.get("tool", "tool")
        namespace = item.get("namespace")
        prefix = f"{namespace}." if namespace else ""
        return f"  [tool] {prefix}{tool}"
    if item_type == "file_change":
        return None
    if item_type == "agent_message":
        text = item.get("text")
        if text:
            return f"assistant> {_clip(text, 1200)}"
    return None


def _format_exec_event(event: dict[str, object]) -> str | None:
    event_type = str(event.get("type", ""))
    if event_type in {"thread.started", "turn.started"}:
        suffix = ""
        if thread_id := event.get("thread_id"):
            suffix = f" {thread_id}"
        return f"[{event_type}]{suffix}"
    if event_type in {"turn.completed", "turn.failed"}:
        usage = event.get("usage")
        suffix = f" usage={usage}" if usage else ""
        return f"[{event_type}]{suffix}"
    if event_type == "error":
        return f"[error] {_clip(event, 1600)}"

    item = event.get("item")
    if isinstance(item, dict):
        if event_type == "item.started":
            return _format_exec_item(item)
        if event_type == "item.completed":
            item_type = str(item.get("type", ""))
            if item_type == "command_execution":
                status = item.get("status", "?")
                exit_code = item.get("exit_code", None)
                output = item.get("output") or item.get("aggregated_output")
                msg = f"  [cmd done] status={status} exit={exit_code}"
                if output and _should_show_command_output(
                    item.get("command", ""), exit_code, output
                ):
                    msg += f"\n  [output]\n{_clip(output, 1800)}"
                elif output:
                    msg += f" output suppressed ({_output_stats(output)})"
                return msg
            if item_type == "file_change":
                return f"  [files] {_clip(item, 800)}"
            if item_type == "agent_message":
                return _format_exec_item(item)
    return None


def _codex_exec_command(config: CodexExecConfig) -> list[str]:
    codex = shutil.which("codex")
    if codex is None:
        raise RuntimeError(
            "`codex` CLI was not found on PATH. Launch agents through "
            "`scripts/rapid-codex-run` from an environment where Codex is "
            "installed."
        )
    if os.environ.get("RAPID_CODEX_PERMISSION_PROFILE") != "1":
        raise RuntimeError(
            "RAPID Codex agents must be launched through `scripts/rapid-codex-run` "
            "so the workspace-local permission profile and run-local CODEX_HOME "
            "are active."
        )

    final_path = config.output_last_message or (AGENT_WORK_DIR / "codex_exec_final.md")
    final_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        codex,
        "exec",
        "--json",
        "--model",
        config.model,
        "--cd",
        str(WORKSPACE),
        "--skip-git-repo-check",
        "-c",
        f'model_reasoning_effort="{config.reasoning_effort}"',
        "-c",
        'model_reasoning_summary="concise"',
        "-c",
        'service_tier="fast"',
        "-c",
        "features.fast_mode=true",
        "-c",
        'approval_policy="never"',
        "-c",
        'web_search="disabled"',
        "--output-last-message",
        str(final_path),
    ]

    for image in config.local_images:
        cmd.extend(["--image", str(image.resolve())])
    cmd.append("-")
    return cmd


def _codex_exec_resume_command(
    config: CodexExecConfig,
    thread_id: str,
    final_path: Path,
) -> list[str]:
    """Build a non-interactive command that continues an existing session."""
    codex = shutil.which("codex")
    if codex is None:  # pragma: no cover - initial command normally catches this
        raise RuntimeError("`codex` CLI was not found on PATH.")

    # ``exec resume`` intentionally does not receive the original images: they
    # are already part of the persisted conversation, and re-attaching them
    # would needlessly increase every retry's request size.
    return [
        codex,
        "exec",
        "resume",
        thread_id,
        "--json",
        "--model",
        config.model,
        "--skip-git-repo-check",
        "-c",
        f'model_reasoning_effort="{config.reasoning_effort}"',
        "-c",
        'model_reasoning_summary="concise"',
        "-c",
        'service_tier="fast"',
        "-c",
        "features.fast_mode=true",
        "-c",
        'approval_policy="never"',
        "-c",
        'web_search="disabled"',
        "--output-last-message",
        str(final_path),
        "-",
    ]


def _is_retryable_capacity_event(event: object) -> bool:
    """Return whether an event indicates a transient model-capacity failure."""
    if not isinstance(event, dict):
        return False
    event_type = str(event.get("type", ""))
    if event_type not in {"error", "turn.failed"}:
        return False
    # Keep matching broad enough for provider wording variants, but do not
    # retry ordinary agent/tool failures that require a code or prompt fix.
    message = json.dumps(event, ensure_ascii=False).lower()
    return any(
        marker in message
        for marker in (
            "selected model is at capacity",
            "model is at capacity",
            "at capacity",
            "capacity exceeded",
            "temporarily overloaded",
            "service unavailable",
        )
    )


def _print_final_message(final_path: Path, returncode: int) -> None:
    print()
    print("=" * 60)
    print(f"[result] exit={returncode}")
    if final_path.is_file():
        final_text = final_path.read_text(encoding="utf-8").strip()
        if final_text:
            print(final_text)
    print("=" * 60)


def run_codex_exec(
    prompt: str,
    config: CodexExecConfig,
    *,
    resume_thread_id: str | None = None,
) -> str | None:
    """Run or resume Codex CLI and return its thread identifier when emitted."""
    full_prompt = (
        prompt.rstrip()
        + "\n\n## Run budget\n\n"
        + f"The launcher requested `--max-turns {config.max_turns}`. Treat this "
        + "as the maximum number of substantial debug/tool-use cycles before "
        + "you stop and report progress."
    )
    final_path = config.output_last_message or (AGENT_WORK_DIR / "codex_exec_final.md")
    effective_config = replace(config, output_last_message=final_path)
    thread_id = resume_thread_id
    retry_count = 0
    retry_delay = max(0.0, config.retry_delay)
    retry_delay_cap = max(MAX_RETRY_DELAY, retry_delay)

    print(f"[system] codex exec cwd: {WORKSPACE}")
    print(f"[system] codex exec log mode: {LOG_MODE or 'compact'}")
    while True:
        is_retry = retry_count > 0
        is_resume = thread_id is not None and (
            resume_thread_id is not None or is_retry
        )
        cmd = (
            _codex_exec_resume_command(effective_config, thread_id, final_path)
            if is_resume
            else _codex_exec_command(effective_config)
        )
        if is_resume and is_retry:
            attempt_prompt = (
                "The previous attempt was interrupted by a transient model-capacity "
                "error. Continue the invention task from the existing session and "
                "workspace state; inspect current files and finish the remaining work."
            )
            print(
                f"[retry] resuming Codex session (attempt {retry_count}/"
                f"{config.max_retries}) after {retry_delay:g}s delay",
                flush=True,
            )
        else:
            attempt_prompt = full_prompt
            if is_retry:
                print(
                    "[retry] no session ID was emitted; starting the Codex request "
                    f"again (attempt {retry_count}/{config.max_retries}) after "
                    f"{retry_delay:g}s delay",
                    flush=True,
                )

        if is_retry:
            time.sleep(retry_delay)

        child_environment = os.environ.copy()
        child_environment.update(config.environment)
        process = subprocess.Popen(
            cmd,
            cwd=WORKSPACE,
            env=child_environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        assert process.stdin is not None
        process.stdin.write(attempt_prompt)
        process.stdin.close()

        retryable_failure = False
        assert process.stdout is not None
        for line in process.stdout:
            text = line.strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                print(text)
                continue
            if isinstance(event, dict) and event.get("thread_id"):
                thread_id = str(event["thread_id"])
            retryable_failure = (
                _is_retryable_capacity_event(event) or retryable_failure
            )
            if _is_verbose_log():
                print(text)
                continue
            rendered = _format_exec_event(event)
            if rendered:
                print(rendered)

        returncode = process.wait()
        if returncode == 0:
            _print_final_message(final_path, returncode)
            return thread_id
        if not retryable_failure or retry_count >= config.max_retries:
            _print_final_message(final_path, returncode)
            raise SystemExit(returncode)

        if is_retry:
            retry_delay = min(retry_delay * 2, retry_delay_cap)
        retry_count += 1
