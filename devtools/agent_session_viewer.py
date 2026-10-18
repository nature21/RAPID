#!/usr/bin/env python3
"""Render a persisted Codex session JSONL file for terminal or HTML inspection.

This module deliberately uses only the Python standard library.  Codex session
files are operational artifacts rather than an RAPID data contract, so parsing
is defensive and unknown record types remain available through ``--raw`` and
HTML raw-record drawers instead of causing the report to fail.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import webbrowser
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# Keep direct-file invocation available alongside rapid-session-view / -m.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from suites.staged.experiment_paths import (
    add_experiment_root_argument,
    configure_experiment_root,
    experiment_root,
)


SESSION_GLOB = "rollout-*.jsonl"
_SAFE_IMAGE_DATA_URI = re.compile(
    r"^data:image/(?:png|jpe?g|gif|webp);base64,[A-Za-z0-9+/=\r\n]+$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LoadedSession:
    path: Path
    records: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class TerminalEntry:
    index: int
    timestamp: Any
    category: str
    title: str
    body: str
    subtitle: str = ""
    mirror: bool = False
    raw_records: tuple[dict[str, Any], ...] = ()


class TerminalStyle:
    _CODES = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "user": "\033[38;5;75m",
        "assistant": "\033[38;5;78m",
        "reasoning": "\033[38;5;177m",
        "tool": "\033[38;5;220m",
        "system": "\033[38;5;245m",
        "event": "\033[38;5;203m",
        "context": "\033[38;5;141m",
    }

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def apply(self, text: str, *styles: str) -> str:
        if not self.enabled or not text:
            return text
        prefix = "".join(self._CODES[style] for style in styles)
        return f"{prefix}{text}{self._CODES['reset']}"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _source_candidates(source: str | os.PathLike[str], root: Path) -> list[Path]:
    supplied = Path(source).expanduser()
    candidates = [supplied if supplied.is_absolute() else Path.cwd() / supplied]
    if not supplied.is_absolute():
        candidates.append(root / supplied)
        if len(supplied.parts) == 1:
            candidates.append(experiment_root(root) / supplied)
    return candidates


def find_session_files(
    source: str | os.PathLike[str] | None,
    *,
    root: Path | None = None,
) -> list[Path]:
    """Resolve a JSONL file, run/workspace directory, or short suite-run name."""

    root = (root or repository_root()).resolve()
    if source is None:
        targets = [experiment_root(root)]
    else:
        candidates = _source_candidates(source, root)
        targets = [candidate for candidate in candidates if candidate.exists()]
        if not targets:
            checked = "\n  ".join(str(path) for path in candidates)
            raise FileNotFoundError(f"session source does not exist; checked:\n  {checked}")
        targets = [targets[0]]

    found: set[Path] = set()
    for target in targets:
        target = target.resolve()
        if target.is_file():
            found.add(target)
            continue

        search_roots = [target]
        if target.name == "workspace" and (target.parent / "codex_home").is_dir():
            search_roots.insert(0, target.parent / "codex_home")
        for search_root in search_roots:
            found.update(path.resolve() for path in search_root.rglob(SESSION_GLOB))

    return sorted(found, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def select_session(files: Sequence[Path], selector: str | None = None) -> Path:
    if not files:
        raise FileNotFoundError("no rollout-*.jsonl Codex sessions found")
    if selector is None:
        return files[-1]

    matches = [path for path in files if selector in path.name or selector in str(path)]
    if not matches:
        raise FileNotFoundError(f"no session matched {selector!r}")
    if len(matches) > 1:
        choices = "\n  ".join(str(path) for path in matches)
        raise ValueError(f"session selector {selector!r} is ambiguous:\n  {choices}")
    return matches[0]


def load_session(path: Path) -> LoadedSession:
    """Load JSONL defensively, including an actively-written partial last line."""

    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"line {line_number}: {exc.msg} (possibly still being written)")
                continue
            if not isinstance(value, dict):
                warnings.append(f"line {line_number}: expected a JSON object")
                continue
            records.append(value)
    return LoadedSession(path=path.resolve(), records=tuple(records), warnings=tuple(warnings))


def _payload(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("payload")
    return value if isinstance(value, dict) else {}


def _iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _display_timestamp(value: Any) -> str:
    parsed = _iso_datetime(value)
    if parsed is None:
        return str(value or "")
    return parsed.astimezone().strftime("%H:%M:%S.%f")[:-3]


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    minutes, remainder = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h {minutes}m {remainder:.0f}s"
    if minutes:
        return f"{minutes}m {remainder:.0f}s"
    return f"{remainder:.1f}s"


def _format_number(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) else str(value or "—")


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, dict):
                for key in ("text", "output", "transcript", "refusal"):
                    if isinstance(item.get(key), str):
                        chunks.append(item[key])
                        break
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunks)
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _summary_text(payload: dict[str, Any]) -> str:
    return _content_text(payload.get("summary"))


def _message_text(payload: dict[str, Any]) -> str:
    return _content_text(payload.get("content"))


def _sanitized_raw(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        return {item_key: _sanitized_raw(item, item_key) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_sanitized_raw(item, key) for item in value]
    if isinstance(value, str):
        if key == "encrypted_content":
            return f"<encrypted reasoning state: {len(value):,} characters>"
        if value.startswith("data:image/") and len(value) > 256:
            media_type = value.partition(";")[0].removeprefix("data:")
            return f"<{media_type} data URI: {len(value):,} characters; rendered above>"
    return value


def _raw_details(record: dict[str, Any], *, label: str = "Raw record") -> str:
    raw = json.dumps(_sanitized_raw(record), ensure_ascii=False, indent=2, default=str)
    return (
        '<details class="raw-details"><summary>'
        + html.escape(label)
        + "</summary><pre>"
        + html.escape(raw)
        + "</pre></details>"
    )


def _render_text(text: str, *, css_class: str = "prose") -> str:
    if not text:
        return '<div class="empty">No readable text in this record.</div>'
    return f'<div class="{css_class}">{html.escape(text)}</div>'


def _render_content_parts(content: Any, *, include_images: bool) -> str:
    if not isinstance(content, list):
        return _render_text(_content_text(content))
    rendered: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            rendered.append(_render_text(str(part)))
            continue
        part_type = str(part.get("type") or "content")
        text = next(
            (
                part[key]
                for key in ("text", "output", "transcript", "refusal")
                if isinstance(part.get(key), str)
            ),
            None,
        )
        if text is not None:
            rendered.append(_render_text(text))
            continue
        image_url = part.get("image_url")
        if isinstance(image_url, str):
            if include_images and _SAFE_IMAGE_DATA_URI.fullmatch(image_url):
                detail = html.escape(str(part.get("detail") or "auto"))
                rendered.append(
                    '<figure class="session-image">'
                    f'<img loading="lazy" src="{html.escape(image_url, quote=True)}" '
                    f'alt="Session input image ({detail})">'
                    f"<figcaption>Input image · {detail}</figcaption></figure>"
                )
            else:
                size = f"{len(image_url):,} characters" if image_url.startswith("data:") else image_url
                rendered.append(
                    '<div class="attachment">Image omitted · '
                    + html.escape(size)
                    + "</div>"
                )
            continue
        rendered.append(
            '<details class="content-details"><summary>'
            + html.escape(part_type)
            + "</summary><pre>"
            + html.escape(json.dumps(part, ensure_ascii=False, indent=2, default=str))
            + "</pre></details>"
        )
    return "".join(rendered) or '<div class="empty">Empty content list.</div>'


def _render_kv(items: Iterable[tuple[str, Any]]) -> str:
    rows = []
    for key, value in items:
        if value is None or value == "":
            continue
        rows.append(
            f"<dt>{html.escape(key)}</dt><dd>{html.escape(str(value))}</dd>"
        )
    return '<dl class="metadata">' + "".join(rows) + "</dl>"


def _mirror_signatures(records: Sequence[dict[str, Any]]) -> tuple[set[str], set[str], set[str]]:
    assistant: set[str] = set()
    user: set[str] = set()
    reasoning: set[str] = set()
    for record in records:
        if record.get("type") != "response_item":
            continue
        payload = _payload(record)
        if payload.get("type") == "message":
            text = _message_text(payload).strip()
            if payload.get("role") == "assistant" and text:
                assistant.add(text)
            elif payload.get("role") == "user" and text:
                user.add(text)
        elif payload.get("type") == "reasoning":
            text = _summary_text(payload).strip()
            if text:
                reasoning.add(text)
    return assistant, user, reasoning


def _tool_maps(
    records: Sequence[dict[str, Any]],
) -> tuple[dict[str, tuple[int, dict[str, Any]]], dict[str, list[tuple[int, dict[str, Any]]]]]:
    outputs: dict[str, tuple[int, dict[str, Any]]] = {}
    events: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, record in enumerate(records):
        payload = _payload(record)
        payload_type = payload.get("type")
        call_id = payload.get("call_id")
        if not isinstance(call_id, str):
            continue
        if payload_type in {"custom_tool_call_output", "function_call_output"}:
            outputs[call_id] = (index, record)
        elif payload_type in {"patch_apply_end", "mcp_tool_call_end"}:
            events.setdefault(call_id, []).append((index, record))
    return outputs, events


def _entry(
    *,
    index: int,
    timestamp: Any,
    category: str,
    title: str,
    body: str,
    raw: str,
    subtitle: str = "",
    badges: Sequence[str] = (),
    mirror: bool = False,
) -> str:
    badge_html = "".join(f"<span>{html.escape(str(badge))}</span>" for badge in badges if badge)
    searchable = " ".join((category, title, subtitle, *badges))
    return (
        f'<article class="entry category-{html.escape(category)}" data-category="{html.escape(category)}" '
        f'data-mirror="{int(mirror)}" data-search="{html.escape(searchable.lower(), quote=True)}">'
        '<div class="rail-dot"></div><header class="entry-header">'
        f'<time title="{html.escape(str(timestamp or ""), quote=True)}">{html.escape(_display_timestamp(timestamp))}</time>'
        f'<div class="entry-heading"><h2>{html.escape(title)}</h2>'
        + (f"<p>{html.escape(subtitle)}</p>" if subtitle else "")
        + f'</div><div class="badges">{badge_html}<span>#{index + 1}</span></div></header>'
        f'<div class="entry-body">{body}{raw}</div></article>'
    )


def _render_tool(
    index: int,
    record: dict[str, Any],
    output_record: dict[str, Any] | None,
    event_records: Sequence[dict[str, Any]],
) -> str:
    payload = _payload(record)
    name = str(payload.get("name") or "tool")
    call_id = str(payload.get("call_id") or "")
    raw_input = payload.get("input") if "input" in payload else payload.get("arguments")
    tool_input = _content_text(raw_input)
    blocks = [
        '<details class="content-details tool-block" open><summary>Input</summary><pre>'
        + html.escape(tool_input or "(empty input)")
        + "</pre></details>"
    ]
    raw_records = [record]
    status = str(payload.get("status") or "")
    if output_record is not None:
        output_payload = _payload(output_record)
        output = _content_text(output_payload.get("output"))
        blocks.append(
            '<details class="content-details tool-block" open><summary>Output</summary><pre>'
            + html.escape(output or "(empty output)")
            + "</pre></details>"
        )
        raw_records.append(output_record)
    else:
        blocks.append('<div class="attachment pending">Output not recorded yet.</div>')
    for event_record in event_records:
        event_payload = _payload(event_record)
        if not status and event_payload.get("status"):
            status = str(event_payload["status"])
        raw_records.append(event_record)
    raw = _raw_details({"paired_records": raw_records}, label="Raw paired records")
    return _entry(
        index=index,
        timestamp=record.get("timestamp"),
        category="tool",
        title=name,
        subtitle=f"Tool call {call_id}" if call_id else "Tool call",
        badges=(status,),
        body="".join(blocks),
        raw=raw,
    )


def _render_record(
    index: int,
    record: dict[str, Any],
    *,
    include_images: bool,
    assistant_messages: set[str],
    user_messages: set[str],
    reasoning_summaries: set[str],
) -> str:
    record_type = str(record.get("type") or "unknown")
    payload = _payload(record)
    payload_type = str(payload.get("type") or "")
    timestamp = record.get("timestamp")
    raw = _raw_details(record)

    if record_type == "session_meta":
        git = payload.get("git") if isinstance(payload.get("git"), dict) else {}
        body = _render_kv(
            (
                ("Session ID", payload.get("session_id") or payload.get("id")),
                ("Working directory", payload.get("cwd")),
                ("CLI version", payload.get("cli_version")),
                ("Origin", payload.get("originator")),
                ("Source", payload.get("source")),
                ("Model provider", payload.get("model_provider")),
                ("Git branch", git.get("branch")),
                ("Git commit", git.get("commit_hash")),
            )
        )
        return _entry(index=index, timestamp=timestamp, category="context", title="Session metadata", body=body, raw=raw)

    if record_type == "turn_context":
        body = _render_kv(
            (
                ("Turn ID", payload.get("turn_id")),
                ("Model", payload.get("model")),
                ("Reasoning effort", payload.get("effort")),
                ("Working directory", payload.get("cwd")),
                ("Date / timezone", f'{payload.get("current_date", "")} {payload.get("timezone", "")}'.strip()),
                ("Approval policy", payload.get("approval_policy")),
            )
        )
        if payload.get("summary"):
            body += '<details class="content-details"><summary>Compacted context summary</summary>' + _render_text(str(payload["summary"])) + "</details>"
        return _entry(index=index, timestamp=timestamp, category="context", title="Turn context", body=body, raw=raw)

    if record_type == "world_state":
        state = payload.get("state")
        body = _render_text(_content_text(state), css_class="prose compact")
        return _entry(index=index, timestamp=timestamp, category="context", title="World state", body=body, raw=raw)

    if record_type == "compacted":
        body = _render_kv(
            (
                ("Window", payload.get("window_number")),
                ("Window ID", payload.get("window_id")),
                ("Previous window", payload.get("previous_window_id")),
            )
        )
        if payload.get("message"):
            body += _render_text(str(payload["message"]))
        return _entry(index=index, timestamp=timestamp, category="context", title="Context compacted", body=body, raw=raw)

    if record_type == "response_item" and payload_type == "message":
        role = str(payload.get("role") or "message")
        category = role if role in {"user", "assistant"} else "system"
        phase = str(payload.get("phase") or "")
        return _entry(
            index=index,
            timestamp=timestamp,
            category=category,
            title=role.title(),
            subtitle=f"{phase} message" if phase else "Message",
            badges=(phase,),
            body=_render_content_parts(payload.get("content"), include_images=include_images),
            raw=raw,
        )

    if record_type == "response_item" and payload_type == "reasoning":
        summary = _summary_text(payload)
        encrypted = payload.get("encrypted_content")
        body = _render_text(summary)
        if encrypted:
            body += (
                '<div class="attachment encrypted">Encrypted continuation state is present '
                f'({len(str(encrypted)):,} characters). It is not human-readable reasoning.</div>'
            )
        return _entry(
            index=index,
            timestamp=timestamp,
            category="reasoning",
            title="Reasoning summary",
            badges=("encrypted state" if encrypted else "",),
            body=body,
            raw=raw,
        )

    if record_type == "event_msg":
        if payload_type == "agent_reasoning":
            text = str(payload.get("text") or "")
            return _entry(
                index=index,
                timestamp=timestamp,
                category="reasoning",
                title="Reasoning summary stream",
                badges=("stream mirror",),
                body=_render_text(text),
                raw=raw,
                mirror=any(text.strip() in summary for summary in reasoning_summaries),
            )
        if payload_type == "agent_message":
            text = str(payload.get("message") or "")
            return _entry(
                index=index,
                timestamp=timestamp,
                category="assistant",
                title="Assistant stream",
                badges=(str(payload.get("phase") or ""), "stream mirror"),
                body=_render_text(text),
                raw=raw,
                mirror=any(text.strip() in message for message in assistant_messages),
            )
        if payload_type == "user_message":
            text = str(payload.get("message") or "")
            mirrored = any(text.strip() == candidate or text.strip() in candidate for candidate in user_messages)
            return _entry(
                index=index,
                timestamp=timestamp,
                category="user",
                title="User stream",
                badges=("stream mirror",),
                body=_render_text(text),
                raw=raw,
                mirror=mirrored,
            )
        if payload_type in {"task_started", "task_complete", "turn_aborted"}:
            title = {
                "task_started": "Turn started",
                "task_complete": "Turn completed",
                "turn_aborted": "Turn aborted",
            }[payload_type]
            duration_ms = payload.get("duration_ms")
            body = _render_kv(
                (
                    ("Turn ID", payload.get("turn_id")),
                    ("Duration", _format_duration(duration_ms / 1000) if isinstance(duration_ms, (int, float)) else None),
                    ("Time to first token", _format_duration(payload.get("time_to_first_token_ms") / 1000) if isinstance(payload.get("time_to_first_token_ms"), (int, float)) else None),
                    ("Reason", payload.get("reason")),
                    ("Context window", _format_number(payload.get("model_context_window")) if payload.get("model_context_window") else None),
                )
            )
            return _entry(index=index, timestamp=timestamp, category="event", title=title, body=body, raw=raw, badges=(payload_type.removeprefix("task_"),))
        if payload_type == "token_count":
            info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
            usage = info.get("total_token_usage") if isinstance(info.get("total_token_usage"), dict) else {}
            body = _render_kv(
                (
                    ("Total tokens", _format_number(usage.get("total_tokens"))),
                    ("Input", _format_number(usage.get("input_tokens"))),
                    ("Cached input", _format_number(usage.get("cached_input_tokens"))),
                    ("Output", _format_number(usage.get("output_tokens"))),
                    ("Reasoning output", _format_number(usage.get("reasoning_output_tokens"))),
                )
            )
            return _entry(index=index, timestamp=timestamp, category="event", title="Token usage snapshot", body=body, raw=raw, mirror=True)
        title = payload_type.replace("_", " ").title() if payload_type else "Event"
        return _entry(index=index, timestamp=timestamp, category="event", title=title, body="", raw=raw)

    title = f"{record_type}: {payload_type}".rstrip(": ")
    return _entry(index=index, timestamp=timestamp, category="event", title=title, body="", raw=raw)


def _session_metadata(records: Sequence[dict[str, Any]], path: Path) -> dict[str, Any]:
    meta = next((record for record in records if record.get("type") == "session_meta"), {})
    meta_payload = _payload(meta)
    contexts = [_payload(record) for record in records if record.get("type") == "turn_context"]
    last_context = contexts[-1] if contexts else {}
    timestamps = [parsed for record in records if (parsed := _iso_datetime(record.get("timestamp"))) is not None]
    duration = (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) >= 2 else None
    final_usage: dict[str, Any] = {}
    for record in records:
        payload = _payload(record)
        if payload.get("type") == "token_count" and isinstance(payload.get("info"), dict):
            usage = payload["info"].get("total_token_usage")
            if isinstance(usage, dict):
                final_usage = usage
    started = {
        _payload(record).get("turn_id")
        for record in records
        if _payload(record).get("type") == "task_started"
    }
    finished = {
        _payload(record).get("turn_id")
        for record in records
        if _payload(record).get("type") in {"task_complete", "turn_aborted"}
    }
    return {
        "session_id": meta_payload.get("session_id") or meta_payload.get("id") or path.stem,
        "cwd": meta_payload.get("cwd") or last_context.get("cwd"),
        "cli_version": meta_payload.get("cli_version"),
        "model": last_context.get("model") or meta_payload.get("model_provider"),
        "effort": last_context.get("effort"),
        "duration": duration,
        "tokens": final_usage,
        "turns": len(started),
        "active": bool(started - finished),
    }


def render_html(session: LoadedSession, *, include_images: bool = True, refresh_seconds: float | None = None) -> str:
    records = session.records
    metadata = _session_metadata(records, session.path)
    assistant_messages, user_messages, reasoning_summaries = _mirror_signatures(records)
    tool_outputs, tool_events = _tool_maps(records)
    consumed: set[int] = set()
    entries: list[str] = []
    category_counts: Counter[str] = Counter()

    for index, record in enumerate(records):
        if index in consumed:
            continue
        payload = _payload(record)
        payload_type = payload.get("type")
        if record.get("type") == "response_item" and payload_type in {"custom_tool_call", "function_call"}:
            call_id = payload.get("call_id")
            output_pair = tool_outputs.get(call_id) if isinstance(call_id, str) else None
            event_pairs = tool_events.get(call_id, []) if isinstance(call_id, str) else []
            if output_pair:
                consumed.add(output_pair[0])
            consumed.update(event_index for event_index, _ in event_pairs)
            entries.append(
                _render_tool(
                    index,
                    record,
                    output_pair[1] if output_pair else None,
                    [event_record for _, event_record in event_pairs],
                )
            )
            category_counts["tool"] += 1
            continue
        if record.get("type") == "response_item" and payload_type in {"custom_tool_call_output", "function_call_output"}:
            # An orphaned output remains visible rather than disappearing.
            body = _render_text(_content_text(payload.get("output")), css_class="prose tool-output")
            entries.append(
                _entry(
                    index=index,
                    timestamp=record.get("timestamp"),
                    category="tool",
                    title="Unpaired tool output",
                    subtitle=str(payload.get("call_id") or ""),
                    body=body,
                    raw=_raw_details(record),
                )
            )
            category_counts["tool"] += 1
            continue

        rendered = _render_record(
            index,
            record,
            include_images=include_images,
            assistant_messages=assistant_messages,
            user_messages=user_messages,
            reasoning_summaries=reasoning_summaries,
        )
        entries.append(rendered)
        if 'data-category="' in rendered:
            category = rendered.split('data-category="', 1)[1].split('"', 1)[0]
            category_counts[category] += 1

    warnings = "".join(f"<li>{html.escape(warning)}</li>" for warning in session.warnings)
    warning_panel = f'<aside class="warning"><strong>Read warnings</strong><ul>{warnings}</ul></aside>' if warnings else ""
    tokens = metadata["tokens"]
    active_label = "live / incomplete" if metadata["active"] else "saved"
    status_class = "active" if metadata["active"] else "saved"
    report_generated = datetime.now().astimezone().isoformat(timespec="seconds")
    refresh_script = ""
    if refresh_seconds:
        milliseconds = max(500, int(refresh_seconds * 1000))
        refresh_script = f"window.setTimeout(() => window.location.reload(), {milliseconds});"

    counts_json = html.escape(json.dumps(category_counts, sort_keys=True), quote=True)
    source_attr = html.escape(str(session.path), quote=True)
    title = f"Codex session {metadata['session_id']}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark light">
<title>{html.escape(title)}</title>
<style>
:root {{ --bg:#0b1020; --panel:#11182b; --panel2:#172038; --text:#e7ecf6; --muted:#97a3ba; --line:#2b3651; --accent:#7dd3fc; --user:#60a5fa; --assistant:#34d399; --reasoning:#c084fc; --tool:#fbbf24; --system:#94a3b8; --event:#fb7185; --context:#a78bfa; --shadow:0 18px 45px rgba(0,0,0,.22); }}
@media (prefers-color-scheme: light) {{ :root {{ --bg:#f4f7fb; --panel:#fff; --panel2:#f7f9fc; --text:#172033; --muted:#647089; --line:#dce3ee; --accent:#0369a1; --shadow:0 12px 32px rgba(25,40,70,.09); }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.55 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
.page {{ width:min(1180px,calc(100% - 32px)); margin:0 auto 80px; }}
.hero {{ padding:38px 0 24px; }}
.eyebrow {{ color:var(--accent); font-size:12px; font-weight:750; letter-spacing:.12em; text-transform:uppercase; }}
h1 {{ font-size:clamp(26px,4vw,42px); line-height:1.12; letter-spacing:-.03em; margin:8px 0 12px; overflow-wrap:anywhere; }}
.source {{ color:var(--muted); font-family:ui-monospace,SFMono-Regular,Consolas,monospace; overflow-wrap:anywhere; }}
.source button {{ margin-left:8px; }}
button {{ border:1px solid var(--line); border-radius:8px; color:var(--text); background:var(--panel2); padding:6px 10px; cursor:pointer; }}
button:hover {{ border-color:var(--accent); }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(135px,1fr)); gap:10px; margin-top:24px; }}
.stat {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:13px 15px; box-shadow:var(--shadow); }}
.stat b {{ display:block; font-size:19px; }} .stat span {{ color:var(--muted); font-size:12px; }}
.status {{ display:inline-flex; align-items:center; gap:7px; }} .status::before {{ content:""; width:8px; height:8px; border-radius:50%; background:var(--assistant); }} .status.saved::before {{ background:var(--system); }}
.warning {{ border:1px solid #f59e0b; background:rgba(245,158,11,.11); border-radius:12px; padding:12px 16px; margin:0 0 16px; }} .warning ul {{ margin:5px 0 0; }}
.controls {{ position:sticky; top:0; z-index:10; display:flex; gap:10px; flex-wrap:wrap; align-items:center; padding:12px; margin-bottom:24px; background:color-mix(in srgb,var(--bg) 88%,transparent); backdrop-filter:blur(14px); border:1px solid var(--line); border-radius:14px; }}
.controls input[type=search] {{ flex:1 1 250px; min-width:180px; color:var(--text); background:var(--panel); border:1px solid var(--line); border-radius:9px; padding:8px 11px; }}
.filter {{ display:inline-flex; align-items:center; gap:5px; padding:5px 8px; border-radius:8px; background:var(--panel); white-space:nowrap; }}
.visible-count {{ color:var(--muted); margin-left:auto; }}
.timeline {{ position:relative; margin-left:10px; }} .timeline::before {{ content:""; position:absolute; left:10px; top:5px; bottom:5px; width:1px; background:var(--line); }}
.entry {{ --entry-color:var(--event); position:relative; margin:0 0 14px 36px; border:1px solid var(--line); border-left:3px solid var(--entry-color); background:var(--panel); border-radius:12px; box-shadow:var(--shadow); overflow:hidden; }}
.entry[hidden] {{ display:none; }} .rail-dot {{ position:absolute; left:-35px; top:21px; width:10px; height:10px; border-radius:50%; background:var(--entry-color); box-shadow:0 0 0 5px var(--bg); }}
.category-user {{ --entry-color:var(--user); }} .category-assistant {{ --entry-color:var(--assistant); }} .category-reasoning {{ --entry-color:var(--reasoning); }} .category-tool {{ --entry-color:var(--tool); }} .category-system {{ --entry-color:var(--system); }} .category-context {{ --entry-color:var(--context); }}
.entry-header {{ display:grid; grid-template-columns:92px minmax(0,1fr) auto; gap:13px; align-items:start; padding:13px 16px; background:var(--panel2); }}
.entry-header time {{ color:var(--muted); font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace; }} .entry-heading h2 {{ margin:0; font-size:15px; }} .entry-heading p {{ margin:1px 0 0; color:var(--muted); font-size:12px; }}
.badges {{ display:flex; justify-content:flex-end; flex-wrap:wrap; gap:5px; }} .badges span {{ color:var(--muted); border:1px solid var(--line); border-radius:999px; padding:1px 7px; font-size:10px; }}
.entry-body {{ padding:16px; }} .prose {{ white-space:pre-wrap; overflow-wrap:anywhere; }} .prose.compact {{ max-height:300px; overflow:auto; }}
pre {{ margin:0; padding:12px; border-radius:9px; background:#080d18; color:#d9e2f2; white-space:pre-wrap; overflow-wrap:anywhere; overflow:auto; font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace; }}
@media (prefers-color-scheme: light) {{ pre {{ background:#f0f3f8; color:#263349; }} }}
details {{ margin-top:12px; }} summary {{ color:var(--muted); cursor:pointer; user-select:none; }} details[open]>summary {{ margin-bottom:8px; }}
.raw-details {{ border-top:1px solid var(--line); padding-top:10px; }} .raw-details pre {{ max-height:500px; }}
.tool-block {{ border:1px solid var(--line); border-radius:10px; padding:8px 10px; }}
.metadata {{ display:grid; grid-template-columns:minmax(100px,160px) 1fr; gap:5px 14px; margin:0; }} .metadata dt {{ color:var(--muted); }} .metadata dd {{ margin:0; overflow-wrap:anywhere; }}
.attachment {{ margin-top:12px; padding:9px 12px; color:var(--muted); border:1px dashed var(--line); border-radius:9px; }} .attachment.encrypted {{ border-color:var(--reasoning); }} .attachment.pending {{ border-color:var(--tool); }}
.session-image {{ margin:14px 0 0; }} .session-image img {{ display:block; max-width:100%; max-height:560px; border-radius:10px; border:1px solid var(--line); }} .session-image figcaption {{ color:var(--muted); font-size:11px; margin-top:5px; }}
.empty {{ color:var(--muted); font-style:italic; }}
footer {{ margin:28px 0; color:var(--muted); text-align:center; font-size:12px; }}
@media (max-width:700px) {{ .page {{ width:min(100% - 18px,1180px); }} .entry {{ margin-left:22px; }} .timeline::before {{ left:2px; }} .rail-dot {{ left:-25px; }} .entry-header {{ grid-template-columns:1fr; gap:4px; }} .badges {{ justify-content:flex-start; }} .metadata {{ grid-template-columns:1fr; }} .metadata dd {{ margin-bottom:5px; }} .visible-count {{ width:100%; margin-left:0; }} }}
</style>
</head>
<body data-counts="{counts_json}">
<main class="page">
<section class="hero">
<div class="eyebrow">Codex persisted session</div>
<h1>{html.escape(str(metadata['session_id']))}</h1>
<div class="source">{html.escape(str(session.path))}<button id="copy-source" data-copy="{source_attr}">Copy path</button></div>
<div class="stats">
<div class="stat"><b class="status {status_class}">{active_label}</b><span>session state</span></div>
<div class="stat"><b>{html.escape(str(metadata['model'] or '—'))}</b><span>model{(' · ' + html.escape(str(metadata['effort']))) if metadata['effort'] else ''}</span></div>
<div class="stat"><b>{metadata['turns']}</b><span>turns started</span></div>
<div class="stat"><b>{len(records):,}</b><span>JSONL records</span></div>
<div class="stat"><b>{_format_duration(metadata['duration'])}</b><span>recorded span</span></div>
<div class="stat"><b>{_format_number(tokens.get('total_tokens'))}</b><span>tokens · {_format_number(tokens.get('reasoning_output_tokens'))} reasoning</span></div>
</div>
</section>
{warning_panel}
<section class="controls" aria-label="Timeline filters">
<input id="search" type="search" placeholder="Search visible session text…" autocomplete="off">
<label class="filter"><input type="checkbox" data-filter="user" checked>User</label>
<label class="filter"><input type="checkbox" data-filter="assistant" checked>Assistant</label>
<label class="filter"><input type="checkbox" data-filter="reasoning" checked>Reasoning</label>
<label class="filter"><input type="checkbox" data-filter="tool" checked>Tools</label>
<label class="filter"><input type="checkbox" data-filter="system">System</label>
<label class="filter"><input type="checkbox" data-filter="event" checked>Events</label>
<label class="filter"><input type="checkbox" data-filter="context">Context</label>
<label class="filter"><input id="mirrors" type="checkbox">Stream mirrors</label>
<button id="expand">Expand details</button><button id="collapse">Collapse details</button>
<span class="visible-count" id="visible-count"></span>
</section>
<section class="timeline">{''.join(entries)}</section>
<footer>Generated {html.escape(report_generated)} · Full prompts and tool outputs may contain sensitive data. Keep this report local.</footer>
</main>
<script>
(() => {{
  const entries = [...document.querySelectorAll('.entry')];
  const search = document.querySelector('#search');
  const mirrors = document.querySelector('#mirrors');
  const filters = [...document.querySelectorAll('[data-filter]')];
  const storageKey = 'rapid-session-view:' + document.querySelector('h1').textContent;
  let saved = {{}};
  try {{ saved = JSON.parse(localStorage.getItem(storageKey) || '{{}}'); }} catch (_) {{}}
  if (typeof saved.search === 'string') search.value = saved.search;
  for (const filter of filters) if (saved.filters && filter.dataset.filter in saved.filters) filter.checked = saved.filters[filter.dataset.filter];
  if (typeof saved.mirrors === 'boolean') mirrors.checked = saved.mirrors;
  const apply = () => {{
    const enabled = new Set(filters.filter(x => x.checked).map(x => x.dataset.filter));
    const terms = search.value.toLowerCase().trim().split(/\\s+/).filter(Boolean);
    let count = 0;
    for (const entry of entries) {{
      const haystack = (entry.dataset.search + ' ' + entry.textContent.toLowerCase());
      const visible = enabled.has(entry.dataset.category) && (mirrors.checked || entry.dataset.mirror !== '1') && terms.every(term => haystack.includes(term));
      entry.hidden = !visible;
      if (visible) count++;
    }}
    document.querySelector('#visible-count').textContent = `${{count}} / ${{entries.length}} entries`;
    const state = {{ search: search.value, mirrors: mirrors.checked, filters: Object.fromEntries(filters.map(x => [x.dataset.filter, x.checked])) }};
    try {{ localStorage.setItem(storageKey, JSON.stringify(state)); }} catch (_) {{}}
  }};
  search.addEventListener('input', apply); mirrors.addEventListener('change', apply); filters.forEach(x => x.addEventListener('change', apply));
  document.querySelector('#expand').addEventListener('click', () => document.querySelectorAll('.entry:not([hidden]) details').forEach(x => x.open = true));
  document.querySelector('#collapse').addEventListener('click', () => document.querySelectorAll('.entry details').forEach(x => x.open = false));
  document.querySelector('#copy-source').addEventListener('click', async event => {{ await navigator.clipboard.writeText(event.currentTarget.dataset.copy); event.currentTarget.textContent = 'Copied'; }});
  apply();
  if (saved.scroll) requestAnimationFrame(() => window.scrollTo(0, saved.scroll));
  window.addEventListener('beforeunload', () => {{ try {{ const state = JSON.parse(localStorage.getItem(storageKey) || '{{}}'); state.scroll = window.scrollY; localStorage.setItem(storageKey, JSON.stringify(state)); }} catch (_) {{}} }});
  {refresh_script}
}})();
</script>
</body>
</html>
"""


def _human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"


def _terminal_content(content: Any) -> str:
    if not isinstance(content, list):
        return _content_text(content)
    chunks: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            chunks.append(str(part))
            continue
        text = next(
            (
                part[key]
                for key in ("text", "output", "transcript", "refusal")
                if isinstance(part.get(key), str)
            ),
            None,
        )
        if text is not None:
            chunks.append(text)
            continue
        image_url = part.get("image_url")
        if isinstance(image_url, str):
            if image_url.startswith("data:"):
                header, _, encoded = image_url.partition(",")
                media_type = header.partition(";")[0].removeprefix("data:")
                approximate_size = len(encoded.rstrip("=")) * 3 // 4
                chunks.append(f"[embedded {media_type} image · {_human_bytes(approximate_size)}]")
            else:
                chunks.append(f"[image: {image_url}]")
            continue
        chunks.append(f"[{part.get('type') or 'attachment'}]")
    return "\n\n".join(chunk for chunk in chunks if chunk)


def _event_user_texts(records: Sequence[dict[str, Any]]) -> set[str]:
    return {
        str(_payload(record).get("message") or "").strip()
        for record in records
        if record.get("type") == "event_msg" and _payload(record).get("type") == "user_message"
    } - {""}


def _terminal_record_entry(
    index: int,
    record: dict[str, Any],
    *,
    assistant_messages: set[str],
    event_user_messages: set[str],
    reasoning_summaries: set[str],
) -> TerminalEntry:
    record_type = str(record.get("type") or "unknown")
    payload = _payload(record)
    payload_type = str(payload.get("type") or "")
    timestamp = record.get("timestamp")
    raw = (record,)

    if record_type == "session_meta":
        body = "\n".join(
            filter(
                None,
                (
                    f"session: {payload.get('session_id') or payload.get('id') or '—'}",
                    f"cwd: {payload.get('cwd')}" if payload.get("cwd") else "",
                    f"Codex CLI: {payload.get('cli_version')}" if payload.get("cli_version") else "",
                ),
            )
        )
        return TerminalEntry(index, timestamp, "context", "SESSION METADATA", body, raw_records=raw)

    if record_type == "turn_context":
        values = (
            f"turn: {payload.get('turn_id')}",
            f"model: {payload.get('model')}" if payload.get("model") else "",
            f"reasoning effort: {payload.get('effort')}" if payload.get("effort") else "",
            f"cwd: {payload.get('cwd')}" if payload.get("cwd") else "",
        )
        return TerminalEntry(
            index,
            timestamp,
            "context",
            "TURN CONTEXT",
            "\n".join(filter(None, values)),
            raw_records=raw,
        )

    if record_type == "world_state":
        return TerminalEntry(
            index,
            timestamp,
            "context",
            "WORLD STATE",
            _content_text(payload.get("state")),
            raw_records=raw,
        )

    if record_type == "compacted":
        body = "\n".join(
            filter(
                None,
                (
                    f"window: {payload.get('window_number')}",
                    str(payload.get("message") or ""),
                ),
            )
        )
        return TerminalEntry(index, timestamp, "context", "CONTEXT COMPACTED", body, raw_records=raw)

    if record_type == "response_item" and payload_type == "message":
        role = str(payload.get("role") or "message")
        category = role if role in {"user", "assistant"} else "system"
        # Codex records injected images, skills, and launch context as user-role
        # response items.  The event user_message is the clean human prompt.
        if role == "user" and event_user_messages:
            category = "system"
        phase = str(payload.get("phase") or "")
        return TerminalEntry(
            index,
            timestamp,
            category,
            role.upper(),
            _terminal_content(payload.get("content")),
            subtitle=phase,
            raw_records=raw,
        )

    if record_type == "response_item" and payload_type == "reasoning":
        encrypted = payload.get("encrypted_content")
        subtitle = "encrypted continuation state present" if encrypted else ""
        return TerminalEntry(
            index,
            timestamp,
            "reasoning",
            "REASONING",
            _summary_text(payload),
            subtitle=subtitle,
            raw_records=raw,
        )

    if record_type == "event_msg":
        if payload_type == "user_message":
            attachments = []
            for key in ("images", "local_images", "audio", "local_audio"):
                value = payload.get(key)
                if isinstance(value, list) and value:
                    attachments.append(f"{len(value)} {key.replace('_', ' ')}")
            body = str(payload.get("message") or "")
            if attachments:
                body += "\n\n[attachments: " + ", ".join(attachments) + "]"
            return TerminalEntry(index, timestamp, "user", "USER", body, raw_records=raw)
        if payload_type == "agent_reasoning":
            text = str(payload.get("text") or "")
            return TerminalEntry(
                index,
                timestamp,
                "reasoning",
                "REASONING STREAM",
                text,
                mirror=any(text.strip() in summary for summary in reasoning_summaries),
                raw_records=raw,
            )
        if payload_type == "agent_message":
            text = str(payload.get("message") or "")
            return TerminalEntry(
                index,
                timestamp,
                "assistant",
                "ASSISTANT STREAM",
                text,
                subtitle=str(payload.get("phase") or ""),
                mirror=any(text.strip() in message for message in assistant_messages),
                raw_records=raw,
            )
        if payload_type in {"task_started", "task_complete", "turn_aborted"}:
            title = {
                "task_started": "TURN STARTED",
                "task_complete": "TURN COMPLETED",
                "turn_aborted": "TURN ABORTED",
            }[payload_type]
            values = [f"turn: {payload.get('turn_id')}"]
            duration_ms = payload.get("duration_ms")
            if isinstance(duration_ms, (int, float)):
                values.append(f"duration: {_format_duration(duration_ms / 1000)}")
            first_token_ms = payload.get("time_to_first_token_ms")
            if isinstance(first_token_ms, (int, float)):
                values.append(f"first token: {_format_duration(first_token_ms / 1000)}")
            if payload.get("reason"):
                values.append(f"reason: {payload['reason']}")
            return TerminalEntry(index, timestamp, "event", title, " · ".join(values), raw_records=raw)
        if payload_type == "token_count":
            info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
            usage = info.get("total_token_usage") if isinstance(info.get("total_token_usage"), dict) else {}
            body = (
                f"total: {_format_number(usage.get('total_tokens'))} · "
                f"output: {_format_number(usage.get('output_tokens'))} · "
                f"reasoning: {_format_number(usage.get('reasoning_output_tokens'))}"
            )
            return TerminalEntry(
                index,
                timestamp,
                "context",
                "TOKEN SNAPSHOT",
                body,
                mirror=True,
                raw_records=raw,
            )
        category = "tool" if payload_type in {"patch_apply_end", "mcp_tool_call_end"} else "event"
        return TerminalEntry(
            index,
            timestamp,
            category,
            (payload_type or "event").replace("_", " ").upper(),
            "",
            raw_records=raw,
        )

    return TerminalEntry(
        index,
        timestamp,
        "event",
        f"{record_type}: {payload_type}".rstrip(": ").upper(),
        "",
        raw_records=raw,
    )


def _terminal_tool_entry(
    index: int,
    record: dict[str, Any],
    output_record: dict[str, Any] | None,
    event_records: Sequence[dict[str, Any]],
) -> TerminalEntry:
    payload = _payload(record)
    name = str(payload.get("name") or "tool")
    call_id = str(payload.get("call_id") or "")
    raw_input = payload.get("input") if "input" in payload else payload.get("arguments")
    blocks = ["INPUT\n" + (_content_text(raw_input) or "(empty)")]
    raw_records = [record]
    if output_record is not None:
        blocks.append("OUTPUT\n" + (_content_text(_payload(output_record).get("output")) or "(empty)"))
        raw_records.append(output_record)
    else:
        blocks.append("OUTPUT\n(pending)")
    raw_records.extend(event_records)
    return TerminalEntry(
        index,
        record.get("timestamp"),
        "tool",
        f"TOOL · {name}",
        "\n\n".join(blocks),
        subtitle=call_id,
        raw_records=tuple(raw_records),
    )


def _truncate_terminal(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return text[:max_chars].rstrip() + f"\n\n… {omitted:,} characters omitted; use --full to show them"


def _indent_terminal(text: str) -> str:
    if not text:
        return ""
    return "\n".join("    " + line for line in text.splitlines())


def _format_terminal_entry(
    entry: TerminalEntry,
    *,
    style: TerminalStyle,
    max_chars: int | None,
    show_raw: bool,
) -> str:
    color = entry.category if entry.category in TerminalStyle._CODES else "event"
    timestamp = style.apply(_display_timestamp(entry.timestamp), "dim")
    heading = style.apply(entry.title, color, "bold")
    subtitle = style.apply(f" · {entry.subtitle}", "dim") if entry.subtitle else ""
    record_number = style.apply(f"  #{entry.index + 1}", "dim")
    lines = [f"{timestamp}  {heading}{subtitle}{record_number}"]
    body = _truncate_terminal(entry.body, max_chars)
    if body:
        lines.append(_indent_terminal(body))
    if show_raw:
        raw = json.dumps(
            _sanitized_raw({"records": entry.raw_records}),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        lines.append(style.apply("    RAW", "dim"))
        lines.append(_indent_terminal(_truncate_terminal(raw, max_chars)))
    return "\n".join(lines)


def render_terminal(
    session: LoadedSession,
    *,
    color: bool = False,
    show_system: bool = False,
    show_context: bool = False,
    show_stream_mirrors: bool = False,
    show_raw: bool = False,
    max_chars: int | None = 12_000,
    start_index: int = 0,
    include_header: bool = True,
) -> str:
    """Render a readable terminal transcript, optionally from a record offset."""

    records = session.records
    metadata = _session_metadata(records, session.path)
    assistant_messages, _, reasoning_summaries = _mirror_signatures(records)
    event_user_messages = _event_user_texts(records)
    tool_outputs, tool_events = _tool_maps(records)
    tool_calls = {
        str(_payload(record).get("call_id")): index
        for index, record in enumerate(records)
        if record.get("type") == "response_item"
        and _payload(record).get("type") in {"custom_tool_call", "function_call"}
        and _payload(record).get("call_id")
    }
    consumed: set[int] = set()
    entries: list[TerminalEntry] = []

    for index, record in enumerate(records):
        if index < start_index or index in consumed:
            continue
        payload = _payload(record)
        payload_type = payload.get("type")
        if record.get("type") == "response_item" and payload_type in {"custom_tool_call", "function_call"}:
            call_id = payload.get("call_id")
            output_pair = tool_outputs.get(call_id) if isinstance(call_id, str) else None
            event_pairs = tool_events.get(call_id, []) if isinstance(call_id, str) else []
            if output_pair:
                consumed.add(output_pair[0])
            consumed.update(event_index for event_index, _ in event_pairs)
            entries.append(
                _terminal_tool_entry(
                    index,
                    record,
                    output_pair[1] if output_pair else None,
                    [event for _, event in event_pairs],
                )
            )
            continue
        if record.get("type") == "response_item" and payload_type in {
            "custom_tool_call_output",
            "function_call_output",
        }:
            call_id = str(payload.get("call_id") or "")
            call_index = tool_calls.get(call_id)
            if call_index is not None and call_index >= start_index:
                continue
            entries.append(
                TerminalEntry(
                    index,
                    record.get("timestamp"),
                    "tool",
                    "TOOL OUTPUT",
                    _content_text(payload.get("output")),
                    subtitle=call_id,
                    raw_records=(record,),
                )
            )
            continue
        entries.append(
            _terminal_record_entry(
                index,
                record,
                assistant_messages=assistant_messages,
                event_user_messages=event_user_messages,
                reasoning_summaries=reasoning_summaries,
            )
        )

    visible_categories = {"user", "assistant", "reasoning", "tool", "event"}
    if show_system:
        visible_categories.add("system")
    if show_context:
        visible_categories.add("context")
    entries = [
        entry
        for entry in entries
        if entry.category in visible_categories and (show_stream_mirrors or not entry.mirror)
    ]

    style = TerminalStyle(color)
    sections: list[str] = []
    if include_header:
        width = min(max(shutil.get_terminal_size((100, 24)).columns, 72), 120)
        state = "LIVE / INCOMPLETE" if metadata["active"] else "SAVED"
        tokens = metadata["tokens"]
        sections.append(
            "\n".join(
                (
                style.apply("CODEX SESSION", "bold", "assistant"),
                style.apply("═" * width, "dim"),
                str(metadata["session_id"]),
                style.apply(str(session.path), "dim"),
                (
                    f"{state} · {metadata['model'] or 'unknown model'}"
                    f"{f' / {metadata["effort"]}' if metadata['effort'] else ''}"
                    f" · {metadata['turns']} turn(s) · {len(records):,} records"
                    f" · {_format_duration(metadata['duration'])}"
                    f" · {_format_number(tokens.get('total_tokens'))} tokens"
                ),
                )
            )
        )
    elif entries:
        sections.append(style.apply(f"UPDATE · {len(records) - start_index} new record(s)", "bold", "event"))

    sections.extend(
        _format_terminal_entry(
            entry,
            style=style,
            max_chars=max_chars,
            show_raw=show_raw,
        )
        for entry in entries
    )
    if session.warnings:
        warning_text = "\n".join(f"- {warning}" for warning in session.warnings)
        sections.append(style.apply("READ WARNINGS\n" + warning_text, "event", "bold"))
    return "\n\n".join(section for section in sections if section) + "\n"


def _terminal_color_enabled(mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never" or "NO_COLOR" in os.environ:
        return False
    return sys.stdout.isatty()


def _print_terminal(rendered: str, *, pager: str, watch: bool) -> None:
    use_pager = not watch and pager != "never" and sys.stdout.isatty() and shutil.which("less") is not None
    if pager == "always" and shutil.which("less") is None:
        raise SystemExit("error: --pager=always requested but 'less' is not installed")
    if pager == "always" and not watch:
        use_pager = True
    if use_pager:
        subprocess.run(["less", "-R"], input=rendered, text=True, check=False)
    else:
        sys.stdout.write(rendered)
        sys.stdout.flush()


def write_report(output: Path, rendered: str) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as stream:
            stream.write(rendered)
            temporary = Path(stream.name)
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def default_output_path(session_path: Path) -> Path:
    """Keep generated reports beside the selected run, outside runtime source."""

    for parent in session_path.parents:
        if parent.name == "sessions":
            return parent.parent / "session_reports" / f"{session_path.stem}.html"
    return session_path.parent / "session_reports" / f"{session_path.stem}.html"


def _list_sessions(files: Sequence[Path]) -> None:
    for path in reversed(files):
        stat = path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds")
        print(f"{modified}  {stat.st_size / (1024 * 1024):8.1f} MiB  {path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a persisted Codex rollout JSONL session in the terminal or as HTML.",
        epilog=(
            "SOURCE may be a JSONL file, suite run directory, workspace directory, "
            "or a short name under the experiment root. With no SOURCE, the newest current suite run session is used."
        ),
    )
    parser.add_argument("source", nargs="?", help="Session JSONL, run/workspace directory, or short run name.")
    add_experiment_root_argument(parser)
    parser.add_argument("--session", help="Select a session whose filename/path contains this ID or substring.")
    parser.add_argument("--list", action="store_true", help="List matching sessions without rendering.")
    parser.add_argument("--html", action="store_true", help="Generate HTML instead of printing to the terminal.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="HTML output path; specifying it implies --html.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Omit embedded input images from HTML reports.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        dest="open_report",
        help="Open the HTML report in a browser; implies --html.",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Terminal color policy (default: auto).",
    )
    parser.add_argument(
        "--pager",
        choices=("auto", "always", "never"),
        default="auto",
        help="Use less -R for terminal output when interactive (default: auto).",
    )
    parser.add_argument("--show-system", action="store_true", help="Include developer/system and injected context messages.")
    parser.add_argument("--show-context", action="store_true", help="Include session, turn, compaction, and token context records.")
    parser.add_argument("--show-stream", action="store_true", help="Include duplicate streaming mirrors of saved messages/reasoning.")
    parser.add_argument("--raw", action="store_true", help="Include sanitized raw JSON after every terminal entry.")
    parser.add_argument("--full", action="store_true", help="Do not truncate long terminal entries.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12_000,
        metavar="N",
        help="Maximum characters per terminal entry (default: 12000).",
    )
    parser.add_argument(
        "--watch",
        nargs="?",
        const=2.0,
        type=float,
        metavar="SECONDS",
        help="Follow new terminal records or refresh HTML continuously (default: 2 seconds).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_experiment_root(args, repository_root())
    try:
        files = find_session_files(args.source)
        if args.list:
            if not files:
                raise FileNotFoundError("no rollout-*.jsonl Codex sessions found")
            _list_sessions(files)
            return 0
        session_path = select_session(files, args.session)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    output = args.output or default_output_path(session_path)
    output = output.expanduser().resolve()
    opened = False

    if args.watch is not None and args.watch <= 0:
        raise SystemExit("error: --watch interval must be positive")
    if args.max_chars < 1:
        raise SystemExit("error: --max-chars must be positive")

    html_mode = args.html or args.output is not None or args.open_report
    if not html_mode:
        color = _terminal_color_enabled(args.color)
        max_chars = None if args.full else args.max_chars
        seen = 0
        first = True
        try:
            while True:
                loaded = load_session(session_path)
                if len(loaded.records) < seen:
                    seen = 0
                    first = True
                if first or len(loaded.records) > seen:
                    rendered = render_terminal(
                        loaded,
                        color=color,
                        show_system=args.show_system,
                        show_context=args.show_context,
                        show_stream_mirrors=args.show_stream,
                        show_raw=args.raw,
                        max_chars=max_chars,
                        start_index=seen,
                        include_header=first,
                    )
                    _print_terminal(rendered, pager=args.pager, watch=args.watch is not None)
                    seen = len(loaded.records)
                    first = False
                if args.watch is None:
                    break
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("stopped watching", flush=True)
        return 0

    try:
        while True:
            loaded = load_session(session_path)
            rendered = render_html(
                loaded,
                include_images=not args.no_images,
                refresh_seconds=args.watch,
            )
            write_report(output, rendered)
            print(
                f"rendered {len(loaded.records):,} records"
                f"{f' ({len(loaded.warnings)} warning(s))' if loaded.warnings else ''}: {output}",
                flush=True,
            )
            if args.open_report and not opened:
                webbrowser.open(output.as_uri())
                opened = True
            if args.watch is None:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("stopped watching", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
