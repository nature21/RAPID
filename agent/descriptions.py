"""Helpers for optional natural-language task descriptions."""

from __future__ import annotations

from pathlib import Path


def load_language_description(
    description: str | None,
    description_file: Path | None,
) -> str:
    """Load and combine optional inline/file descriptions."""
    parts: list[str] = []
    if description and description.strip():
        parts.append(description.strip())
    if description_file is not None:
        path = description_file.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"description file not found: {description_file}")
        file_text = path.read_text(encoding="utf-8").strip()
        if file_text:
            parts.append(file_text)
    return "\n\n".join(parts)


def format_language_description_for_prompt(kind: str, description: str) -> str:
    """Build a prompt section for a user-provided language description."""
    if not description:
        return ""
    return f"""
## Language description

The user provided this natural-language description of the {kind}. Treat it
as an intended-behavior constraint alongside the demonstration frames. If it
is ambiguous or appears to conflict with the frames, state the assumption you
choose before writing code.

{description}
"""
