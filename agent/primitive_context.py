"""Run-specific context for primitive invention."""

from __future__ import annotations

from pathlib import Path

from agent.descriptions import (
    format_language_description_for_prompt,
)
from agent.env_modules import (
    format_env_modules_for_prompt,
)
from agent.invention_context import (
    format_workspace_paths,
    invention_run_clause,
    search_budget_clause,
)

from .codex_exec import PRIMITIVES_DIR


def build_primitive_run_prompt(
    demo_frames: list[Path],
    demo_sources: list[Path],
    primitive_name: str | None,
    num_variants: int,
    variant_seed: int,
    max_iters: int,
    env_modules: list[str],
    language_description: str,
) -> str:
    frame_lines = format_workspace_paths(demo_frames)
    source_lines = format_workspace_paths(demo_sources)
    if primitive_name:
        name_clause = (
            f"- Primitive name: `{primitive_name}`. Use this exact name."
        )
    else:
        name_clause = (
            "- Primitive name: not supplied. Choose a short snake_case name, "
            "then stop for user confirmation before writing files."
        )

    env_section = format_env_modules_for_prompt(env_modules)
    description_section = format_language_description_for_prompt(
        "primitive to learn",
        language_description,
    )
    evaluation_clause = invention_run_clause("primitive", num_variants)
    budget_clause = search_budget_clause(max_iters)

    return f"""
Use `$rapid-construct-primitive` for this run.

Follow the workflow, file layout, validation rules, and quality bar in the
staged workspace instructions and `$rapid-construct-primitive` skill. This prompt
supplies only run-specific facts; do not restate those instructions.

## Demo frames

These keyframes live inside the workspace and are also attached to this
turn as local images:

{frame_lines}

## Original demo sources

The original demo inputs were copied into the workspace. Extract more frames
from these files if the sampled frames do not make contact timing or motion
direction clear:

{source_lines}

{env_section}

{description_section}

## Run parameters

{name_clause}
- Requested generalization variants after `example_0`: {num_variants}.
- Deterministic generalization seed: {variant_seed}.
{budget_clause}
{evaluation_clause}

After the Codex run, the launcher separately reports workflow completion,
behavioral predicate success, and audit-advisory status for every requested
example.
""".strip()


def existing_primitives() -> set[str]:
    if not PRIMITIVES_DIR.is_dir():
        return set()
    return {
        path.name
        for path in PRIMITIVES_DIR.iterdir()
        if path.is_dir() and not path.name.startswith(".") and path.name != "__pycache__"
    }
