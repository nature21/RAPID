"""Run-specific context for strategy invention."""

from __future__ import annotations

import re
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

from .codex_exec import PRIMITIVES_DIR, STRATEGIES_DIR


_STRATEGY_PATTERN = re.compile(r"^strategy_(\d+)$")


def next_free_strategy_name() -> str:
    """Return strategy_N with N one higher than the current max."""
    if not STRATEGIES_DIR.exists():
        return "strategy_1"
    used: list[int] = []
    for child in STRATEGIES_DIR.iterdir():
        if not child.is_dir():
            continue
        match = _STRATEGY_PATTERN.match(child.name)
        if match:
            used.append(int(match.group(1)))
    return f"strategy_{max(used) + 1 if used else 1}"


def list_existing_primitives() -> list[str]:
    if not PRIMITIVES_DIR.exists():
        return []
    out: list[str] = []
    for child in sorted(PRIMITIVES_DIR.iterdir()):
        if child.name.startswith("_") or child.name == "__pycache__":
            continue
        if not child.is_dir() or not (child / "__init__.py").exists():
            continue
        out.append(child.name)
    return out


def list_existing_strategies() -> list[str]:
    if not STRATEGIES_DIR.exists():
        return []
    out: list[str] = []
    for child in sorted(STRATEGIES_DIR.iterdir()):
        if child.name.startswith("_") or child.name == "__pycache__":
            continue
        if child.is_dir() and (child / "__init__.py").exists():
            out.append(child.name)
    return out


def build_strategy_run_prompt(
    demo_frames: list[Path],
    demo_sources: list[Path],
    strategy_name: str,
    available_primitives: list[str],
    existing_strategies: list[str],
    max_iters: int,
    num_variants: int,
    variant_seed: int,
    library_mode: str,
    env_modules: list[str],
    language_description: str,
) -> str:
    frame_lines = format_workspace_paths(demo_frames)
    source_lines = format_workspace_paths(demo_sources)
    primitive_lines = (
        "\n".join(f"  - `{primitive}`" for primitive in available_primitives)
        or "  - (none)"
    )
    strategy_lines = (
        "\n".join(f"  - `{strategy}`" for strategy in existing_strategies)
        or "  - (none)"
    )
    if library_mode == "closed":
        library_policy = (
            "Closed library: do not add or edit files under "
            "`src/primitives/` or `src/primitive_core/`."
        )
    else:
        library_policy = (
            "Open library: existing learned primitives are editable development "
            "inputs. If the library cannot express a target-changing phase, "
            "distinguish a missing primitive from an over-specific matching "
            "contract, implement and validate the narrowest corresponding "
            "primitive addition or repair, record it, and continue the strategy "
            "run without waiting for approval."
        )
    description_section = format_language_description_for_prompt(
        "strategy to learn",
        language_description,
    )
    env_section = format_env_modules_for_prompt(env_modules)
    evaluation_clause = invention_run_clause("strategy", num_variants)
    budget_clause = search_budget_clause(max_iters)

    return f"""
Use `$rapid-compose-strategy` for this run.

Follow the workflow, file layout, physical-fidelity rules, and quality bar in
the staged workspace instructions and `$rapid-compose-strategy` skill. This
prompt supplies only run-specific facts; do not restate those instructions.

## Demo frames

These keyframes live inside the workspace and are also attached to this
turn as local images:

{frame_lines}

## Original demo sources

The original demo inputs were copied into the workspace. Extract more frames
from these files if the sampled frames do not make phase ordering, contact
timing, or motion direction clear:

{source_lines}

{env_section}

{description_section}

## Run parameters

- Strategy name: `{strategy_name}`.
- Assigned-scene instantiation name: `example_0`.
- Generate {num_variants} geometry/pose variants after `example_0` with seed
  {variant_seed}.
{budget_clause}
- Library mode: `{library_mode}`. {library_policy}
{evaluation_clause}

After the Codex run, the launcher separately reports workflow completion,
behavioral predicate success, and audit-advisory status for every requested
example.

## Primitives currently available under `src/primitives/`

{primitive_lines}

## Strategies already in the repo

{strategy_lines}
""".strip()
