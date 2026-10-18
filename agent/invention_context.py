"""Concise run-specific context shared by invention launchers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

from .codex_exec import WORKSPACE


InventionKind = Literal["primitive", "strategy"]


def format_workspace_paths(paths: Sequence[Path]) -> str:
    """Format paths relative to the active workspace for a run prompt."""

    return "\n".join(f"  - `{path.relative_to(WORKSPACE)}`" for path in paths)


def invention_run_clause(kind: InventionKind, num_variants: int) -> str:
    """Describe the single autonomous invention phase."""

    ownership_clause = (
        "Define the primitive's intrinsic effect-goal parameter interface and "
        "infer the demonstration invocation values. Every construction example "
        "must record a complete invocation in both VARIANT_SPEC.goal_parameters "
        "and binding inputs."
        if kind == "primitive"
        else
        "Use a role-only strategy contract. Express every primitive parameter "
        "as a learned connecting constraint evaluated from live bound geometry "
        "and the propagated phase-start state by execute_sequential_strategy."
    )
    return f"""
This is one autonomous invention phase. Treat the user's natural-language
description as qualitative semantic guidance and interpret it together with
the demonstration and assigned scene. Infer the goal `G` and measurable
predicate `Psi`, commit them to the learned {kind} description before the first
simulator run, and keep them stable unless later physical evidence demonstrates
a semantic mistake. Search difficulty or budget exhaustion does not justify
lowering a threshold or deleting a conjunct; apply the active skill's retained
counterexample protocol before calibrating `Psi`. Derive its measurable
thresholds and formulas.

{ownership_clause}

Create, statically validate, develop, tune, run, and record all requested
examples `example_0` through `example_{num_variants}` in this same Codex run.
All examples are development evidence: you may use any of them to diagnose and
improve the shared learned program. The completion target is successful
behavior on every required example, including required mechanism and retention
conditions. Finishing the workflow or exhausting one optimizer configuration
does not establish that target. Continue evidence-guided program improvement
while the overall run budget and a promising next experiment remain.
Preserve the best retained trajectory for each example. If the overall budget
is exhausted, or diagnosis leaves no promising next experiment within it,
report partial completion with the remaining failures, attempts, and reason
for stopping. Keep audit disagreements and any replaced development cases
visible separately; a replacement does not solve the original case. Follow
AGENTS.md for scene review and bounded resampling.
""".strip()


def search_budget_clause(iteration_ceiling: int) -> str:
    """Describe the deterministic bounded-search policy for a run prompt."""

    return f"""
- Per-attempt optimizer iteration ceiling: {iteration_ceiling}. Follow the
  skill's deterministic nominal/minimum/standard/expanded search ladder and
  do not exceed this run ceiling. Nominal-only and 1x1 diagnostic runs never
  count toward stagnation; stagnation accounting starts only at the minimum
  meaningful search stage. This ceiling limits each optimizer attempt, not
  the number of evidence-guided program revisions within the overall run
  budget. Expanded-search failure calls for diagnosis, not automatic stopping
  or scene replacement.
""".strip()


__all__ = [
    "InventionKind",
    "format_workspace_paths",
    "invention_run_clause",
    "search_budget_clause",
]
