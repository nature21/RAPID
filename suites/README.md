# Suites

This package defines staged agent evaluation suites for a zero-primitive baseline.
No concrete primitive implementations are present in the clean source tree.

## Staged Agent Suites

`primitive_suites/` and `strategy_suites/` describe the two-stage setting:

- **Construction stage**: an agent sees construction demos and assigned task
  environments, then
  writes learned primitives into its workspace checkpoint.
- **composition stage**: an agent starts from a learned or merged workspace
  checkpoint plus a composition demo/environment, then builds a strategy that
  composes the existing learned primitives.

Learned primitives and strategies are scene-agnostic and publish declarative
binding contracts. Concrete scene packages live under `src/instantiations/`;
one-shot generalization helpers create new bindings without editing learned
programs.

The `suites.staged` package provides helpers for creating isolated
candidate workspaces, creating new workspaces from prior checkpoints,
merging learned workspaces, and running modules against an isolated
`src/` tree. See the root
`SUITE_RUNS.md` for the operational workflow.
