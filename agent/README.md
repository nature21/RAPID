# `agent/` - RAPID invention agents

This package contains the automated invention agents used by RAPID. The project
now uses Codex CLI noninteractive runs (`codex exec`) as the only automatic
agent backend.

| Path | Purpose |
|---|---|
| `construct_primitive_exec.py` | Invents a primitive through `codex exec` and `$rapid-construct-primitive`. |
| `compose_strategy_exec.py` | Invents a multi-phase strategy through `codex exec` and `$rapid-compose-strategy`. |
| `instantiate_primitive_exec.py` | One-shot binding of an existing primitive through `$rapid-instantiate-primitive`. |
| `instantiate_strategy_exec.py` | Invention-time strategy example evaluation through `$rapid-instantiate-strategy`. |
| `deploy_strategy_exec.py` | Fast held-out binding and one frozen-policy execution through `$rapid-deploy-strategy`. |
| `codex_exec.py` | Shared Codex exec runtime: demo normalization, guidance staging, and event rendering. |
| `primitive_context.py` | Dynamic primitive run facts: demo frames, source paths, assigned tasks, name, and budgets. |
| `strategy_context.py` | Dynamic strategy run facts plus workspace primitive/strategy inventory. |
| `primitive_validation.py` | Post-run package layout validation for generated primitives. |
| `templates/invent_workspace/` | Workspace-local `AGENTS.md` and Codex skill templates staged into each invention workspace. |
| `descriptions.py` | Shared loading/formatting for optional language descriptions. |
| `env_modules.py` | Validation and prompt formatting for assigned task environment modules. |

Suite workspace creation stages an invent-agent instruction bundle from
`agent/templates/invent_workspace/` into the active workspace. The invention
launchers refresh the same bundle before starting Codex:

```text
workspace/AGENTS.md
workspace/.agents/skills/rapid-construct-primitive/SKILL.md
workspace/.agents/skills/rapid-compose-strategy/SKILL.md
workspace/.agents/skills/rapid-instantiate-primitive/SKILL.md
workspace/.agents/skills/rapid-instantiate-strategy/SKILL.md
workspace/.agents/skills/rapid-deploy-strategy/SKILL.md
```

The dynamic launcher prompt contains only run-specific facts such as demo
frames, demo source paths, environment modules, names, budgets, and available
primitives. Durable workflow guidance belongs in the staged `SKILL.md` files,
not in Python string prompts.

`launch_experiments.sh` freezes shared source in `runtime_<batch>/workspace`
and exports `RAPID_RUNTIME_WORKSPACE` to all stages. Shared runtime edits during
invention are publication failures; the session writes `runtime_integrity.json`
and exits nonzero. Learned-program provenance remains separate from this check.
Workspace copies verify their external `runtime_manifest.json`, and library
assembly rejects mismatched runtimes before copying primitives.

The strategy composition stage merges the constructed primitives into a shared
seed library before starting strategy invention. Strategy array elements reuse
the completed library under a shared filesystem lock.

Run agents through the permission-profile launcher from the repo root.

For example, the following commands invent a push primitive and a strategy.
Replace quoted placeholders with actual run inputs:

```bash
./scripts/rapid-codex-run \
  --workspace "<workspace_path>" \
  -- agent/construct_primitive_exec.py "<demonstration_path>" \
     --env-module "<environment_module>" \
     --name push

./scripts/rapid-codex-run \
  --workspace "<workspace_path>" \
  -- agent/compose_strategy_exec.py "<demonstration_path>" \
     --name "<strategy_name>" \
     --library-mode closed
```

Both launchers accept optional language guidance through `--description "..."`
or `--description-file path.txt`. `--max-turns` is an advisory run budget
passed into the Codex prompt. If Codex reports a transient model-capacity
failure, the launcher resumes the same session up to 30 times. Retry waits
start at 5 seconds and double after each failed retry, capped at 120 seconds.
Invention, instantiation, and deployment share these defaults. Use
`--max-retries` to change the retry limit and `--retry-delay` to change the
initial wait; an initial wait above 120 seconds also raises the cap to that
value. Other failures are not retried.

Each invention uses one Codex phase for the baseline and all requested
variants. The agent infers and commits `G` and `Psi` before simulation, then
develops against every example. `--max-iters` is the per-attempt ceiling for
the staged 32x6, 64x10, and 96x16 search ladder; nominal/1x1 diagnostics do not
count toward stagnation. On completion, `agent_work/invention_status.json`
reports workflow completion, behavioral predicate success, audit status and
fixed development qualification separately. Qualification uses frozen probe
inputs and source fingerprints; missing, failing or stale evidence cannot
count as a passing qualification. Its geometry and interface checks do not
replace held-out behavioral evaluation.

Learned primitives expose intrinsic effect-goal parameters such as displacement
or direction. Construction examples infer and record complete primitive invocations.
Learned strategies have role-only bindings: typed connecting constraints derive
primitive parameters from live phase-start state, and the shared sequential
executor runs one local optimization per primitive before propagating state to
the next constraint.

Bind an unchanged learned program to a new scene with a required goal
reference. For example:

```bash
./scripts/rapid-codex-run \
  --workspace "<workspace_path>" \
  -- agent/instantiate_primitive_exec.py \
     --primitive push \
     --env-module "<environment_module>" \
     --name "<instance_name>" \
     --goal "<description of the requested displacement and reference frame>"

./scripts/rapid-codex-run \
  --workspace "<workspace_path>" \
  -- agent/deploy_strategy_exec.py \
     --strategy "<strategy_name>" \
     --env-module "<environment_module>" \
     --name "<instance_name>" \
     --goal "<description of the intended outcome>"
```

Primitive and invention-example instantiation retain the smoke-driven workflow.
Strategy deployment supplies Codex a compact contract, scene snapshot and
structurally filtered role candidates. The host resolves body roles uniquely
implied by selected owned entities, validates the complete mapping, writes the
manifest and executes the frozen learned policies exactly once. Ambiguous or
conflicting selections require agent correction. The default
binding request uses `gpt-6-luna` at low reasoning effort and permits at most
one validator-directed correction. Pass `--bind-only` to stop after validation,
or `--goal-file` for longer goal metadata.
