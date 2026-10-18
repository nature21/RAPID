"""Post-run integrity and layout checks for generated instantiations."""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from instantiation_core.contracts import (
    BindingContract,
    validate_scene_binding,
)
from primitive_core.common import (
    GLOBAL_MAX_OPTIMIZATION_BUDGET,
    OptimizationBudget,
    OptimizationPolicy,
)
from instantiation_core.manifest import ManifestError, load_binding_manifest
from instantiation_core.json_values import to_json_value
from instantiation_core.scene import load_scene_module
from instantiation_core.generalization import (
    SceneVariantSpec,
    generalization_metadata,
    make_variant_data,
    make_variant_model,
    runtime_scene_fingerprint,
    validate_variant_initial_state,
)

from .codex_exec import (
    PRIMITIVES_DIR,
    REPO_ROOT,
    SRC_DIR,
    STRATEGIES_DIR,
    WORKSPACE,
)
from .instantiation_context import (
    CANONICAL_INIT,
    CANONICAL_MAIN,
    CANONICAL_PARENT_INIT,
    instantiation_dir,
    program_dir,
)
from .sandbox_validation import run_sandboxed_validation


STRATEGY_TRANSIT_EXECUTION_FINDING_PREFIX = (
    "learned strategy must use typed shared transit execution; "
)


def scene_interface_findings(tree: ast.AST, path: Path) -> list[str]:
    """Keep development provenance out of learned execution dependencies."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute) and node.attr == "VARIANT_SPEC"
            or isinstance(node, ast.Name) and node.id == "VARIANT_SPEC"
            or isinstance(node, ast.ImportFrom)
            and any(alias.name == "VARIANT_SPEC" for alias in node.names)
            or isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "hasattr"}
            and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "VARIANT_SPEC"
        ):
            return [
                "learned execution must use the runtime scene interface; "
                f"development metadata is owned by the artifact writer: {path}"
            ]
    return []


def partition_program_contract_findings(
    kind: str,
    findings: Sequence[str],
    *,
    execution_policy: str,
) -> tuple[list[str], list[str]]:
    """Split hard failures from attempt-mode strategy transit advisories."""

    if kind != "strategy" or execution_policy != "attempt":
        return list(findings), []
    advisories = [
        finding
        for finding in findings
        if finding.startswith(STRATEGY_TRANSIT_EXECUTION_FINDING_PREFIX)
    ]
    blocking = [finding for finding in findings if finding not in advisories]
    return blocking, advisories


def digest_tree(path: Path) -> str:
    """Return a stable content digest for one source tree."""

    digest = hashlib.sha256()
    if not path.exists():
        return digest.hexdigest()
    if path.is_file():
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        return digest.hexdigest()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        if "__pycache__" in file_path.parts or file_path.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def learned_source_digests() -> dict[str, str]:
    """Snapshot learned programs and canonical runtime used by binding."""

    return {
        "primitives": digest_tree(PRIMITIVES_DIR),
        "strategies": digest_tree(STRATEGIES_DIR),
        "primitive_core": digest_tree(SRC_DIR / "primitive_core"),
        "instantiation_core": digest_tree(SRC_DIR / "instantiation_core"),
    }


def learned_source_changes(before: dict[str, str]) -> list[str]:
    after = learned_source_digests()
    return sorted(name for name, digest in before.items() if after[name] != digest)


def validate_instantiation_main(path: Path) -> list[str]:
    """Require the immutable canonical manifest entrypoint."""

    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"could not read instantiation entrypoint {path}: {exc}"]
    if source == CANONICAL_MAIN:
        return []
    return [f"noncanonical instantiation entrypoint: {path}"]


def record_learned_source_digests(
    output_path: Path,
    before: dict[str, str],
) -> list[str]:
    """Record before/after learned-tree digests and return changed tree names."""

    after = learned_source_digests()
    changed = sorted(name for name, digest in before.items() if after[name] != digest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "before": before,
                "after": after,
                "changed": changed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return changed


def validate_program_contract(kind: str, program: str) -> list[str]:
    """Return diagnostics for one learned program's public binding contract."""

    findings: list[str] = []
    path = program_dir(kind, program) / "binding_contract.py"
    if not path.is_file():
        return [f"missing learned-program binding contract: {path}"]
    package = "primitives" if kind == "primitive" else "strategies"
    module_name = f"{package}.{program}.binding_contract"
    importlib.invalidate_caches()
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return [f"could not import {module_name}: {exc}"]
    contract = getattr(module, "BINDING_CONTRACT", None)
    if not isinstance(contract, BindingContract):
        findings.append(f"{module_name}.BINDING_CONTRACT must be a BindingContract")
        return findings
    if contract.program_kind != kind:
        findings.append(
            f"{module_name}.BINDING_CONTRACT.program_kind is "
            f"{contract.program_kind!r}; expected {kind!r}"
        )
    if contract.program_name != program:
        findings.append(
            f"{module_name}.BINDING_CONTRACT.program_name is "
            f"{contract.program_name!r}; expected {program!r}"
        )
    source_root = program_dir(kind, program)
    banned_roots = {"assigned_tasks", "instantiations", "tasks", "test_scenes"}
    for source_path in sorted(source_root.rglob("*.py")):
        try:
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            findings.append(f"could not parse {source_path}: {exc}")
            continue
        findings.extend(scene_interface_findings(tree, source_path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            for imported in modules:
                if imported.split(".", 1)[0] in banned_roots:
                    findings.append(
                        "learned source may not import assigned task or "
                        f"instantiation module {imported!r}: {source_path}"
                    )
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.split(".", 1)[0] in banned_roots
            ):
                findings.append(
                    "learned source may not reference assigned task or "
                    f"instantiation module {node.value!r}: {source_path}"
                )
            if isinstance(node, ast.Call):
                called_name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else ""
                )
                if called_name == "FR3SafetyLimits":
                    findings.append(
                        "learned source may not construct or tune shared FR3 "
                        f"motion/audit limits: {source_path}"
                    )
                if called_name == "FR3SafetySpec" and any(
                    keyword.arg == "limits" for keyword in node.keywords
                ):
                    findings.append(
                        "learned FR3SafetySpec may declare audit contact pairs but "
                        f"may not override shared limits: {source_path}"
                    )
        if kind == "primitive" and source_path.name == "types.py":
            forbidden_fields = {
                "nominal_actions",
                "nominal_knots",
                "sigma_actions",
                "sigma_init",
                "waypoint_actions",
                "waypoints",
            }
            declared_fields = {
                node.target.id
                for node in ast.walk(tree)
                if isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
            }
            leaked_fields = sorted(forbidden_fields & declared_fields)
            if leaked_fields:
                findings.append(
                    f"primitive public types leak action-level policy data: "
                    f"{leaked_fields}"
                )
        if kind == "primitive" and source_path.name == "main.py":
            policy_values = [
                keyword.value
                for call in ast.walk(tree)
                if isinstance(call, ast.Call)
                for keyword in call.keywords
                if keyword.arg == "optimization_policy"
            ]
            uses_policy = any(
                any(
                    (
                        isinstance(reference, ast.Name)
                        and reference.id == "OPTIMIZATION_POLICY"
                    )
                    or (
                        isinstance(reference, ast.Attribute)
                        and reference.attr == "OPTIMIZATION_POLICY"
                    )
                    for reference in ast.walk(value)
                )
                for value in policy_values
            )
            if not uses_policy:
                findings.append(
                    "primitive main must pass its frozen OPTIMIZATION_POLICY "
                    "as InstantiationSpec.optimization_policy"
                )
            budget_values = [
                keyword.value
                for call in ast.walk(tree)
                if isinstance(call, ast.Call)
                for keyword in call.keywords
                if keyword.arg == "reference_optimization_budget"
            ]
            uses_budget = any(
                any(
                    (
                        isinstance(reference, ast.Name)
                        and reference.id == "REFERENCE_OPTIMIZATION_BUDGET"
                    )
                    or (
                        isinstance(reference, ast.Attribute)
                        and reference.attr == "REFERENCE_OPTIMIZATION_BUDGET"
                    )
                    for reference in ast.walk(value)
                )
                for value in budget_values
            )
            if not uses_budget:
                findings.append(
                    "primitive main must pass its "
                    "REFERENCE_OPTIMIZATION_BUDGET as "
                    "InstantiationSpec.reference_optimization_budget"
                )
        if kind == "strategy":
            forbidden_transit_symbols = {
                "realize_transit_waypoints",
                "rollout_controls",
                "rollout_fr3_transit",
                "waypoint_actions",
                "waypoints",
            }
            referenced = {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name)
            } | {
                node.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)
            }
            leaked_transit = sorted(forbidden_transit_symbols & referenced)
            if leaked_transit:
                findings.append(
                    STRATEGY_TRANSIT_EXECUTION_FINDING_PREFIX
                    + f"found {leaked_transit} in {source_path}"
                )
    if kind == "strategy":
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            pass
        else:
            banned_callback_fragments = {
                "compile_phase",
                "evaluate_success",
                "phase_compiler",
                "predicate_callback",
                "success_predicate",
            }
            exposed: set[str] = set()
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    exposed.add(node.name)
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                    exposed.update(
                        target.id
                        for target in targets
                        if isinstance(target, ast.Name)
                    )
            leaked = sorted(
                name
                for name in exposed
                if any(
                    fragment in name.lower()
                    for fragment in banned_callback_fragments
                )
            )
            if leaked:
                findings.append(
                    "strategy binding contract exposes forbidden callbacks: "
                    f"{leaked}"
                )
        try:
            from strategies.program import StrategyProgram, validate_strategy_program

            phase_module = importlib.import_module(f"strategies.{program}.program")
            phase_program = getattr(phase_module, "PHASE_PROGRAM", None)
            if not isinstance(phase_program, StrategyProgram):
                findings.append(
                    f"strategies.{program}.program.PHASE_PROGRAM must be a "
                    "StrategyProgram"
                )
            else:
                primitive_contracts: dict[str, BindingContract] = {}
                primitive_reference_budgets: dict[str, OptimizationBudget] = {}
                for phase in phase_program.phases:
                    primitive_module = importlib.import_module(
                        f"primitives.{phase.primitive}.binding_contract"
                    )
                    primitive_contract = getattr(
                        primitive_module,
                        "BINDING_CONTRACT",
                        None,
                    )
                    if isinstance(primitive_contract, BindingContract):
                        primitive_contracts[phase.primitive] = primitive_contract
                    nominal_module = importlib.import_module(
                        f"primitives.{phase.primitive}.nominal"
                    )
                    reference_budget = getattr(
                        nominal_module,
                        "REFERENCE_OPTIMIZATION_BUDGET",
                        None,
                    )
                    if isinstance(reference_budget, OptimizationBudget):
                        primitive_reference_budgets[
                            phase.primitive
                        ] = reference_budget
                findings.extend(
                    validate_strategy_program(
                        phase_program,
                        contract,
                        primitive_contracts,
                        primitive_reference_budgets,
                    )
                )
        except Exception as exc:
            findings.append(f"could not validate frozen strategy program: {exc}")
        for suffix, symbols in {
            "predicate": ("evaluate_final_psi",),
            "main": ("make_parser", "compile_strategy"),
        }.items():
            try:
                learned_module = importlib.import_module(
                    f"strategies.{program}.{suffix}"
                )
            except Exception as exc:
                findings.append(
                    f"could not import strategies.{program}.{suffix}: {exc}"
                )
                continue
            for symbol in symbols:
                value = getattr(learned_module, symbol, None)
                if not callable(value):
                    findings.append(
                        f"strategies.{program}.{suffix}.{symbol} must be callable"
                    )
                elif symbol == "compile_strategy" and list(
                    inspect.signature(value).parameters
                )[:2] != ["binding", "args"]:
                    findings.append(
                        f"strategies.{program}.main.compile_strategy must begin "
                        "with (binding, args)"
                    )
    else:
        required_modules = {
            "features": "resolve_geometry",
            "nominal": "make_nominal_plan",
            "transit": "default_transit_spec",
            "cost": "effect_predicate",
            "main": "compile_primitive",
        }
        for suffix, symbol in required_modules.items():
            try:
                learned_module = importlib.import_module(
                    f"primitives.{program}.{suffix}"
                )
            except Exception as exc:
                findings.append(
                    f"could not import primitives.{program}.{suffix}: {exc}"
                )
                continue
            value = getattr(learned_module, symbol, None)
            if not callable(value):
                findings.append(
                    f"primitives.{program}.{suffix}.{symbol} must be callable"
                )
            elif suffix == "main" and list(
                inspect.signature(value).parameters
            )[:2] != ["binding", "args"]:
                findings.append(
                    f"primitives.{program}.main.compile_primitive must begin "
                    "with (binding, args)"
                )
            if suffix == "cost":
                if not callable(getattr(learned_module, "effect_cost", None)):
                    findings.append(
                        f"primitives.{program}.cost.effect_cost must be callable"
                    )
                if not isinstance(
                    getattr(learned_module, "EFFECT_TOLERANCES", None),
                    Mapping,
                ):
                    findings.append(
                        f"primitives.{program}.cost.EFFECT_TOLERANCES must be a mapping"
                    )
            if suffix == "nominal":
                policy = getattr(learned_module, "OPTIMIZATION_POLICY", None)
                budget = getattr(
                    learned_module,
                    "REFERENCE_OPTIMIZATION_BUDGET",
                    None,
                )
                if not isinstance(policy, OptimizationPolicy):
                    findings.append(
                        f"primitives.{program}.nominal.OPTIMIZATION_POLICY "
                        "must be an OptimizationPolicy"
                    )
                if not isinstance(budget, OptimizationBudget):
                    findings.append(
                        f"primitives.{program}.nominal."
                        "REFERENCE_OPTIMIZATION_BUDGET must be an "
                        "OptimizationBudget"
                    )
                elif not GLOBAL_MAX_OPTIMIZATION_BUDGET.contains(budget):
                    findings.append(
                        f"primitives.{program}.nominal reference budget "
                        "exceeds GLOBAL_MAX_OPTIMIZATION_BUDGET"
                    )
                elif (
                    isinstance(policy, OptimizationPolicy)
                    and policy.min_iters > budget.max_iters
                ):
                    findings.append(
                        f"primitives.{program}.nominal policy min_iters "
                        "exceeds its reference budget"
                    )
        try:
            primitive_main = importlib.import_module(f"primitives.{program}.main")
        except Exception:
            pass
        else:
            if not callable(getattr(primitive_main, "make_parser", None)):
                findings.append(
                    f"primitives.{program}.main.make_parser must be callable"
                )
        compile_path = source_root / "compile.py"
        if compile_path.is_file():
            try:
                compile_tree = ast.parse(compile_path.read_text(encoding="utf-8"))
            except SyntaxError:
                compile_tree = None
            invokes_nominal = bool(
                compile_tree is not None
                and any(
                    isinstance(node, ast.Call)
                    and (
                        isinstance(node.func, ast.Name)
                        and node.func.id == "make_nominal_plan"
                        or isinstance(node.func, ast.Attribute)
                        and node.func.attr == "make_nominal_plan"
                    )
                    for node in ast.walk(compile_tree)
                )
            )
            if not invokes_nominal:
                findings.append(
                    f"{compile_path} must invoke the learned nominal policy"
                )
    return findings


def report_program_contract_validation(kind: str, program: str) -> None:
    """Validate a program contract without importing it in the host process."""

    findings = run_sandboxed_validation(
        {
            "check": "program_contract",
            "kind": kind,
            "program": program,
        }
    )
    if not findings:
        return
    print(f"error: selected {kind} has an invalid binding contract:", file=sys.stderr)
    for finding in findings:
        print(f"  - {finding}", file=sys.stderr)
    raise SystemExit(2)


def validate_instantiation_layout(
    kind: str,
    program: str,
    instance: str,
    *,
    expected_scene_module: str | None = None,
    expected_goal: str | None = None,
    execution_policy: str = "strict",
) -> list[str]:
    """Return structural/import diagnostics for a generated package."""

    findings: list[str] = []
    program_findings, _advisories = partition_program_contract_findings(
        kind,
        validate_program_contract(kind, program),
        execution_policy=execution_policy,
    )
    findings.extend(program_findings)

    target = instantiation_dir(kind, program, instance)
    parent_init = target.parent / "__init__.py"
    if not parent_init.is_file():
        findings.append(
            f"missing program instantiation package file: "
            f"{parent_init}"
        )
    elif parent_init.read_text(encoding="utf-8") != CANONICAL_PARENT_INIT:
        findings.append(f"noncanonical program instantiation package: {parent_init}")
    for filename in ("__init__.py", "binding.json", "main.py"):
        if not (target / filename).is_file():
            findings.append(f"missing instantiation file: {target / filename}")
    init_path = target / "__init__.py"
    main_path = target / "main.py"
    if init_path.is_file() and init_path.read_text(encoding="utf-8") != CANONICAL_INIT:
        findings.append(f"noncanonical instantiation entrypoint: {init_path}")
    if main_path.is_file():
        findings.extend(validate_instantiation_main(main_path))
    if target.is_dir():
        allowed = {"__init__.py", "binding.json", "main.py", "__pycache__"}
        transient_sandbox_metadata = {".agents", ".codex", ".git"}
        for path in target.iterdir():
            if path.name in allowed:
                continue
            if path.name in transient_sandbox_metadata and (
                path.is_dir() or path.is_symlink()
            ):
                continue
            findings.append(
                "unexpected executable or adapter file in manifest-only "
                f"instantiation: {path}"
            )
    if findings:
        return findings

    importlib.invalidate_caches()
    contract_path = program_dir(kind, program) / "binding_contract.py"
    package = "primitives" if kind == "primitive" else "strategies"
    contract_module_name = f"{package}.{program}.binding_contract"
    try:
        contract_module = importlib.import_module(contract_module_name)
    except Exception as exc:
        findings.append(
            f"could not import {contract_module_name}: {exc}"
        )
        return findings
    contract = getattr(contract_module, "BINDING_CONTRACT", None)
    if not isinstance(contract, BindingContract):
        findings.append(f"{contract_path} must publish BINDING_CONTRACT")
        return findings
    try:
        binding = load_binding_manifest(target / "binding.json", contract=contract)
    except ManifestError as exc:
        findings.append(str(exc))
        return findings
    if binding.instance_name != instance:
        findings.append(
            f"manifest instance_name is {binding.instance_name!r}; "
            f"expected {instance!r}"
        )
    if expected_scene_module is not None and binding.scene_module != expected_scene_module:
        findings.append(
            f"manifest scene_module is {binding.scene_module!r}; "
            f"expected {expected_scene_module!r}"
        )
    if expected_goal is not None and binding.goal_text != expected_goal:
        findings.append("manifest goal_text does not preserve supplied goal metadata")
    if findings:
        return findings
    try:
        scene = load_scene_module(binding.scene_module)
        model = scene.make_model()
        invariant_checks = tuple(getattr(contract_module, "INVARIANT_CHECKS", ()))
        report = validate_scene_binding(
            model,
            contract,
            binding,
            invariant_checks=invariant_checks,
        )
    except Exception as exc:
        findings.append(f"could not validate manifest against scene: {exc}")
        return findings
    if not report.binding_valid:
        detail = "; ".join(issue.message for issue in report.issues)
        findings.append(f"binding validation failed: {detail}")
    return findings


def sandboxed_instantiation_findings(
    kind: str,
    program: str,
    instance: str,
    *,
    scene_module: str,
    goal: str,
) -> list[str]:
    """Validate one binding in the launcher sandbox."""

    return run_sandboxed_validation(
        {
            "check": "instantiation_layout",
            "kind": kind,
            "program": program,
            "instance": instance,
            "expected_scene_module": scene_module,
            "expected_goal": goal,
        }
    )


def report_instantiation_validation(
    kind: str,
    program: str,
    instance: str,
    *,
    learned_before: dict[str, str],
    run_dir: Path,
    scene_module: str,
    goal: str,
    fatal: bool = True,
) -> list[str]:
    """Report binding/layout findings and learned-source provenance."""

    findings = sandboxed_instantiation_findings(
        kind,
        program,
        instance,
        scene_module=scene_module,
        goal=goal,
    )
    digest_record = run_dir / "learned_program_digests.json"
    changed = record_learned_source_digests(digest_record, learned_before)
    if changed:
        findings.append(
            "instantiation modified learned/runtime source tree(s): "
            + ", ".join(changed)
        )
    if not findings:
        print(
            f"[instantiate_{kind}_exec] instantiation layout and binding "
            "validation passed"
        )
        return findings
    disposition = "failed" if fatal else "advisory findings"
    print(
        f"[instantiate_{kind}_exec] validation {disposition}:",
        file=sys.stderr,
    )
    for finding in findings:
        print(f"  - {finding}", file=sys.stderr)
    if fatal:
        raise SystemExit(1)
    return findings


_REQUIRED_PHYSICAL_GENERALIZATION_DIMENSIONS = {
    "object_size",
    "object_pose",
    "object_mass",
    "contact_friction",
}


def validate_generalization_collection(
    specifications: list[SceneVariantSpec],
    physical_fingerprints: list[str],
    runtime_fingerprints: list[str],
    *,
    num_variants: int,
    goal_parameter_names: Sequence[str] = (),
    require_minimum: bool = True,
) -> list[str]:
    """Validate contract-relevant coverage and runtime uniqueness."""

    findings: list[str] = []
    if require_minimum and num_variants < 2:
        findings.append(
            "invention freezing requires at least two geometry/pose variants"
        )
    expected = num_variants + 1
    if len(specifications) != expected:
        findings.append(
            f"expected {expected} generalization specifications; "
            f"found {len(specifications)}"
        )
        return findings
    if len(physical_fingerprints) != expected or len(runtime_fingerprints) != expected:
        findings.append("every example must produce physical and runtime fingerprints")
        return findings

    baseline = specifications[0]
    if baseline.variant_index != 0:
        findings.append("example_0 VARIANT_SPEC.variant_index must be 0")
    if baseline.changed_dimensions:
        findings.append("example_0 must preserve the assigned baseline")

    covered: set[str] = set()
    covered_goal_parameters: set[str] = set()
    declared_goal_parameters = set(map(str, goal_parameter_names))
    baseline_goal_parameters = {
        str(key): to_json_value(value)
        for key, value in baseline.goal_parameters.items()
    }
    if set(baseline_goal_parameters) != declared_goal_parameters:
        findings.append(
            "example_0 goal_parameters must exactly cover the primitive contract: "
            f"expected {sorted(declared_goal_parameters)}, "
            f"found {sorted(baseline_goal_parameters)}"
        )
    if baseline.varied_goal_parameters:
        findings.append("example_0 varied_goal_parameters must be empty")
    same_goal_variant = False
    yaw_values: list[float] = []
    for index, specification in enumerate(specifications[1:], start=1):
        if specification.variant_index != index:
            findings.append(
                f"example_{index} VARIANT_SPEC.variant_index is "
                f"{specification.variant_index}; expected {index}"
            )
        physical = set(specification.changed_physical_dimensions)
        goal_parameters = {
            str(key): to_json_value(value)
            for key, value in specification.goal_parameters.items()
        }
        if set(goal_parameters) != declared_goal_parameters:
            findings.append(
                f"example_{index} goal_parameters must exactly cover the "
                f"primitive contract: expected {sorted(declared_goal_parameters)}, "
                f"found {sorted(goal_parameters)}"
            )
        actually_varied = {
            key
            for key in declared_goal_parameters
            if goal_parameters.get(key) != baseline_goal_parameters.get(key)
        }
        stated_varied = set(specification.varied_goal_parameters)
        if stated_varied != actually_varied:
            findings.append(
                f"example_{index} varied_goal_parameters must be "
                f"{sorted(actually_varied)}; found {sorted(stated_varied)}"
            )
        if not actually_varied:
            same_goal_variant = True
        changed = set(physical)
        if actually_varied:
            changed.add("goal")
        covered.update(changed)
        if len(changed) < 3:
            findings.append(
                f"example_{index} changes {len(changed)} dimensions; expected at least 3"
            )
        if len(physical) < 2:
            findings.append(
                f"example_{index} changes {len(physical)} physical dimensions; "
                "expected at least 2"
            )
        if specification.target_yaw is not None:
            yaw_values.append(float(specification.target_yaw))
        else:
            findings.append(f"example_{index} must vary target yaw")
        covered_goal_parameters.update(actually_varied)
        for key, value in specification.goal_parameters.items():
            try:
                json.dumps(to_json_value(value))
            except TypeError:
                findings.append(
                    f"example_{index} goal parameter {key!r} is not JSON serializable"
                )

    missing = sorted(_REQUIRED_PHYSICAL_GENERALIZATION_DIMENSIONS - covered)
    if missing and num_variants > 0:
        findings.append(
            "generalization set does not cover required dimensions: "
            + ", ".join(missing)
        )
    missing_goal_parameters = sorted(
        declared_goal_parameters - covered_goal_parameters
    )
    if missing_goal_parameters:
        findings.append(
            "generalization set does not vary declared primitive goal "
            "parameter(s): " + ", ".join(missing_goal_parameters)
        )
    if declared_goal_parameters and num_variants >= 2 and not same_goal_variant:
        findings.append(
            "generalization set requires a geometry variant that preserves the "
            "baseline primitive goal parameters"
        )
    if num_variants > 1 and len({round(value, 12) for value in yaw_values}) < 2:
        findings.append("generalization set must contain distinct target yaw values")
    if len(set(physical_fingerprints)) != len(physical_fingerprints):
        findings.append("generated examples contain duplicate physical scene fingerprints")
    if len(set(runtime_fingerprints)) != len(runtime_fingerprints):
        findings.append("generated examples contain duplicate runtime fingerprints")
    return findings


def validate_generated_generalization_layout(
    kind: str,
    program: str,
    num_variants: int,
    *,
    require_minimum: bool = True,
) -> list[str]:
    """Validate every generated example and its concrete derived scene."""

    findings: list[str] = []
    specifications: list[SceneVariantSpec] = []
    physical_fingerprints: list[str] = []
    runtime_fingerprints: list[str] = []
    goal_parameter_names: tuple[str, ...] = ()
    for index in range(num_variants + 1):
        instance = f"example_{index}"
        findings.extend(validate_instantiation_layout(kind, program, instance))
        target = WORKSPACE / "test_scenes" / program / instance
        scene_path = target / "scene.py"
        manifest_path = target / "variant.json"
        if not scene_path.is_file():
            findings.append(f"missing generated test scene: {scene_path}")
            continue
        if not manifest_path.is_file():
            findings.append(f"missing generalization manifest: {manifest_path}")
            continue

        try:
            package = "primitives" if kind == "primitive" else "strategies"
            contract_module = importlib.import_module(
                f"{package}.{program}.binding_contract"
            )
            contract = getattr(contract_module, "BINDING_CONTRACT")
            goal_parameter_names = tuple(
                requirement.name for requirement in contract.parameters
            )
            binding = load_binding_manifest(
                instantiation_dir(kind, program, instance) / "binding.json",
                contract=contract,
            )
        except Exception as exc:
            findings.append(f"could not load {instance} binding: {exc}")
            continue
        expected_scene_module = f"test_scenes.{program}.{instance}.scene"
        if binding.scene_module != expected_scene_module:
            findings.append(
                f"{instance} manifest scene_module is {binding.scene_module!r}; "
                f"expected {expected_scene_module!r}"
            )
            continue
        try:
            metadata = generalization_metadata(binding)
        except Exception as exc:
            findings.append(f"could not fingerprint {instance}: {exc}")
            continue
        if metadata is None:
            findings.append(
                f"{expected_scene_module} must export VARIANT_SPEC: SceneVariantSpec"
            )
            continue
        specification_module = importlib.import_module(expected_scene_module)
        specification = getattr(specification_module, "VARIANT_SPEC")
        specifications.append(specification)
        physical_fingerprints.append(str(metadata["physical_fingerprint"]))
        runtime_fingerprints.append(str(metadata["runtime_fingerprint"]))
        if index == 0:
            expected_model = make_variant_model(specification)
            expected_data = make_variant_data(specification, expected_model)
            expected_baseline = runtime_scene_fingerprint(
                expected_model,
                expected_data,
            )
            if metadata["physical_fingerprint"] != expected_baseline:
                findings.append(
                    "example_0 runtime scene does not preserve its assigned baseline"
                )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            findings.append(f"invalid {manifest_path}: {exc}")
            continue
        expected_manifest = specification.to_dict()
        if manifest != expected_manifest:
            findings.append(
                f"{manifest_path} does not match VARIANT_SPEC.to_dict()"
            )
        specified_goal = {
            str(key): to_json_value(value)
            for key, value in specification.goal_parameters.items()
        }
        bound_goal = {
            str(key): to_json_value(value)
            for key, value in binding.parameters.items()
        }
        if bound_goal != specified_goal:
            findings.append(
                f"{instance} binding parameters must exactly match "
                "VARIANT_SPEC.goal_parameters"
            )
        for diagnostic in validate_variant_initial_state(specification):
            findings.append(f"{instance}: {diagnostic}")

        metrics_path = WORKSPACE / "results" / f"{program}_{instance}_metrics.json"
        video_path = WORKSPACE / "results" / f"{program}_{instance}.mp4"
        if not metrics_path.is_file():
            findings.append(f"missing retained metrics: {metrics_path}")
        else:
            try:
                metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                retained = metrics_payload.get("metrics", {}).get("generalization")
                if retained != metadata:
                    findings.append(
                        f"{metrics_path} does not retain current generalization metadata"
                    )
                diagnostics = metrics_payload.get("metrics", {})
                required_diagnostics = {
                    "nominal_success",
                    "nominal_cost",
                    "optimized_cost",
                    "optimizer_improvement",
                    "nominal_to_optimized_control_rms",
                    "nominal_to_optimized_control_max",
                    "effective_horizon",
                    "num_iters",
                    "failure_signature",
                }
                if diagnostics.get("controller_rollout_backend") == (
                    "native_cpp_fr3_osc"
                ):
                    required_diagnostics.update(
                        {
                            "whole_arm_audit_valid",
                            "trajectory_audit_profile",
                            "retained_minimum_horizon",
                            "retained_timing_valid",
                            "timing_limited_candidate_segments",
                            "minimum_environment_clearance",
                            "minimum_enforced_environment_clearance",
                            "minimum_declared_proximity_clearance",
                            "self_collision_steps",
                            "unapproved_contact_steps",
                            "dominant_unapproved_contact_pair",
                            "gripper_intent",
                            "gripper_optimizer_channel_fixed",
                            "commanded_aperture_range",
                            "actual_aperture_range",
                            "commanded_aperture_total_variation",
                            "actual_aperture_total_variation",
                            "maximum_aperture_tracking_error",
                            "aperture_motion_during_target_contact_steps",
                            "commanded_aperture_motion_during_target_contact_steps",
                            "actual_aperture_motion_during_target_contact_steps",
                            "maximum_commanded_aperture_motion_during_target_contact",
                            "maximum_actual_aperture_velocity_during_target_contact",
                            "retained_gripper_contact_audit_valid",
                        }
                    )
                missing_diagnostics = sorted(
                    required_diagnostics - set(diagnostics)
                )
                if missing_diagnostics:
                    findings.append(
                        f"{metrics_path} is missing nominal diagnostics: "
                        + ", ".join(missing_diagnostics)
                    )
                if not bool(diagnostics.get("success")):
                    findings.append(
                        f"{metrics_path} does not report optimized predicate success"
                    )
            except (OSError, json.JSONDecodeError, AttributeError) as exc:
                findings.append(f"invalid retained metrics {metrics_path}: {exc}")
        if not video_path.is_file() or video_path.stat().st_size == 0:
            findings.append(f"missing retained video: {video_path}")

    findings.extend(
        validate_generalization_collection(
            specifications,
            physical_fingerprints,
            runtime_fingerprints,
            num_variants=num_variants,
            goal_parameter_names=goal_parameter_names,
            require_minimum=require_minimum,
        )
    )
    return findings


def report_generated_generalization_layout(
    kind: str,
    program: str,
    num_variants: int,
    *,
    fatal: bool = True,
) -> list[str]:
    """Run complete invention-suite validation inside the sandbox."""

    findings = run_sandboxed_validation(
        {
            "check": "generalization_layout",
            "kind": kind,
            "program": program,
            "num_variants": num_variants,
        }
    )
    if not findings:
        print(
            f"[invent_{kind}_exec] {num_variants + 1} generalization "
            "examples validated"
        )
        return findings
    disposition = "failed" if fatal else "advisory findings"
    print(
        f"[invent_{kind}_exec] generalization validation {disposition}:",
        file=sys.stderr,
    )
    for finding in findings:
        print(f"  - {finding}", file=sys.stderr)
    if fatal:
        raise SystemExit(1)
    return findings


__all__ = [
    "STRATEGY_TRANSIT_EXECUTION_FINDING_PREFIX",
    "digest_tree",
    "learned_source_changes",
    "learned_source_digests",
    "partition_program_contract_findings",
    "record_learned_source_digests",
    "report_program_contract_validation",
    "report_generated_generalization_layout",
    "report_instantiation_validation",
    "sandboxed_instantiation_findings",
    "validate_instantiation_layout",
    "validate_generated_generalization_layout",
    "validate_generalization_collection",
    "validate_program_contract",
]
