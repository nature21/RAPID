"""Post-run validation for generated scene-agnostic strategies."""

from __future__ import annotations

import ast
import sys

from .codex_exec import STRATEGIES_DIR
from .instantiation_validation import validate_program_contract
from .sandbox_validation import run_sandboxed_validation


def _validate_strategy_layout(strategy_name: str) -> list[str]:
    findings: list[str] = []
    strategy = STRATEGIES_DIR / strategy_name
    if not strategy.is_dir():
        return [f"missing strategy package: {strategy}"]
    for filename in (
        "__init__.py",
        "binding_contract.py",
        "strategy_description.md",
        "program.py",
        "predicate.py",
        "main.py",
    ):
        if not (strategy / filename).is_file():
            findings.append(f"missing canonical file: {strategy / filename}")
    description_path = strategy / "strategy_description.md"
    if description_path.is_file() and "## Phase signature audit" not in (
        description_path.read_text(encoding="utf-8")
    ):
        findings.append(
            f"{description_path} must contain a ## Phase signature audit section"
        )
    main_path = strategy / "main.py"
    if main_path.is_file():
        try:
            tree = ast.parse(main_path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            findings.append(f"could not parse {main_path}: {exc}")
        else:
            functions = {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if "compile_strategy" not in functions:
                findings.append(
                    f"{main_path} must define compile_strategy(binding, args)"
                )
            else:
                compile_node = next(
                    node
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "compile_strategy"
                )
                positional = [
                    argument.arg
                    for argument in (
                        compile_node.args.posonlyargs + compile_node.args.args
                    )
                ]
                if positional[:2] != ["binding", "args"]:
                    findings.append(
                        f"{main_path} compile_strategy must begin with "
                        "(binding, args)"
                    )
                invokes_shared_executor = any(
                    isinstance(node, ast.Call)
                    and (
                        isinstance(node.func, ast.Name)
                        and node.func.id == "execute_sequential_strategy"
                        or isinstance(node.func, ast.Attribute)
                        and node.func.attr == "execute_sequential_strategy"
                    )
                    for node in ast.walk(compile_node)
                )
                if not invokes_shared_executor:
                    findings.append(
                        f"{main_path} compile_strategy must invoke "
                        "execute_sequential_strategy()"
                    )
                policy_builds = [
                    call
                    for call in ast.walk(compile_node)
                    if isinstance(call, ast.Call)
                    and (
                        isinstance(call.func, ast.Name)
                        and call.func.id == "build_config"
                        or isinstance(call.func, ast.Attribute)
                        and call.func.attr == "build_config"
                    )
                    and any(
                        keyword.arg == "learned_policy"
                        for keyword in call.keywords
                    )
                ]
                if not policy_builds:
                    findings.append(
                        f"{main_path} compile_strategy must build each primitive "
                        "phase config from its learned policy"
                    )
                for call in policy_builds:
                    requested_values = [
                        keyword.value
                        for keyword in call.keywords
                        if keyword.arg == "requested_budget"
                    ]
                    uses_phase_budget = any(
                        any(
                            isinstance(reference, ast.Attribute)
                            and reference.attr == "optimization_budget"
                            for reference in ast.walk(value)
                        )
                        for value in requested_values
                    )
                    if not uses_phase_budget:
                        findings.append(
                            f"{main_path} learned-policy build_config calls must "
                            "pass requested_budget from the current "
                            "PrimitivePhase.optimization_budget"
                        )
            if "make_parser" not in functions:
                findings.append(f"{main_path} must define make_parser()")
    program_path = strategy / "program.py"
    if program_path.is_file():
        source = program_path.read_text(encoding="utf-8")
        if "PHASE_PROGRAM" not in source or "StrategyProgram" not in source:
            findings.append(
                f"{program_path} must publish PHASE_PROGRAM: StrategyProgram"
            )
    predicate_path = strategy / "predicate.py"
    if predicate_path.is_file():
        try:
            tree = ast.parse(predicate_path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            findings.append(f"could not parse {predicate_path}: {exc}")
        else:
            if not any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "evaluate_final_psi"
                for node in tree.body
            ):
                findings.append(
                    f"{predicate_path} must define evaluate_final_psi()"
                )
    findings.extend(validate_program_contract("strategy", strategy_name))
    return findings


def report_strategy_layout_validation(
    strategy_name: str, *, fatal: bool = True
) -> list[str]:
    findings = run_sandboxed_validation(
        {
            "check": "strategy_layout",
            "program": strategy_name,
        }
    )
    if not findings:
        print("[compose_strategy_exec] strategy layout validation passed")
        return findings
    disposition = "failed" if fatal else "advisory findings"
    print(
        f"[compose_strategy_exec] strategy layout validation {disposition}:",
        file=sys.stderr,
    )
    for finding in findings:
        print(f"  - {finding}", file=sys.stderr)
    if fatal:
        raise SystemExit(1)
    return findings


__all__ = ["_validate_strategy_layout", "report_strategy_layout_validation"]
