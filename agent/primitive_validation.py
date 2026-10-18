"""Post-run validation for generated primitives."""

from __future__ import annotations

import ast
import sys

from .codex_exec import PRIMITIVES_DIR
from .instantiation_validation import validate_program_contract
from .sandbox_validation import run_sandboxed_validation


def _validate_primitive_layout(primitive_name: str) -> list[str]:
    src_dir = PRIMITIVES_DIR.parent
    findings: list[str] = []
    primitive = PRIMITIVES_DIR / primitive_name

    if not primitive.is_dir():
        findings.append(
            f"missing canonical primitive implementation: {primitive.relative_to(src_dir)}"
        )
    for filename in (
        "__init__.py",
        "binding_contract.py",
        "types.py",
        "features.py",
        "nominal.py",
        "transit.py",
        "cost.py",
        "compile.py",
        "main.py",
        "primitive_description.md",
    ):
        if not (primitive / filename).is_file():
            findings.append(
                f"missing canonical file: {(primitive / filename).relative_to(src_dir)}"
            )

    version_dirs = (
        [
            path
            for path in primitive.iterdir()
            if path.is_dir() and path.name.startswith("v")
        ]
        if primitive.is_dir()
        else []
    )
    for version_dir in version_dirs:
        findings.append(
            f"unexpected version subpackage: {version_dir.relative_to(src_dir)}"
        )

    description_path = primitive / "primitive_description.md"
    if description_path.is_file():
        description = description_path.read_text(encoding="utf-8")
        for section in (
            "Semantic goal G",
            "Mechanism M",
            "Success predicate Psi",
            "Progress representation P(t)",
            "Role necessity audit",
        ):
            heading = f"## {section}"
            if heading not in description:
                findings.append(
                    f"{description_path} must contain a {heading} section"
                )

    required_functions = {
        "features.py": "resolve_geometry",
        "nominal.py": "make_nominal_plan",
        "transit.py": "default_transit_spec",
        "cost.py": "effect_predicate",
        "main.py": "compile_primitive",
    }
    for filename, function_name in required_functions.items():
        path = primitive / filename
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            findings.append(f"could not parse {path}: {exc}")
            continue
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if function_name not in functions:
            findings.append(f"{path} must define {function_name}()")
        if filename == "cost.py":
            if "effect_cost" not in functions:
                findings.append(f"{path} must define effect_cost()")
            assigned = {
                target.id
                for node in tree.body
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                if isinstance(target, ast.Name)
            }
            if "EFFECT_TOLERANCES" not in assigned:
                findings.append(f"{path} must publish EFFECT_TOLERANCES")
        if filename == "nominal.py":
            assigned = {
                target.id
                for node in tree.body
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                if isinstance(target, ast.Name)
            }
            if "OPTIMIZATION_POLICY" not in assigned:
                findings.append(f"{path} must publish OPTIMIZATION_POLICY")
            if "REFERENCE_OPTIMIZATION_BUDGET" not in assigned:
                findings.append(
                    f"{path} must publish REFERENCE_OPTIMIZATION_BUDGET"
                )
        if filename == "main.py" and function_name in functions:
            node = functions[function_name]
            positional = [
                argument.arg
                for argument in node.args.posonlyargs + node.args.args
            ]
            if positional[:2] != ["binding", "args"]:
                findings.append(
                    f"{path} compile_primitive must begin with (binding, args)"
                )
            if "make_parser" not in functions:
                findings.append(f"{path} must define make_parser()")
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
                    f"{path} must pass OPTIMIZATION_POLICY as "
                    "InstantiationSpec.optimization_policy"
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
                    f"{path} must pass REFERENCE_OPTIMIZATION_BUDGET as "
                    "InstantiationSpec.reference_optimization_budget"
                )

    compile_path = primitive / "compile.py"
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
                f"{compile_path} must invoke the learned make_nominal_plan policy"
            )
    types_path = primitive / "types.py"
    if types_path.is_file():
        try:
            tree = ast.parse(types_path.read_text(encoding="utf-8"))
        except SyntaxError:
            pass
        else:
            forbidden = {
                "nominal_actions",
                "nominal_knots",
                "sigma_actions",
                "sigma_init",
                "waypoint_actions",
                "waypoints",
            }
            declared = {
                node.target.id
                for node in ast.walk(tree)
                if isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
            }
            leaked = sorted(forbidden & declared)
            if leaked:
                findings.append(
                    f"{types_path} leaks action-level nominal fields: {leaked}"
                )

    if primitive.is_dir():
        findings.extend(validate_program_contract("primitive", primitive_name))
    return findings


def report_primitive_layout_validation(
    primitive_names: list[str], *, fatal: bool = True
) -> list[str]:
    findings = run_sandboxed_validation(
        {
            "check": "primitive_layout",
            "programs": primitive_names,
        }
    )

    if not findings:
        print("[construct_primitive_exec] primitive layout validation passed")
        return findings

    disposition = "failed" if fatal else "advisory findings"
    print(
        f"[construct_primitive_exec] primitive layout validation {disposition}:",
        file=sys.stderr,
    )
    for finding in findings:
        print(f"  - {finding}", file=sys.stderr)
    if fatal:
        raise SystemExit(1)
    return findings
