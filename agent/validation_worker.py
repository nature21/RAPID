"""Restricted subprocess entry point for dynamic generated-code validation."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from .sandbox_validation import RESULT_PREFIX


def dispatch_validation(request: dict[str, Any]) -> list[str]:
    """Dispatch one validation request inside the permission sandbox."""

    check = request.get("check")
    if check == "program_contract":
        from .instantiation_validation import validate_program_contract

        return validate_program_contract(
            str(request["kind"]),
            str(request["program"]),
        )
    if check == "instantiation_layout":
        from .instantiation_validation import validate_instantiation_layout

        return validate_instantiation_layout(
            str(request["kind"]),
            str(request["program"]),
            str(request["instance"]),
            expected_scene_module=request.get("expected_scene_module"),
            expected_goal=request.get("expected_goal"),
            execution_policy=os.environ.get(
                "RAPID_EXECUTION_POLICY",
                "strict",
            ).strip(),
        )
    if check == "primitive_layout":
        from .primitive_validation import _validate_primitive_layout

        findings: list[str] = []
        for program in request.get("programs", []):
            findings.extend(_validate_primitive_layout(str(program)))
        return findings
    if check == "generalization_layout":
        from .instantiation_validation import validate_generated_generalization_layout

        return validate_generated_generalization_layout(
            str(request["kind"]),
            str(request["program"]),
            int(request["num_variants"]),
            require_minimum=bool(request.get("require_minimum", True)),
        )
    if check == "strategy_layout":
        from .strategy_validation import _validate_strategy_layout

        return _validate_strategy_layout(str(request["program"]))
    raise ValueError(f"unknown validation check: {check!r}")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("validation worker expects one JSON request", file=sys.stderr)
        return 2
    try:
        request = json.loads(args[0])
        if not isinstance(request, dict):
            raise TypeError("validation request must be an object")
        findings = dispatch_validation(request)
        result = {"findings": findings}
    except BaseException as exc:
        result = {
            "findings": [
                f"validation worker failed: {type(exc).__name__}: {exc}"
            ]
        }
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["dispatch_validation", "main"]
