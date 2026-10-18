"""Canonical dispatcher for primitive and strategy instantiations."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from primitive_core.common import OptimizationBudget

from .contracts import BindingContract, validate_scene_binding
from .generalization import attach_generalization_metadata
from .manifest import load_binding_manifest
from .runner import (
    InstantiationSpec,
    apply_optimizer_env_overrides,
    cartesian_goals_path_for_metrics,
    run_instantiation,
)
from .scene import load_scene_module


def instantiation_results_dir(
    manifest_path: str | Path,
    *,
    program_kind: str,
    program_name: str,
    instance_name: str,
) -> Path:
    """Return the stable output directory for a concrete instantiation."""

    workspace_env = os.environ.get("RAPID_WORKSPACE", "").strip()
    if workspace_env:
        workspace = Path(workspace_env).expanduser().resolve()
    else:
        resolved = Path(manifest_path).expanduser().resolve()
        src = next(
            (parent for parent in resolved.parents if parent.name == "src"),
            None,
        )
        workspace = src.parent if src is not None else Path.cwd().resolve()
    plural = "primitives" if program_kind == "primitive" else "strategies"
    return workspace / "results_instantiation" / plural / program_name / instance_name


def _has_option(argv: Sequence[str], option: str) -> bool:
    return any(item == option or item.startswith(f"{option}=") for item in argv)


def run_manifest_entrypoint(
    manifest_path: str | Path,
    argv: Sequence[str] | None = None,
) -> object:
    """Validate a manifest and dispatch only to its frozen learned program."""

    preliminary = load_binding_manifest(manifest_path)
    package = "primitives" if preliminary.program_kind == "primitive" else "strategies"
    contract_module = importlib.import_module(
        f"{package}.{preliminary.program_name}.binding_contract"
    )
    contract = getattr(contract_module, "BINDING_CONTRACT", None)
    if not isinstance(contract, BindingContract):
        raise TypeError("learned program must publish BINDING_CONTRACT")
    binding = load_binding_manifest(manifest_path, contract=contract)

    scene_module = load_scene_module(binding.scene_module)
    model = scene_module.make_model()
    invariant_checks = tuple(getattr(contract_module, "INVARIANT_CHECKS", ()))
    report = validate_scene_binding(
        model,
        contract,
        binding,
        invariant_checks=invariant_checks,
    )
    report = attach_generalization_metadata(report, binding)
    report.require_valid()

    if binding.program_kind == "strategy":
        from strategies.program import StrategyProgram, validate_strategy_program

        phase_module = importlib.import_module(
            f"strategies.{binding.program_name}.program"
        )
        phase_program = getattr(phase_module, "PHASE_PROGRAM", None)
        if not isinstance(phase_program, StrategyProgram):
            raise TypeError("learned strategy must publish PHASE_PROGRAM")
        primitive_contracts: dict[str, BindingContract] = {}
        primitive_reference_budgets: dict[str, OptimizationBudget] = {}
        for phase in phase_program.phases:
            primitive_module = importlib.import_module(
                f"primitives.{phase.primitive}.binding_contract"
            )
            primitive_contract = getattr(primitive_module, "BINDING_CONTRACT", None)
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
            if not isinstance(reference_budget, OptimizationBudget):
                raise TypeError(
                    f"primitive {phase.primitive!r} must publish "
                    "REFERENCE_OPTIMIZATION_BUDGET"
                )
            primitive_reference_budgets[phase.primitive] = reference_budget
        phase_findings = validate_strategy_program(
            phase_program,
            contract,
            primitive_contracts,
            primitive_reference_budgets,
        )
        if phase_findings:
            raise ValueError(
                "learned strategy phase program is invalid: "
                + "; ".join(phase_findings)
            )

    learned_main = importlib.import_module(
        f"{package}.{binding.program_name}.main"
    )
    make_parser = getattr(learned_main, "make_parser", None)
    compile_name = (
        "compile_primitive" if binding.program_kind == "primitive" else "compile_strategy"
    )
    compile_program = getattr(learned_main, compile_name, None)
    if not callable(make_parser) or not callable(compile_program):
        raise TypeError(
            f"learned {binding.program_kind} must expose make_parser() and "
            f"{compile_name}(binding, args)"
        )
    parser = make_parser()
    command_argv = list(argv) if argv is not None else sys.argv[1:]
    output_dir = instantiation_results_dir(
        manifest_path,
        program_kind=binding.program_kind,
        program_name=binding.program_name,
        instance_name=binding.instance_name,
    )
    output_defaults: dict[str, Any] = {
        "metrics_json": str(output_dir / "metrics.json")
    }
    if not _has_option(command_argv, "--replay") and not _has_option(
        command_argv, "--no-record"
    ):
        output_defaults["record"] = str(output_dir / "trajectory.mp4")
    parser.set_defaults(**output_defaults)

    args = parser.parse_args(command_argv)
    if args.cartesian_goals_json is None and args.metrics_json is not None:
        args.cartesian_goals_json = str(
            cartesian_goals_path_for_metrics(args.metrics_json)
        )
    apply_optimizer_env_overrides(args)
    spec = compile_program(binding, args)
    if not isinstance(spec, InstantiationSpec):
        raise TypeError(f"{compile_name}() must return InstantiationSpec")
    spec.binding_report = report
    return run_instantiation(spec, args)


__all__ = [
    "instantiation_results_dir",
    "run_manifest_entrypoint",
]
