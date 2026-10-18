from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent import (
    primitive_context,
    primitive_validation,
    strategy_context,
    strategy_validation,
)


def _touch(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class PrimitiveLayoutTests(unittest.TestCase):
    @staticmethod
    def _make_primitive(primitive: Path) -> None:
        files = {
            "__init__.py": "",
            "binding_contract.py": "",
            "types.py": "",
            "features.py": "def resolve_geometry(context):\n    return {}\n",
            "nominal.py": (
                "OPTIMIZATION_POLICY = object()\n"
                "REFERENCE_OPTIMIZATION_BUDGET = object()\n\n"
                "def make_nominal_plan(context, features):\n    pass\n"
            ),
            "transit.py": "def default_transit_spec():\n    pass\n",
            "cost.py": (
                "EFFECT_TOLERANCES = {'error': 0.1}\n\n"
                "def effect_predicate(states, controls):\n    pass\n\n"
                "def effect_cost(states, controls):\n    pass\n"
            ),
            "compile.py": "def compile_effect():\n    make_nominal_plan()\n",
            "main.py": (
                "def make_parser():\n    pass\n\n"
                "def compile_primitive(binding, args):\n"
                "    return InstantiationSpec(\n"
                "        optimization_policy=OPTIMIZATION_POLICY,\n"
                "        reference_optimization_budget="
                "REFERENCE_OPTIMIZATION_BUDGET,\n"
                "    )\n"
            ),
            "primitive_description.md": (
                "## Semantic goal G\n\n"
                "## Mechanism M\n\n"
                "## Success predicate Psi\n\n"
                "## Progress representation P(t)\n\n"
                "## Role necessity audit\n"
            ),
        }
        for filename, contents in files.items():
            _touch(primitive / filename, contents)

    def test_primitive_validation_accepts_flat_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            primitives = Path(tmp) / "src/primitives"
            primitive = primitives / "push"
            self._make_primitive(primitive)

            with (
                mock.patch.object(primitive_validation, "PRIMITIVES_DIR", primitives),
                mock.patch.object(
                    primitive_validation,
                    "validate_program_contract",
                    return_value=[],
                ),
            ):
                findings = primitive_validation._validate_primitive_layout("push")

            self.assertEqual(findings, [])

    def test_primitive_validation_rejects_version_subpackage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            primitives = Path(tmp) / "src/primitives"
            primitive = primitives / "push"
            self._make_primitive(primitive)
            _touch(primitive / "v2/__init__.py")

            with (
                mock.patch.object(primitive_validation, "PRIMITIVES_DIR", primitives),
                mock.patch.object(
                    primitive_validation,
                    "validate_program_contract",
                    return_value=[],
                ),
            ):
                findings = primitive_validation._validate_primitive_layout("push")

            self.assertEqual(findings, ["unexpected version subpackage: primitives/push/v2"])

    def test_primitive_validation_requires_role_necessity_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            primitives = Path(tmp) / "src/primitives"
            primitive = primitives / "push"
            self._make_primitive(primitive)
            _touch(primitive / "primitive_description.md", "# Push\n")

            with (
                mock.patch.object(primitive_validation, "PRIMITIVES_DIR", primitives),
                mock.patch.object(
                    primitive_validation,
                    "validate_program_contract",
                    return_value=[],
                ),
            ):
                findings = primitive_validation._validate_primitive_layout("push")

        self.assertTrue(any("Role necessity audit" in item for item in findings))

    def test_primitive_validation_requires_semantic_model_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            primitives = Path(tmp) / "src/primitives"
            primitive = primitives / "primitive"
            self._make_primitive(primitive)
            _touch(
                primitive / "primitive_description.md",
                "## Role necessity audit\n",
            )

            with (
                mock.patch.object(primitive_validation, "PRIMITIVES_DIR", primitives),
                mock.patch.object(
                    primitive_validation,
                    "validate_program_contract",
                    return_value=[],
                ),
            ):
                findings = primitive_validation._validate_primitive_layout("primitive")

        for section in (
            "Semantic goal G",
            "Mechanism M",
            "Success predicate Psi",
            "Progress representation P(t)",
        ):
            self.assertTrue(any(section in item for item in findings))

    def test_primitive_validation_rejects_nominal_arrays_in_public_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            primitives = Path(tmp) / "src/primitives"
            primitive = primitives / "push"
            self._make_primitive(primitive)
            _touch(
                primitive / "types.py",
                "from dataclasses import dataclass\n\n"
                "@dataclass\nclass Invocation:\n"
                "    nominal_knots: object\n"
                "    sigma_init: object\n",
            )
            with (
                mock.patch.object(primitive_validation, "PRIMITIVES_DIR", primitives),
                mock.patch.object(
                    primitive_validation,
                    "validate_program_contract",
                    return_value=[],
                ),
            ):
                findings = primitive_validation._validate_primitive_layout("push")

        self.assertTrue(any("action-level nominal fields" in item for item in findings))

    def test_primitive_validation_requires_frozen_optimization_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            primitives = Path(tmp) / "src/primitives"
            primitive = primitives / "push"
            self._make_primitive(primitive)
            _touch(
                primitive / "nominal.py",
                "def make_nominal_plan(context, features):\n    pass\n",
            )
            _touch(
                primitive / "main.py",
                "def make_parser():\n    pass\n\n"
                "def compile_primitive(binding, args):\n    pass\n",
            )
            with (
                mock.patch.object(primitive_validation, "PRIMITIVES_DIR", primitives),
                mock.patch.object(
                    primitive_validation,
                    "validate_program_contract",
                    return_value=[],
                ),
            ):
                findings = primitive_validation._validate_primitive_layout("push")

        self.assertTrue(any("OPTIMIZATION_POLICY" in item for item in findings))
        self.assertTrue(any("optimization_policy" in item for item in findings))
        self.assertTrue(
            any("REFERENCE_OPTIMIZATION_BUDGET" in item for item in findings)
        )
        self.assertTrue(
            any("reference_optimization_budget" in item for item in findings)
        )

    def test_layout_reports_can_return_advisory_findings(self) -> None:
        with mock.patch.object(
            primitive_validation,
            "run_sandboxed_validation",
            return_value=["primitive finding"],
        ):
            primitive_findings = (
                primitive_validation.report_primitive_layout_validation(
                    ["push"],
                    fatal=False,
                )
            )
        with mock.patch.object(
            strategy_validation,
            "run_sandboxed_validation",
            return_value=["strategy finding"],
        ):
            strategy_findings = (
                strategy_validation.report_strategy_layout_validation(
                    "strategy_1",
                    fatal=False,
                )
            )

        self.assertEqual(primitive_findings, ["primitive finding"])
        self.assertEqual(strategy_findings, ["strategy finding"])

    def test_primitive_discovery_returns_names_without_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            primitives = Path(tmp) / "src/primitives"
            _touch(primitives / "push/__init__.py")
            _touch(primitives / "tilt/__init__.py")
            _touch(primitives / "_private/__init__.py")

            with (
                mock.patch.object(primitive_context, "PRIMITIVES_DIR", primitives),
                mock.patch.object(strategy_context, "PRIMITIVES_DIR", primitives),
            ):
                self.assertEqual(
                    primitive_context.existing_primitives(),
                    {"push", "tilt", "_private"},
                )
                self.assertEqual(
                    strategy_context.list_existing_primitives(),
                    ["push", "tilt"],
                )

    def test_strategy_validation_requires_contract_and_binding_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            strategies = Path(tmp) / "src/strategies"
            strategy = strategies / "strategy_1"
            _touch(strategy / "__init__.py")
            _touch(strategy / "binding_contract.py")
            _touch(
                strategy / "strategy_description.md",
                "## Phase signature audit\n",
            )
            _touch(
                strategy / "program.py",
                "from strategies.program import StrategyProgram\n"
                "PHASE_PROGRAM = StrategyProgram\n",
            )
            _touch(
                strategy / "predicate.py",
                "def evaluate_final_psi(states):\n    return True\n",
            )
            _touch(
                strategy / "main.py",
                "def make_parser():\n    pass\n\n"
                "def execute_sequential_strategy():\n    pass\n\n"
                "def compile_strategy(binding, args):\n"
                "    phase = PHASE_PROGRAM.phases[0]\n"
                "    build_config(\n"
                "        Config,\n"
                "        args,\n"
                "        learned_policy=POLICY,\n"
                "        reference_budget=REFERENCE,\n"
                "        requested_budget=phase.optimization_budget,\n"
                "    )\n"
                "    return execute_sequential_strategy()\n",
            )
            with (
                mock.patch.object(
                    strategy_validation,
                    "STRATEGIES_DIR",
                    strategies,
                ),
                mock.patch.object(
                    strategy_validation,
                    "validate_program_contract",
                    return_value=[],
                ),
            ):
                findings = strategy_validation._validate_strategy_layout(
                    "strategy_1"
                )

            self.assertEqual(findings, [])

    def test_strategy_validation_requires_phase_signature_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            strategies = Path(tmp) / "src/strategies"
            strategy = strategies / "strategy_1"
            _touch(strategy / "__init__.py")
            _touch(strategy / "binding_contract.py")
            _touch(strategy / "strategy_description.md", "# Strategy\n")
            _touch(
                strategy / "program.py",
                "from strategies.program import StrategyProgram\n"
                "PHASE_PROGRAM = StrategyProgram\n",
            )
            _touch(
                strategy / "predicate.py",
                "def evaluate_final_psi(states):\n    return True\n",
            )
            _touch(
                strategy / "main.py",
                "def make_parser():\n    pass\n\n"
                "def execute_sequential_strategy():\n    pass\n\n"
                "def compile_strategy(binding, args):\n"
                "    return execute_sequential_strategy()\n",
            )
            with (
                mock.patch.object(
                    strategy_validation,
                    "STRATEGIES_DIR",
                    strategies,
                ),
                mock.patch.object(
                    strategy_validation,
                    "validate_program_contract",
                    return_value=[],
                ),
            ):
                findings = strategy_validation._validate_strategy_layout(
                    "strategy_1"
                )

        self.assertTrue(any("Phase signature audit" in item for item in findings))

    def test_strategy_validation_requires_phase_budget_wiring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            strategies = Path(tmp) / "src/strategies"
            strategy = strategies / "strategy_1"
            _touch(strategy / "__init__.py")
            _touch(strategy / "binding_contract.py")
            _touch(
                strategy / "strategy_description.md",
                "## Phase signature audit\n",
            )
            _touch(
                strategy / "program.py",
                "from strategies.program import StrategyProgram\n"
                "PHASE_PROGRAM = StrategyProgram\n",
            )
            _touch(
                strategy / "predicate.py",
                "def evaluate_final_psi(states):\n    return True\n",
            )
            _touch(
                strategy / "main.py",
                "def make_parser():\n    pass\n\n"
                "def execute_sequential_strategy():\n    pass\n\n"
                "def compile_strategy(binding, args):\n"
                "    build_config(\n"
                "        Config,\n"
                "        args,\n"
                "        learned_policy=POLICY,\n"
                "        reference_budget=REFERENCE,\n"
                "    )\n"
                "    return execute_sequential_strategy()\n",
            )
            with (
                mock.patch.object(
                    strategy_validation,
                    "STRATEGIES_DIR",
                    strategies,
                ),
                mock.patch.object(
                    strategy_validation,
                    "validate_program_contract",
                    return_value=[],
                ),
            ):
                findings = strategy_validation._validate_strategy_layout(
                    "strategy_1"
                )

        self.assertTrue(
            any("PrimitivePhase.optimization_budget" in item for item in findings)
        )


if __name__ == "__main__":
    unittest.main()
