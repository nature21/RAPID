from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent import deploy_strategy_exec
from agent.deployment_context import (
    DEPLOY_RESPONSE_SCHEMA_VERSION,
    build_deployment_prompt,
    compact_scene_inventory,
    parse_deployment_binding_decision,
    serialize_phase_program,
    serialize_strategy_contract,
)
from agent.instantiation_context import CANONICAL_MAIN
from instantiation_core import (
    BindingContract,
    EntityKind,
    RelationKind,
    RelationalInvariant,
    RoleRequirement,
)
from strategies.program import ConnectingConstraint, PrimitivePhase, StrategyProgram


def _response(*, roles: dict[str, object] | None = None) -> str:
    return json.dumps(
        {
            "schema_version": DEPLOY_RESPONSE_SCHEMA_VERSION,
            "compatible": True,
            "roles": roles or {"target_body": "box"},
            "diagnostics": [],
        }
    )


class DeploymentContextTests(unittest.TestCase):
    def test_strict_response_parser_accepts_only_semantic_roles(self) -> None:
        decision = parse_deployment_binding_decision(
            "```json\n"
            + _response(
                roles={
                    "target_body": "box",
                    "obstacles": ["wall", "divider"],
                }
            )
            + "\n```"
        )

        self.assertTrue(decision.compatible)
        self.assertEqual(decision.roles["target_body"], "box")
        self.assertEqual(decision.roles["obstacles"], ["wall", "divider"])
        self.assertEqual(decision.diagnostics, ())

    def test_strict_response_parser_rejects_deployment_tuning(self) -> None:
        payload = json.loads(_response())
        payload["optimizer"] = {"num_rollouts": 8}

        with self.assertRaisesRegex(ValueError, r"unknown=\['optimizer'\]"):
            parse_deployment_binding_decision(json.dumps(payload))

    def test_incompatible_response_requires_diagnostics_and_no_roles(self) -> None:
        payload = {
            "schema_version": DEPLOY_RESPONSE_SCHEMA_VERSION,
            "compatible": False,
            "roles": {},
            "diagnostics": ["No fixture is present."],
        }

        decision = parse_deployment_binding_decision(json.dumps(payload))

        self.assertFalse(decision.compatible)
        self.assertEqual(decision.diagnostics, ("No fixture is present.",))

    def test_context_contains_contract_and_declarative_phase_semantics(self) -> None:
        contract = BindingContract(
            program_kind="strategy",
            program_name="strategy_1",
            roles=(
                RoleRequirement(
                    role="target_body",
                    kind=EntityKind.BODY,
                    description="Object to manipulate.",
                ),
                RoleRequirement(
                    role="fixture_body",
                    kind=EntityKind.BODY,
                    description="Fixture used by the manipulation.",
                ),
            ),
            invariants=("The target must be movable.",),
            relations=(
                RelationalInvariant(
                    relation=RelationKind.DIFFERENT_BODIES,
                    left_role="target_body",
                    right_role="fixture_body",
                    description="The fixture and target are distinct.",
                ),
            ),
        )
        program = StrategyProgram(
            phases=(
                PrimitivePhase(
                    name="push_phase",
                    primitive="push",
                    role_map={"target_body": "target_body"},
                    connecting_constraints=(
                        ConnectingConstraint(
                            primitive_input="displacement",
                            description="Move toward the fixture.",
                            depends_on_roles=("target_body", "fixture_body"),
                            output_frame="world",
                            resolver=lambda _context: [0.1, 0.0, 0.0],
                        ),
                    ),
                ),
            ),
        )

        serialized_contract = serialize_strategy_contract(contract)
        serialized_phases = serialize_phase_program(program)

        self.assertNotIn("parameters", serialized_contract)
        self.assertEqual(
            serialized_contract["relations"][0]["relation"],
            "different_bodies",
        )
        self.assertEqual(serialized_phases[0]["primitive"], "push")
        self.assertEqual(
            serialized_phases[0]["connecting_constraints"][0]["primitive_input"],
            "displacement",
        )
        self.assertNotIn(
            "resolver",
            serialized_phases[0]["connecting_constraints"][0],
        )

    def test_scene_inventory_omits_runtime_state_and_unused_entity_kinds(self) -> None:
        inventory = {
            "summary": {"nbody": 2, "ngeom": 1},
            "bodies": [
                {
                    "name": "box",
                    "parent": "world",
                    "joints": ["box_joint"],
                    "position": [0.1, 0.2, 0.3],
                }
            ],
            "joints": [{"name": "box_joint", "type": "mjJNT_FREE"}],
            "geoms": [
                {
                    "name": "box_geom",
                    "body": "box",
                    "type": "mjGEOM_BOX",
                    "size": [0.1, 0.2, 0.3],
                    "contype": 1,
                    "conaffinity": 1,
                },
                {
                    "name": "visual_mesh",
                    "body": "box",
                    "type": "mjGEOM_MESH",
                    "size": [0.1, 0.2, 0.3],
                    "contype": 0,
                    "conaffinity": 0,
                },
            ],
            "sites": [{"name": "unused_site", "body": "box"}],
        }

        compact = compact_scene_inventory(
            inventory,
            required_kinds={"body", "geom"},
            type_filters={"geom": {"box", "plane"}},
        )

        self.assertEqual(compact["bodies"][0]["joints"][0]["type"], "free")
        self.assertEqual(compact["geoms"][0]["type"], "box")
        self.assertEqual(len(compact["geoms"]), 1)
        self.assertNotIn("position", compact["bodies"][0])
        self.assertNotIn("sites", compact)

    def test_prompt_requires_skill_and_forbids_source_inspection(self) -> None:
        prompt = build_deployment_prompt(
            strategy="strategy_1",
            scene_module="tasks.example.scene",
            instance="held_out",
            goal="Put the box in the corridor.",
            context={"strategy_contract": {"roles": []}},
            has_view=False,
        )

        self.assertIn("$rapid-deploy-strategy", prompt)
        self.assertIn("Put the box in the corridor.", prompt)
        self.assertIn("Do not inspect workspace files", prompt)


class DeployStrategyExecTests(unittest.TestCase):
    def test_host_writes_role_only_binding_with_frozen_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "instantiations/strategies/strategy_1/scene_5"
            manifest = deploy_strategy_exec._write_deployment_package(
                destination,
                strategy="strategy_1",
                instance="scene_5",
                scene_module="tasks.invention_real2sim.scene_5",
                goal="Grasp the cookie box.",
                roles={"target_body": "cookie_box"},
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))

            self.assertEqual(payload["roles"], {"target_body": "cookie_box"})
            self.assertEqual(payload["inputs"], {})
            self.assertEqual(
                (destination / "main.py").read_text(encoding="utf-8"),
                CANONICAL_MAIN,
            )

    def test_bind_only_uses_one_model_decision_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            strategies = root / "src/strategies"
            (strategies / "strategy_1").mkdir(parents=True)
            work = root / "agent_work"
            destination = root / "src/instantiations/strategies/strategy_1/scene_5"
            model_calls: list[str] = []

            def fake_codex_exec(prompt, config, *, resume_thread_id=None):
                del resume_thread_id
                model_calls.append(prompt)
                assert config.output_last_message is not None
                config.output_last_message.write_text(
                    _response(roles={"target_body": "cookie_box"}),
                    encoding="utf-8",
                )
                return "thread-1"

            def fake_view(_scene_module: str, run_dir: Path) -> Path:
                view = run_dir / "view.png"
                view.write_bytes(b"png")
                return view

            context_results = [
                {"ok": True, "context": {"strategy_contract": {"roles": []}}},
                {"ok": True, "findings": []},
            ]
            with (
                mock.patch.object(deploy_strategy_exec, "STRATEGIES_DIR", strategies),
                mock.patch.object(deploy_strategy_exec, "AGENT_WORK_DIR", work),
                mock.patch.object(
                    deploy_strategy_exec,
                    "validate_env_modules_importable",
                    return_value=[],
                ),
                mock.patch.object(
                    deploy_strategy_exec,
                    "stage_workspace_codex_guidance",
                    return_value=[],
                ),
                mock.patch.object(
                    deploy_strategy_exec,
                    "_run_context_worker",
                    side_effect=context_results,
                ),
                mock.patch.object(
                    deploy_strategy_exec,
                    "_prepare_initial_view",
                    side_effect=fake_view,
                ),
                mock.patch.object(
                    deploy_strategy_exec,
                    "run_codex_exec",
                    side_effect=fake_codex_exec,
                ),
                mock.patch.object(
                    deploy_strategy_exec,
                    "instantiation_dir",
                    return_value=destination,
                ),
                mock.patch.object(
                    deploy_strategy_exec,
                    "sandboxed_instantiation_findings",
                    return_value=[],
                ),
                mock.patch.object(deploy_strategy_exec, "_run_frozen_strategy") as run,
            ):
                returncode = deploy_strategy_exec.run(
                    [
                        "--strategy",
                        "strategy_1",
                        "--env-module",
                        "tasks.invention_real2sim.scene_5",
                        "--name",
                        "scene_5",
                        "--goal",
                        "Grasp the cookie box.",
                        "--bind-only",
                    ]
                )

            self.assertEqual(returncode, 0)
            self.assertEqual(len(model_calls), 1)
            run.assert_not_called()
            self.assertTrue((destination / "binding.json").is_file())


if __name__ == "__main__":
    unittest.main()
