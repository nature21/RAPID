from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import mujoco

from agent import deployment_context_worker
from instantiation_core.contracts import (
    BindingContract, EntityKind, RelationKind, RelationalInvariant,
    RoleRequirement, SceneBinding, resolve_binding_ownership,
    structural_role_candidates, validate_scene_binding,
)


class BindingCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = mujoco.MjModel.from_xml_string('''
        <mujoco><worldbody><body name="assembly">
          <body name="effector"><geom name="surface" type="box" size=".02 .03 .04"/>
            <site name="tip"/></body>
          <body name="sensor"><site name="sensor_site"/></body>
        </body><body name="object"><freejoint name="motion"/>
          <geom name="shape" type="box" size=".01 .02 .03"/>
        </body></worldbody></mujoco>''')
        self.contract = BindingContract(
            program_kind="strategy", program_name="transfer",
            roles=(
                RoleRequirement("tool", EntityKind.BODY, "Effector body"),
                RoleRequirement("tip", EntityKind.SITE, "Control site", same_body_as="tool"),
                RoleRequirement("surface", EntityKind.GEOM, "Tool surface", allowed_types=("box",), same_body_as="tool"),
                RoleRequirement("target", EntityKind.BODY, "Movable entity", required_body_joint_types=("free",)),
            ),
            relations=(RelationalInvariant(RelationKind.DIFFERENT_BODIES, "tool", "target", "Distinct entities"),),
        )

    def binding(self, **roles: str) -> SceneBinding:
        return SceneBinding("strategy", "transfer", "sample", "test.scene", "Transfer the entity", roles, {})

    def test_joint_and_ownership_constraints_prune_ancestor_and_unrelated_site(self) -> None:
        domains = structural_role_candidates(self.model, self.contract)
        self.assertEqual(domains["target"], ["object"])
        self.assertEqual(domains["tool"], ["effector"])
        self.assertEqual(domains["tip"], ["tip"])
        self.assertEqual(domains["surface"], ["surface"])

    def test_selected_entities_uniquely_resolve_their_direct_owner(self) -> None:
        original = self.binding(tool="assembly", tip="tip", surface="surface", target="object")
        resolved = resolve_binding_ownership(self.model, self.contract, original)
        self.assertEqual(resolved.one("tool"), "effector")
        self.assertEqual(original.one("tool"), "assembly")
        self.assertTrue(validate_scene_binding(self.model, self.contract, resolved).binding_valid)

    def test_conflicting_selected_entities_are_not_silently_replaced(self) -> None:
        original = self.binding(tool="assembly", tip="sensor_site", surface="surface", target="object")
        self.assertEqual(resolve_binding_ownership(self.model, self.contract, original), original)

    def test_invalid_entity_does_not_trigger_speculative_repair(self) -> None:
        original = self.binding(tool="assembly", tip="missing", surface="surface", target="object")
        self.assertEqual(resolve_binding_ownership(self.model, self.contract, original), original)

    def test_optional_role_does_not_eliminate_valid_required_candidates(self) -> None:
        contract = BindingContract("strategy", "optional", (
            RoleRequirement("body", EntityKind.BODY, "Entity"),
            RoleRequirement("site", EntityKind.SITE, "Optional feature", min_count=0, same_body_as="body"),
        ))
        self.assertIn("object", structural_role_candidates(self.model, contract)["body"])

    def test_worker_returns_complete_resolved_names_and_rechecks_invariants(self) -> None:
        roles = dict(tool="assembly", tip="tip", surface="surface", target="object")
        with (
            mock.patch.object(deployment_context_worker, "_load_strategy", return_value=(SimpleNamespace(), self.contract, None)),
            mock.patch.object(deployment_context_worker, "load_scene_module", return_value=SimpleNamespace(make_model=lambda: self.model)),
        ):
            result = deployment_context_worker.validate_roles(strategy="transfer", scene_module="test.scene", instance="sample", goal="Transfer", roles=roles)
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["roles"], dict(roles, tool="effector"))


if __name__ == "__main__":
    unittest.main()
