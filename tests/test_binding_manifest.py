from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.instantiation_context import (
    CANONICAL_INIT,
    CANONICAL_MAIN,
    stage_canonical_instantiation,
)
from instantiation_core import (
    BINDING_MANIFEST_SCHEMA,
    BindingContract,
    ManifestError,
    ParameterRequirement,
    ParameterType,
    load_binding_manifest,
    parse_binding_manifest,
)


def _contract() -> BindingContract:
    return BindingContract(
        program_kind="primitive",
        program_name="push",
        roles=(),
        parameters=(
            ParameterRequirement(
                name="distance",
                value_type=ParameterType.FLOAT,
                description="Requested object displacement.",
                minimum=0.01,
                maximum=0.5,
            ),
        ),
    )


def _payload() -> dict[str, object]:
    return {
        "schema_version": "rapid.binding/v1",
        "program_kind": "primitive",
        "program_name": "push",
        "instance_name": "renamed",
        "scene_module": "tasks.example.scene",
        "goal_text": "Push the object.",
        "roles": {},
        "inputs": {"distance": 0.2},
    }


class BindingManifestTests(unittest.TestCase):
    def test_published_schema_matches_runtime_schema(self) -> None:
        schema_path = (
            Path(__file__).parents[1]
            / "workspace/src/instantiation_core/binding.schema.json"
        )
        self.assertEqual(
            json.loads(schema_path.read_text(encoding="utf-8")),
            BINDING_MANIFEST_SCHEMA,
        )

    def test_strict_manifest_parses_declared_task_input(self) -> None:
        binding = parse_binding_manifest(_payload(), contract=_contract())
        self.assertEqual(binding.parameters["distance"], 0.2)

    def test_unknown_top_level_motion_data_is_rejected(self) -> None:
        payload = _payload()
        payload["waypoints"] = [[0.0, 1.0]]
        with self.assertRaisesRegex(ManifestError, "unknown keys"):
            parse_binding_manifest(payload, contract=_contract())

    def test_undeclared_policy_input_is_rejected(self) -> None:
        payload = _payload()
        payload["inputs"] = {"distance": 0.2, "clearance": 0.08}
        with self.assertRaisesRegex(ManifestError, "undeclared inputs"):
            parse_binding_manifest(payload, contract=_contract())

    def test_wrong_type_and_range_are_rejected_without_a_scene(self) -> None:
        wrong = _payload()
        wrong["inputs"] = {"distance": "far"}
        outside = _payload()
        outside["inputs"] = {"distance": 0.8}
        with self.assertRaisesRegex(ManifestError, "must be float"):
            parse_binding_manifest(wrong, contract=_contract())
        with self.assertRaisesRegex(ManifestError, "at most"):
            parse_binding_manifest(outside, contract=_contract())

    def test_launcher_generates_canonical_entrypoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "push/renamed"
            manifest = stage_canonical_instantiation(
                target,
                kind="primitive",
                program="push",
                instance="renamed",
                scene_module="tasks.example.scene",
                goal="Push the object.",
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))

            self.assertEqual(
                (target / "__init__.py").read_text(encoding="utf-8"),
                CANONICAL_INIT,
            )
            self.assertEqual(
                (target / "main.py").read_text(encoding="utf-8"),
                CANONICAL_MAIN,
            )
            self.assertEqual(
                (target.parent / "__init__.py").read_text(encoding="utf-8"),
                "",
            )
            self.assertEqual(payload["roles"], {})
            self.assertEqual(payload["inputs"], {})
            with self.assertRaisesRegex(ManifestError, "missing required inputs"):
                load_binding_manifest(manifest, contract=_contract())


if __name__ == "__main__":
    unittest.main()
