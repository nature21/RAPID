from __future__ import annotations

import tempfile
import shutil
import unittest
from pathlib import Path

from suites.staged.workspaces import (
    audit_zero_primitive_baseline,
    create_isolated_workspace,
    merge_workspaces,
    migrate_workspace_to_current_format,
)
from agent.instantiation_context import CANONICAL_INIT, CANONICAL_MAIN


def _touch(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_repo(root: Path) -> Path:
    repo = root / "repo"
    canonical_src = repo / "workspace/src"
    for filename in ("__init__.py", "common.py", "compile.py", "primitive.md"):
        _touch(canonical_src / "primitive_core" / filename, f"# {filename}\n")
    _touch(canonical_src / "primitive_core/README.md", "# Primitive Core\n")
    _touch(canonical_src / "primitives/__init__.py", "")
    _touch(canonical_src / "primitives/README.md", "# Primitives\n")
    _touch(canonical_src / "instantiation_core/__init__.py", "")
    _touch(canonical_src / "instantiations/__init__.py", "")
    _touch(canonical_src / "instantiations/primitives/__init__.py", "")
    _touch(canonical_src / "instantiations/strategies/__init__.py", "")
    _touch(canonical_src / "strategies/explanation.md", "# Strategies\n")
    _touch(canonical_src / "scene_io/__init__.py", "")
    _touch(repo / "AGENTS.md", "# test\n")
    _touch(repo / "SUITE_RUNS.md", "# test\n")
    _touch(repo / "scripts/rapid-python", "#!/usr/bin/env bash\n")
    _touch(repo / "agent/templates/invent_workspace/AGENTS.md", "# invent\n")
    _touch(
        repo
        / "agent/templates/invent_workspace/.agents/skills/"
        "rapid-construct-primitive/SKILL.md",
        "---\nname: rapid-construct-primitive\ndescription: test\n---\n",
    )
    _touch(
        repo
        / "agent/templates/invent_workspace/.agents/skills/"
        "rapid-compose-strategy/SKILL.md",
        "---\nname: rapid-compose-strategy\ndescription: test\n---\n",
    )
    return repo


def _make_learned_workspace(
    root: Path,
    *,
    primitive: str,
    marker: str,
) -> Path:
    workspace = root / "workspace"
    shutil.copytree(root.parent / "repo/workspace/src", workspace / "src")
    _touch(workspace / "src/primitives/__init__.py", "")
    _touch(workspace / f"src/primitives/{primitive}/__init__.py", f"MARKER = {marker!r}\n")
    _touch(
        workspace / f"src/primitives/{primitive}/binding_contract.py",
        f"MARKER = {marker!r}\n",
    )
    _touch(
        workspace
        / f"src/instantiations/primitives/{primitive}/example_0/main.py",
        CANONICAL_MAIN,
    )
    _touch(
        workspace
        / f"src/instantiations/primitives/{primitive}/example_0/__init__.py",
        CANONICAL_INIT,
    )
    _touch(
        workspace
        / f"src/instantiations/primitives/{primitive}/example_0/binding.json",
        (
            "{\n"
            '  "schema_version": "rapid.binding/v1",\n'
            '  "program_kind": "primitive",\n'
            f'  "program_name": "{primitive}",\n'
            '  "instance_name": "example_0",\n'
            '  "scene_module": "test_scenes.example_0.scene",\n'
            '  "goal_text": "test",\n'
            '  "roles": {},\n'
            '  "inputs": {}\n'
            "}\n"
        ),
    )
    return workspace


class PrimitiveWorkspaceTests(unittest.TestCase):
    def test_create_workspace_preserves_flat_primitive_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = _make_repo(root)
            source = _make_learned_workspace(
                root / "source",
                primitive="push",
                marker="source",
            )

            created = create_isolated_workspace(
                root / "run",
                repo_root=repo,
                from_workspace=source,
            )

            self.assertTrue((created.src / "primitives/push/__init__.py").is_file())
            self.assertTrue(
                (
                    created.src
                    / "instantiations/primitives/push/example_0/main.py"
                ).is_file()
            )
            self.assertTrue(
                (created.src / "instantiation_core/__init__.py").is_file()
            )
            self.assertTrue((created.src / "primitive_core/common.py").is_file())

    def test_merge_renames_conflicting_named_primitives_and_imports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = _make_repo(root)
            first = _make_learned_workspace(
                root / "source_a",
                primitive="push",
                marker="a",
            )
            second = _make_learned_workspace(
                root / "source_b",
                primitive="push",
                marker="b",
            )

            merged = merge_workspaces(
                root / "merged",
                [first, second],
                repo_root=repo,
            )

            renamed = "push__from_source_b"
            self.assertTrue((merged.src / "primitives/push/__init__.py").is_file())
            self.assertTrue(
                (merged.src / f"primitives/{renamed}/__init__.py").is_file()
            )
            renamed_manifest = (
                merged.src
                / f"instantiations/primitives/{renamed}/example_0/binding.json"
            ).read_text()
            self.assertIn(f'"program_name": "{renamed}"', renamed_manifest)
    def test_merge_deduplicates_identical_named_primitives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = _make_repo(root)
            first = _make_learned_workspace(
                root / "source_a",
                primitive="push",
                marker="same",
            )
            second = _make_learned_workspace(
                root / "source_b",
                primitive="push",
                marker="same",
            )

            merged = merge_workspaces(
                root / "merged",
                [first, second],
                repo_root=repo,
            )

            primitive_dirs = sorted(
                path.name
                for path in (merged.src / "primitives").iterdir()
                if path.is_dir()
            )
            self.assertEqual(primitive_dirs, ["push"])
            index = (merged.workspace / "merge_index.yaml").read_text()
            self.assertIn('"primitive": "push"', index)

    def test_zero_baseline_ignores_core_but_flags_primitives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            _touch(src / "primitive_core/helpers/internal.py", "")

            self.assertEqual(audit_zero_primitive_baseline(src), [])

            _touch(src / "primitives/push/__init__.py", "")
            self.assertEqual(audit_zero_primitive_baseline(src), ["primitives/push"])

    def test_zero_baseline_flags_concrete_instantiations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            _touch(src / "instantiations/__init__.py", "")
            _touch(src / "instantiations/primitives/__init__.py", "")
            _touch(src / "instantiations/strategies/__init__.py", "")
            _touch(
                src / "instantiations/primitives/push/example_0/main.py",
                "",
            )

            self.assertEqual(
                audit_zero_primitive_baseline(src),
                ["instantiations/primitives/push"],
            )

    def test_migration_adds_instantiation_scaffold_and_removes_legacy_solutions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = _make_repo(root)
            workspace = root / "legacy/workspace"
            _touch(workspace / "src/solutions/push/example_0/main.py", "LEGACY = True\n")
            _touch(
                workspace / "src/instantiations/primitives/push/example_0/main.py",
                "EXISTING = True\n",
            )

            record = migrate_workspace_to_current_format(
                workspace,
                repo_root=repo,
            )

            self.assertTrue(
                (workspace / "src/instantiations/__init__.py").is_file()
            )
            self.assertTrue(
                (workspace / "src/instantiations/primitives/__init__.py").is_file()
            )
            self.assertTrue(
                (workspace / "src/instantiations/strategies/__init__.py").is_file()
            )
            self.assertEqual(
                (
                    workspace
                    / "src/instantiations/primitives/push/example_0/main.py"
                ).read_text(),
                "EXISTING = True\n",
            )
            self.assertFalse((workspace / "src/solutions").exists())
            self.assertIn("instantiations", record["copied_infra"])
            self.assertIn("solutions", record["removed_legacy_src"])


if __name__ == "__main__":
    unittest.main()
