from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from agent import invention_session
from agent.codex_exec import CodexExecConfig
from suites.staged.runtime_integrity import (
    require_shared_source, shared_source_hashes, verify_runtime_manifest,
)
from suites.staged.workspaces import create_isolated_workspace, merge_workspaces
from tests.test_workspace_primitive_layout import _make_repo, _touch


class RuntimeIntegrityTests(unittest.TestCase):
    def test_learned_edits_allowed_but_shared_add_delete_change_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp)
            _touch(src / "primitive_core/compile.py", "original")
            _touch(src / "strategies/runtime.py", "original")
            expected = shared_source_hashes(src)
            _touch(src / "primitives/new/nominal.py", "learned")
            _touch(src / "strategies/new/program.py", "learned")
            _touch(src / "primitive_core/__pycache__/compile.pyc", "cache")
            require_shared_source(src, expected)
            for path in (src / "primitive_core/compile.py", src / "strategies/runtime.py"):
                with self.subTest(path=path):
                    path.write_text("changed")
                    with self.assertRaisesRegex(RuntimeError, "shared runtime mismatch"):
                        require_shared_source(src, expected)
                    path.write_text("original")
            _touch(src / "primitive_core/extension.py", "added")
            with self.assertRaisesRegex(RuntimeError, "extension.py"):
                require_shared_source(src, expected)
            (src / "primitive_core/extension.py").unlink()
            (src / "primitive_core/compile.py").unlink()
            with self.assertRaisesRegex(RuntimeError, "compile.py"):
                require_shared_source(src, expected)

    def test_pinned_runtime_survives_repo_changes_across_stages(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            repo = _make_repo(root)
            baseline = create_isolated_workspace(root / "runtime", repo_root=repo)
            expected = verify_runtime_manifest(baseline.workspace)
            _touch(repo / "workspace/src/primitive_core/compile.py", "new repo version")
            _touch(repo / "workspace/src/primitive_core/new.py", "new file")
            with mock.patch.dict(os.environ, RAPID_RUNTIME_WORKSPACE=str(baseline.workspace)):
                construction = create_isolated_workspace(root / "construction", repo_root=repo)
                _touch(construction.src / "primitives/example/__init__.py", "")
                library = merge_workspaces(root / "library", [construction.workspace], repo_root=repo)
                strategy = create_isolated_workspace(root / "strategy", repo_root=repo,
                                                     from_workspace=library.workspace)
                evaluation = create_isolated_workspace(root / "eval", repo_root=repo,
                                                       from_workspace=strategy.workspace)
                for ws in (construction, library, strategy, evaluation):
                    self.assertEqual(verify_runtime_manifest(ws.workspace), expected)

    def test_merge_and_eval_copy_reject_changed_runtime_before_creation(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            repo = _make_repo(root)
            construction = create_isolated_workspace(root / "construction", repo_root=repo)
            _touch(construction.src / "primitive_core/compile.py", "private extension")
            with self.assertRaisesRegex(RuntimeError, "compile.py"):
                merge_workspaces(root / "library", [construction.workspace], repo_root=repo)
            with self.assertRaisesRegex(RuntimeError, "compile.py"):
                create_isolated_workspace(root / "eval", repo_root=repo,
                                          from_workspace=construction.workspace)
            self.assertFalse((root / "library/workspace").exists())
            self.assertFalse((root / "eval/workspace").exists())

    def test_invention_runtime_change_fails_and_writes_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            path = src / "primitive_core/compile.py"
            _touch(path, "original")
            session = invention_session.InventionSession(
                kind="primitive", run_dir=root,
                config=CodexExecConfig("model", "high", 400), iteration_ceiling=1,
            )
            with mock.patch.object(invention_session, "SRC_DIR", src), mock.patch.object(
                invention_session, "PRIMITIVES_DIR", src / "primitives"
            ), mock.patch.object(
                invention_session, "run_codex_exec", side_effect=lambda *a: path.write_text("edited")
            ):
                with self.assertRaises(SystemExit):
                    session.run("prompt", [])
            report = json.loads((root / "runtime_integrity.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["changed_files"], ["primitive_core/compile.py"])
