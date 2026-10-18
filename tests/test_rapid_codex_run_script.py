from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


class NpplCodexRunScriptTests(unittest.TestCase):
    def test_stale_ambient_workspace_does_not_block_launcher_bootstrap(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        missing_workspace = repo_root / "suite_runs/removed_run/workspace"
        env = os.environ.copy()
        env.update(
            {
                "RAPID_WORKSPACE": str(missing_workspace),
                "RAPID_ACTIVE_SRC_DIR": str(missing_workspace / "src"),
                "RAPID_SRC_DIR": str(missing_workspace / "src"),
            }
        )

        completed = subprocess.run(
            [str(repo_root / "scripts/rapid-codex-run"), "--help"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--workspace", completed.stdout)
        self.assertNotIn(str(missing_workspace), completed.stderr)


if __name__ == "__main__":
    unittest.main()
