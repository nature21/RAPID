"""Run generated-code validation inside the workspace permission sandbox."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from typing import Any

from .codex_exec import REPO_ROOT, WORKSPACE


RESULT_PREFIX = "RAPID_VALIDATION_RESULT="
DEFAULT_TIMEOUT_SECONDS = 300.0
PERMISSION_PROFILE = "rapid-experiment"


def _sandbox_validation_command(
    request: Mapping[str, Any],
    *,
    codex: str | None = None,
) -> list[str]:
    """Build a safe-path validation command using the active permission profile."""

    executable = codex or shutil.which("codex")
    if executable is None:
        raise RuntimeError("`codex` CLI was not found on PATH")
    return [
        executable,
        "sandbox",
        "-C",
        str(WORKSPACE),
        "-P",
        PERMISSION_PROFILE,
        "--",
        str(REPO_ROOT / "scripts" / "rapid-python"),
        "-P",
        "-m",
        "agent.validation_worker",
        json.dumps(dict(request), sort_keys=True),
    ]


def run_sandboxed_validation(
    request: Mapping[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[str]:
    """Return findings emitted by a generated-code validator in the sandbox."""

    if os.environ.get("RAPID_CODEX_PERMISSION_PROFILE") != "1":
        return [
            "refusing to execute generated validation code outside "
            "scripts/rapid-codex-run"
        ]

    env = os.environ.copy()
    cache_dir = WORKSPACE / "agent_work" / "tmp" / "rapid_cache"
    env["TMPDIR"] = str(WORKSPACE / "agent_work" / "tmp")
    env["XDG_CACHE_HOME"] = str(cache_dir)
    env["MPLCONFIGDIR"] = str(cache_dir / "matplotlib")
    env["PYTHONSAFEPATH"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            _sandbox_validation_command(request),
            cwd=WORKSPACE,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return [f"sandboxed validation could not run: {exc}"]

    payload_text: str | None = None
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            payload_text = line.removeprefix(RESULT_PREFIX)

    if completed.returncode != 0 or payload_text is None:
        detail = (completed.stderr or completed.stdout).strip()
        if detail:
            detail = detail[-2000:]
        else:
            detail = "validation worker produced no diagnostic output"
        return [
            f"sandboxed validation failed with exit {completed.returncode}: {detail}"
        ]

    try:
        payload = json.loads(payload_text)
        findings = payload["findings"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        return [f"sandboxed validation returned an invalid result: {exc}"]
    if not isinstance(findings, list) or not all(
        isinstance(finding, str) for finding in findings
    ):
        return ["sandboxed validation findings must be a list of strings"]
    return findings


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "PERMISSION_PROFILE",
    "RESULT_PREFIX",
    "_sandbox_validation_command",
    "run_sandboxed_validation",
]
