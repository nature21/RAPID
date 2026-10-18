"""One-shot Codex binding of an existing RAPID primitive to a new scene."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agent.instantiate_common import run
else:  # pragma: no cover
    from .instantiate_common import run


if __name__ == "__main__":
    run("primitive")
