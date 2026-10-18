"""Private strategy-specific scene generators.

Each module owns the relational layout and requirements for exactly one
strategy.  The aggregate CLI imports these modules dynamically; invention
workspaces cannot read this package.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from ..specs import StrategyContract
from ..strategy_common import ContextResult, SceneEditor


def generator_for(
    strategy_id: int,
) -> Callable[[SceneEditor, StrategyContract], ContextResult]:
    module = importlib.import_module(f"{__name__}.strategy_{strategy_id}")
    return module.apply


__all__ = ["generator_for"]
