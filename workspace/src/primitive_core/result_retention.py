"""Scope completed execution results to their enclosing artifact writer."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from .common import CompileResult


_results: ContextVar[list[CompileResult] | None] = ContextVar("compile_results", default=None)


@contextmanager
def capture_execution_results() -> Iterator[list[CompileResult]]:
    """Keep completed execution evidence if learned postprocessing raises."""
    results: list[CompileResult] = []
    token = _results.set(results)
    try:
        yield results
    finally:
        _results.reset(token)


def retain_execution_result(result: CompileResult) -> None:
    """Publish a completed execution to the active artifact writer, if any."""
    results = _results.get()
    if results is not None:
        results[:] = [result]
