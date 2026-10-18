"""Inclusive wall times for execution stages, including failed searches."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import time

_timings = ContextVar("execution_timings", default=None)


@contextmanager
def capture_timings():
    values = {}
    token = _timings.set(values)
    try:
        yield values
    finally:
        _timings.reset(token)


@contextmanager
def execution_timer(stage):
    started = time.perf_counter()
    try:
        yield
    finally:
        values = _timings.get()
        if values is not None:
            record = values.setdefault(stage, {"calls": 0, "wall_seconds": 0.0})
            record["calls"] += 1
            record["wall_seconds"] += time.perf_counter() - started


def timed_execution(stage):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with execution_timer(stage):
                return function(*args, **kwargs)
        return wrapped
    return decorate
