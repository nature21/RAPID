import os
import sys

from suites.staged.benchmark_runner import _attach_execution_diagnostics
from suites.staged.strategy_test import _run


def test_execution_log_preserves_process_status_and_exception(tmp_path):
    log = tmp_path / "execution.log"
    status = _run(
        [sys.executable, "-c", "print('started', flush=True); raise RuntimeError('failed before trajectory')"],
        cwd=tmp_path, env=os.environ.copy(), log_path=log,
    )
    assert status == 1
    assert "started" in log.read_text()
    record = _attach_execution_diagnostics({
        "binding_written": True, "binding_valid": True,
        "program_success": False, "benchmark_success": False,
        "returncode": status, "execution_log_path": str(log),
    }, {})
    assert record["failure_type"] == "RuntimeError"
    assert record["failure_message"] == "failed before trajectory"
    assert record["failure_stage"] == "execution"
    assert record["execution_completed"] is False
    assert record["metrics_available"] is False
    assert record["benchmark_success"] is False
    assert record["returncode"] == status


def test_transit_traceback_is_not_misreported_as_preflight(tmp_path):
    log = tmp_path / "execution.log"
    log.write_text(
        "Traceback (most recent call last):\n"
        "primitive_core.fr3_transit_planner.FR3TransitPlanningFailure: unreachable\n"
        "The above exception was the direct cause of the following exception:\n"
        "Traceback (most recent call last):\n"
        "primitive_core.fr3_transit.FR3TransitViolation: no collision-free handoff\n"
    )
    record = {"binding_written": True, "execution_log_path": str(log)}
    result = _attach_execution_diagnostics(record, {})
    assert result["failure_type"] == "FR3TransitViolation"
    assert result["failure_stage"] == "transit"
    assert result["failure_segment"] is None
    # Existing structured metrics take precedence over traceback fallback.
    result = _attach_execution_diagnostics(record, {"metrics": {
        "partial_strategy": True, "failure_type": "PhysicalPreconditionError",
        "failure_message": "insufficient overhang", "failed_segment": "pickup",
    }})
    assert result["failure_type"] == "PhysicalPreconditionError"
    assert result["failure_segment"] == "pickup"


def test_missing_traceback_keeps_unknown_preflight_classification(tmp_path):
    log = tmp_path / "execution.log"
    log.write_text("Loading scene\nWarning: renderer unavailable\n")
    result = _attach_execution_diagnostics({
        "binding_written": True, "execution_log_path": str(log),
    }, {})
    assert result["failure_type"] == "StrategyPreflightFailure"
