from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from instantiation_core.runner import _print_metrics


def test_standalone_diagnostics_preserve_behavior_and_metrics(capsys):
    metrics = {
        "success": True,
        "effect_success": True,
        "mechanism_valid": True,
        "stability_valid": True,
        "transit": {
            "transit_convergence_valid": False,
            "terminal_position_error": 0.097,
            "terminal_orientation_error": np.deg2rad(19),
        },
    }
    original = deepcopy(metrics)

    _print_metrics("grasp", SimpleNamespace(metrics=metrics, initial_pose=None))

    output = capsys.readouterr().out
    assert "grasp / transit: convergence=false" in output
    assert "terminal_position_error=97.00 mm" in output
    assert "terminal_orientation_error=19.00 deg" in output
    assert "grasp / effect: effect_success=true; mechanism_valid=true" in output
    assert metrics == original


def test_strategy_distinguishes_projection_and_each_transit_phase(capsys):
    phase_name = "pickup_with_a_long_name_that_compact_repr_would_truncate"
    metrics = {
        "phase_metrics": {
            phase_name: {
                "transit": {
                    "initial_knot_advisory": {
                        "translation": 0.025,
                        "orientation": np.deg2rad(5),
                    },
                    "transit_phases": [
                        {
                            "phase": "configure_gripper",
                            "transit_convergence_valid": True,
                            "terminal_position_error": 0.001,
                            "terminal_orientation_error": 0.0,
                        },
                        {
                            "phase": "precontact",
                            "transit_convergence_valid": False,
                            "terminal_position_error": 0.097,
                            "terminal_orientation_error": np.deg2rad(19),
                        },
                    ],
                },
                "effect_success": False,
                "mechanism_valid": False,
            },
            "release": {
                "transit_convergence_valid": True,
                "terminal_position_error": 0.002,
                "terminal_orientation_error": 0.0,
            },
        },
    }
    original = deepcopy(metrics)

    _print_metrics("strategy", SimpleNamespace(metrics=metrics, initial_pose=None))

    output = capsys.readouterr().out.split("--- Execution diagnostics ---")[1]
    assert f"strategy / {phase_name} / transit / handoff projection: translation=25.00 mm; rotation=5.00 deg" in output
    assert "configure_gripper: convergence=true; terminal_position_error=1.00 mm" in output
    assert "precontact: convergence=false; terminal_position_error=97.00 mm" in output
    assert f"strategy / {phase_name} / effect: effect_success=false" in output
    assert "strategy / release / transit: convergence=true" in output
    assert output.count("handoff projection:") == 1
    assert metrics == original


def test_partial_failure_does_not_claim_an_effect_was_executed(capsys):
    metrics = {
        "phase_metrics": {
            "expose": {"effect_success": True, "mechanism_valid": True},
            "pick_up__rejected": {
                "success": False,
                "failure_type": "FR3TransitViolation",
                "failure_message": "No collision-free handoff candidate found",
            },
        },
    }

    _print_metrics("strategy", SimpleNamespace(metrics=metrics, initial_pose=None))

    output = capsys.readouterr().out.split("--- Execution diagnostics ---")[1]
    assert "expose / effect: effect_success=true" in output
    assert "pick_up__rejected / failure: FR3TransitViolation: No collision-free handoff candidate found" in output
    assert "pick_up__rejected / effect" not in output


def test_missing_and_nonfinite_diagnostics_remain_unavailable(capsys):
    metrics = {
        "transit": {
            "initial_knot_advisory": {"translation": float("inf")},
            "transit_phases": [
                {"phase": "precontact", "terminal_position_error": float("nan")},
            ],
        },
        "effect_success": False,
    }

    _print_metrics("grasp", SimpleNamespace(metrics=metrics, initial_pose=None))

    output = capsys.readouterr().out.split("--- Execution diagnostics ---")[1]
    assert "handoff projection: translation=unavailable; rotation=unavailable" in output
    assert "precontact: convergence=unavailable; terminal_position_error=unavailable; terminal_orientation_error=unavailable" in output
    assert "effect_success=false; mechanism_valid=unavailable; stability_valid=unavailable" in output
