"""Regression checks for the causal PAULA/MuJoCo motor primitive."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulations.active_inference.experiments.paula_motor_causal import CASES, _summary, run_case


def _run(name):
    return _summary(run_case(11, name, CASES[name], steps=600, substeps=6))


def test_paula_cpg_drives_graded_muscles_and_forward_mujoco_motion():
    summary = _run("forward")
    assert summary["cpg_spikes"] > 0
    assert summary["max_abs_muscle_state"] > 0
    assert summary["max_abs_actuator_control"] > 0
    assert summary["displacement"] > 0.75
    assert abs(summary["heading_change_degrees"]) < 10.0


def test_descending_neural_currents_produce_opposite_turns():
    left = _run("left_turn")
    right = _run("right_turn")
    assert left["displacement"] > 0.6 and left["heading_change_degrees"] < -20.0
    assert right["displacement"] > 0.6 and right["heading_change_degrees"] > 20.0


def test_cpg_output_and_nmj_ablations_leave_the_body_still_but_preserve_cpg_spikes():
    for name in ("cpg_to_muscle_ablation", "nmj_ablation"):
        summary = _run(name)
        assert summary["cpg_spikes"] > 0
        assert summary["max_abs_actuator_control"] == 0.0
        assert summary["displacement"] == 0.0
