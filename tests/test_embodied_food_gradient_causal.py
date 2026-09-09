"""Regression tests for the nonvisual food-gradient steering circuit."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulations.active_inference.experiments.embodied_food_gradient_causal import CASES, _summary, run_case


def _run(name):
    return _summary(run_case(11, name, CASES[name], steps=6, substeps=4))


def test_opposite_physical_food_gradients_drive_opposite_paula_turn_populations_without_vision():
    positive = _run("food_positive_y")
    negative = _run("food_negative_y")
    assert positive["food_sensor_left_spikes"] + positive["food_sensor_right_spikes"] > 150
    assert negative["food_sensor_left_spikes"] + negative["food_sensor_right_spikes"] > 150
    assert positive["turn_imbalance_TL_minus_TR"] < -3
    assert negative["turn_imbalance_TL_minus_TR"] > 3
    assert positive["max_abs_actuator_control"] > 0.0
    assert negative["max_abs_actuator_control"] > 0.0


def test_food_sensor_to_turn_ablation_preserves_sensor_activity_but_removes_turn_output():
    full = _run("food_positive_y")
    ablated = _run("food_path_ablation")
    assert ablated["food_sensor_left_spikes"] + ablated["food_sensor_right_spikes"] > 150
    assert full["turn_TR_spikes"] > 0
    assert ablated["turn_TL_spikes"] == 0
    assert ablated["turn_TR_spikes"] == 0
