"""Regression test for nonvisual lateral toxin avoidance steering."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulations.active_inference.experiments.embodied_toxin_gradient_causal import CASES, _summary, run_case


def test_mirrored_physical_toxin_gradients_reverse_paula_turns_and_w_tox_ablation_removes_only_output():
    positive = _summary(run_case(11, "toxin_positive_y", CASES["toxin_positive_y"], steps=10, substeps=8))
    negative = _summary(run_case(11, "toxin_negative_y", CASES["toxin_negative_y"], steps=10, substeps=8))
    ablated = _summary(run_case(11, "toxin_path_ablation", CASES["toxin_path_ablation"], steps=10, substeps=8))
    assert positive["toxin_sensor_left_spikes"] + positive["toxin_sensor_right_spikes"] > 400
    assert negative["toxin_sensor_left_spikes"] + negative["toxin_sensor_right_spikes"] > 400
    assert positive["turn_imbalance_TL_minus_TR"] > 6
    assert negative["turn_imbalance_TL_minus_TR"] < -6
    assert ablated["toxin_sensor_left_spikes"] + ablated["toxin_sensor_right_spikes"] > 400
    assert ablated["turn_TL_spikes"] == 0 and ablated["turn_TR_spikes"] == 0
