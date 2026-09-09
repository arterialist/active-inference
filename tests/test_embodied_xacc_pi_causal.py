"""Fast regression checks for the isolated XACC/YACC neural PI path."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulations.active_inference import central_complex as cc
from simulations.active_inference.experiments.embodied_xacc_pi_causal import replay_accumulator


def _straight_heading_stream(index: int, ticks: int = 120):
    return [
        {
            "segment": "outbound_straight",
            "heading_index": index,
            "speed_physical": 0.4,
            "speed_current": 1.0,
            "pose": {"x": tick * np.cos(cc.PHI[index]), "y": tick * np.sin(cc.PHI[index])},
        }
        for tick in range(ticks)
    ]


def test_xacc_yacc_integrates_heading_code_retains_state_and_kpi_ablation_preserves_pg():
    stream = _straight_heading_stream(index=9)
    full = replay_accumulator(stream, k_pi=64.0, retain_ticks=120)
    ablated = replay_accumulator(stream, k_pi=0.0, retain_ticks=120)

    x, y = full["final_vector"]["x"], full["final_vector"]["y"]
    assert full["final_vector"]["magnitude"] > 0.1
    assert abs(np.degrees(np.angle(np.exp(1j * (np.arctan2(y, x) - cc.PHI[9]))))) < 1e-3
    assert full["post_retention_vector"]["magnitude"] / full["final_vector"]["magnitude"] > 0.98
    assert sum(row["pg_spikes"] for row in full["tick_trace"]) > 0
    assert sum(row["pg_spikes"] for row in full["tick_trace"]) == sum(row["pg_spikes"] for row in ablated["tick_trace"])
    assert ablated["final_vector"]["magnitude"] == 0.0
