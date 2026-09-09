"""Causal regression tests for the PAULA T-maze circuit."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulations/active_inference"))
import paula_aif as p


SEEDS = (11, 23, 44, 77)


def _outcomes(seed, **build_kwargs):
    np.random.seed(seed)
    brain = p.Brain(**build_kwargs)
    outcomes = [p.run_episode(i % 2, brain) for i in range(12)]
    return sum(row["visited_cue"] for row in outcomes), sum(row["reward"] for row in outcomes)


def test_full_paula_circuit_reaches_cue_and_correct_arm_on_four_seeds():
    for seed in SEEDS:
        assert _outcomes(seed) == (12, 12)


def test_cue_action_pathway_ablation_leaves_the_agent_still():
    for seed in SEEDS:
        assert _outcomes(seed, w_epi=0.0) == (0, 0)


def test_belief_evidence_ablation_visits_cue_but_cannot_choose_the_reward_arm():
    for seed in SEEDS:
        assert _outcomes(seed, w_ev=0.0) == (12, 0)


def test_tick_trace_contains_only_neural_drivers_and_observational_outputs():
    np.random.seed(11)
    trace = []
    result = p.run_episode(p.RL, p.Brain(), tick_log=trace)
    assert result == {"visited_cue": True, "reward": True}
    assert trace
    assert {"evidence_left", "evidence_right", "tonic_uncertainty", "spikes", "selected_action"} <= trace[0].keys()
