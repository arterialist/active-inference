"""Instrument checks and finite-protocol regressions, not an organism benchmark."""
from collections import Counter
import json

import pytest

from simulations.active_inference.experiments.adaptive_association import (
    build_association, run, schedule, stimuli,
)
from simulations.active_inference.experiments.association_analysis import audit
from simulations.active_inference.experiments.association_replay import replay
from simulations.active_inference.experiments.composition_analysis import first_difference


def test_environment_mapping_is_not_encoded_in_initial_weights(tmp_path):
    configs = []
    for variant in ("corrective", "corrective_window"):
        path = tmp_path / f"{variant}.json"
        net, _ = build_association(path, 23, "closed_loop", variant=variant)
        configs.append(json.loads(path.read_text()))
        assert len(net.network.neurons) == 8
        assert all(n.params.eta_post == n.params.eta_retro == 1e-5 for n in net.network.neurons.values())
        assert all(n.lower_t_ref_bound <= n.t_ref <= n.upper_t_ref_bound for n in net.network.neurons.values())
    # The causal comparison changes only a local modulatory response, not cue
    # mapping, initial cue weights, connectivity, or port-count t_ref bounds.
    for config in configs:
        for n in config["neurons"]:
            n["params"]["w_tref"] = [0., 0.]
    assert configs[0] == configs[1]


def test_unpaired_control_preserves_each_sensory_pulse_count():
    trials = schedule(training_trials=2)
    counts = []
    for mode in ("closed_loop", "unpaired"):
        counts.append(Counter(tuple(p) for tr in trials for t in range(tr["start"], tr["stop"]) for p in stimuli(tr, t, mode)))
    assert counts[0] == counts[1]
    assert all(tr["stop"]-tr["start"] == 160 for tr in trials)
    swapped = schedule(training_trials=2, mapping=1)
    assert all(a["cue"] == b["cue"] and (a["outcome"] is None or a["outcome"] != b["outcome"])
               for a, b in zip(trials, swapped, strict=True))


@pytest.fixture(scope="module")
def preparations(tmp_path_factory):
    root = tmp_path_factory.mktemp("association")
    for name, options in (
        ("window", dict(variant="corrective_window_resolved_retro", challenge="reversal")),
        ("cut_window", dict(variant="corrective_resolved_retro", challenge="reversal")),
        ("underresolved", dict(variant="corrective_window_slow_retro", challenge="reversal")),
        ("timing", dict(variant="corrective_window_resolved_retro", challenge="timing_recovery")),
        ("reference", dict()),
        ("yoked", dict(mode="yoked", reference=root / "reference")),
        ("shifted", dict(mode="yoked_shifted", reference=root / "reference")),
    ):
        run(root / name, seed=23, training_trials=12, **options)
    return root


def test_reversal_requires_correct_credit_window_in_this_preparation(preparations):
    good = audit(preparations / "window")
    bad = audit(preparations / "cut_window")
    assert good["phases"]["retention"]["correct_only"] == 6
    assert good["phases"]["reversal_probe"]["correct_only"] == 6
    assert not good["false_settlement_trials"]
    assert bad["phases"]["reversal_probe"].get("ambiguous", 0) > 0
    assert bad["false_settlement_trials"]
    assert good["first_modulated_depression_of_old_mapping"]["t_ref"] == 4.


def test_packet_replay_matches_all_neurons_but_shifted_packets_do_not(preparations):
    project = lambda state: state["neurons"]
    assert first_difference(preparations / "reference", preparations / "yoked", project) is None
    assert first_difference(preparations / "reference", preparations / "shifted", project) is not None
    a = audit(preparations / "reference")
    b = audit(preparations / "shifted")
    assert a["phases"]["retention"]["correct_only"] == 6
    assert b["phases"]["retention"].get("correct_only", 0) == 0


def test_saved_config_and_actual_inputs_reproduce_tick_state(preparations):
    assert replay(preparations / "window")["exact_replay"]


def test_positive_parameter_does_not_prove_effective_basal_adaptation(preparations):
    good = audit(preparations / "window")["prediction_terminal_updates"]
    bad = audit(preparations / "underresolved")["prediction_terminal_updates"]
    assert good["basal_changed_ticks"] == good["basal_update_ticks"] > 0
    assert good["basal_unrepresented_ticks"] == 0
    assert bad["basal_nonzero_requested_ticks"] > 0
    assert bad["basal_changed_ticks"] == 0
    assert bad["basal_unrepresented_ticks"] == bad["basal_nonzero_requested_ticks"]


def test_timing_transfer_failure_is_distinct_from_memory_erasure(preparations):
    result = audit(preparations / "timing")
    assert result["phases"]["transfer_period2"]["correct_only"] == 6
    assert result["phases"]["transfer_period6"]["silent_or_insufficient"] == 6
    assert all(p["maximum_threshold_margin"] < 0. for p in result["probe_response_margins"]
               if p["phase"] == "transfer_period6" and p["expected_prediction"])
    assert not result["challenge_passed"]


def test_auditor_rejects_fabricated_trial_spike_count(preparations):
    file = preparations / "window" / "summary.json"
    original = file.read_text()
    summary = json.loads(original)
    summary["trials"][0]["spikes"]["5"] = 10
    try:
        file.write_text(json.dumps(summary))
        with pytest.raises(AssertionError):
            audit(file.parent)
    finally:
        file.write_text(original)


def test_auditor_rejects_false_challenge_pass_flag(preparations):
    file = preparations / "cut_window" / "summary.json"
    original = file.read_text()
    summary = json.loads(original)
    summary["challenge_passed"] = True
    try:
        file.write_text(json.dumps(summary))
        with pytest.raises(AssertionError):
            audit(file.parent)
    finally:
        file.write_text(original)
