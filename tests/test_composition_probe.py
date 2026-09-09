"""Research-instrument checks, not acceptance tests for embodied agents."""
import json

import pytest

from simulations.active_inference.experiments.composition_analysis import (
    analyze, analyze_gate, expected_ring_ticks, first_difference, rows,
)
from simulations.active_inference.experiments.composition_probe import run
from simulations.active_inference.experiments.composition_interventions import run as intervene


@pytest.mark.parametrize("phase", [0, 1])
def test_native_reader_tracks_relation_without_external_drive(tmp_path, phase):
    path = tmp_path / "native"
    run(path, 11, "frozen", "shared", phase, 420)
    result = analyze(path)
    assert result["ring_exact"] and result["readout_exact"]
    records = list(rows(path))
    assert all(not r["external"] for r in records[1:] if r["executed_tick"] != 3)
    assert expected_ring_ticks(4+phase, phase, 50) == [4, 25, 46]


@pytest.mark.parametrize("mode", ["frozen", "both"])
def test_input_observer_is_noninterfering(tmp_path, mode):
    a = run(tmp_path / "observed", 11, mode, "shared", 0, 420)
    b = run(tmp_path / "unobserved", 11, mode, "shared", 0, 420, observed=False)
    assert a["state_digest"] == b["state_digest"]


def test_silent_reader_is_not_a_success(tmp_path):
    run(tmp_path / "silent", 11, "frozen", "none", 0, 420)
    r = analyze(tmp_path / "silent")
    assert r["ring_exact"] and not r["readout_exact"]
    assert r["last_spike"]["7"] is None and r["last_spike"]["8"] is None


def test_attachment_effect_is_removed_by_specific_feedback_cut(tmp_path):
    for name, attachment, cut in (("none", "none", False), ("shared", "shared", False),
                                   ("separate", "separate", False), ("cut", "shared", True)):
        run(tmp_path / name, 11, "both", attachment, 0, 420, cut_consumer_retro=cut)
    assert first_difference(tmp_path / "none", tmp_path / "shared") == 6
    assert first_difference(tmp_path / "none", tmp_path / "separate") is None
    assert first_difference(tmp_path / "none", tmp_path / "cut") is None
    a = analyze(tmp_path / "none")
    b = analyze(tmp_path / "shared")
    assert a["first_missing"]["1"]["tick"] == 340
    assert b["first_missing"]["5"]["tick"] == 284
    assert b["first_missing"]["5"]["cause"] == "subthreshold_arrival"
    assert b["first_missing"]["5"]["reconstruction_error"] < 1e-6
    assert a["max_retro_error"] < 1e-5


def test_analyzer_rejects_false_summary(tmp_path):
    path = tmp_path / "corrupt"
    run(path, 11, "frozen", "shared", 0, 105)
    file = path / "summary.json"
    summary = json.loads(file.read_text())
    summary["spike_ticks"]["7"] = []
    file.write_text(json.dumps(summary))
    with pytest.raises(AssertionError):
        analyze(path)


def test_early_freeze_preserves_dynamics_but_late_gain_restore_does_not(tmp_path):
    early = intervene(tmp_path / "early", 11, 0, "freeze", 160, 630)
    late = intervene(tmp_path / "late", 11, 0, "restore_gains", 400, 630)
    control = intervene(tmp_path / "control", 11, 0, "none", 0, 630)
    assert early["windows"][-1]["both_rings_active"]
    assert early["windows"][-1]["selectivity"] == 1.
    assert not late["windows"][-1]["both_rings_active"]
    assert late["spike_ticks"] == control["spike_ticks"]
    assert first_difference(tmp_path / "early", tmp_path / "control") >= 160
    late_rows = list(rows(tmp_path / "late"))
    actions = [r for r in late_rows[1:] if r["intervention"]]
    assert len(actions) == 1 and actions[0]["executed_tick"] == 400
    assert actions[0]["intervention"]["changes"]


def test_reseed_is_recorded_as_external_cue_not_spontaneous_memory(tmp_path):
    s = intervene(tmp_path / "reseed", 11, 0, "restore_gains_and_reseed", 400, 630)
    assert any(t > 400 for t in s["spike_ticks"][1])
    assert any(t > 400 for t in s["spike_ticks"][7])
    pulses = [r["executed_tick"] for r in rows(tmp_path / "reseed") if r.get("external")]
    assert pulses == [3, 400]


def test_neuron_delivered_plasticity_burst_changes_weights_without_erasing_phase(tmp_path):
    from simulations.active_inference.experiments.composition_plasticity_gate import run as gated
    baseline = gated(tmp_path / "basal", 11, 0, "basal", 840)
    burst = gated(tmp_path / "burst", 11, 0, "burst", 840)
    cut = gated(tmp_path / "cut", 11, 0, "receptor_cut", 840)
    for name in ("basal", "burst", "cut"):
        checked = analyze_gate(tmp_path / name)
        assert checked["ring_exact"] and checked["readout_exact"]
        assert all(v != 0 for v in checked["recurrent_weight_changes"].values())
    assert burst["peak_multiplier"] > 800.
    assert burst["final_state"]["plasticity_rates"]["1"]["used_multiplier"] < 1.+1e-10
    assert burst["recurrent_weight_changes"]["1"] > 10*baseline["recurrent_weight_changes"]["1"]
    assert baseline["recurrent_weight_changes"] == cut["recurrent_weight_changes"]
    assert baseline["peak_multiplier"] == cut["peak_multiplier"] == 1.
