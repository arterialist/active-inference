"""Scientific current-injection boundaries and rate observation operators."""
import numpy as np
import pytest

from simulations.drosophila.electrophysiology import CurrentElectrode
from simulations.drosophila.pn_intrinsic import (
    coarse_curve, continuous_rate, fit_proposal, window_rates, run_ramp,
    joint_synaptic_probe, audit_ramp_trace,
    prepared_cell,
)
from simulations.drosophila.paula import Neuron
from simulations.drosophila.connectome import Subgraph
from neuron.neuron import NeuronParameters, PostsynapticPoint, PostsynapticInputVector, RetrogradeSignalEvent


def cell():
    c = Neuron(1, NeuronParameters(num_inputs=2, eta_post=1e-6, eta_retro=1e-6), log_level="CRITICAL")
    for p, weight in enumerate([2., -1.]):
        c.postsynaptic_points[p] = PostsynapticPoint(PostsynapticInputVector(weight, 0, np.zeros(2)))
        c.distances[p] = 2
        c.register_source(p, p+10, 0)
    return c


def test_zero_electrode_is_bit_exact_and_other_cells_are_not_injected():
    native, measured, other = cell(), cell(), cell()
    with CurrentElectrode(measured, np.zeros(30)):
        for tick in range(30):
            for c in (native, measured):
                if tick % 5 == 0:
                    c.input_buffer[0, 0] = 4
            a, b = native.tick({}, tick), measured.tick({}, tick)
            other.tick({}, tick)
            assert native.S == measured.S and native.O == measured.O
            assert repr(a) == repr(b)
            assert other.S == other.O == 0
    for p in (0, 1):
        assert native.postsynaptic_points[p].u_i.info == measured.postsynaptic_points[p].u_i.info


def test_signed_electrical_current_is_not_a_plastic_receptor():
    c = cell()
    initial = [p.u_i.info for p in c.postsynaptic_points.values()]
    course = np.r_[np.full(20, .2), np.full(20, -.1)]
    expected = 0.
    with CurrentElectrode(c, course) as instrument:
        for t, current in enumerate(course):
            events = c.tick({}, t)
            expected += (-expected+current)/20
            assert c.S == pytest.approx(expected, abs=1e-15)
            assert not events and not c.input_buffer.any()
    np.testing.assert_array_equal(instrument.native_current, 0)
    np.testing.assert_array_equal(instrument.total_current, course)
    assert [p.u_i.info for p in c.postsynaptic_points.values()] == initial
    assert c.params.eta_post > 0 and c.params.eta_retro > 0 and not c._ablation


def test_real_synaptic_input_and_plasticity_continue_during_injection():
    c = cell()
    count = 0
    with CurrentElectrode(c, np.ones(15)*.2) as instrument:
        for t in range(15):
            if t == 3:
                c.input_buffer[0, 0] = 1
            events = c.tick({}, t)
            count += sum(isinstance(e, RetrogradeSignalEvent) for e in events)
    assert count == 1 and c.postsynaptic_points[0].u_i.info != 2
    assert np.flatnonzero(instrument.native_current).tolist() == [5]


def test_incomplete_or_out_of_order_injection_restores_model_method():
    original = Neuron._hillock_current
    c = cell()
    with pytest.raises(ValueError, match="Incomplete"):
        with CurrentElectrode(c, [1, 1]):
            c.tick({}, 0)
    assert Neuron._hillock_current is original
    with pytest.raises(ValueError, match="consecutive"):
        with CurrentElectrode(c, [1, 1]):
            c.tick({}, 4)
    assert Neuron._hillock_current is original


def test_firing_window_has_declared_half_open_boundaries():
    np.testing.assert_array_equal(window_rates([0, 25, 50, 75], [25, 50]), [40, 40])
    np.testing.assert_array_equal(window_rates([0, 25, 50, 75], [0], alignment="trailing"), [0])
    np.testing.assert_array_equal(window_rates([0, 25, 50, 75], [0], alignment="leading"), [40])


def test_known_effective_curve_is_recovered_and_missing_bins_fail():
    x = np.linspace(0, 100, 891)
    y = continuous_rate(x, 31.2, 73.)
    result = fit_proposal(x, np.stack([y, y, y]))
    assert result["rheobase_pa"] == pytest.approx(31.2, rel=1e-6)
    assert result["lambda_ms"] == pytest.approx(73., rel=1e-6)
    fixed = fit_proposal(x, np.stack([y]*3), fixed_lambda_ms=73.)
    assert fixed["rheobase_pa"] == pytest.approx(31.2, rel=1e-6)
    assert fixed["lambda_ms"] == 73.
    assert coarse_curve(x, np.stack([y, y])).shape == (2, 18)
    with pytest.raises(ValueError, match="support"):
        coarse_curve(x[:30], y[:30])


def miniature():
    def node(r, typ, label):
        return {"global_index": int(r), "annotation": {"root_id": r, "cell_class": typ, "hemibrain_type": label}}
    return Subgraph(("2",), {"1": node("1", "olfactory", "ORN_DL5"),
                             "2": node("2", "ALPN", "DL5_adPN"),
                             "3": node("3", "olfactory", "ORN_DL5")},
        np.array([[1, 2, 1, 2, 20, 1, 20, 0, 101], [3, 2, 3, 2, 60, 1, 60, 1, 102]], dtype=np.int64), {})


def test_actual_neuron_ramp_and_all_partner_unitary_calibration():
    proposal = {"rheobase_pa": 25., "lambda_ms": 60.}
    tail = {"decay_ms": [10., 48.], "peak_fractions": [.82, .18]}
    full = miniature()
    full.selected = ("2", "1")
    with pytest.raises(ValueError, match="isolated"):
        prepared_cell(full, "2", proposal, 1., tail)
    trace, rates, result = run_ramp(miniature(), "2", proposal, 1., tail, np.arange(101.))
    assert trace.shape[0] > 22000 and np.isfinite(trace).all()
    assert not np.any(trace[:, 2])  # No native chemical current.
    np.testing.assert_array_equal(trace[:, 3], trace[:, 1]/25)
    assert result["spike_count"] > 0 and result["adaptation_enabled"]
    assert result["equation_audit"]["command_and_spikes_exact"]
    false = trace.copy()
    # A reset can hide an altered current in voltage alone, but not this audit.
    spike = int(np.flatnonzero(trace[:, 6])[0])
    false[spike, 3] += .01
    with pytest.raises(AssertionError):
        audit_ramp_trace(false, proposal, 1., 3)
    false = trace.copy()
    false[40, 5] += .01
    with pytest.raises(AssertionError):
        audit_ramp_trace(false, proposal, 1., 3)
    assert set(rates) == {"center", "leading", "trailing"}
    joint, unitary = joint_synaptic_probe(miniature(), "2", proposal, tail, 35.)
    peaks = [r["peak_current_pa"] for r in joint["records"]]
    for name in ("legacy", "gain_only", "integration_only", "joint"):
        assert unitary[f"train_{name}"].shape == (800, 6)
    assert joint["train"]["source_row"] == 101
    assert not joint["train"]["synaptic_depression_included"]
    assert np.mean(peaks) == pytest.approx(35, rel=1e-6)
    assert peaks[1]/peaks[0] == pytest.approx(3)
    for record in joint["records"]:
        assert record["weight_initial"] != record["weight_final"]
        assert len(unitary[f"row_{record['source_row']}"]) == 250
