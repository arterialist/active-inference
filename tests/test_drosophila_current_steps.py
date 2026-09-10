"""Keep electrical interventions, cellular calibration and closed-loop lesions separate."""
import json

import numpy as np
import pytest

from simulations.drosophila.connectome import Subgraph
from simulations.drosophila.electrophysiology import CurrentElectrode
from simulations.drosophila.paula import Dynamics, build_paula
from simulations.drosophila.pn_current_steps import (
    step_course, prepare, release_observer, audit_trace, run,
)
from simulations.drosophila.prisco import dump_new
from simulations.drosophila.pn_step_analysis import replay_condition


def fixture():
    def node(r, cls, typ):
        return {"global_index": int(r), "annotation": {"root_id": r, "cell_class": cls, "hemibrain_type": typ}}
    graph = Subgraph(("2", "3"), {"1": node("1", "olfactory", "ORN_DL5"),
        "2": node("2", "ALPN", "DL5_adPN"), "3": node("3", "APL", "APL")},
        np.array([[1, 2, 1, 2, 40, 1, 40, 0, 101],
                  [2, 3, 2, 3, 100, 1, 100, 1, 102],
                  [3, 2, 3, 2, 100, -1, -100, 2, 103]], dtype=np.int64), {})
    intrinsic = {"root": "2", "proposals": {"pooled_control": {"rheobase_pa": 25., "lambda_ms": 60.}},
                 "joint_unitary": {"per_count_weight": .037}, "source_hashes": {}}
    tail = {"decay_ms": [10., 48.], "peak_fractions": [.82, .18]}
    return graph, intrinsic, tail


def test_course_is_complete_and_uses_half_open_one_second_epochs():
    command, epochs = step_course()
    assert len(command) == 9000
    for e in epochs:
        assert e["stop"]-e["start"] == 1000
        np.testing.assert_array_equal(command[e["start"]:e["stop"]], e["current_pa"])
    assert command[999] == 0 and command[1000] == 25 and command[1999] == 25 and command[2000] == 0
    for bad in ([], [np.nan], [-1]):
        with pytest.raises(ValueError):
            step_course(bad)


def test_override_does_not_change_other_cells_or_any_bindings():
    graph, intrinsic, tail = fixture()
    original = build_paula(graph, Dynamics(weight_per_count=.075))
    prep, target = prepare(graph, intrinsic, tail)
    for name in ("edge_bindings", "incoming_boundary_ports", "outgoing_boundary_terminals"):
        np.testing.assert_array_equal(getattr(prep, name), getattr(original, name))
    assert target.params.lambda_param == 60
    assert target.postsynaptic_points[0].u_i.info == pytest.approx(40*.037)
    assert target.postsynaptic_points[1].u_i.info == -7.5
    other = prep.network.network.neurons[3]
    assert other.params.lambda_param == 20
    assert other.postsynaptic_points[0].u_i.info == 7.5
    assert prep.network.current_tick == 0 and target.params.eta_post > 0
    intrinsic["root"] = "3"
    with pytest.raises(ValueError, match="different neuron"):
        prepare(graph, intrinsic, tail)


def test_closed_loop_release_block_retains_apl_activity_and_return_events():
    graph, intrinsic, tail = fixture()
    results = []
    for blocked in (False, True):
        prep, target = prepare(graph, intrinsic, tail)
        apl = prep.network.network.neurons[3]
        samples, events = [], []
        original = type(apl).tick
        with CurrentElectrode(target, np.full(500, 4.)) as electrode, \
                release_observer(apl, blocked, lambda t, a, d, r: events.append((a,d,r))):
            for t in range(500):
                prep.network.run_tick()
                samples.append((target.S, target.O, apl.S, apl.O))
        assert type(apl).tick is original
        samples, events = np.array(samples), np.array(events)
        assert np.any(samples[:, 3] > 0) and events[:, 0].sum() > 0 and events[:, 2].sum() > 0
        assert apl.params.eta_post > 0 and not apl._ablation
        if blocked:
            np.testing.assert_array_equal(events[:, 1], 0)
            np.testing.assert_array_equal(electrode.native_current, 0)
        else:
            np.testing.assert_array_equal(events[:, 0], events[:, 1])
            assert np.any(electrode.native_current < 0)
        results.append(samples)
    assert not np.array_equal(results[0][:, :2], results[1][:, :2])


def test_recorded_steps_are_audited_per_tick_and_forgery_fails(tmp_path):
    graph, intrinsic, tail = fixture()
    graph.save(tmp_path/"graph")
    dump_new(tmp_path/"intrinsic.json", intrinsic)
    dump_new(tmp_path/"tail.json", {"fits": {"2": {"all_cells": tail}}})
    result = run(tmp_path/"graph", tmp_path/"intrinsic.json", tmp_path/"tail.json",
                 tmp_path/"run", "isolated", levels=(100.,), chunk=311)
    assert len(result["chunks"]) == 10 and result["selected_cells"] == 1
    assert not result["protocol"]["measured_step_responses_available"]
    a = np.concatenate([np.load(tmp_path/"run"/r["file"])["trace"] for r in result["chunks"]])
    command, _ = step_course((100.,))
    audit_trace(a, command, intrinsic["proposals"]["pooled_control"])
    false = a.copy()
    tick = np.flatnonzero(a[:, 6])[0]
    false[tick, 3] += .1  # The reset can conceal forged current from a voltage-only audit.
    with pytest.raises(AssertionError):
        audit_trace(false, command, intrinsic["proposals"]["pooled_control"])
    replay, _, _ = replay_condition(graph, intrinsic, tail, tmp_path/"run")
    assert replay["exact_replay_ticks"] == 3000
    assert replay["active_target_inputs"] == []
    assert replay["target_post_coefficients_changed"] == 0
    # A self-consistent hash cannot certify fabricated postsynaptic learning.
    from simulations.drosophila.prisco import digest
    path = tmp_path/"run"/result["chunks"][0]["file"]
    with np.load(path) as f:
        changed = {key: f[key] for key in f.files}
    changed["post_weight"][20, 0] += .01
    np.savez_compressed(path, **changed)
    result["chunks"][0]["sha256"] = digest(path)
    (tmp_path/"run/analysis.json").write_text(json.dumps(result))
    with pytest.raises(AssertionError):
        replay_condition(graph, intrinsic, tail, tmp_path/"run")
    false = a.copy()
    false[42, 5] += .01
    with pytest.raises(AssertionError):
        audit_trace(false, command, intrinsic["proposals"]["pooled_control"])
