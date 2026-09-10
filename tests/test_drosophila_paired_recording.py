import copy

import numpy as np
import pytest

from simulations.drosophila.connectome import Subgraph
from simulations.drosophila.paula import Dynamics, build_paula
from simulations.drosophila.paired_recording import pair_cut, course, simulate, audit, summarize


def fixture():
    def node(r, cls, typ):
        return {"global_index": int(r), "annotation": {"root_id": r, "cell_class": cls, "hemibrain_type": typ}}
    g = Subgraph(("2", "3", "4"), {"1": node("1", "olfactory", "ORN_DL5"),
        "2": node("2", "ALPN", "DL5_adPN"), "3": node("3", "ALLN", "lLN2T_e"),
        "4": node("4", "Kenyon_Cell", "KC")}, np.array([
        [1, 2, 1, 2, 40, 1, 40, 0, 101], [2, 3, 2, 3, 94, 1, 94, 0, 102],
        [3, 2, 3, 2, 36, 1, 36, 0, 103], [3, 4, 3, 4, 20, 1, 20, 0, 104],
        [4, 3, 4, 3, 5, 1, 5, 0, 105]], dtype=np.int64), {})
    intrinsic = {"root": "2", "proposals": {"pooled_control": {"rheobase_pa": 25., "lambda_ms": 60.}},
                 "joint_unitary": {"per_count_weight": .037}, "source_hashes": {}}
    return g, intrinsic, {"decay_ms": [10., 48.], "peak_fractions": [.82, .18]}


def test_pair_cut_keeps_actual_interconnections_and_boundary_port_identity():
    g, _, _ = fixture()
    cut = pair_cut(g, ("2", "3"))
    np.testing.assert_array_equal(cut.edges, g.edges)
    assert cut.summary()["internal"]["synapses"] == 130
    a, b = build_paula(g), build_paula(cut, electrical_roots=cut.selected)
    for r in cut.selected:
        ac, bc = (p.network.network.neurons[p.root_to_id[r]] for p in (a, b))
        assert ac.params.num_inputs == bc.params.num_inputs
        assert ac.distances == bc.distances
        assert ac.upper_t_ref_bound == bc.upper_t_ref_bound
        assert a.drive_ports[r] == b.drive_ports[r]
    with pytest.raises(ValueError, match="boundary"):
        pair_cut(g, ("1", "2"))
    for roots in (("2", "2"), ("1", "2")):
        with pytest.raises(ValueError):
            build_paula(g, electrical_roots=roots)
    assert "electrical_input" not in a.assumptions


def test_signed_course_and_subthreshold_polarity_are_symmetric():
    g, intrinsic, tail = fixture(); g = pair_cut(g, ("2", "3"))
    data = []
    for level in (-.5, .5):
        a, meta = simulate(g, intrinsic, tail, "2", level, .03, False, trials=1, period=600)
        assert not a["trace"][:, :, 2].any()
        assert not a["inputs"].any()
        summary = summarize(a, meta)[0]
        assert 0 < summary["subthreshold_transfer_ratio"] < .03/1.03
        data.append(a)
    np.testing.assert_array_equal(data[0]["trace"][:, :, :2], -data[1]["trace"][:, :, :2])
    command, epochs = course(-.5)
    assert len(command) == 10500 and len(epochs) == 2
    assert command[499] == 0 and command[500] == -.5 and command[1000] == 0


def test_spiking_parity_and_release_block_preserve_native_adaptation():
    g, intrinsic, tail = fixture(); g = pair_cut(g, ("2", "3"))
    for blocked in (False, True):
        args = (g, intrinsic, tail, "2", 4., 0., blocked)
        a, meta = simulate(*args, trials=1, duration=500, period=800, baseline=100)
        b, _ = simulate(*args, trials=1, duration=500, period=800, baseline=100, ordinary=True)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        assert a["events"][:, 0, 0].sum() > 0
        if blocked:
            assert not a["inputs"].any() and meta["changed_weights"] == 0
        else:
            assert a["inputs"].any() and a["events"][:, :, 2].sum() > 0
            assert meta["changed_weights"] > 0
        assert summarize(a, meta)[0]["subthreshold_transfer_ratio"] is None


def test_equation_audit_rejects_forged_gap_and_current_hidden_by_spike_reset():
    g, intrinsic, tail = fixture(); g = pair_cut(g, ("2", "3"))
    a, meta = simulate(g, intrinsic, tail, "2", 4., .03, False,
                       trials=1, duration=100, period=200, baseline=100)
    spike = np.flatnonzero(a["trace"][:, 0, 2])[0]
    for col in (8, 9):
        bad = copy.deepcopy(a)
        bad["trace"][spike, 0, col] += .1
        with pytest.raises(AssertionError):
            audit(bad, .03, meta["lambda_ticks"], False)


def test_full_record_replay_rejects_a_fabricated_weight_even_with_updated_hash(tmp_path):
    import json
    from simulations.drosophila.paired_recording import run
    from simulations.drosophila.electrical_analysis import replay_pairs
    from simulations.drosophila.prisco import dump_new, digest
    g, intrinsic, tail = fixture()
    g.save(tmp_path/"graph")
    dump_new(tmp_path/"intrinsic.json", intrinsic)
    dump_new(tmp_path/"tail.json", {"fits": {"2": {"all_cells": tail}}})
    meta = run(tmp_path/"graph", tmp_path/"intrinsic.json", tmp_path/"tail.json", tmp_path/"run",
        roots=("2", "3"), conductances=(.03,), levels=(4.,),
        trials=1, duration=100, period=200, baseline=100)
    replay = replay_pairs(tmp_path/"run", tmp_path/"intrinsic.json", tmp_path/"tail.json")
    assert replay["courses"] == 4 and replay["exact_replayed_cell_ticks"] == 2400
    record = meta["records"][0]; path = tmp_path/"run"/record["file"]
    with np.load(path) as f:
        arrays = {key: f[key] for key in f.files}
    arrays["weights"][75, 0, 0] += .1
    np.savez_compressed(path, **arrays)
    record["sha256"] = digest(path)
    (tmp_path/"run/analysis.json").write_text(json.dumps(meta))
    with pytest.raises(AssertionError):
        replay_pairs(tmp_path/"run", tmp_path/"intrinsic.json", tmp_path/"tail.json")
