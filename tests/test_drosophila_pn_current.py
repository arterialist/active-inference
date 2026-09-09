"""Current-shape fit independence and unchanged anatomical port contracts."""
import json

import numpy as np
import pytest

from simulations.drosophila.connectome import Subgraph
from simulations.drosophila.paula import build_paula, Dynamics, PNCurrentKernel, Neuron
from simulations.drosophila.pn_current import fit_shape, kernel, compare_shapes, probe
from neuron.extensions.experimental.input_current import InputCurrentNeuron


def graph():
    def node(root, index, kind, label):
        return {"global_index": index, "annotation": {
            "root_id": root, "cell_class": kind, "hemibrain_type": label}}
    return Subgraph(("2", "3", "4"), {
        "1": node("1", 1, "olfactory", "ORN_DL5"),
        "2": node("2", 2, "ALPN", "DL5_adPN"),
        "3": node("3", 3, "Kenyon_Cell", ""),
        "4": node("4", 4, "ALPN", "VA6_adPN"),
        "5": node("5", 5, "olfactory", "ORN_VA6")},
        np.array([[1, 2, 1, 2, 44, 1, 44, 0, 101], [2, 3, 2, 3, 2, 1, 2, 1, 102],
                  [3, 2, 3, 2, 1, 1, 1, 2, 103], [5, 2, 5, 2, 3, 1, 3, 3, 104],
                  [1, 4, 1, 4, 3, 1, 3, 4, 105]], dtype=np.int64), {})


def test_fit_recovers_known_shapes_without_relabeling_ms_as_ticks():
    t = np.linspace(0, 100, 251)
    for taus, fractions in (([14], [1]), ([10, 48], [.82, .18])):
        wave = kernel(t, taus, fractions)
        fit = fit_shape(t, np.stack([wave]*3), len(taus))
        np.testing.assert_allclose(fit["decay_ms"], taus, rtol=1e-5)
        np.testing.assert_allclose(fit["peak_fractions"], fractions, rtol=1e-5)
        assert not fit["near_bound"]


def test_held_out_cell_does_not_influence_its_fitted_parameters():
    t = np.arange(101.)
    waves = np.stack([kernel(t, [10+i], [1]) for i in range(12)])
    a = compare_shapes(t, waves)
    waves[4] = kernel(t, [100], [1])
    b = compare_shapes(t, waves)
    for count in ("1", "2"):
        aa, bb = a[count]["leave_one_cell_out"][4], b[count]["leave_one_cell_out"][4]
        assert aa["decay_ms"] == bb["decay_ms"]
        assert aa["peak_fractions"] == bb["peak_fractions"]
        assert aa["held_out_rmse"] != bb["held_out_rmse"]


def test_selected_type_only_without_changing_anatomy_weights_or_experimental_drive():
    g = graph()
    native = build_paula(g)
    spec = PNCurrentKernel("2", "ORN_DL5", (10, 48), (.82, .18))
    new = build_paula(g, current_kernel=spec)
    for field in ("edge_bindings", "incoming_boundary_ports", "outgoing_boundary_terminals"):
        np.testing.assert_array_equal(getattr(native, field), getattr(new, field))
    for nid, cell in native.network.network.neurons.items():
        other = new.network.network.neurons[nid]
        for name, value in vars(cell.params).items():
            np.testing.assert_equal(value, getattr(other.params, name))
        assert cell.distances == other.distances
        assert [p.u_i.info for p in cell.postsynaptic_points.values()] == [p.u_i.info for p in other.postsynaptic_points.values()]
    cell = new.network.network.neurons[2]
    assert isinstance(cell, InputCurrentNeuron)
    assert cell.current_ports.tolist() == [0]
    assert new.drive_ports["2"] not in cell.current_ports
    assert type(new.network.network.neurons[4]) is Neuron
    assert new.assumptions["input_current"]["source_rows"] == [101]
    assert "input_current" not in native.assumptions
    assert new.network.network.fast_connection_cache.keys() == native.network.network.fast_connection_cache.keys()
    for root in ("3", "99"):
        with pytest.raises(ValueError, match="ALPN"):
            build_paula(g, current_kernel=PNCurrentKernel(root, "ORN_DL5", (10,), (1,)))


def test_tick_recordings_match_independent_convolution_and_charge_accounting(tmp_path):
    g = graph()
    g.save(tmp_path / "graph")
    fit = {"fits": {"2": {"all_cells": {"decay_ms": [10, 48], "peak_fractions": [.82, .18]}}}}
    (tmp_path / "fit.json").write_text(json.dumps(fit))
    result = probe(tmp_path / "graph", tmp_path / "fit.json", tmp_path / "probe")
    assert result["all_incoming_pairs"] == 3  # Includes recurrent KC port in isolated cut.
    assert result["all_outgoing_pairs"] == 1
    with np.load(tmp_path / "probe/per_tick.npz") as trace:
        for record in result["records"]:
            data = trace[record["key"]]
            assert len(data) == 800 and np.isfinite(data).all()
            if record["normalization"] in {"area", "peak"}:
                response = kernel(np.arange(800)*record["ms_per_tick_hypothesis"], [10, 48], [.82, .18])
                if record["normalization"] == "area":
                    response /= record["peak_kernel_discrete_charge_gain"]
                    assert record["total_current"] + record["unrecorded_future_current_sum"] == pytest.approx(record["total_impulse"], abs=1e-10)
                if record["stimulus"] != "current_step":
                    np.testing.assert_allclose(data[:, 3], np.convolve(data[:, 2], response)[:800], atol=1e-12)
                    np.testing.assert_allclose(data[:, 3], data[:, 9:11].sum(axis=1), atol=1e-12)
            if record["stimulus"] == "current_step":
                assert record["spike_ticks"] == next(r["spike_ticks"] for r in result["records"] if r["key"] == "ms1_native_current_step")
    assert result["calibrated_physical_clock"] is None
    with pytest.raises(FileExistsError):
        probe(tmp_path / "graph", tmp_path / "fit.json", tmp_path / "probe")


def test_reunion_preserves_recurrent_ports_and_records_actual_downstream_consumers(tmp_path):
    from simulations.drosophila.pn_current_reunion import run, analyze
    g = graph()
    g.save(tmp_path / "graph")
    fit = {"fits": {"2": {"all_cells": {"decay_ms": [10, 48], "peak_fractions": [.82, .18]}}}}
    (tmp_path / "fit.json").write_text(json.dumps(fit))
    result = run(tmp_path / "graph", tmp_path / "fit.json", tmp_path / "reunion", ticks=240)
    probe(tmp_path / "graph", tmp_path / "fit.json", tmp_path / "isolated")
    checked = analyze(tmp_path / "graph", tmp_path / "reunion", tmp_path / "isolated", tmp_path / "attribution.json")
    assert all(r["exact_recorded_soma_output_replay"] for r in checked["conditions"])
    assert result["anatomy"]["selected_neurons"] == 3
    assert result["anatomy"]["internal"]["directed_pairs"] == 2
    assert result["protocol"]["all_connections_preserved"]
    with np.load(tmp_path / "reunion/peak.npz") as f:
        assert f["soma"].shape == (241, 3, 3)
        assert f["pn_inputs"].shape == (240, 4, 4)
        assert np.any(f["soma"][1:, 0, 1] > 0)
        # Subthreshold KC voltage is an actual native consumer of PN outputs.
        assert np.any(f["soma"][1:, 1, 0] > 0)
        np.testing.assert_array_equal(f["pn_inputs"][:, 0, 0],
            (np.arange(240) < 200) & (np.arange(240) % 5 == 0))
        assert np.count_nonzero(f["pn_post_weight"][-1] != f["pn_post_weight"][0]) > 0
    # A changed trace cannot pass merely because its final totals look plausible.
    with (tmp_path / "reunion/peak.npz").open("ab") as stream:
        stream.write(b"altered")
    with pytest.raises(ValueError, match="hash"):
        analyze(tmp_path / "graph", tmp_path / "reunion", tmp_path / "isolated", tmp_path / "bad.json")
    # Even an internally consistent manifest must not hide a false voltage.
    from simulations.drosophila.prisco import digest
    path = tmp_path / "reunion/peak.npz"
    with np.load(path) as f:
        altered = {key: f[key].copy() for key in f.files}
    altered["soma"][20, 0, 0] += .01
    np.savez_compressed(path, **altered)
    metadata = tmp_path / "reunion/analysis.json"
    manifest = json.loads(metadata.read_text())
    next(r for r in manifest["conditions"] if r["condition"] == "peak")["trace_sha256"] = digest(path)
    metadata.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        analyze(tmp_path / "graph", tmp_path / "reunion", tmp_path / "isolated", tmp_path / "false-voltage.json")
