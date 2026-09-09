"""Independent source-arithmetic, waveform and native-kernel checks."""
import json

import numpy as np
import pytest

from simulations.drosophila.gugel import check_formulas, waveform_metrics


def test_formula_dependencies_resolved_from_constants():
    values = {"A1": 3.0, "A2": 2.8875, "A3": 2.775}
    formulas = {"A3": "=A2-0.1125", "A2": "=A1-0.1125"}
    result = check_formulas(values, formulas)
    assert result["formula_count"] == 2
    assert result["maximum_absolute_residual"] < 1e-14
    assert values["A3"] == 2.775
    with pytest.raises(ValueError, match="disagrees"):
        check_formulas({**values, "A3": 2.7}, formulas)


@pytest.mark.parametrize("values,formulas,message", [
    ({"A1": 0, "A2": 0}, {"A1": "=A2+0", "A2": "=A1+0"}, "cycle"),
    ({"A1": 2}, {"A1": "=A2+1"}, "Missing"),
    ({"A1": None, "A2": 1}, {"A1": "=A2+1"}, "Missing/non-numeric"),
    ({"A1": 2}, {"A1": "=SUM(A2:A3)"}, "Unsupported"),
])
def test_formula_failures_do_not_become_numbers(values, formulas, message):
    with pytest.raises(ValueError, match=message):
        check_formulas(values, formulas)


def test_exponential_waveform_against_analytic_integral():
    time = np.arange(0, 201, 0.1)
    inward = np.where(time >= 50, 30*np.exp(-(time-50)/15), 0)
    current = 2-inward
    result = waveform_metrics(time, current)
    assert result["baseline_pa"] == 2
    assert result["peak_inward_pa"] == 30
    assert result["first_1_over_e_crossing_ms"] == pytest.approx(15, abs=1e-10)
    for end in (50, 100):
        analytic = 30*15*(1-np.exp(-end/15))
        expected_tail = (np.exp(-5/15)-np.exp(-end/15))/(1-np.exp(-end/15))
        measured = result["postpeak_windows_ms"][str(end)]
        assert measured["postpeak_charge_pa_ms"] == pytest.approx(analytic, rel=4e-6)
        assert measured["charge_after_5ms_fraction"] == pytest.approx(expected_tail, abs=1e-12)
    np.testing.assert_array_equal(current, 2-inward)


def test_waveform_is_not_rectified_and_preserves_real_zeros():
    time = np.arange(201, dtype=float)
    current = np.zeros(201)
    current[50:60] = -10
    current[70:100] = 1
    result = waveform_metrics(time, current)
    # Signed trapezoidal postpeak area = 95 - 30, not 95 or 125.
    assert result["postpeak_windows_ms"]["50"]["postpeak_charge_pa_ms"] == 65
    with pytest.raises(ValueError, match="No inward|Insufficient"):
        waveform_metrics(time, np.zeros(201))


@pytest.mark.parametrize("time,current", [
    ([0, 1, 1], [0, -1, 0]), ([0, 1, 2], [0, np.nan, 0]),
    ([0, 1, 2], [0, -1]), ([0, 1, 2], [0, -1, 0]),
])
def test_missing_or_bad_waveform_support_fails(time, current):
    with pytest.raises(ValueError):
        waveform_metrics(time, current)


def test_native_probe_preserves_ports_and_records_current_separately(tmp_path):
    from simulations.drosophila.connectome import Subgraph
    from simulations.drosophila.gugel import native_input_probe

    def node(root, index, kind, label):
        return {"global_index": index, "annotation": {
            "root_id": root, "cell_class": kind, "hemibrain_type": label}}
    graph = Subgraph(("2",), {"1": node("1", 1, "olfactory", "ORN_DL5"),
                              "2": node("2", 2, "ALPN", "DL5_adPN"),
                              "3": node("3", 3, "Kenyon_Cell", "")},
                     np.array([[1, 2, 1, 2, 44, 1, 44, 0, 101],
                               [2, 3, 2, 3, 2, 1, 2, 1, 102]], dtype=np.int64), {})
    graph.save(tmp_path / "graph")
    result = native_input_probe(tmp_path / "graph", tmp_path / "probe")
    assert result["effective_current_nonzero_ticks"] == [2]
    assert result["preserved_incoming_pairs"] == 1
    assert result["preserved_outgoing_pairs"] == 1
    assert result["num_inputs_with_experimental_port"] == 2
    assert result["weight_initial"] != result["weight_final"]
    assert result["physical_seconds_per_tick"] is None
    with np.load(tmp_path / "probe/per_tick.npz") as f:
        trace = f["trace"]
    assert trace[3, 4] == 0
    assert trace[3, 2] > 0
    # Full recorded response agrees with a separately computed passive decay.
    np.testing.assert_allclose(trace[2:, 2], trace[2, 2] * 0.95**np.arange(158), rtol=2e-6)
    assert json.loads((tmp_path / "probe/analysis.json").read_text())["source_row"] == 101
    with pytest.raises(FileExistsError):
        native_input_probe(tmp_path / "graph", tmp_path / "probe")
