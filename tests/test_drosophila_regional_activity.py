import json

import numpy as np
import pytest

from simulations.drosophila.connectome import sha256
from simulations.drosophila.regional_activity import SOURCES, analyze, contact_projection, regional_balance, weighted_mean
from test_drosophila_cable import cable_record


def binding(record, tmp_path):
    directory = tmp_path / "regions"
    directory.mkdir()
    m = json.loads((record / "manifest.json").read_text())
    np.savez_compressed(directory / "node_regions.npz", node_ids=[101, 102, 103],
        xyz_nm=[[0, 0, 0], [10000, 0, 0], [20000, 0, 0]],
        inside=np.array([[True, False, False], [False, False, False], [False, True, False]]),
        ambiguous=np.zeros((3, 3), dtype=bool), region_names=["input_region", "output_region", "empty_region"])
    (directory / "analysis.json").write_text(json.dumps({
        "schema": "flywire-apl-neuropil-binding-v1", "root": m["assumptions"]["spatial"]["root"],
        "anatomy_sha256": m["assumptions"]["spatial"]["anatomy_sha256"],
        "node_regions_sha256": sha256(directory / "node_regions.npz"), "fixture": True}))
    return directory


def test_contact_fractions_use_all_contacts_and_keep_overlaps():
    masks = np.array([[1, 0], [0, 1], [1, 1]], dtype=bool)
    fractions, counts = contact_projection(np.array([0, 0, 1, 2]), np.array([0, 0, 0, 1]), 3, masks)
    np.testing.assert_array_equal(counts, [3, 1, 0])
    np.testing.assert_allclose(fractions, [[2/3, 1/3], [1, 1], [0, 0]])


def test_area_weighted_voltage_is_not_node_average_or_peak_normalized():
    values = np.array([[0, 10, 20], [5, 10, 0]])
    weights = np.array([[1, 0], [3, 0], [0, 0]])
    result = weighted_mean(values, weights)
    np.testing.assert_allclose(result[:, 0], [7.5, 8.75])
    assert np.isnan(result[:, 1]).all()  # Absence is not silently reported as silence.


def test_axial_inflow_sign_and_whole_cell_conservation():
    voltage = np.array([[0., 0.], [.3, .1]])
    capacity = np.array([.5, .5])
    masks = np.array([[1, 0, 1], [0, 1, 1]])
    currents = np.array([[1., 0., 1.]])
    np.testing.assert_allclose(regional_balance(voltage, capacity, masks, currents, .2), [[-.25, .25, 0]], atol=1e-15)


def test_full_readout_retains_ticks_sources_and_missing_regions(cable_record, tmp_path):
    regions = binding(cable_record, tmp_path)
    output = tmp_path / "readout"
    result = analyze(cable_record, regions, output)
    assert result["max_regional_input_reconstruction_error"] < 1e-10
    with np.load(output / "per_tick.npz") as f:
        assert f["state_after_tick"].tolist() == list(range(-1, 224))
        assert f["tick"].tolist() == list(range(224))
        current = f["input_signed"]
        assert np.any(current[:, 0, SOURCES.index("KC")])
        assert not current[:, 1].any()  # No direct current at the output attachment.
        assert np.any(f["axial_inflow"][:, 1] > 0)
        assert np.nanmax(f["kc_output_release_mean"][:, 1]) > 0
        assert np.isnan(f["voltage_mean"][:, 2]).all()
        assert not current[:, :, SOURCES.index("experimental")].any()
        np.testing.assert_allclose(f["axial_inflow"][:, -1], 0, atol=1e-10)
        np.testing.assert_allclose(f["input_positive"] + f["input_negative"], current, atol=1e-12)
    audit = json.loads((output / "full_record_audit.json").read_text())
    assert audit["cable"]["checks_passed"]
    with pytest.raises(FileExistsError):
        analyze(cable_record, regions, output)


def test_rehashed_binding_cannot_swap_recorded_node_identities(cable_record, tmp_path):
    regions = binding(cable_record, tmp_path)
    path = regions / "node_regions.npz"
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files}
    arrays["node_ids"] = arrays["node_ids"][::-1]
    np.savez_compressed(path, **arrays)
    metadata = json.loads((regions / "analysis.json").read_text())
    metadata["node_regions_sha256"] = sha256(path)
    (regions / "analysis.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="identities or coordinates"):
        analyze(cable_record, regions, tmp_path / "bad")


def test_no_negative_pn_current_produces_zero_component(cable_record, tmp_path):
    regions = binding(cable_record, tmp_path)
    output = tmp_path / "split"
    analyze(cable_record, regions, output, decompose_negative_pn=True)
    with np.load(output / "per_tick.npz") as f:
        assert np.nanmax(np.abs(f["negative_pn_voltage_mean"])) == 0
        np.testing.assert_allclose(f["other_recorded_current_voltage_mean"], f["voltage_mean"], equal_nan=True)
        np.testing.assert_allclose(f["other_recorded_current_all_output_release_mean"],
                                   f["all_output_release_mean"], equal_nan=True)


def test_negative_pn_decomposition_matches_independent_dense_replay(tmp_path):
    from test_drosophila_cable import spatial_fixture
    from test_drosophila_input_panels import recruited_pn_graph
    from simulations.drosophila.intervention_probe import run_intervention
    from neuron.extensions.experimental.passive_cable import PassiveCable

    g = recruited_pn_graph()
    spatial = spatial_fixture(tmp_path)
    path = spatial / "anatomy.npz"
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files}
    # The recruited inhibitory PN has one contact at the middle cable node.
    arrays["contacts"] = np.vstack([arrays["contacts"],
        [10, int(g.selected[-1]), int(g.selected[1]), 500, 102, -1, 1]])
    arrays["pair_source_rows"] = np.r_[arrays["pair_source_rows"], 4]
    np.savez_compressed(path, **arrays)
    meta = json.loads((spatial / "analysis.json").read_text())
    meta["anatomy_sha256"] = sha256(path)
    (spatial / "analysis.json").write_text(json.dumps(meta))
    record, output = tmp_path / "course", tmp_path / "split"
    run_intervention(g, record, "intact", .5, spatial=spatial,
        apl_representation="local_cable", pn_drive_panel="without_inhibitory_drive")
    result = analyze(record, binding(record, tmp_path), output, decompose_negative_pn=True)
    assert "not a connected-network lesion" in result["negative_pn_decomposition"]["scope"]
    audit = json.loads((output / "full_record_audit.json").read_text())
    delivery, = audit["pn_input_selection"]["first_undriven_PN_spike"]["apl_deliveries"]
    assert delivery["recorded_cable_arrived_current"] == pytest.approx(-.5 * .95**2)

    cable = PassiveCable(arrays["parents"], arrays["xyz_nm"] / 1000, arrays["radius_nm"] / 1000, 25000)
    alpha = 1/20
    matrix = np.diag(cable.capacity) + alpha * cable.laplacian.toarray()
    manifest = json.loads((record / "manifest.json").read_text())
    with np.load(record / "columns.npz") as c:
        edge, = c["edge_bindings"][c["edge_bindings"][:, 1] == 3]
        pn_port = int(edge[4])
    negative, remainder = np.zeros(3), np.zeros(3)
    expected_negative, expected_remainder = [negative.copy()], [remainder.copy()]
    for chunk in manifest["recording"]["chunks"]:
        with np.load(record / chunk["file"]) as f:
            for current, arrived in zip(f["cable_current"], f["cable_arrived_current"], strict=True):
                selected = np.array([0, min(0, arrived[pn_port]), 0])
                negative = np.linalg.solve(matrix, (1-alpha)*cable.capacity*negative + alpha*selected)
                remainder = np.linalg.solve(matrix, (1-alpha)*cable.capacity*remainder + alpha*(current-selected))
                expected_negative.append(negative.copy())
                expected_remainder.append(remainder.copy())
    with np.load(output / "per_tick.npz") as f:
        assert np.nanmin(f["negative_pn_voltage_mean"]) < 0
        np.testing.assert_allclose(f["negative_pn_voltage_mean"][:, :2], np.array(expected_negative)[:, [0, 2]], atol=1e-10)
        np.testing.assert_allclose(f["other_recorded_current_voltage_mean"][:, :2], np.array(expected_remainder)[:, [0, 2]], atol=1e-10)
        np.testing.assert_allclose(f["negative_pn_voltage_mean"] + f["other_recorded_current_voltage_mean"],
                                   f["voltage_mean"], atol=1e-10, equal_nan=True)
        # Rectify the reconstructed node voltage before region averaging.
        np.testing.assert_allclose(f["other_recorded_current_all_output_release_mean"][:, 1],
                                   np.clip(np.array(expected_remainder)[:, 2] * .01, 0, 1), atol=1e-10)
        assert np.isnan(f["negative_pn_voltage_mean"][:, 2]).all()
