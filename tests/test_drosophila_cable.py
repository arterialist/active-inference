from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from simulations.drosophila.connectome import sha256
from simulations.drosophila.spatial import CONTACT_COLUMNS
from simulations.drosophila.paula import Dynamics, LocalCableGradedNeuron, build_paula
from simulations.drosophila.intervention_probe import run_intervention
from simulations.drosophila.intervention_analysis import compare_apl_models, inspect_record, verify_unobserved
from test_drosophila_interventions import ROOTS, graph


def spatial_fixture(tmp_path):
    directory = tmp_path / "anatomy"
    directory.mkdir()
    contacts = [[i+1, int(ROOTS[0]), int(ROOTS[1]), 201, 101, -1, 0] for i in range(3)]
    contacts += [[i+4, int(ROOTS[1]), int(ROOTS[0]), 103, 301, 2, -1] for i in range(4)]
    contacts += [[8, 0, int(ROOTS[1]), 0, 102, -1, 1],
                 [9, int(ROOTS[1]), 0, 103, 0, 2, -1]]
    np.savez_compressed(directory / "anatomy.npz", node_ids=np.array([101, 102, 103]),
        parents=np.array([-1, 0, 1]), xyz_nm=np.array([[0, 0, 0], [10000, 0, 0], [20000, 0, 0]]),
        radius_nm=np.array([200, 200, 0]), contacts=np.array(contacts, dtype=np.int64),
        pair_source_rows=np.array([1]*3 + [2]*4 + [-1, -1]))
    (directory / "analysis.json").write_text(json.dumps({
        "schema": "flywire-spatial-audit-v1", "root": ROOTS[1], "contact_columns": CONTACT_COLUMNS,
        "anatomy_sha256": sha256(directory / "anatomy.npz"), "fixture": True}))
    return directory


def test_opt_in_keeps_graph_ids_ports_and_coefficients_unchanged(tmp_path):
    g = graph()
    global_p = build_paula(g)
    local_p = build_paula(g, replace(Dynamics(), apl_representation="local_cable"), spatial=spatial_fixture(tmp_path))
    assert global_p.root_to_id == local_p.root_to_id
    assert global_p.drive_ports == local_p.drive_ports
    assert global_p.network.network.connections == local_p.network.network.connections
    for name in ("edge_bindings", "incoming_boundary_ports", "outgoing_boundary_terminals"):
        np.testing.assert_array_equal(getattr(global_p, name), getattr(local_p, name))
    for nid, native in global_p.network.network.neurons.items():
        local = local_p.network.network.neurons[nid]
        for name, value in asdict(native.params).items():
            np.testing.assert_array_equal(value, getattr(local.params, name))
        assert native.upper_t_ref_bound == local.upper_t_ref_bound
        assert native.distances == local.distances
        assert native.synapse_sources == local.synapse_sources
        assert [s.u_i.info for s in native.postsynaptic_points.values()] == [s.u_i.info for s in local.postsynaptic_points.values()]
    apl = local_p.network.network.neurons[1]
    assert isinstance(apl, LocalCableGradedNeuron)
    assert apl.cable_node_ids.tolist() == [101, 102, 103]
    np.testing.assert_array_equal(np.asarray(apl.input_projection.sum(axis=0)), [[1, 0]])
    assert local_p.assumptions["spatial"]["unpaired_contacts_not_driven_or_connected"] == 2


def test_missing_or_mismatched_geometry_cannot_silently_fall_back(tmp_path):
    with pytest.raises(ValueError, match="requires a verified"):
        build_paula(graph(), replace(Dynamics(), apl_representation="local_cable"))
    spatial = spatial_fixture(tmp_path)
    with pytest.raises(ValueError, match="without selecting"):
        build_paula(graph(), spatial=spatial)
    m = json.loads((spatial / "analysis.json").read_text())
    m["root"] = ROOTS[0]
    (spatial / "analysis.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="identify this APL"):
        build_paula(graph(), replace(Dynamics(), apl_representation="local_cable"), spatial=spatial)


@pytest.fixture
def cable_record(tmp_path):
    output = tmp_path / "course"
    run_intervention(graph(), output, "intact", .5, spatial=spatial_fixture(tmp_path), apl_representation="local_cable")
    return output


def test_cable_full_trace_and_unobserved_replay(cable_record):
    report = inspect_record(cable_record)[0]
    assert report["cable"]["checks_passed"]
    assert report["positive_terminal_coefficients"]
    replay = verify_unobserved(graph(), cable_record)
    assert replay["equal_values"]["cable_voltage"] == 225 * 3
    assert replay["equal_values"]["cable_current"] == 224 * 3
    assert replay["equal_values"]["cable_arrived_current"] == 224 * 2
    assert replay["equal_values"]["cable_terminal_release"] == 225
    assert replay["equal_values"]["cable_checks"] == 224 * 2


@pytest.mark.parametrize("field,message", [("cable_voltage", "current conservation"),
                                            ("cable_terminal_release", "Local release")])
def test_recomputed_file_hash_cannot_hide_false_cable_measurement(cable_record, field, message):
    m = json.loads((cable_record / "manifest.json").read_text())
    chunk = m["recording"]["chunks"][4]
    path = cable_record / chunk["file"]
    with np.load(path) as data:
        changed = {key: data[key] for key in data.files}
    changed[field][2, 0] += 1
    np.savez_compressed(path, **changed)
    chunk["sha256"] = sha256(path)
    (cable_record / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match=message):
        inspect_record(cable_record)


def test_local_release_lesion_keeps_intracellular_activity_and_returns(tmp_path):
    output = tmp_path / "blocked"
    run_intervention(graph(), output, "apl_release_block", .5,
                     spatial=spatial_fixture(tmp_path), apl_representation="local_cable")
    report = inspect_record(output)[0]
    assert report["blocked_forward_events"] > 0
    assert report["native_return_events_from_blocked_cells"] > 0
    assert max(report["cable"]["voltage_max_each_tick"]) > 0
    assert report["postsynaptic_coefficients_changed"] > 0


def test_blocked_model_comparison_proves_downstream_equivalence(tmp_path):
    local, global_p = tmp_path / "local", tmp_path / "global"
    run_intervention(graph(), local, "apl_release_block", .5,
                     spatial=spatial_fixture(tmp_path), apl_representation="local_cable")
    run_intervention(graph(), global_p, "apl_release_block", .5)
    result = compare_apl_models(global_p, local)
    assert result["non_apl_all_recorded_fields_identical"]
    assert result["first_exact_field_divergence"]["APL"]["soma"] is not None
    assert result["first_exact_field_divergence"]["APL"]["inputs"] is None


def test_model_comparison_rejects_undeclared_parameter_change(cable_record, tmp_path):
    other = tmp_path / "global"
    run_intervention(graph(), other, "intact", .4)
    with pytest.raises(ValueError, match="Other dynamical parameters"):
        compare_apl_models(other, cable_record)


def test_conserved_total_cannot_hide_wrong_spatial_current(cable_record):
    m = json.loads((cable_record / "manifest.json").read_text())
    chunk = m["recording"]["chunks"][4]
    path = cable_record / chunk["file"]
    with np.load(path) as data:
        changed = {key: data[key] for key in data.files}
    changed["cable_current"][2, 0] += 1
    changed["cable_current"][2, 1] -= 1
    np.savez_compressed(path, **changed)
    chunk["sha256"] = sha256(path)
    (cable_record / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Per-node passive cable equation"):
        inspect_record(cable_record)


def test_satisfied_pde_cannot_hide_wrong_input_port_placement(cable_record):
    m = json.loads((cable_record / "manifest.json").read_text())
    chunk = m["recording"]["chunks"][4]
    path = cable_record / chunk["file"]
    with np.load(path) as data:
        changed = {key: data[key] for key in data.files}
    # Current sum and every node equation still agree. The input-port account
    # is nevertheless inconsistent with where the measured contacts lie.
    changed["cable_arrived_current"][2, 0] += 1
    changed["cable_arrived_current"][2, 1] -= 1
    np.savez_compressed(path, **changed)
    chunk["sha256"] = sha256(path)
    (cable_record / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Spatial input placement"):
        inspect_record(cable_record)


def test_first_input_refinement_retains_every_substep_and_port_contribution(cable_record, tmp_path):
    from simulations.drosophila.cable_response_probe import refine_first_response
    from neuron.extensions.experimental.passive_cable import PassiveCable
    output = tmp_path / "refinement"
    result = refine_first_response(cable_record, output, divisions=(1, 2, 8))
    assert result["exact_original_transition"]
    with np.load(output / "inputs.npz") as data:
        np.testing.assert_allclose(data["per_port_peak_voltage_contribution"].sum(axis=1),
                                   data["recorded_voltage"][data["peak_node_indices"]], rtol=1e-12, atol=1e-12)
        current = data["current"]
    geometry = cable_record.parent / "anatomy" / "anatomy.npz"
    with np.load(geometry) as g:
        cable = PassiveCable(g["parents"], g["xyz_nm"] / 1000, g["radius_nm"] / 1000, 25000)
    for divisions in (1, 2, 8):
        state = np.zeros(3)
        time = 0.0
        a = 1 / (20 * divisions)
        matrix = np.diag(cable.capacity) + a * cable.laplacian.toarray()
        for chunk in [c for c in result["chunks"] if c["divisions"] == divisions]:
            with np.load(output / chunk["file"]) as data:
                assert data["time"][0] == time
                np.testing.assert_allclose(data["voltage"][0], state, atol=1e-11)
                for voltage, t in zip(data["voltage"][1:], data["time"][1:], strict=True):
                    assert t == pytest.approx(time + 1/divisions)
                    state = np.linalg.solve(matrix, (1-a) * cable.capacity * state + a * current)
                    np.testing.assert_allclose(voltage, state, atol=1e-11)
                    time = t
        assert time == 1
    with pytest.raises(FileExistsError):
        refine_first_response(cable_record, output, divisions=(1, 2))
    with pytest.raises(ValueError, match="strictly increasing"):
        refine_first_response(cable_record, output / "invalid", divisions=(2, 1))
