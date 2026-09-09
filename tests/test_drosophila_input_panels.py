import json
import numpy as np
import pytest

from simulations.drosophila.input_panels import make_panel, recorded_drive_rows
from simulations.drosophila.intervention_probe import make_course, run_intervention
from simulations.drosophila.intervention_analysis import inspect_record, compare, verify_unobserved
from simulations.drosophila.paula import build_paula
from test_drosophila_interventions import graph, ROOTS


def negative_pn_graph():
    g = graph()
    g.edges[0, 5:7] *= -1
    # Deliberately conflicting label: the control uses the declared model sign.
    g.nodes[ROOTS[2]]["annotation"]["known_nt"] = "acetylcholine"
    return g


def test_model_sign_control_does_not_guess_from_transmitter_label():
    panel = make_panel(negative_pn_graph(), "without_inhibitory_drive")
    assert panel["drive_roots"] == []
    assert panel["excluded_roots"] == [ROOTS[2]]
    assert panel["catalog"][0]["known_nt"] == "acetylcholine"
    assert panel["catalog"][0]["model_sign"] == -1


@pytest.mark.parametrize("label,excluded", [("VP2_adPN", True), ("DA1_lPN,VP1m+_lvPN", True),
                                          ("VP11_adPN", False), ("", False), ("M_vPNml50", False)])
def test_vp_control_uses_explicit_labels_not_a_claim_of_exclusive_modality(label, excluded):
    g = graph()
    g.nodes[ROOTS[2]]["annotation"]["hemibrain_type"] = label
    panel = make_panel(g, "without_vp_drive")
    assert bool(panel["excluded_roots"]) == excluded


def test_input_panel_does_not_rebuild_or_silence_neurons():
    g = negative_pn_graph()
    p = build_paula(g)
    objects = list(p.network.network.neurons.values())
    connections = list(p.network.network.connections)
    baseline = make_course(p, g)
    selected = make_course(p, g, "without_inhibitory_drive")
    assert baseline[0].any() and not selected[0].any()
    for i in (1, 2):
        np.testing.assert_array_equal(baseline[i], selected[i])
    assert list(p.network.network.neurons.values()) == objects
    assert p.network.network.connections == connections
    assert all(c.params.eta_post > 0 and c.params.eta_retro > 0 for c in objects)
    assert all(not c._ablation for c in objects)


def test_record_audit_replay_and_comparison_enforce_selected_inputs(tmp_path):
    g = negative_pn_graph()
    baseline, selected = tmp_path / "all", tmp_path / "subset"
    run_intervention(g, baseline, "intact", .5)
    run_intervention(g, selected, "intact", .5, pn_drive_panel="without_inhibitory_drive")
    assert inspect_record(selected)[0]["recording_checks_passed"]
    assert verify_unobserved(g, selected)["ticks"] == 224
    with np.load(baseline / "columns.npz") as a, np.load(selected / "columns.npz") as b:
        assert a.files == b.files
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    with pytest.raises(ValueError, match="driven PN subsets"):
        compare(baseline, selected)
    m = json.loads((selected / "manifest.json").read_text())
    m["protocol"]["pn_drive_panel"]["catalog"][0]["model_sign"] = 1
    (selected / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="contradicts"):
        inspect_record(selected)


def test_legacy_records_remain_explicitly_all_pns():
    np.testing.assert_array_equal(recorded_drive_rows({"pn_rows": [2]}, ROOTS), [2])
    with pytest.raises(ValueError, match="lacks"):
        recorded_drive_rows({"pn_rows": [2], "driven_pn_rows": []}, ROOTS)


def test_unknown_panel_is_not_silently_all_pns():
    with pytest.raises(ValueError, match="Unknown PN"):
        make_panel(graph(), "olfactory_guessed")


def recruited_pn_graph():
    from simulations.drosophila.connectome import Subgraph
    g = graph()
    root = str(int(ROOTS[-1])+1)
    g.nodes[root] = {"global_index": 3, "annotation": {"root_id": root, "cell_class": "ALPN", "hemibrain_type": "M_test"}}
    extra = np.array([[int(ROOTS[2]), int(root), 2, 3, 80, 1, 80, 3, 3],
                      [int(root), int(ROOTS[1]), 3, 1, 1, -1, -1, 4, 4]], dtype=np.int64)
    return Subgraph((*ROOTS, root), g.nodes, np.vstack([g.edges, extra]), g.provenance)


def test_undriven_pn_can_be_recruited_and_each_incoming_potential_is_named(tmp_path):
    g = recruited_pn_graph()
    root = g.selected[-1]
    output = tmp_path / "recruited"
    run_intervention(g, output, "intact", .5, pn_drive_panel="without_inhibitory_drive")
    report = inspect_record(output)[0]
    witness = report["pn_input_selection"]["first_undriven_PN_spike"]
    assert witness["root"] == root and witness["external_current"] == 0
    assert witness["tick"] == 66
    assert witness["previous_membrane"] == witness["recorded_post_reset_voltage"] == 0
    assert witness["reconstructed_unclipped_pre_reset_voltage"] > witness["active_threshold"]
    assert len(witness["incoming_potentials"]) == 1
    arrival = witness["incoming_potentials"][0]
    assert arrival["source_root"] == ROOTS[2]
    assert arrival["source_output_tick"] == 63 and arrival["receiving_tick"] == 64
    assert arrival["local_potential_before_learning"] == 40
    assert arrival["potential_at_spike_tick"] == pytest.approx(40 * .95**2)
    delivery, = witness["apl_deliveries"]
    assert delivery["source_output_tick"] == 66
    assert delivery["receiving_tick"] == 67 and delivery["integration_tick"] == 69
    assert delivery["source_output"] == delivery["source_terminal_info_at_emission"] == delivery["arriving_info"] == 1
    assert delivery["local_potential_before_learning"] == -.5
    assert delivery["predicted_delayed_potential_float64"] == pytest.approx(-.5 * .95**2)
    assert "recorded_cable_arrived_current" not in delivery  # This fixture has global graded APL.
