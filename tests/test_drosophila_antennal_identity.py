import copy

import numpy as np
import pytest

from simulations.drosophila.antennal_identity import identity_audit
from simulations.drosophila.connectome import Subgraph
from simulations.drosophila.orn_onset import negative_gaba_control
from simulations.drosophila.paula import build_paula


def graph():
    nodes = {}
    for i, (top, known, source) in enumerate([
        ("serotonin", "acetylcholine", "Shang et al., 2007"),
        ("gaba", "gaba, myoinhibitory peptide", "Sizemore et al., 2023"),
        ("gaba", "", "")]):
        root = str(i+1)
        nodes[root] = {"global_index": i, "annotation": {"root_id": root, "cell_class": "ALLN",
            "hemibrain_type": "test", "top_nt": top, "top_nt_conf": ".34",
            "known_nt": known, "known_nt_source": source}}
    edges = np.array([[1, 2, 0, 1, 2, 1, 2, 0, 0], [2, 1, 1, 0, 3, 1, 3, 1, 1],
                      [2, 3, 1, 2, 4, 1, 4, 2, 2], [3, 2, 2, 1, 1, -1, -1, 3, 3]], dtype=np.int64)
    return Subgraph(("1", "2", "3"), nodes, edges, {})


def test_audit_separates_prediction_curated_evidence_and_current_sign():
    g = graph(); original = copy.deepcopy(g)
    report = identity_audit(g)
    assert report["gaba_positive_candidates"] == ["2"]
    assert report["prediction_curated_conflicts"] == 1
    assert report["cells"][0]["top_nt"] == "serotonin"
    assert not report["cells"][1]["prediction_outside_curated_transmitter_set"]
    np.testing.assert_array_equal(g.edges, original.edges)
    assert g.nodes == original.nodes


def test_sign_control_changes_only_internal_receiving_signs_before_ticks():
    g = graph(); prep = build_paula(g); original = copy.deepcopy(g)
    old = [[p.u_i.info for p in c.postsynaptic_points.values()] for c in prep.network.network.neurons.values()]
    report = negative_gaba_control(prep, g)
    assert report["source_roots"] == ["2"]
    assert len(report["changed_receiving_coefficients"]) == 2
    for row, pre, terminal, post, port, before, after in report["changed_receiving_coefficients"]:
        assert before == old[post][port] and after == -before
        assert prep.network.network.neurons[post].postsynaptic_points[port].u_i.info == after
    assert all(c.params.eta_post > 0 and c.params.eta_retro > 0 for c in prep.network.network.neurons.values())
    np.testing.assert_array_equal(g.edges, original.edges)
    np.testing.assert_array_equal(prep.edge_bindings, build_paula(g).edge_bindings)
    prep.network.current_tick = 1
    with pytest.raises(ValueError, match="first tick"):
        negative_gaba_control(prep, g)
