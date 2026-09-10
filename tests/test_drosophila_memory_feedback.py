"""Causal-assay contracts, independent of the downloaded biological graph."""
import numpy as np

from simulations.drosophila.connectome import Subgraph
from simulations.drosophila.memory_feedback.acquisition import build, protocol


def fixture():
    roots = tuple(str(720575940000000001+i) for i in range(4))
    nodes = {r: dict(global_index=i, annotation=dict(root_id=r, side="left",
             cell_class=cl, hemibrain_type=ty))
             for i, (r, cl, ty) in enumerate(zip(roots,
                 ("Kenyon_Cell", "DAN", "MBON", "MBON"),
                 ("KCab-s", "PAM11", "MBON07", "MBON14")))}
    pairs = [(0, 2, 10, 1), (1, 2, 5, 1), (2, 1, 3, -1),
             (3, 2, 2, 1), (2, 3, 1, -1)]
    edges = np.array([[int(roots[a]), int(roots[b]), a, b, n, s, n*s, i, i]
                      for i, (a, b, n, s) in enumerate(pairs)], dtype=np.int64)
    return Subgraph(roots[:3], nodes, edges, {})


def test_receptor_control_preserves_fast_currents_reciprocity_and_learning():
    graph = fixture()
    enabled, selected = build(graph)
    blocked, _ = build(graph, dopamine=False)
    assert enabled.edge_bindings.tolist() == blocked.edge_bindings.tolist()
    assert len(enabled.edge_bindings) == 3
    assert len(enabled.incoming_boundary_ports) == len(enabled.outgoing_boundary_terminals) == 1
    assert selected[:, 1:].tolist() == [[0, 0, 2, 0]]
    for nid, cell in enabled.network.network.neurons.items():
        other = blocked.network.network.neurons[nid]
        assert cell.params.eta_post > 0 and cell.params.eta_retro > 0
        for sid, syn in cell.postsynaptic_points.items():
            assert syn.u_i.info == other.postsynaptic_points[sid].u_i.info
        for sid, terminal in cell.presynaptic_points.items():
            np.testing.assert_array_equal(terminal.u_o.mod, other.presynaptic_points[sid].u_o.mod)
    receptor = enabled.network.network.neurons[2].postsynaptic_points[1]
    assert receptor.u_i.adapt[1] == .1
    assert blocked.network.network.neurons[2].postsynaptic_points[1].u_i.adapt[1] == 0


def test_displaced_benefit_matches_dose_and_has_no_nearby_cue():
    def expand(paired):
        p = protocol(paired, 2)
        cue = np.concatenate([np.repeat(x["cue"], x["ticks"]) for x in p])
        nutrient = np.concatenate([np.repeat(x["nutrient"], x["ticks"]) for x in p])
        return cue, nutrient
    pc, pn = expand(True); uc, un = expand(False)
    np.testing.assert_array_equal(pc, uc)
    assert pn.sum() == un.sum() == 120
    assert np.all(pc[pn] == "A")
    for tick in np.flatnonzero(un):
        assert np.all(uc[max(0, tick-1000):tick+1001] == "")


def test_selected_rule_matches_existing_equation_without_reversing_other_inputs():
    from simulations.drosophila.memory_feedback.input_rule import build as selected_build
    graph = fixture()
    native, _ = build(graph, "native")
    whole, _ = build(graph, "dopamine_hebb")
    selected, _ = selected_build(graph)
    # Identical receiving events and local dopamine. Compare one actual tick:
    # selected KC update must equal the existing whole-cell alternative, while
    # an unselected negative reciprocal input must equal the native update.
    cells = [p.network.network.neurons[2] for p in (native, whole, selected)]
    for c in cells:
        c.t_last_fire = 0
        c.M_vector[1] = .3
        c.postsynaptic_points[2].u_i.info = -.5
        c.input_buffer[0, 0] = 1.
        c.input_buffer[2, 0] = 1.
        c.tick({}, 1)
    assert cells[2].postsynaptic_points[0].u_i.info == cells[1].postsynaptic_points[0].u_i.info
    assert cells[2].postsynaptic_points[2].u_i.info == cells[0].postsynaptic_points[2].u_i.info
    assert cells[2].postsynaptic_points[2].u_i.info != cells[1].postsynaptic_points[2].u_i.info
