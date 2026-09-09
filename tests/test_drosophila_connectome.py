"""Small adversarial fixtures, independent of downloads and biological outcomes."""
from copy import deepcopy
import json

import numpy as np
import pytest

from simulations.drosophila.connectome import (
    Catalog, EDGE_COLUMNS, RAW_COLUMNS, Subgraph, alpn_providers,
    extract_subgraph, iter_edges, kc_apl_roots, read_catalog, verify_sources,
)
from simulations.drosophila.paula import Dynamics, PresynapticPoint, PresynapticOutputVector, build_paula

ROOTS = tuple(str(720575940000000001 + i) for i in range(5))


def catalog():
    # A KC with no cell_type and an ALPN from the other hemisphere must survive.
    names = ["Kenyon_Cell", "MBIN", "ALPN", "MBON", "ALPN"]
    annotations = {root: {"root_id": root, "side": "right" if i == 2 else "left",
                          "cell_class": names[i], "cell_type": "",
                          "hemibrain_type": "APL" if i == 1 else "",
                          "top_nt": "gaba" if i == 1 else "ach", "known_nt": "",
                          "top_nt_conf": "0.6"}
                   for i, root in enumerate(ROOTS)}
    return Catalog(ROOTS, annotations)


def row(pre, post, count, sign, source_row):
    return [int(ROOTS[pre]), int(ROOTS[post]), pre, post, count, sign,
            count * sign, source_row + 20, source_row]


def graph():
    c = catalog()
    return Subgraph(ROOTS[:3], {r: {"global_index": i, "annotation": c.annotations[r]}
                               for i, r in enumerate(ROOTS)}, np.array([
        row(2, 0, 10, 1, 0), row(0, 1, 3, 1, 1), row(1, 0, 4, -1, 2),
        row(3, 0, 2, 1, 3), row(0, 3, 5, 1, 4),
    ], dtype=np.int64), {"fixture": True})


def parquet(tmp_path, rows):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    data = np.array(rows, dtype=np.int64)
    path = tmp_path / "edges.parquet"
    table = pa.table({name: pa.array(data[:, i], type=pa.int64()) for i, name in enumerate(RAW_COLUMNS)})
    pq.write_table(table, path, row_group_size=1)
    return path


def test_anatomical_selection_includes_untyped_kc_and_actual_cross_side_provider(tmp_path):
    c = catalog()
    core = kc_apl_roots(c, "left")
    assert core == ROOTS[:2]
    path = parquet(tmp_path, [row(2, 0, 1, 1, 0), row(4, 3, 5, 1, 1)])
    assert alpn_providers(path, c, core) == {ROOTS[2]}
    with pytest.raises(ValueError, match="Expected one"):
        kc_apl_roots(c, "right")


def test_extraction_preserves_recurrence_boundary_and_one_count_edges(tmp_path):
    rows = [row(2, 0, 1, 1, 0), row(0, 0, 2, 1, 1), row(1, 0, 4, -1, 2),
            row(3, 0, 2, 1, 3), row(0, 3, 5, 1, 4), row(3, 4, 17, 1, 5)]
    g = extract_subgraph(parquet(tmp_path, rows), catalog(), ROOTS[:3], {})
    assert g.edges.tolist() == rows[:5]
    assert len(g.internal) == 3
    assert g.summary()["incoming_boundary"]["synapses"] == 2
    assert g.summary()["outgoing_boundary"]["synapses"] == 5
    assert g.provenance["source_synapses"] == 31
    assert g.provenance["source_rows_checked"] == 6


@pytest.mark.parametrize("column,value,message", [
    (2, 10, "outside global"), (0, int(ROOTS[3]), "Root/index"),
    (4, 0, "Non-positive"), (5, 0, "Unknown source"), (6, 99, "Signed-count"),
])
def test_invalid_raw_rows_fail_instead_of_being_repaired(tmp_path, column, value, message):
    r = row(0, 1, 2, 1, 0)
    r[column] = value
    path = parquet(tmp_path, [r])
    with pytest.raises(ValueError, match=message):
        list(iter_edges(path, catalog(), batch_size=1))


def test_duplicates_across_batches_are_not_summed(tmp_path):
    path = parquet(tmp_path, [row(0, 1, 2, 1, 0), row(0, 1, 3, 1, 1)])
    with pytest.raises(ValueError, match="Duplicate directed pair"):
        extract_subgraph(path, catalog(), ROOTS[:2], {})


def test_exact_ids_and_known_vs_predicted_annotation_survive_disk(tmp_path):
    g = graph()
    g.nodes[ROOTS[1]]["annotation"]["known_nt"] = "gaba, Gpb5"
    output = tmp_path / "cut"
    g.save(output)
    reloaded = Subgraph.load(output)
    assert reloaded.selected == g.selected
    assert reloaded.nodes == g.nodes
    assert np.array_equal(reloaded.edges, g.edges)
    raw = json.loads((output / "nodes.json").read_text())
    assert isinstance(raw["selected"][0], str)
    assert int(raw["selected"][0]) > 2**53
    with pytest.raises(FileExistsError):
        g.save(output)
    with (output / "nodes.json").open("a") as out:
        out.write(" ")
    with pytest.raises(ValueError, match="Changed artifact"):
        Subgraph.load(output)


def test_source_verification_rejects_wrong_release(tmp_path):
    (tmp_path / "Connectivity_783.parquet").write_bytes(b"wrong release")
    with pytest.raises(ValueError, match="Source mismatch"):
        verify_sources(tmp_path)


def test_annotation_and_index_validation():
    c = catalog()
    with pytest.raises(ValueError, match="Duplicate neuron identity"):
        Catalog((ROOTS[0], ROOTS[0]), c.annotations)
    with pytest.raises(ValueError, match="Missing annotation"):
        Catalog(ROOTS, {})
    with pytest.raises(ValueError, match="exact decimal string"):
        Catalog((float(ROOTS[0]),), c.annotations)
    bad = graph()
    bad.nodes[ROOTS[1]]["global_index"] = 0
    with pytest.raises(ValueError, match="duplicate global index"):
        bad.validate()


def test_adapter_preserves_exact_graph_identity_and_one_signed_scaling():
    g = graph()
    before = deepcopy(g.nodes)
    p = build_paula(g)
    assert p.root_to_id == {ROOTS[0]: 0, ROOTS[1]: 1, ROOTS[2]: 2}
    assert len(p.network.network.connections) == len(g.internal) == 3
    assert len(p.edge_bindings) == 3
    for edge, binding in zip(g.internal, p.edge_bindings, strict=True):
        source_row, pre, terminal, post, synapse = map(int, binding)
        assert source_row == edge[8]
        assert (pre, post) == tuple(edge[2:4])
        a, b = p.network.network.neurons[pre], p.network.network.neurons[post]
        assert a.presynaptic_points[terminal].u_o.info == 1
        assert b.postsynaptic_points[synapse].u_i.info == edge[6] * 0.02
        assert b.postsynaptic_points[synapse].u_i.plast == 0
        assert b.synapse_sources[synapse] == (pre, terminal)
        assert b.distances[synapse] == 2
        assert a.params.eta_retro > 0 and b.params.eta_post > 0
        assert not a._ablation and not b._ablation
    assert g.nodes == before
    assert p.network.network.neurons[1]._gg == 0.01


def test_global_ids_survive_different_cuts():
    g = graph()
    smaller = deepcopy(g)
    smaller.selected = ROOTS[:2]
    a, b = build_paula(g), build_paula(smaller)
    assert all(a.root_to_id[r] == b.root_to_id[r] for r in smaller.selected)
    for root in smaller.selected:
        ac = a.network.network.neurons[a.root_to_id[root]]
        bc = b.network.network.neurons[b.root_to_id[root]]
        assert ac.params.num_inputs == bc.params.num_inputs
        assert ac.upper_t_ref_bound == bc.upper_t_ref_bound
        assert ac.distances == bc.distances
        assert a.drive_ports[root] == b.drive_ports[root]
    by_row = {int(x[0]): x.tolist() for x in a.edge_bindings}
    assert all(x.tolist() == by_row[int(x[0])] for x in b.edge_bindings)
    assert len(a.incoming_boundary_ports) == 1
    assert len(a.outgoing_boundary_terminals) == 1
    assert len(b.incoming_boundary_ports) == 2


def test_port_binding_uses_source_row_order_not_array_iteration_order():
    g = graph()
    a = build_paula(g)
    g.edges = g.edges[::-1].copy()
    b = build_paula(g)
    np.testing.assert_array_equal(a.edge_bindings, b.edge_bindings)
    np.testing.assert_array_equal(a.incoming_boundary_ports, b.incoming_boundary_ports)
    np.testing.assert_array_equal(a.outgoing_boundary_terminals, b.outgoing_boundary_terminals)


def test_real_negative_current_and_latency_with_positive_release():
    # A source fires at tick 0, crossing cleft at 1 and dendrite at 3.
    g = graph()
    g.selected = ROOTS[:2]
    g.edges = np.asarray([row(1, 0, 4, -1, 2)], dtype=np.int64)
    d = Dynamics(apl_representation="spiking_null", weight_per_count=0.5,
                 lambda_ticks=1, cooldown_ticks=1)
    p = build_paula(g, d)
    p.stimulate(ROOTS[1], 2)
    states = []
    for _ in range(5):
        p.network.run_tick()
        states.append(p.network.network.neurons[0].S)
    assert states[:3] == [0, 0, 0]
    assert states[3] == pytest.approx(-2 * d.signal_decay**2)
    assert states[4] == 0
    syn = p.network.network.neurons[0].postsynaptic_points[0]
    terminal = p.network.network.neurons[1].presynaptic_points[0]
    assert syn.u_i.info != -2  # positive adaptation actually executes
    assert terminal.u_o.info != 1
    assert terminal.u_o.info > 0


@pytest.mark.parametrize("kwargs", [{"eta_post": 0}, {"eta_retro": 0}, {"weight_per_count": float("nan")},
                                    {"dendritic_delay_ticks": -1}, {"lambda_ticks": 0.5}])
def test_invalid_dynamics_fail(kwargs):
    with pytest.raises(ValueError):
        build_paula(graph(), Dynamics(**kwargs))


def test_no_silent_strength_clipping():
    with pytest.raises(ValueError, match="no clipping"):
        build_paula(graph(), Dynamics(weight_per_count=100))


def test_no_silent_port_truncation():
    # 4096 genuine incoming pairs leave no room for the declared drive port.
    roots = tuple(str(720575940000000001 + i) for i in range(4097))
    nodes = {r: {"global_index": i, "annotation": {"root_id": r, "hemibrain_type": "", "cell_class": "fixture"}}
             for i, r in enumerate(roots)}
    edges = np.array([[int(r), int(roots[0]), i, 0, 1, 1, 1, i, i]
                      for i, r in enumerate(roots[1:], 1)], dtype=np.int64)
    g = Subgraph(roots, nodes, edges, {})
    with pytest.raises(ValueError, match="port overflow"):
        build_paula(g)


def test_drive_rejects_missing_neuron_and_negative_release():
    p = build_paula(graph())
    with pytest.raises(ValueError, match="outside"):
        p.stimulate(ROOTS[3], 1)
    with pytest.raises(ValueError, match="non-negative"):
        p.stimulate(ROOTS[0], -1)


def test_build_has_no_global_random_side_effect():
    np.random.seed(928)
    before = np.random.get_state()
    build_paula(graph())
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_positive_native_return_rate_can_still_round_to_no_change():
    error = np.array([-0.94, 0.0, 0.0, 0.0], dtype=np.float32)
    tiny = PresynapticPoint(PresynapticOutputVector(info=1.0, mod=np.zeros(2)), u_i_retro=1.0)
    observable = PresynapticPoint(PresynapticOutputVector(info=1.0, mod=np.zeros(2)), u_i_retro=1.0)
    for _ in range(32):
        tiny.process_retrograde_signal(error, 1e-8)
        observable.process_retrograde_signal(error, Dynamics().eta_retro)
    assert tiny.u_o.info == 1.0
    assert observable.u_o.info < 1.0


def test_execution_observer_preserves_native_trajectory(tmp_path):
    from simulations.drosophila.execution_probe import SOMA_FIELDS, run_probe

    g = graph()
    result = run_probe(g, tmp_path / "observed")
    p = build_paula(g)
    raw_soma = []
    for t in range(32):
        if t == 4:
            p.stimulate(ROOTS[0], 40)
        if t == 20:
            p.stimulate(ROOTS[1], 40)
        p.network.run_tick()
        raw_soma.append([[getattr(c, f) for f in SOMA_FIELDS] for c in p.network.network.neurons.values()])
    with np.load(tmp_path / "observed/trace.npz", allow_pickle=False) as trace:
        np.testing.assert_array_equal(trace["soma"][1:], np.array(raw_soma))
        final_post = [s.u_i.info for c in p.network.network.neurons.values() for s in c.postsynaptic_points.values()]
        final_pre = [s.u_o.info for c in p.network.network.neurons.values() for s in c.presynaptic_points.values()]
        np.testing.assert_array_equal(trace["post_weight"][-1], final_post)
        np.testing.assert_array_equal(trace["terminal_info"][-1], final_pre)
        assert trace["inputs"][4, 3, 0] == 40  # Three anatomical input ports, including boundary.
    assert result["checks"]["changed_postsynaptic_coefficients"] > 0
    assert result["checks"]["changed_terminal_coefficients"] > 0
