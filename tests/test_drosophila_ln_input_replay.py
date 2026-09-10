import copy

import numpy as np
import pytest

from simulations.drosophila.connectome import Subgraph
from simulations.drosophila.paula import Dynamics, build_paula
from simulations.drosophila.ln_input_replay import cut_cells, port_groups, replay, response_summary


def fixture():
    nodes = {r: {"global_index": int(r), "annotation": {"root_id": r,
             "hemibrain_type": "test", "cell_class": "ALPN" if r == "1" else "ALLN"}}
             for r in ("1", "2", "3")}
    g = Subgraph(("1", "2", "3"), nodes, np.array([
        [1, 2, 1, 2, 40, 1, 40, 0, 101], [3, 2, 3, 2, 20, -1, -20, 0, 102],
        [2, 3, 2, 3, 10, 1, 10, 0, 103]], dtype=np.int64), {})
    cut = cut_cells(g, ("2",))
    prep = build_paula(cut, Dynamics(weight_per_count=.075))
    return cut, prep.network.network.neurons[2]


def test_cut_retains_every_incident_port_and_rejects_boundary_promotion():
    g, cell = fixture()
    assert cell.params.num_inputs == 3 and len(cell.presynaptic_points) == 1
    assert g.summary()["incoming_boundary"]["synapses"] == 60
    pn, ln, rows = port_groups(g, "2", pn_root="1")
    np.testing.assert_array_equal(pn, [True, False, False])
    np.testing.assert_array_equal(ln, [False, True, False])
    assert [int(r[8]) for r in rows] == [101, 102]
    with pytest.raises(ValueError, match="boundary"):
        cut_cells(g, ("1",))


def test_condition_masks_input_without_erasing_weights_or_mutating_record():
    g, c = fixture(); pn, ln, _ = port_groups(g, "2", pn_root="1")
    inputs = np.zeros((100, 3, 4), dtype=np.float32)
    inputs[10:50:4, 0, 0] = 8.
    inputs[20:60:5, 1, 0] = 1.
    original = inputs.copy()
    empty = replay(copy.deepcopy(c), inputs, pn, ln, "none")
    np.testing.assert_array_equal(empty["trace"][:, :3], 0)
    np.testing.assert_array_equal(empty["weights"], np.broadcast_to(empty["initial_weights"], (100, 3)))
    a = replay(copy.deepcopy(c), inputs, pn, ln, "DL5_only")
    b = replay(copy.deepcopy(c), inputs, pn, ln, "without_ALLN")
    np.testing.assert_array_equal(a["trace"], b["trace"])
    np.testing.assert_array_equal(a["weights"], b["weights"])
    assert response_summary(a)["spikes"] > 0
    assert np.all(a["weights"][:, 1] == -1.5)
    np.testing.assert_array_equal(inputs, original)


def test_gain_is_initial_receiving_coefficient_not_changed_input_amplitude():
    g, c = fixture(); pn, ln, _ = port_groups(g, "2", pn_root="1")
    inputs = np.zeros((30, 3, 4), dtype=np.float32); inputs[0, 0, 0] = 1.
    a = replay(copy.deepcopy(c), inputs, pn, ln, "all")
    b = replay(copy.deepcopy(c), inputs, pn, ln, "DL5_gain_half")
    assert a["initial_weights"][0] == 3. and b["initial_weights"][0] == 1.5
    assert b["trace"][2, 6] == a["trace"][2, 6]/2
    assert c.postsynaptic_points[0].u_i.info == 3.
    with pytest.raises(ValueError): replay(copy.deepcopy(c), inputs, pn, ln, "fake")
