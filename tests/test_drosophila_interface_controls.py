"""A-memory intervention must not directly replace B's student memory."""
from types import SimpleNamespace

import numpy as np

from simulations.drosophila.memory_feedback.interface_controls import mask_for, swap


def test_teacher_mask_and_swap_preserve_b_terminals_and_original_scalar_types():
    ids=np.array([1, 2, 3, 4, 5]); roots=np.array(["a1", "ag", "bg", "teacher", "student"])
    selected=np.array([[10, 1, 0, 4, 0], [11, 2, 0, 5, 0], [12, 3, 0, 5, 1]])
    m=dict(codes=dict(A=["a1", "ag"], B=["bg"]), roles=dict(MBON07=["teacher"], MBON04=["student"]))
    mask=mask_for(m, ids, roots, selected, "A", ("MBON07", "MBON04"))
    np.testing.assert_array_equal(mask, [True, True, False])
    np.testing.assert_array_equal(mask_for(m, ids, roots, selected, "A", ("MBON07",)), [True, False, False])
    def branch(values):
        neurons={n: SimpleNamespace(presynaptic_points={0: SimpleNamespace(u_o=SimpleNamespace(info=v))}) for n, v in enumerate(values, 1)}
        return SimpleNamespace(network=SimpleNamespace(network=SimpleNamespace(neurons=neurons)))
    a=branch([np.float32(.2), np.float64(.3), .7]); b=branch([1., 1., .1])
    changes, originals=swap(a,b,selected,mask)
    assert [c["source_row"] for c in changes] == [10, 11]
    assert a.network.network.neurons[3].presynaptic_points[0].u_o.info == .7
    for point, value in originals: point.u_o.info=value
    assert type(a.network.network.neurons[1].presynaptic_points[0].u_o.info) is np.float32
