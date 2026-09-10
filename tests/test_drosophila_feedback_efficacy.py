"""An efficacy experiment must not silently become a global gain change."""
from types import SimpleNamespace

import numpy as np

from simulations.drosophila.memory_feedback import feedback_efficacy


def test_only_measured_forward_teacher_projection_is_scaled(monkeypatch):
    def point(w): return SimpleNamespace(u_i=SimpleNamespace(info=w))
    def cell(points): return SimpleNamespace(postsynaptic_points=points, params=SimpleNamespace(w_max=100.))
    neurons = {1: cell({0: point(.08)}), 2: cell({0: point(.02), 1: point(.04)}), 3: cell({0: point(.06)})}
    bindings = np.array([[10, 1, 0, 2, 0], [11, 3, 0, 2, 1], [12, 1, 1, 3, 0], [13, 2, 0, 1, 0]])
    prep = SimpleNamespace(root_to_id={"s": 1, "d": 2, "k": 3}, edge_bindings=bindings,
        network=SimpleNamespace(network=SimpleNamespace(neurons=neurons)), assumptions={})
    graph = SimpleNamespace(selected=["s", "d", "k"], nodes={r: {"annotation": {"hemibrain_type": t}}
        for r, t in (("s", "SMP108"), ("d", "PAM08"), ("k", "KCg-m"))})
    monkeypatch.setattr(feedback_efficacy.terminal_course, "build", lambda _: (prep, np.empty((0, 5))))
    result, _ = feedback_efficacy.build(graph)
    assert result.network.network.neurons[2].postsynaptic_points[0].u_i.info == .02*64
    assert neurons[2].postsynaptic_points[1].u_i.info == .04
    assert neurons[3].postsynaptic_points[0].u_i.info == .06
    assert neurons[1].postsynaptic_points[0].u_i.info == .08
    assert len(result.assumptions["feedback_efficacy"]["changed_pairs"]) == 1
    np.testing.assert_array_equal(result.edge_bindings, bindings)


def test_student_output_preserves_nonstudent_thresholds_and_coefficients(monkeypatch):
    from simulations.drosophila.memory_feedback import student_output
    def cell():
        return SimpleNamespace(params=SimpleNamespace(r_base=1., b_base=1.2), r=1., b=1.2)
    neurons = {1: cell(), 2: cell()}
    prep = SimpleNamespace(root_to_id={"student": 1, "other": 2},
        network=SimpleNamespace(network=SimpleNamespace(neurons=neurons)), assumptions={})
    selected = np.array([[10, 2, 0, 1, 0]])
    monkeypatch.setattr(student_output.feedback_efficacy, "build", lambda _: (prep, selected))
    monkeypatch.setattr(student_output.student_course, "groups", lambda _: {"MBON04": ["student"]})
    result, actual = student_output.build(None)
    assert neurons[1].r == .5 and neurons[1].b == .6
    assert neurons[1].params.r_base == .5 and neurons[1].params.b_base == .6
    assert neurons[2].r == 1. and neurons[2].b == 1.2
    assert actual is selected and result is prep
