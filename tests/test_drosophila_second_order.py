"""The intervention must isolate one forward projection, not silence a cell."""
from types import SimpleNamespace

from simulations.drosophila.memory_feedback.second_order import ProjectionGate, protocol


def test_gate_removes_only_forward_projection_at_delivery_and_preserves_reverse():
    forward = SimpleNamespace(event=(1, 10, .8))
    other_target = SimpleNamespace(event=(1, 11, .7))
    reverse = SimpleNamespace(event=(2, 20, .6))
    retro = object()
    net = SimpleNamespace(current_tick=3, wheel_size=2,
        presynaptic_wheel=[[], [forward, other_target, reverse]],
        retrograde_wheel=[[], [retro]],
        network=SimpleNamespace(connections=[(1, 10, 2, 0), (1, 11, 3, 0), (2, 20, 1, 0)]))
    observe = ProjectionGate(net, {1}, {2}, False)
    observe.before_step()
    assert net.presynaptic_wheel[1] == [forward, other_target, reverse]
    assert observe.observed == 1 and observe.removed == 0
    gate = ProjectionGate(net, {1}, {2}, True)
    gate.before_step()
    assert net.presynaptic_wheel[1] == [other_target, reverse]
    assert net.retrograde_wheel[1] == [retro]
    assert len(net.network.connections) == 3
    assert gate.removed == gate.observed == 1


def test_temporal_control_matches_exposure_and_elapsed_time():
    a, b = protocol(), protocol(displaced=True)
    for cue in ("", "A", "B"):
        assert sum(t for _, c, t in a if c == cue) == sum(t for _, c, t in b if c == cue)
    assert a[1][2] == 20 and b[1][2] == 1000
    assert a[-1] == b[-1] == ("retention", "", 1000)
