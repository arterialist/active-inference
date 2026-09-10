"""The interface must be cue-neutral and driven by actual neural output."""
from types import SimpleNamespace

from simulations.drosophila.memory_feedback import student_interface as interface


def test_equal_presence_and_neural_motor_source_without_nutrient_shortcut():
    roots = ("ka", "kb", "da", "d7", "d8", "s353", "s108")
    ids = {r: i for i, r in enumerate(roots)}
    cells = {i: SimpleNamespace(id=i, O=float(r == "s108"), params=SimpleNamespace(num_inputs=2)) for r, i in ids.items()}
    inputs = {}
    net = SimpleNamespace(network=SimpleNamespace(neurons=cells), run_tick=lambda: None,
        set_external_input=lambda n, port, value: inputs.__setitem__((n, port), value))
    roles = dict(PAM11=["da"], PAM07=["d7"], PAM08=["d8"], SMP353=["s353"], SMP108=["s108"])
    codes = dict(A=["ka"], B=["kb"])
    body = SimpleNamespace(offer=lambda pump, well: (0., 0.), step=lambda spike, p, w: (spike, p, w))
    for cue in ("A", "B"):
        result = interface.step(net, ids, roles, codes, cue, body, factor=32., well=True)
        assert result == (1., 0., 0.)
        assert inputs[(ids["s353"], 1)] == inputs[(ids["s108"], 1)] == 1.4/32.
        assert all(inputs[(ids[r], 1)] == 0 for r in ("da", "d7", "d8"))
    result = interface.step(net, ids, roles, codes, "B", body, factor=32., motor_role="SMP353")
    assert result == (0., 0., 0.)
    interface.step(net, ids, roles, codes, "", body, factor=32.)
    assert inputs[(ids["s353"], 1)] == inputs[(ids["s108"], 1)] == 0.
