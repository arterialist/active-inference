"""Broader teaching must depend on ingested nutrients, never cue or phase."""
from types import SimpleNamespace

from simulations.drosophila.memory_feedback import primary_boundary as boundary


def test_primary_inputs_follow_actual_ingestion_and_clear_without_nutrients():
    names = ("kc", "alpha", "gamma7", "gamma8", "motor")
    ids = dict(zip(names, range(5)))
    cells = {n: SimpleNamespace(id=n, O=0., params=SimpleNamespace(num_inputs=3)) for n in range(5)}
    inputs = {}
    net = SimpleNamespace(network=SimpleNamespace(neurons=cells), run_tick=lambda: None,
        set_external_input=lambda n, port, value: inputs.__setitem__((n, port), value))
    roles = dict(PAM11=["alpha"], PAM07=["gamma7"], PAM08=["gamma8"], SMP353=["motor"])
    codes = dict(A=["kc"], B=[])
    body = SimpleNamespace(offer=lambda pump, well: (boundary.feeding.DOSE_J, 0.), step=lambda *a: a)
    boundary.step(net, ids, roles, codes, "", body, factor=32.)
    assert [inputs[(ids[r], 2)] for r in ("alpha", "gamma7", "gamma8")] == [40.]*3
    # A requested pump that delivers no energy must not teach; cue A alone
    # cannot leave the previous dopamine input active either.
    body.offer = lambda pump, well: (0., 0.)
    boundary.step(net, ids, roles, codes, "A", body, factor=32., pump=True)
    assert inputs[(ids["kc"], 2)] == 40.
    assert [inputs[(ids[r], 2)] for r in ("alpha", "gamma7", "gamma8")] == [0.]*3
