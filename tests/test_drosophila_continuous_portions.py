"""Finite food replenishes by physical disengagement, never elapsed time."""
import numpy as np

from simulations.drosophila.memory_feedback.continuous_portions import PortionBody, PORTION_J
from simulations.drosophila.memory_feedback.feeding import DOSE_J
from simulations.active_inference.components.body.loaded_hinge import afferents


def test_depleted_portion_requires_withdrawal_then_new_contact():
    body = PortionBody()
    body.body.data.qpos[0] = .1
    assert np.isclose(sum(body.offer(False, True)[1] for _ in range(100)), PORTION_J)
    for _ in range(1000):
        assert body.offer(False, True) == (0., 0.)
    assert body.remaining_j == 0 and body.portions_loaded == 1
    body.body.data.qpos[0] = .021
    assert body.offer(False, True) == (0., 0.)
    assert body.remaining_j == 0
    body.body.data.qpos[0] = .02
    assert body.offer(False, True) == (0., 0.)
    assert body.portions_loaded == 2 and body.remaining_j == PORTION_J
    body.body.data.qpos[0] = .04
    assert body.offer(False, True) == (0., DOSE_J)


def test_inventory_tracks_accepted_food_and_does_not_top_up_early():
    body = PortionBody()
    body.body.data.qpos[0] = .1
    body.organs.gut_j = 48.
    assert body.offer(False, True) == (0., 0.)
    assert body.remaining_j == PORTION_J
    body.organs.gut_j = 0.
    body.offer(False, True)
    body.body.data.qpos[0] = .01
    body.offer(False, True)
    assert body.remaining_j == PORTION_J-DOSE_J
    assert body.portions_loaded == 1


def test_checkpoint_preserves_physical_inventory_and_next_transition(tmp_path):
    old = PortionBody()
    old.body.data.qpos[0] = .1
    for _ in range(100):
        old.offer(False, True)
    old.body.data.qpos[0] = .02
    path = tmp_path / "body-state.npz"
    old.save(path)
    new = PortionBody()
    new.restore(path)
    assert np.array_equal(afferents(old.body), afferents(new.body))
    assert old.offer(False, True) == new.offer(False, True)
    assert old.remaining_j == new.remaining_j == PORTION_J
    assert old.samples == new.samples == 101
    assert old.portions_loaded == new.portions_loaded == 2
