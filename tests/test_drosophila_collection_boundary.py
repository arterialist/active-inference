"""The relocation changes collection geometry, not the afferent or actuator."""
import numpy as np

from simulations.drosophila.memory_feedback.collection_boundary import CollectionBoundaryBody
from simulations.drosophila.memory_feedback.feeding import FeedingBody, DOSE_J
from simulations.active_inference.components.body.loaded_hinge import afferents


def test_default_environment_preserves_old_body_trajectory():
    old, new = FeedingBody(), CollectionBoundaryBody()
    for tick in range(160):
        spike = float(tick % 5 == 0)
        old_food, new_food = old.offer(False, True), new.offer(False, True)
        assert old_food == new_food
        assert np.array_equal(old.step(spike, *old_food), new.step(spike, *new_food))


def test_collection_move_keeps_sensory_calibration_and_pre_action_contact():
    old, moved = FeedingBody(), CollectionBoundaryBody(.08)
    for angle, expected_old, expected_moved in ((.03, 0., 0.), (.04, DOSE_J, 0.),
        (.079999, DOSE_J, 0.), (.08, DOSE_J, DOSE_J)):
        old.body.data.qpos[0] = moved.body.data.qpos[0] = angle
        assert np.array_equal(afferents(old.body), afferents(moved.body))
        assert old.offer(False, True) == (0., expected_old)
        assert moved.offer(False, True) == (0., expected_moved)
    assert moved.offer(False, False) == (0., 0.)
    assert moved.offer(True, False) == (DOSE_J, 0.)
