"""The expression bound must not include untouched or non-B terminals."""
import numpy as np
from simulations.drosophila.memory_feedback.plastic_reach import reached_mask


def test_reached_mask_requires_selected_depressive_feedback_contrast():
    chosen = reached_mask(np.array([True, True, True, False, True]),
                          [0.9, 1., 1.1, 0.2, 1.-1e-8], np.ones(5))
    assert chosen.tolist() == [True, False, False, False, False]
