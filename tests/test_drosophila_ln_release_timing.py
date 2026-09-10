import numpy as np
import pytest

from simulations.drosophila.ln_release_timing import audit_block, differences


def test_block_checks_every_cell_before_and_after_onset():
    a = np.ones((8, 3, 3), dtype=int)
    a[4:, 1, 1] = 0
    assert audit_block(a, 1, 4) == {"withheld_forward_events": 4, "retained_return_events_after_onset": 4}
    for tick, cell in ((3, 1), (5, 0)):
        wrong = a.copy(); wrong[tick, cell, 1] = 0
        with pytest.raises(AssertionError): audit_block(wrong, 1, 4)
    wrong = a.copy(); wrong[4, 1, 1] = 1
    with pytest.raises(AssertionError): audit_block(wrong, 1, 4)


def test_state_and_spike_divergence_are_distinguished():
    a = np.zeros((8, 3, 3)); b = a.copy()
    b[3, 1, 0] = .1; b[5, 2, 1] = 1
    assert differences(a, b) == {"first_state_difference": 3, "first_spike_difference": 5, "cells_with_changed_spikes": 1}
    assert differences(a, a)["first_state_difference"] is None
