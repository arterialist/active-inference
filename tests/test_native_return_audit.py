import numpy as np
import pytest

from simulations.active_inference.experiments.native_return_audit import information_updates


def test_return_uses_previous_modulation_and_float32_updates():
    cfg = dict(neurons=[dict(id=7, params=dict(eta_retro=1e-7), metadata=dict(
        plasticity_rate_index=0, plasticity_rate_boost=499., plasticity_rate_half_saturation=.1))])
    cells = np.zeros((2, 1, 8)); cells[0, 0, 3] = .4
    initial = np.array([[.1, 0.]])
    events = np.array([[12, 8, 0, 7, 900, 1., 1., float(np.float32(.1)), 0., 0., 0.],
                       [13, 8, 0, 7, 900, 1., 1., float(np.float32(-.1)), 0., 0., 0.]])
    expected, wrong, rates = information_updates(events, cells, initial, cfg, 12)
    assert np.allclose(rates, np.array([250.5, 400.2])*1e-7, rtol=0, atol=1e-19)
    assert np.all(expected != wrong)
    assert np.array_equal(expected, np.float32(expected))
    events[0, 5] = .1  # Python float .1 is not the native float32 value.
    with pytest.raises(ValueError, match='float32'):
        information_updates(events, cells, initial, cfg, 12)
