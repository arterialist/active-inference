import numpy as np
import pytest

from simulations.active_inference.experiments.eligibility_exposure_audit import affine_step


def test_conditional_decomposition_matches_direct_updates_with_varying_drive():
    initial = np.array([.2, .7, .3])
    q = initial.copy(); a = np.ones(3); b = np.zeros(3)
    for t in range(30):
        plus = np.array([0., (t % 3)/2, .4])
        minus = np.array([0., (t % 5)/4, .4])
        eta = np.array([.01, .001*(t+1), .004])
        total = plus+minus
        decay = np.exp(-eta*total)
        equilibrium = np.divide(plus, total, out=np.zeros(3), where=total > 0)
        q = q*decay+equilibrium*(1-decay)
        a, b = affine_step(a, b, plus, minus, eta)
        assert np.allclose(a*initial+b, q, atol=1e-14, rtol=0)
    assert a[0] == 1 and b[0] == 0  # positive learning rate without eligible events
    assert np.all((a > 0) & (a <= 1))


def test_no_frozen_rates_or_malformed_inputs():
    one, zero = np.ones(2), np.zeros(2)
    with pytest.raises(ValueError):
        affine_step(one, zero, one, one, zero)
    with pytest.raises(ValueError):
        affine_step(one, zero, one, one, np.ones(3))
