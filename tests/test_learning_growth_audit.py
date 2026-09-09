import numpy as np
import pytest

from simulations.active_inference.experiments.learning_growth_audit import (
    attributed_step, growth_modes,
)


def test_attribution_reconstructs_sequential_positive_and_negative_learning():
    q0 = np.array([.2, .4, .7])
    q = q0.copy()
    terms = np.zeros((3, 3))
    rng = np.random.default_rng(17)
    for t in range(100):
        plus, minus = rng.uniform(0, 1, (2, 3))
        eta = rng.uniform(.00001, .03, 3)
        terms = attributed_step(terms, q0, plus, minus, eta, t % 3)
        factor = np.exp(-eta * (plus + minus))
        q = factor * q + (1 - factor) * plus / (plus + minus)
        np.testing.assert_allclose(q, q0 + terms.sum(axis=0), atol=2e-14, rtol=0)
    assert (terms < 0).any()
    np.testing.assert_array_equal(attributed_step(terms, q0, np.zeros(3), np.zeros(3), eta, 0), terms)


def test_relabelling_does_not_change_total_or_preserve_earlier_terms_unattenuated():
    birth = np.array([.3])
    first = attributed_step(np.zeros((3, 1)), birth, [1.], [0.], [.2], 0)
    second = attributed_step(first, birth, [0.], [1.], [.2], 1)
    alternate = attributed_step(first, birth, [0.], [1.], [.2], 0)
    np.testing.assert_allclose(second.sum(axis=0), alternate.sum(axis=0))
    assert 0 < second[0, 0] < first[0, 0]
    assert second[1, 0] < 0


def test_growth_modes_are_orthogonal_and_invariant_to_port_order():
    x = np.array([.1, .2, .3, .4, -.2, 0.])
    targets = np.array([10, 10, 20, 20, 30, 30])
    modes = growth_modes(x, targets)
    np.testing.assert_allclose(sum(modes), x)
    for i in range(3):
        for j in range(i):
            assert abs(modes[i] @ modes[j]) < 1e-14
    order = np.array([3, 5, 1, 0, 2, 4])
    np.testing.assert_allclose(growth_modes(x[order], targets[order]), np.array(modes)[:, order])
    assert np.isclose(sum(v @ v for v in modes), x @ x)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        growth_modes([1, 2, 3], [0, 1, 1])
    with pytest.raises(ValueError):
        attributed_step(np.zeros((3, 2)), [.1, .2], [0, 0], [0, 0], [0, .1], 0)
    with pytest.raises(ValueError):
        attributed_step(np.zeros((3, 2)), [.1, .2], [0, 0], [0, 0], [.1, .1], -1)
