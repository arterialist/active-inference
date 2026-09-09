import math

import numpy as np
import pytest

from simulations.active_inference.experiments.crossed_av_credit import (
    credit_components, differential, projected_bounds,
)
from simulations.active_inference.experiments.crossed_av_continuation_analysis import (
    analyze, trajectory, factorial_modes, FACTORIAL,
)
from simulations.active_inference.experiments import context_organization as base


def local_record():
    q = np.array([[.0001, .3], [.4, .9999]])
    x = np.array([[.2, .8], [.6, .1]])
    e = np.array([-.8, .7])
    data = dict(weights_initial=q.copy(), context_initial=x.copy(), error_initial=e.copy())
    arrivals = np.random.default_rng(1).random((24, 2, 2))
    errors = []; weights = []; rates = []
    for t, a in enumerate(arrivals):
        rate = 1e-5 * (1 + 499 * abs(e) / (.01 + abs(e)))
        x = math.exp(-1/64) * x + (1-math.exp(-1/64)) * a
        q = np.minimum(1., np.maximum(0., q + rate[:, None] * e[:, None] * x))
        new_error = np.array([-.1, .2]) if t < 12 else np.array([.3, -.4])
        next_e = math.exp(-1/4) * e + (1-math.exp(-1/4)) * new_error
        errors.append(np.column_stack((e, new_error, next_e)))
        weights.append(q.copy()); rates.append(rate); e = next_e
    data.update(arrivals=arrivals, errors=np.array(errors), weights=np.array(weights), eta=np.array(rates))
    return data


def test_credit_factors_reconstruct_every_update_and_keep_clipping():
    z = local_record(); pieces = credit_components(z)
    dq = np.diff(np.concatenate((z['weights_initial'][None], z['weights'])), axis=0)
    np.testing.assert_allclose(sum(pieces.values()), dq, rtol=0, atol=2e-15)
    assert np.any(abs(pieces['bound_correction']) > 1e-4)
    assert np.any(pieces['old_context']) and np.any(pieces['new_context'])
    z['weights'][7, 0, 1] += 1e-5
    with pytest.raises(ValueError, match='learning audit'):
        credit_components(z)


def test_source_reordering_and_full_projection():
    q = np.array([[[1., 2.], [3., 4.]], [[2., 5.], [7., 8.]]])
    ids = np.array([[4, 9], [9, 4]])
    np.testing.assert_array_equal(differential(q, ids), [[-3, -1], [-6, -2]])
    np.testing.assert_array_equal(differential(q[0], ids), [-3, -1])
    basis = np.random.default_rng(4).normal(size=(4, 64, 2))
    delta = differential(q, ids)
    full = np.array([[basis[p] @ w for p in range(4)] for w in delta])
    expected = np.stack((full[:, :, 16:64].min(axis=2), full[:, :, 16:64].max(axis=2)), axis=2)
    np.testing.assert_allclose(projected_bounds(delta, basis), expected, atol=1e-14)
    with pytest.raises(ValueError, match='identities'):
        differential(q, [[4, 4], [4, 9]])


def test_trajectory_column_identity_is_independent_of_neuron_order():
    data = local_record(); ticks = len(data['weights'])
    data['neuron_ids'] = np.array([20, 10])
    data['cells'] = np.zeros((ticks, 2, len(base.FIELDS)))
    data['cells'][:, 0, base.FIELDS.index('O')] = .7
    data['cells'][:, 1, base.FIELDS.index('O')] = .2
    data['body'] = np.tile([.1, -.3, .4, .5, -.2], (ticks, 1))
    x = trajectory(data, dict(prediction=[10, 20]))
    assert x.shape == (ticks, 15)
    np.testing.assert_allclose(x[:, 5], -.1)
    np.testing.assert_allclose(x[:, 6:10], np.tile([.2, .7, .5, .3], (ticks, 1)))
    np.testing.assert_array_equal(x[:, 10:12], data['errors'][:, :, 1])
    np.testing.assert_array_equal(x[:, 12:14], data['eta'])


def test_continuation_analysis_rejects_absent_or_invalid_evidence(tmp_path):
    with pytest.raises(ValueError, match='No evidence'):
        analyze([], tmp_path/'empty')
    for blocks in (True, 4, 17, 7.0):
        with pytest.raises(ValueError, match='block count'):
            analyze([], tmp_path/'bad', blocks)
    assert not (tmp_path/'empty').exists()


def test_factorial_reconstruction_and_bias_are_distinct():
    response = np.random.default_rng(15).normal(size=(96, 4))
    modes = factorial_modes(response)
    np.testing.assert_allclose(modes @ (4 * FACTORIAL), response, atol=5e-16)
    # An interaction can grow while EVERY response remains negative.
    before = np.array([[-1., -1., -1., -1.]])
    after = before + np.array([[.2, -.2, -.2, .2]])
    assert np.all(after < 0)
    np.testing.assert_allclose(factorial_modes(after), [[-1, 0, 0, .2]], atol=1e-16)
    assert factorial_modes(after)[0, 3] > factorial_modes(before)[0, 3]
    with pytest.raises(ValueError, match='finite'):
        factorial_modes(np.full((2, 4), np.nan))
