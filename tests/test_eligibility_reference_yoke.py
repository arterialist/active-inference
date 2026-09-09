from copy import deepcopy

import numpy as np
import pytest

from test_eligibility_reference import fixture
from simulations.active_inference.experiments import context_organization as base
from simulations.active_inference.experiments.eligibility_reference_probe import record
from simulations.active_inference.experiments.eligibility_reference_yoke import YokedAfferents, verify_yoked
from simulations.active_inference.experiments.eligibility_reference_intervention_analysis import changed_ticks, onset
from simulations.active_inference.experiments.opponent_context import AfferentDelay


def test_yoke_retains_physical_history_and_rejects_bad_course():
    delay = AfferentDelay(2)
    samples = np.array([[1., 0., .2, 0.], [0., 1., 0., .4]])
    yoke = YokedAfferents(delay, samples)
    samples[:] = 99
    np.testing.assert_array_equal(yoke.step([0., 1., 0., 2.]), [1., 0., .2, 0.])
    np.testing.assert_array_equal(yoke.state(), [[0., 0., 0., 0.], [0., 1., 0., 2.]])
    np.testing.assert_array_equal(yoke.step([1., 0., 3., 0.]), [0., 1., 0., .4])
    before = yoke.state().copy()
    with pytest.raises(ValueError, match='exhausted'):
        yoke.step([0., 0., 0., 0.])
    np.testing.assert_array_equal(before, yoke.state())
    for bad in ([1., 2.], [[np.nan]*4], [[-1.]*4]):
        with pytest.raises(ValueError):
            YokedAfferents(AfferentDelay(2), bad)


def test_identical_yoke_is_exact_and_auditor_catches_corruption(tmp_path):
    net, g, selected, _ = fixture(tmp_path)
    branch = deepcopy(net)
    rng = np.random.default_rng(8)
    features = [dict(visual=rng.uniform(0., .5, (300, 96)),
                     auditory=rng.uniform(0., .5, (300, 96))) for _ in (0, 1)]
    reference = record(net, base.Arm(), AfferentDelay(64), features, g, selected, 0, 1, ticks=96)
    yoked = record(branch, base.Arm(), YokedAfferents(AfferentDelay(64), reference['drive'][:, 194:198]),
                   features, g, selected, 0, 1, ticks=96)
    for key in reference:
        np.testing.assert_array_equal(reference[key], yoked[key])
    actual = verify_yoked(yoked, reference, g)
    np.testing.assert_array_equal(actual, reference['drive'][:, 194:198])
    for field in ('raw_afferents', 'drive', 'delay_final', 'body', 'weights'):
        bad = {k: v.copy() for k, v in yoked.items()}
        bad[field][20] += .01
        with pytest.raises(ValueError):
            verify_yoked(bad, reference, g)


def test_divergence_is_per_tick_not_an_average():
    a = np.zeros((6, 2, 3)); b = a.copy()
    b[2, 0, 0] = 1.; b[3, 0, 0] = -1.
    assert a.mean() == b.mean()
    assert onset(a, b) == 2
    np.testing.assert_array_equal(changed_ticks(a, b), [False, False, True, True, False, False])
    assert onset(a, a) is None
    with pytest.raises(ValueError):
        changed_ticks(a, b[:3])
