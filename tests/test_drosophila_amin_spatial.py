import numpy as np
import pytest

from simulations.drosophila.amin_spatial import (
    backbone_cable, exponential_profile, finite, merge_profiles,
)


def test_shared_stem_counts_once_but_distinct_junction_segments_remain():
    hi = np.arange(26)
    vi = np.r_[np.arange(19), np.arange(26, 35)]
    expected = np.arange(35, dtype=float)
    np.testing.assert_array_equal(merge_profiles(expected[hi], expected[vi], hi, vi), expected)
    corrupt = expected[vi].copy()
    corrupt[0] += .1
    with pytest.raises(ValueError, match="Shared source segment"):
        merge_profiles(expected[hi], corrupt, hi, vi)


def test_missing_region_cannot_be_zero_imputed():
    with pytest.raises(ValueError, match="Not all 35"):
        merge_profiles([0.], [0.], [0], [0])
    with pytest.raises(ValueError, match="Missing"):
        finite([1, None])


@pytest.mark.parametrize("length", [25., 50., 75.])
def test_exponential_calculation_matches_independent_scalar_equation(length):
    xy = np.array([[-10., 0.], [0., 0.], [10., 0.], [0., 10.]])
    drive = np.array([1., .5, .1, 0.])
    direct = np.array([sum(drive[j] * np.exp(-sum(abs(a-b) for a, b in zip(p, q)) / length)
                           for j, q in enumerate(xy)) for p in xy])
    np.testing.assert_allclose(exponential_profile(xy, drive, length), direct / direct.max(), atol=1e-15)


@pytest.mark.parametrize("length", [25., 50., 75.])
def test_uniform_density_has_uniform_whole_cable_trajectory(length):
    xy = np.array([[-20., 0.], [-10., 0.], [0., 0.], [0., 0.], [10., 0.], [0., 10.]])
    r = backbone_cable(xy, np.ones(len(xy)), length)
    assert r["voltage"].shape == (401, 5)
    # Two source ROIs share the physical junction; neither acquires a doubled
    # membrane-density input or disappears from the observation mapping.
    assert r["source_to_node"][2] == r["source_to_node"][3]
    assert np.ptp(r["voltage"], axis=1).max() < 1e-12
    expected = 1 - .95 ** np.arange(401)
    np.testing.assert_allclose(r["voltage"], np.tile(expected[:, None], (1, 5)), atol=1e-12)
    assert r["equation_residual"].max() < 1e-12


def test_backbone_geometry_gap_is_not_silently_bridged():
    with pytest.raises(ValueError, match="connected 10 um"):
        backbone_cable(np.array([[0., 0.], [20., 0.]]), np.ones(2), 50)


@pytest.mark.parametrize("length", [0., -1., float("nan")])
def test_invalid_length_is_rejected(length):
    xy = np.array([[0., 0.], [10., 0.]])
    with pytest.raises(ValueError):
        exponential_profile(xy, np.ones(2), length)
    with pytest.raises(ValueError):
        backbone_cable(xy, np.ones(2), length)


def test_incomplete_settling_does_not_become_a_stationary_result():
    with pytest.raises(ValueError, match="did not reach"):
        backbone_cable(np.array([[0., 0.], [10., 0.]]), np.ones(2), 50, ticks=1)
