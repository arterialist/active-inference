import numpy as np
import pytest
from scipy.sparse import csc_matrix

from simulations.drosophila.amin_input_physics import stationary


def test_stationary_conductance_matches_independent_dense_equation():
    c = np.array([.2, .3, .5])
    l = np.array([[2., -2., 0.], [-2., 5., -3.], [0., -3., 3.]])
    i, g = np.array([.1, 0., 0.]), np.array([0., 2., .5])
    expected = np.linalg.solve(np.diag(c+g)+l, i+g*.8)
    v, residual, mass = stationary(c, csc_matrix(l), i, g, .8)
    np.testing.assert_allclose(v, expected, atol=1e-14)
    assert residual < 1e-14 and mass < 1e-14


def test_uniform_ligand_gives_g_over_one_plus_g_at_every_node():
    c = np.array([.2, .3, .5])
    l = csc_matrix([[2., -2., 0.], [-2., 5., -3.], [0., -3., 3.]])
    for gain in (.1, 1., 10., 100.):
        v, _, _ = stationary(c, l, np.zeros(3), c*gain)
        np.testing.assert_allclose(v, gain/(1+gain), atol=1e-13)


def test_weak_conductance_tends_to_matched_current_response():
    c = np.array([.2, .3, .5])
    l = csc_matrix([[2., -2., 0.], [-2., 5., -3.], [0., -3., 3.]])
    drive = c*np.array([1., .2, 0.])
    reference = stationary(c, l, drive)[0]
    weak = stationary(c, l, np.zeros(3), drive*1e-8)[0]/1e-8
    np.testing.assert_allclose(weak, reference, rtol=1e-8)


@pytest.mark.parametrize("bad", [-1., float("nan"), float("inf")])
def test_invalid_conductance_is_rejected(bad):
    with pytest.raises(ValueError, match="Invalid"):
        stationary(np.ones(2)/2, csc_matrix([[1., -1.], [-1., 1.]]), np.zeros(2), np.array([bad, 0.]))
