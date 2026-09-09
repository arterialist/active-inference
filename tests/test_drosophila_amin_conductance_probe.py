import numpy as np
import pytest

from simulations.drosophila.amin_conductance_probe import check_transition, independent_geometry


def test_independent_equation_checker_accepts_solve_and_rejects_corrupted_state():
    geometry = {"parents": np.array([-1, 0, 1]),
        "xyz_um": np.array([[0., 0., 0.], [2., 0., 0.], [2., 3., 0.]]),
        "radius_um": np.array([.2, .3, .25])}
    c, l = independent_geometry(geometry, 20.)
    v = np.array([.3, .1, .2])
    g = c*np.array([5., 0., 1.])
    expected = np.linalg.solve(np.diag(c+.05*g)+.05*l.toarray(), .95*c*v+.05*g)
    error, mass = check_transition(v, expected, c, l, g, .05)
    assert error < 1e-12 and mass < 1e-12
    corrupted = expected.copy()
    corrupted[1] += .001
    with pytest.raises(ValueError, match="equation mismatch"):
        check_transition(v, corrupted, c, l, g, .05)
    with pytest.raises(ValueError, match="equation mismatch"):
        check_transition(v, expected, c, l, g*2, .05)


def test_zero_axial_coupling_retains_membrane_measure():
    geometry = {"parents": np.array([-1, 0, 1]),
        "xyz_um": np.array([[0., 0., 0.], [2., 0., 0.], [2., 3., 0.]]),
        "radius_um": np.array([.2, .3, .25])}
    c, l = independent_geometry(geometry, 0.)
    c1, _ = independent_geometry(geometry, 20.)
    np.testing.assert_array_equal(c, c1)
    np.testing.assert_array_equal(l.toarray(), np.zeros((3, 3)))
