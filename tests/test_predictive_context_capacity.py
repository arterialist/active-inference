import numpy as np
import pytest

from simulations.active_inference.experiments.predictive_context_capacity import cone_projection


def test_positive_cone_and_witness():
    points=np.array([[1.,1.],[2.,1.]])
    pred,witness=cone_projection(points,[0.,1.])
    np.testing.assert_allclose(pred,[.5,.5])
    np.testing.assert_allclose(points.T@witness,pred)
    assert np.all(witness>=0)
    pred,witness=cone_projection(points,[3.,2.])
    np.testing.assert_allclose(pred,[3.,2.])
    np.testing.assert_allclose(points.T@witness,pred)


def test_sensory_gain_cannot_expand_the_cone_but_opponent_codes_can():
    points=np.array([[1.,1.],[2.,1.]])
    a,_=cone_projection(points,[0.,1.]);b,_=cone_projection(points*np.array([[7.],[.03]]),[0.,1.])
    np.testing.assert_allclose(a,b)
    expanded=np.vstack((points,[0.,1.]))
    pred,_=cone_projection(expanded,[0.,1.]);np.testing.assert_allclose(pred,[0.,1.])


def test_degenerate_zero_and_collinear_inputs():
    for points in (np.zeros((3,2)),np.array([[0.,0.]])):
        pred,_=cone_projection(points,[1.,2.]);np.testing.assert_array_equal(pred,[0.,0.])
    pred,_=cone_projection([[1.,1.],[2.,2.]],[2.,1.]);np.testing.assert_allclose(pred,[1.5,1.5])
    pred,_=cone_projection([[1.,0.],[0.,1.]],[0.,0.]);np.testing.assert_array_equal(pred,[0.,0.])
    with pytest.raises(ValueError):cone_projection([[-1.,2.]],[1.,1.])
