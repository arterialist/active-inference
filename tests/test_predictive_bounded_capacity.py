import numpy as np
import pytest

from simulations.active_inference.experiments.predictive_bounded_capacity import bounded_projection


def test_bounded_axis_targets_and_independent_lower_bound():
    result=bounded_projection([[1.,0.],[0.,1.]],[2.,.4])
    np.testing.assert_allclose(result['closest'],[1.,.4],atol=1e-12)
    assert result['error_lower']==pytest.approx(1.)
    assert result['error_upper']==pytest.approx(1.)
    assert result['gap']<1e-12


def test_correlated_features_cannot_invent_a_missing_cue_axis():
    result=bounded_projection([[1.,1.],[2.,2.]],[1.,0.])
    np.testing.assert_allclose(result['closest'],[.5,.5],atol=1e-12)
    assert result['error_lower']==pytest.approx(.5)
    feasible=bounded_projection([[.5,0.],[0.,.5]],[.2,.3])
    assert feasible['error_upper']<1e-24
    with pytest.raises(ValueError):bounded_projection([[1.,0.]],[1.,0.],cap=0.)


def test_roundoff_repair_recomputes_the_feasible_certificate(monkeypatch):
    from types import SimpleNamespace
    from simulations.active_inference.experiments import predictive_bounded_capacity as module
    monkeypatch.setattr(module,'lsq_linear',lambda *a,**kw:SimpleNamespace(success=True,x=np.array([-1e-17,1.])))
    result=bounded_projection([[1.,0.],[0.,1.]],[0.,2.])
    np.testing.assert_array_equal(result['weights'],[0.,1.])
    assert result['error_upper']==result['error_lower']==1.
