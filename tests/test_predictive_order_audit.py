from copy import deepcopy

import numpy as np
import pytest

from simulations.active_inference.experiments.predictive_order_audit import combine


def test_balancing_does_not_hide_order_reversal():
    a=dict(physical_profile_axis=np.array([1.,-1.]),
           prediction_assignment_by_cue_interaction=np.array([[0.,0.],[2.,0.],[1.,0.]]),
           consumer_assignment_by_cue_interaction=np.array([[0.],[1.],[2.]]))
    b=deepcopy(a);b['prediction_assignment_by_cue_interaction']*=-.5
    result=combine([a,b])
    assert np.all(result['projection_balanced'][1:]>0)
    assert np.all(result['projection_by_order'][0,1:]>0)
    assert np.all(result['projection_by_order'][1,1:]<0)
    np.testing.assert_array_equal(result['prediction_balanced'],a['prediction_assignment_by_cue_interaction']/4)
    b['physical_profile_axis'][0]=2
    with pytest.raises(ValueError,match='reference axes'):combine([a,b])
