import numpy as np

from simulations.active_inference.experiments.predictive_transplant_audit import interactions


def test_shared_weight_gain_is_not_mistaken_for_feature_assignment():
    records={}
    for mapping,sign in (('paired',1),('swapped',-1)):
        for cue,c in ((0,1),(1,-1)):
            for layer in ('prediction','consumer'):
                records[mapping,'birth',cue,layer]=np.zeros((5,2))
                learned=np.tile([3.+sign*c,3.-sign*c],(5,1))
                records[mapping,'learned',cue,layer]=learned
                records[mapping,'shuffled',cue,layer]=learned.copy()
    effects=interactions(records,[1.,-1.])
    assert np.all(effects['minus_birth_projection']>0)
    np.testing.assert_array_equal(effects['minus_shuffled_projection'],0)
    records['paired','shuffled',0,'prediction']-=np.array([1.,-1.])
    effects=interactions(records,[1.,-1.])
    assert np.all(effects['minus_shuffled_projection']>0)
