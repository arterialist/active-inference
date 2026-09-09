import numpy as np

from simulations.active_inference.experiments.sensory_timing_audit import expected_external,feature_effect
from simulations.active_inference.experiments.sensory_timing_transfer import timed_inputs


def test_independent_input_reconstruction_and_preserved_feature_contrast():
    fs=[dict(ticks=8,visual=np.arange(16).reshape(8,2)/20,auditory=np.zeros((8,2)))]
    groups=dict(vision=[1,2],touch=[3,4]);tr=dict(start=0,stop=8,visual_clip=0,audio_clip=None)
    for mode in ('native','continuous','phase_shift_1'):
        actual=np.zeros((8,4))
        for t in range(8):
            for n,v in timed_inputs(fs,groups,tr,t,mode):actual[t,n-1]=v
        np.testing.assert_array_equal(actual,expected_external(fs,[1,2,3,4],[1,2],0,mode,8))
    records={(m,c,cue):np.zeros((8,2)) for m in ('paired','swapped') for c in ('learned','shuffled') for cue in (0,1)}
    records['paired','learned',0][:]=[1.,-1.]
    np.testing.assert_array_equal(feature_effect(records),records['paired','learned',0])
    records['paired','shuffled',0][:]=[1.,-1.]
    np.testing.assert_array_equal(feature_effect(records),np.zeros((8,2)))
