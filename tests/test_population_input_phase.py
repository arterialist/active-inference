import numpy as np
import pytest

from simulations.active_inference.experiments.population_input_phase_probe import drive


def test_real_encoder_and_continuous_control_match_dose_not_timing():
    groups=dict(vision=[1,2,3,4],touch=[])
    features=[dict(ticks=16,visual=np.full((16,4),.6),auditory=np.zeros((16,0)))]
    trial=dict(start=0,stop=16,visual_clip=0,audio_clip=None)
    arrays=[]
    for mode in ('four_phase','continuous_mean'):
        a=np.zeros((16,4))
        for t in range(16):
            for n,v in drive(groups,trial,t,features,mode):a[t,n-1]=v
        arrays.append(a)
    np.testing.assert_allclose(arrays[0].sum(axis=0),arrays[1].sum(axis=0),rtol=0,atol=1e-14)
    assert not np.array_equal(*arrays)
    assert np.all(np.count_nonzero(arrays[0],axis=1)==1)
    assert np.all(np.count_nonzero(arrays[1],axis=1)==4)
    with pytest.raises(ValueError):drive(groups,trial,0,features,'unknown')
