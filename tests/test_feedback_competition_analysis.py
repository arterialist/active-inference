import numpy as np
import pytest

from simulations.active_inference.experiments.feedback_competition_analysis import representation


def test_shared_activity_is_not_joint_representation():
    a=np.broadcast_to(np.linspace(.1,1.,64)[None,:,None],(4,64,7)).copy()
    filtered,modes,gram=representation(a)
    np.testing.assert_allclose(modes[0],filtered[0])
    assert not np.any(modes[1:]) and not np.any(gram[:,1:])
    a[0,:,0]+=.2;a[3,:,0]+=.2
    _,modes,gram=representation(a)
    assert np.all(modes[3,2:,0]>0)
    np.testing.assert_allclose(gram[:,3,3],(modes[3]**2).sum(1))
    with pytest.raises(ValueError):representation(a[:3])
    a[0,0,0]=np.nan
    with pytest.raises(ValueError):representation(a)
