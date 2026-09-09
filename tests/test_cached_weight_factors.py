import numpy as np
import pytest
from simulations.active_inference.experiments.cached_weight_factors import growth_factors
from simulations.active_inference.experiments.cached_weight_factors_audit import acquired_contrasts


def test_mean_growth_keeps_birth_differences_and_learned_total():
    ports=[(n,s,100+s) for n in (10,11) for s in range(4)]
    q0=np.arange(8.)*.01+.2; q1=q0+np.array([0,.1,.2,.3,.04,.03,.02,.01])
    factors=growth_factors(ports,q0,q1)
    for indices in (slice(0,4),slice(4,8)):
        assert np.isclose(factors['mean_growth'][indices].sum(),q1[indices].sum())
        assert np.allclose(np.diff(factors['mean_growth'][indices]),np.diff(q0[indices]))
        for i in (1,2,3):
            assert np.array_equal(np.sort(factors[f'birth_cycle{i}'][indices]),q0[indices])


def test_do_not_clip_invalid_mean_growth_into_a_different_intervention():
    ports=[(1,i,10+i) for i in range(4)]
    with pytest.raises(ValueError):
        growth_factors(ports,np.array([.01,.8,.8,.8]),np.zeros(4))


def test_birth_placement_is_removed_from_acquired_placement_contrast():
    names=('intact','initial','cycle1','cycle2','cycle3','birth_cycle1','birth_cycle2','birth_cycle3','mean_growth')
    x={(name,cue):np.full((8,2),10.-cue) for name in names for cue in (0,1)}
    x['intact',0]+=3; x['initial',0]+=3
    d=acquired_contrasts(x)
    assert (d['birth_placement']==3).all()
    assert not d['acquired_placement'].any()
    x['intact',0]+=2
    assert (acquired_contrasts(x)['acquired_placement']==2).all()
