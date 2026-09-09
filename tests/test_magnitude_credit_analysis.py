import numpy as np
import pytest

from simulations.active_inference.experiments import context_organization as base
from simulations.active_inference.experiments import crossed_av_world as world
from simulations.active_inference.experiments.temporal_verification import configure
from simulations.active_inference.experiments.opponent_context import AfferentDelay
from simulations.active_inference.experiments.crossed_av_credit import differential
from simulations.active_inference.experiments.crossed_av_continuation_analysis import FACTORIAL
from simulations.active_inference.experiments.magnitude_credit_analysis import update_effects
from neuron.extensions.experimental.magnitude_retrograde import MagnitudeRetrogradeNeuron


@pytest.fixture
def episode(tmp_path):
    cfg,g,s=configure(width=8)
    for n in cfg['neurons']:n['metadata']['retrograde_magnitude_error']=True
    p=tmp_path/'config.json';p.write_text(base.encode(cfg))
    net=base.fresh(p,11,MagnitudeRetrogradeNeuron)[0]
    rng=np.random.default_rng(8)
    features=[dict(visual=rng.random((300,96)),auditory=rng.random((300,96))) for _ in (0,1)]
    return world.course(net,base.Arm(),features,g,s,0,0,ticks=144,delay=AfferentDelay(64))


def test_full_tick_projection_and_component_reconstruction(episode):
    basis=np.random.default_rng(9).random((4,64,16))
    z=episode;result=update_effects(z,basis)
    q=differential(z['weights'],z['context_source_ids'])
    q0=differential(z['weights_initial'],z['context_source_ids'])
    pair_response=np.array([[b@v for b in basis] for v in np.r_[q0[None],q]])
    expected=np.einsum('mp,tpk->tmk',FACTORIAL,np.diff(pair_response,axis=0))
    np.testing.assert_allclose(result['update_modes'],expected,rtol=0,atol=2e-14)
    assert np.any(result['update_modes'])
    np.testing.assert_allclose(result['component_modes_at63'].sum(1),expected[:,:,63],rtol=0,atol=2e-14)


def test_shared_activity_has_no_joint_contrast(episode):
    one=np.random.default_rng(5).random((64,16));basis=np.stack([one]*4)
    result=update_effects(episode,basis)
    assert np.any(result['update_modes'][:,0])
    np.testing.assert_array_equal(result['update_modes'][:,1:],0)
    with pytest.raises(ValueError,match='finite four-pair'):
        update_effects(episode,basis[:3])
