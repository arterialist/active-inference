from copy import deepcopy

import numpy as np
import pytest

from simulations.active_inference.experiments import context_organization as base
from simulations.active_inference.experiments.opponent_context import AfferentDelay
from simulations.active_inference.experiments import crossed_av_world as world
from simulations.active_inference.experiments.temporal_verification import configure
from neuron.neuron import RetrogradeSignalEvent
from neuron.extensions.experimental.predictive_receptor import PredictiveReceptorNeuron
from neuron.extensions.experimental.magnitude_retrograde import MagnitudeRetrogradeNeuron
from simulations.active_inference.experiments.magnitude_feedback_learning import observed_course


def loaded(tmp_path, enabled=False, cls=MagnitudeRetrogradeNeuron):
    cfg,g,s=configure(width=8)
    if enabled:
        for n in cfg['neurons']:n['metadata']['retrograde_magnitude_error']=True
    p=tmp_path/'config.json';p.write_text(base.encode(cfg))
    return base.fresh(p,11,cls)[0],g,s


def test_default_path_exact_with_body_and_ongoing_learning(tmp_path):
    a,g,s=loaded(tmp_path,cls=PredictiveReceptorNeuron)
    b,_,_=loaded(tmp_path)
    rng=np.random.default_rng(24)
    features=[dict(visual=rng.random((300,96)),auditory=rng.random((300,96))) for _ in (0,1)]
    x=world.course(a,base.Arm(),features,g,s,0,1,ticks=144,delay=AfferentDelay(64))
    y=observed_course(b,base.Arm(),AfferentDelay(64),features,g,s,0,1,ticks=144)
    assert all(np.array_equal(x[k],y[k]) for k in x)
    assert all(n.magnitude_retrograde_events==0 for n in b.network.neurons.values())


def test_only_inhibitory_information_return_changes_at_a_local_tick(tmp_path):
    net,g,_=loaded(tmp_path)
    native=net.network.neurons[g['mixed_1'][0]]
    changed=deepcopy(native);changed.retrograde_magnitude_error=True
    for sid in native.postsynaptic_points:
        native.input_buffer[sid,0]=np.float32(.3)
        changed.input_buffer[sid,0]=np.float32(.3)
    before={sid:p.u_i.info for sid,p in native.postsynaptic_points.items()}
    a=native.tick({},0);b=changed.tick({},0)
    for key in ('S','O','t_ref','t_last_fire'):assert getattr(native,key)==getattr(changed,key)
    for sid,p in native.postsynaptic_points.items():
        assert p.u_i.info==changed.postsynaptic_points[sid].u_i.info
    assert len(a)==len(b)
    corrected=0
    for e,f in zip(a,b):
        if isinstance(e,RetrogradeSignalEvent):
            sid=e.source_synapse_id
            np.testing.assert_array_equal(e.error_vector[1:],f.error_vector[1:])
            if before[sid]<0:
                assert f.error_vector[0]==-(np.float32(.3)-abs(before[sid]))
                assert np.sign(f.error_vector[0])==np.sign(abs(before[sid])-np.float32(.3))
                assert e.error_vector[0]<0
                corrected+=1
            else:np.testing.assert_array_equal(e.error_vector,f.error_vector)
        else:assert e==f
    assert changed.magnitude_retrograde_events==corrected>0


@pytest.mark.parametrize('value', [1, 'true', None])
def test_nonboolean_option_rejected(tmp_path,value):
    cfg,_,_=configure(width=8)
    cfg['neurons'][0]['metadata']['retrograde_magnitude_error']=value
    p=tmp_path/'invalid.json';p.write_text(base.encode(cfg))
    with pytest.raises(ValueError,match='must be boolean'):
        base.fresh(p,11,MagnitudeRetrogradeNeuron)


def test_invalid_opt_in_and_inhibitory_plastic_throughput(tmp_path):
    net,g,_=loaded(tmp_path,True)
    n=net.network.neurons[g['mixed_1'][0]]
    sid=next(i for i,p in n.postsynaptic_points.items() if p.u_i.info<0)
    n.postsynaptic_points[sid].u_i.plast=.1;n.input_buffer[sid,0]=.3
    with pytest.raises(ValueError,match='zero plastic throughput'):n.tick({},0)


def test_checkpoint_retains_extension_and_pending_returns(tmp_path):
    net,g,s=loaded(tmp_path,True)
    features=[dict(visual=np.ones((300,96))*.3,auditory=np.ones((300,96))*.2) for _ in (0,1)]
    arm=base.Arm();delay=AfferentDelay(64)
    world.course(net,arm,features,g,s,0,1,ticks=32,delay=delay)
    p=tmp_path/'checkpoint.paula';base.save_checkpoint(net,p)
    other=base.load_checkpoint(p,trusted=True).network
    arm2=base.Arm();arm2.restore(arm.state())
    history=AfferentDelay(64,delay.state())
    a=world.course(net,arm,features,g,s,0,0,ticks=96,delay=delay)
    b=world.course(other,arm2,features,g,s,0,0,ticks=96,delay=history)
    assert all(np.array_equal(a[k],b[k]) for k in a)
    assert all(n.retrograde_magnitude_error for n in other.network.neurons.values())
