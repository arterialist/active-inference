from copy import deepcopy

import numpy as np
import pytest

from simulations.active_inference.experiments import context_organization as base
from simulations.active_inference.experiments.active_sweep_probe import configure, PhysicalDelay, record, verify
from simulations.active_inference.components.body.loaded_hinge import LoadedHinge, afferents
from neuron.extensions.experimental.magnitude_retrograde import MagnitudeRetrogradeNeuron


def preparation(tmp_path):
    cfg,g,_ = configure(11,width=8)
    path=tmp_path/'brain.json';path.write_text(base.encode(cfg))
    net=base.fresh(path,11,MagnitudeRetrogradeNeuron)[0]
    rng=np.random.default_rng(90)
    features=dict(visual=rng.uniform(0,.5,(300,96)),auditory=rng.uniform(0,.5,(300,96)))
    return net,g,features


def test_loaded_world_requires_motion_and_has_passive_force():
    body=LoadedHinge(.8)
    for _ in range(300):
        body.step(0.)
    assert body.crossings==0
    np.testing.assert_array_equal(afferents(body),np.zeros(6))
    body.data.qpos[0]=.1;body.data.qvel[0]=.2
    assert body.environmental_torque()==pytest.approx(-.175)
    assert body.environmental_torque()*body.data.qvel[0]<0
    for bad in (-.1,2.,np.nan):
        with pytest.raises(ValueError):LoadedHinge(bad)


def test_gate_counts_alternating_physical_crossings_only():
    body=LoadedHinge()
    body.data.qpos[0]=.02
    body.step(0.)
    assert body.crossings==1 and body.next_gate==-1
    for _ in range(5):body.step(0.)
    assert body.crossings==1
    body.data.qpos[0]=-.02;body.data.qvel[0]=0.
    body.step(0.)
    assert body.crossings==2 and body.next_gate==1


def test_wiring_changes_only_declared_context_weights():
    a,g,_=configure(11,True,width=8);b,h,_=configure(11,False,width=8)
    assert g==h and a['connections']==b['connections'] and a['neurons']==b['neurons']
    changed=[]
    for x,y in zip(a['synaptic_points'],b['synaptic_points']):
        if x!=y:
            assert x['type']=='postsynaptic' and x['neuron_id'] in g['mixed_0']+g['mixed_1']
            assert abs(x['u_i']['info'])==.125 and y['u_i']['info']==0.
            changed.append(x)
    assert len(changed)==16*6
    assert len({n['id'] for n in a['neurons']})==len(a['neurons'])
    assert all(n['params']['eta_post']>0 and n['params']['eta_retro']>0 for n in a['neurons'])


def test_observer_is_exact_and_physical_learning_audit_rejects_corruption(tmp_path):
    net,g,features=preparation(tmp_path);other=deepcopy(net)
    a=record(net,LoadedHinge(.8),PhysicalDelay(),features,g,ticks=256)
    b=record(other,LoadedHinge(.8),PhysicalDelay(),features,g,ticks=256,observe=False)
    for k in b:np.testing.assert_array_equal(a[k],b[k])
    assert verify(a,g,features)==0.
    assert np.any(a['neural_command']) and np.any(a['raw_afferents']) and np.any(a['weights'])
    from simulations.active_inference.experiments.active_sweep_analysis import mechanics, verify_predictive_arrivals
    cfg,_,_=configure(11,width=8)
    mechanics(a);verify_predictive_arrivals(a,cfg,g)
    bad={k:v.copy() for k,v in a.items()};bad['arrivals'][100,0,1]+=.1
    with pytest.raises(ValueError,match='source-neuron'):
        verify_predictive_arrivals(bad,cfg,g)
    ids=list(a['neuron_ids'])
    clock=a['cells'][:,[ids.index(n) for n in g['cpg']],base.FIELDS.index('O')]
    assert np.all((clock>0).sum(axis=0)>=1) and (clock[:,0]>0).sum()>=2
    for field in ('body','physical_states','drive','raw_afferents','weights','birth_input','gate'):
        bad={k:v.copy() for k,v in a.items()};bad[field][100]+=.01 if bad[field].dtype.kind=='f' else 1
        with pytest.raises(ValueError):verify(bad,g,features)


def test_motor_omission_prevents_motion_without_freezing_brain(tmp_path):
    net,g,features=preparation(tmp_path)
    z=record(net,LoadedHinge(.8),PhysicalDelay(),features,g,ticks=256,coupling=0.)
    assert verify(z,g,features)==0.
    assert np.any(z['neural_command'])
    assert not np.any(z['body'][:,1:]) and not np.any(z['raw_afferents'])
    assert np.all(z['eta']>0)
    assert not np.any(z['weights'])


def test_delay_preserves_all_six_physical_channels():
    delay=PhysicalDelay();pulse=np.arange(6,dtype=float)
    assert not np.any(delay.step(pulse))
    for _ in range(63):assert not np.any(delay.step(np.zeros(6)))
    np.testing.assert_array_equal(delay.step(np.zeros(6)),pulse)


def test_executable_checkpoint_replays_the_coupled_continuation(tmp_path):
    import random
    from simulations.active_inference.experiments.crossed_av_continuation import isolated_rng
    net,g,features=preparation(tmp_path);body=LoadedHinge(.8);delay=PhysicalDelay()
    record(net,body,delay,features,g,ticks=128)
    path=tmp_path/'acquired.paula';base.save_checkpoint(net,path,sources=[])
    state=body.state();history=delay.state();crossings=body.crossings;next_gate=body.next_gate
    with isolated_rng():expected=record(net,body,delay,features,g,ticks=96)
    snapshot=base.load_checkpoint(path,trusted=True)
    random.setstate(snapshot.python_rng);np.random.set_state(snapshot.numpy_rng)
    restored=LoadedHinge(.8);restored.restore(state,next_gate=next_gate,crossings=crossings)
    actual=record(snapshot.network,restored,PhysicalDelay(history),features,g,ticks=96)
    assert set(actual)==set(expected)
    for key in expected:np.testing.assert_array_equal(actual[key],expected[key])
    assert verify(actual,g,features)==0.
