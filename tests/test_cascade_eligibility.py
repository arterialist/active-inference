from copy import deepcopy
import math

import numpy as np
import pytest

from simulations.active_inference.experiments import context_organization as base
from simulations.active_inference.experiments.active_sweep_probe import configure as original,record,PhysicalDelay
from simulations.active_inference.experiments.active_sweep_credit import configure,kernel_means,record_credit,CONDITIONS
from simulations.active_inference.experiments.active_sweep_credit_analysis import audit
from simulations.active_inference.components.body.loaded_hinge import LoadedHinge
from neuron.extensions.experimental.cascade_eligibility import CascadeEligibilityNeuron,cascade_step
from neuron.extensions.experimental.magnitude_retrograde import MagnitudeRetrogradeNeuron


def prepare(tmp_path,condition,cls=CascadeEligibilityNeuron):
    cfg,g,_=original(11,width=8);cfg=configure(cfg,g,condition)
    path=tmp_path/f'{condition}-{cls.__name__}.json';path.write_text(base.encode(cfg))
    net=base.fresh(path,11,cls)[0]
    rng=np.random.default_rng(90)
    features=dict(visual=rng.uniform(0,.5,(300,96)),auditory=rng.uniform(0,.5,(300,96)))
    return net,g,cfg,features


def test_cascade_impulse_mass_mean_and_shape():
    d0=math.exp(-1/64);mean=d0/(1-d0)
    for stages in (1,8):
        d=mean/(mean+stages);state=np.zeros((stages,1));impulse=[]
        for t in range(6000):
            state=cascade_step(state,np.array([float(t==0)]),d);impulse.append(state[-1,0])
        impulse=np.array(impulse)
        assert impulse.sum()==pytest.approx(1.,abs=2e-13)
        assert (np.arange(len(impulse))*impulse).sum()==pytest.approx(mean,abs=1e-10)
        assert (np.argmax(impulse)==0)==(stages==1)
    with pytest.raises(ValueError):cascade_step(np.zeros((2,1)),np.array([-1.]),.9)


def test_factorial_changes_only_selected_metadata(tmp_path):
    cfg,g,_=original(11,width=8);old,matched=kernel_means(cfg,g)
    assert old==pytest.approx(63.501302078)
    assert matched==pytest.approx(82.520811664)
    for condition in CONDITIONS:
        new=configure(cfg,g,condition)
        assert new['connections']==cfg['connections'] and new['synaptic_points']==cfg['synaptic_points']
        assert new['external_inputs']==cfg['external_inputs']
        for a,b in zip(new['neurons'],cfg['neurons']):
            assert a['params']==b['params']
            md=deepcopy(a['metadata']);md.pop('prediction_credit_stages',None);md.pop('prediction_credit_mean',None)
            assert md==b['metadata']


def test_default_is_exact_and_audit_detects_corruption(tmp_path):
    n,g,cfg,features=prepare(tmp_path,'old')
    legacy,_,_,_=prepare(tmp_path,'old',MagnitudeRetrogradeNeuron)
    a=record_credit(n,LoadedHinge(.8),PhysicalDelay(),features,g,256)
    b=record(legacy,LoadedHinge(.8),PhysicalDelay(),features,g,ticks=256)
    for key in b:np.testing.assert_array_equal(a[key],b[key])
    assert audit(a,cfg,g,features)==0.
    for key in ('credit_states','weights','errors','raw_afferents','drive','physical_states','gate'):
        bad={k:v.copy() for k,v in a.items()};bad[key][100]+=.01 if bad[key].dtype.kind=='f' else 1
        with pytest.raises(ValueError):audit(bad,cfg,g,features)
    bad={k:v.copy() for k,v in a.items()}
    source=list(a['neuron_ids']).index(g['error_positive'][0])
    bad['cells'][150,source,base.FIELDS.index('O')]+=.1
    with pytest.raises(ValueError,match='Teaching differs from neural release'):audit(bad,cfg,g,features)


@pytest.mark.parametrize('condition',['mean_only','shape_only','matched_cascade'])
def test_enabled_local_cascades_in_the_coupled_body(tmp_path,condition):
    n,g,cfg,features=prepare(tmp_path,condition)
    a=record_credit(n,LoadedHinge(.8),PhysicalDelay(),features,g,256)
    assert audit(a,cfg,g,features)==0.
    assert np.any(a['weights']) and np.all(a['eta']>0)
    assert np.any(a['body'][:,1])
    # Exact executable continuation includes every added intracellular stage.
    import random
    from simulations.active_inference.experiments.crossed_av_continuation import isolated_rng
    state=a['physical_states'][-1];history=a['delay_final'];gates=a['gate'][-1]
    path=tmp_path/f'{condition}.paula';base.save_checkpoint(n,path,sources=[])
    body=LoadedHinge(.8);body.restore(state,crossings=int(gates[0]),next_gate=int(gates[1]))
    with isolated_rng():expected=record_credit(n,body,PhysicalDelay(history),features,g,64)
    saved=base.load_checkpoint(path,trusted=True)
    random.setstate(saved.python_rng);np.random.set_state(saved.numpy_rng)
    body=LoadedHinge(.8);body.restore(state,crossings=int(gates[0]),next_gate=int(gates[1]))
    actual=record_credit(saved.network,body,PhysicalDelay(history),features,g,64)
    for key in expected:np.testing.assert_array_equal(actual[key],expected[key])


def test_continuing_course_and_reset_keep_nonweight_state(tmp_path):
    from simulations.active_inference.experiments.active_sweep_acquisition import continuation,replay_prefix
    from simulations.active_inference.experiments.active_sweep_memory import restore,reset_selected
    n,g,cfg,features=prepare(tmp_path,'matched_cascade')
    body=LoadedHinge(.8);delay=PhysicalDelay()
    acquired=record_credit(n,body,delay,features,g,256)
    neural=tmp_path/'acquired.paula';physical=tmp_path/'body.npz'
    base.save_checkpoint(n,neural,sources=[])
    np.savez_compressed(physical,state=body.state(),delay=delay.state(),gate=[body.crossings,body.next_gate])
    intact=record_credit(n,body,delay,features,g,96)
    continuation(acquired,intact)
    n,body,delay=restore(neural,physical)
    short=record_credit(n,body,delay,features,g,64)
    replay_prefix(intact,short)
    bad={k:v.copy() for k,v in short.items()};bad['credit_states'][20,0,0,0]+=.001
    with pytest.raises(ValueError):replay_prefix(intact,bad)
    bad={k:v.copy() for k,v in intact.items()};bad['gate_initial'][0]+=1
    with pytest.raises(ValueError):continuation(acquired,bad)
    n,body,delay=restore(neural,physical);reset_selected(n,g)
    reset=record_credit(n,body,delay,features,g,96)
    continuation(acquired,reset,reset=True)
    assert audit(reset,cfg,g,features)==0.
    assert np.any(reset['weights'])  # Reset diagnostic continues acquiring.
