from copy import deepcopy
import math

import numpy as np
import pytest

from simulations.active_inference.experiments import context_organization as base
from simulations.active_inference.experiments.temporal_verification import configure
from simulations.active_inference.components.learning.eligibility_reference import append_eligibility_reference
from neuron.extensions.experimental.contrast_eligibility import ContrastEligibilityNeuron
from neuron.extensions.experimental.magnitude_retrograde import MagnitudeRetrogradeNeuron


def fixture(tmp_path,strength=1.,cls=ContrastEligibilityNeuron):
    cfg,g,s=configure(11,width=8)
    for n in cfg['neurons']:n['metadata']['retrograde_magnitude_error']=True
    cfg,added=append_eligibility_reference(cfg,{k:g[k] for k in ('mixed_0','mixed_1')},g['prediction'],
                                         enabled=True,strength=strength)
    p=tmp_path/'config.json';p.write_text(base.encode(cfg))
    return base.fresh(p,11,cls)[0],dict(g,**added),s,cfg


def test_zero_strength_exact_inherited_dynamics(tmp_path):
    a,g,_,cfg=fixture(tmp_path,0.);b,_,_,_=fixture(tmp_path,0.,MagnitudeRetrogradeNeuron)
    for t in range(80):
        for net in (a,b):
            for nid in g['vision'][:8]:net.set_external_input(nid,0,.4+.1*math.sin(t/7))
            net.set_external_input(g['force'][0],0,.3)
            net.run_tick()
        assert np.array_equal(base.cellular(list(a.network.neurons.values())),base.cellular(list(b.network.neurons.values())))
        for nid in a.network.neurons:
            an,bn=a.network.neurons[nid],b.network.neurons[nid]
            assert [p.u_i.info for p in an.postsynaptic_points.values()]==[p.u_i.info for p in bn.postsynaptic_points.values()]


def test_signed_eligibility_changes_write_not_current_forward_state(tmp_path):
    net,g,_,_=fixture(tmp_path);n=net.network.neurons[g['prediction'][0]]
    for sid in n.prediction_ports:n.postsynaptic_points[sid].u_i.info=.4
    n.prediction_context[:]=np.linspace(.1,.9,len(n.prediction_ports));n.prediction_error=.3
    n.prediction_reference[:]=.5
    control=deepcopy(n);control.prediction_reference_strength=0.
    events=n.tick({},0);other=control.tick({},0)
    assert n.O==control.O and n.S==control.S and len(events)==len(other)
    actual=np.array([n.postsynaptic_points[s].u_i.info for s in n.prediction_ports])
    np.testing.assert_array_equal(actual,np.clip(.4+n.prediction_eta*.3*n.prediction_eligibility,0,1))
    assert actual.min()<.4<actual.max() and n.prediction_eta>0
    assert all(n.postsynaptic_points[s].potential==0 for s in n.prediction_reference_ports)


def test_reference_cannot_be_host_or_somatic_drive(tmp_path):
    net,g,_,_=fixture(tmp_path);n=net.network.neurons[g['prediction'][0]];sid=n.prediction_reference_ports[0]
    with pytest.raises(ValueError,match='driven by neurons'):n.tick({sid:{'info':1.}},0)
    n.postsynaptic_points[sid].u_i.info=.1
    with pytest.raises(ValueError,match='zero somatic'):n.tick({},0)


def test_disabled_builder_preserves_graph_and_bad_banks_rejected():
    cfg,g,_=configure(11,width=8);old=deepcopy(cfg)
    result,added=append_eligibility_reference(cfg,{},g['prediction'])
    assert result==old==cfg and added=={}
    with pytest.raises(ValueError,match='disjoint'):
        append_eligibility_reference(cfg,{'a':[1,1]},g['prediction'],enabled=True)


def test_embodied_record_audits_and_observer_does_not_change_state(tmp_path):
    from simulations.active_inference.experiments.eligibility_reference_probe import record,verify_learning
    from simulations.active_inference.experiments import crossed_av_world as world
    from simulations.active_inference.experiments.opponent_context import AfferentDelay
    net,g,selected,cfg=fixture(tmp_path);control=deepcopy(net)
    rng=np.random.default_rng(11)
    features=[dict(visual=rng.uniform(0,.5,(300,96)),auditory=rng.uniform(0,.5,(300,96))) for _ in (0,1)]
    z=record(net,base.Arm(),AfferentDelay(64),features,g,selected,0,1,ticks=96)
    expected=world.course(control,base.Arm(),features,g,selected,0,1,ticks=96,delay=AfferentDelay(64))
    for key in expected:np.testing.assert_array_equal(z[key],expected[key])
    assert verify_learning(z)<2e-12
    from simulations.active_inference.experiments.eligibility_reference_pool_audit import audit_pools
    checked=audit_pools(z,cfg)
    assert checked['checked'][:1].sum()==0
    assert checked['checked'][2:].all()
    for field,column in [('pool_weights',None),('cells',base.FIELDS.index('S')),('cells',base.FIELDS.index('O'))]:
        bad={k:v.copy() for k,v in z.items()}
        if column is None:bad[field][30,0,0]+=.01
        else:bad[field][30,list(z['neuron_ids']).index(z['pool_ids'][0]),column]+=.01
        with pytest.raises(ValueError,match='intracellular'):audit_pools(bad,cfg)
    for field in ('reference_arrivals','reference_trace','effective_eligibility','weights','eta'):
        bad={k:v.copy() for k,v in z.items()};bad[field][70]+=.01
        with pytest.raises(ValueError):verify_learning(bad)


def test_checkpoint_keeps_reference_state_and_inflight_signals(tmp_path):
    from simulations.active_inference.experiments.crossed_av_continuation import isolated_rng
    import random
    net,g,_,_=fixture(tmp_path)
    def advance(network,start,count):
        rows=[]
        for t in range(start,start+count):
            for nid in g['vision'][:12]:network.set_external_input(nid,0,.3+.2*np.sin(t/8))
            network.set_external_input(g['force'][t%2],0,.5)
            network.run_tick()
            for nid in g['prediction']:
                n=network.network.neurons[nid]
                rows.append(np.r_[n.S,n.O,n.prediction_reference,n.prediction_eligibility,
                                  [n.postsynaptic_points[s].u_i.info for s in n.prediction_ports]])
        return np.array(rows)
    advance(net,0,40);p=tmp_path/'state.paula';base.save_checkpoint(net,p,sources=[])
    with isolated_rng():expected=advance(net,40,48)
    saved=base.load_checkpoint(p,trusted=True)
    random.setstate(saved.python_rng);np.random.set_state(saved.numpy_rng)
    np.testing.assert_array_equal(advance(saved.network,40,48),expected)


def test_independent_audit_requires_each_executable_checkpoint():
    from simulations.active_inference.experiments.eligibility_reference_analysis import verify_checkpoint_family
    m=dict(conditions={'wired':None,'slow':None,'contrast':None},normal_blocks=1,reversal_blocks=1)
    checkpoints=[dict(condition=c,reverse=r,block=1) for c in m['conditions'] for r in (False,True)]
    verify_checkpoint_family(m,checkpoints)
    for bad in (checkpoints[:-1],checkpoints+[checkpoints[0]]):
        with pytest.raises(ValueError,match='checkpoint family'):verify_checkpoint_family(m,bad)


def test_opposed_write_requires_an_actual_change_and_positive_input():
    from simulations.active_inference.experiments.eligibility_reference_analysis import opposed_writes
    z=dict(weights_initial=np.array([[.2,.3]]),weights=np.array([[[.3,.2]],[[.3,.4]]]),
           errors=np.array([[[-1.,0.,0.]],[[1.,0.,0.]]]),arrivals=np.array([[[1.,1.]],[[1.,0.]]]))
    np.testing.assert_array_equal(opposed_writes(z),[[[True,False]],[[False,False]]])
