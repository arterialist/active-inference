from copy import deepcopy

import numpy as np
import pytest

from simulations.active_inference.experiments.context_organization import (
    Arm, configure, course, encode, fresh, physical_drive, verify_learning,
    verify_physics, PredictiveReceptorNeuron, save_checkpoint, load_checkpoint,
)


def features():
    rng=np.random.default_rng(411)
    return [dict(visual=rng.uniform(0,1,(300,96)),auditory=rng.uniform(0,1,(300,96))) for _ in range(2)]


def test_context_is_world_contingency_not_wired_assignment():
    cfg,g,selected=configure(width=8)
    blind,gb,sb=configure(width=8,contextual=False)
    assert g==gb and selected==sb
    assert cfg['connections']==blind['connections']
    assert all(n['params']['eta_post']>0 and n['params']['eta_retro']>0 for n in cfg['neurons'])
    f=features()
    for clip in (0,1):
        a,fa=physical_drive(f,0,clip,19)
        b,fb=physical_drive(f,1,clip,19)
        assert np.array_equal(a[:192],b[:192]) and fa==-fb
        _,fr=physical_drive(f,0,clip,19,reverse=True)
        assert fr==-fa


def test_learning_physics_and_executable_replay(tmp_path):
    cfg,g,s=configure(width=8)
    path=tmp_path/'config.json';path.write_text(encode(cfg))
    net,_,_,_=fresh(path,11,PredictiveReceptorNeuron)
    body=Arm();f=features()
    prefix=course(net,body,f,g,s,0,0,ticks=40)
    assert verify_learning(prefix)<2e-12
    assert verify_physics(prefix)<2e-12
    cp=tmp_path/'state.paula';save_checkpoint(net,cp)
    other=load_checkpoint(cp,trusted=True).network
    body2=Arm();body2.restore(body.state())
    a=course(net,body,f,g,s,1,1,ticks=40)
    b=course(other,body2,f,g,s,1,1,ticks=40)
    assert all(np.array_equal(a[k],b[k]) for k in a)
    bad=deepcopy(a);bad['weights'][10,0,0]+=.01
    with pytest.raises(ValueError,match='Learning'):verify_learning(bad)
    bad=deepcopy(a);bad['physical_states'][10,1]+=.01
    with pytest.raises(ValueError,match='Physical'):verify_physics(bad)


def test_missing_context_cannot_choose_force_direction(tmp_path):
    cfg,g,s=configure(width=8,contextual=False)
    path=tmp_path/'config.json';path.write_text(encode(cfg))
    net,_,_,_=fresh(path,11,PredictiveReceptorNeuron)
    # Before any feedback/learning divergence, bank anatomy has identical
    # sensory weights for corresponding mixed neurons in both contexts.
    nodes=net.network.neurons
    for a,b in zip(g['mixed_0'],g['mixed_1']):
        assert [p.u_i.info for p in nodes[a].postsynaptic_points.values()]==[
            p.u_i.info for p in nodes[b].postsynaptic_points.values()]


def test_teaching_lesion_is_explicit_and_restores_class(tmp_path):
    from simulations.active_inference.experiments.context_organization_expression import teaching_path_lesion
    cfg,g,s=configure(width=8);path=tmp_path/'config.json';path.write_text(encode(cfg))
    net,_,_,_=fresh(path,11,PredictiveReceptorNeuron)
    original=PredictiveReceptorNeuron.tick
    with teaching_path_lesion(net,True):
        data=course(net,Arm(),features(),g,s,0,0,ticks=40)
    assert PredictiveReceptorNeuron.tick is original
    assert np.all(data['errors'][:,:,1]==0)
    assert np.all(data['eta']>0)
    assert verify_learning(data)==0
    with pytest.raises(RuntimeError),teaching_path_lesion(net,True):raise RuntimeError('test cleanup')
    assert PredictiveReceptorNeuron.tick is original
