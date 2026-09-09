from copy import deepcopy
import json
import random
import sys

import numpy as np
import pytest

from simulations.active_inference.experiments.eligibility_association_probe import config,dynamic_snapshot
from neuron.neuron import Neuron
from neuron.extensions.experimental.eligibility_trace import EligibilityTraceNeuron
from simulations.active_inference.experiments.multimodal_pairing_probe import fresh
from simulations.active_inference.experiments.composition_probe import encode
from simulations.active_inference.experiments.threshold_release_probe import ThresholdObserver,ReleaseDriver
from simulations.active_inference.experiments.threshold_release_audit import check_thresholds,check_release
from simulations.active_inference.experiments.population_hierarchy import cellular


def networks(tmp_path):
    cfg,groups=config();p=tmp_path/'config.json';p.write_text(encode(cfg))
    net,_,_,_=fresh(p,11,EligibilityTraceNeuron)
    return net,deepcopy(net),groups


def test_threshold_observer_reads_pre_reset_and_preserves_complete_state(tmp_path):
    a,b,g=networks(tmp_path);nid=g['vision'][0]
    observer=ThresholdObserver(b.network.neurons[nid]);old=Neuron.tick
    for t in range(8):
        for net in (a,b):net.set_external_input(nid,0,1.)
        a.run_tick()
        with observer.observe():b.run_tick()
        assert dynamic_snapshot(a)==dynamic_snapshot(b)
    assert Neuron.tick is old and sys.gettrace() is None
    assert len(observer.rows)==8
    for row in observer.rows:
        assert row['will_fire']==(row['pre_S']>=row['active_threshold'] and (row['since_last'] is None or row['since_last']>=row['c']))
    assert any(row['will_fire'] and row['pre_S']>0 for row in observer.rows)


def test_release_block_changes_only_wheel_not_soma_retro_or_rng(tmp_path):
    a,b,g=networks(tmp_path);nid=g['vision'][0]
    for net in (a,b):
        for target in g['vision'][:2]:net.set_external_input(target,0,1.)
        net.run_tick()  # Configured one-tick dendritic transit precedes firing.
    rng=random.getstate();sham=ReleaseDriver(a,nid,1,False);sham.do_tick();after=random.getstate()
    random.setstate(rng);block=ReleaseDriver(b,nid,1,True);result=block.do_tick()
    assert random.getstate()==after
    assert result['traveling_signals']==sum(map(len,b.presynaptic_wheel))+sum(map(len,b.retrograde_wheel))
    assert block.evidence['removed'] and sham.evidence['found']==block.evidence['found']
    original=json.loads(dynamic_snapshot(a));expected=deepcopy(original)
    for row in block.evidence['removed']:expected['presynaptic_wheel'].remove(row)
    assert json.loads(dynamic_snapshot(b))==expected
    assert b.network.neurons[nid].O>0
    assert original['presynaptic_wheel'] and expected['presynaptic_wheel']
    check_release(block.evidence,nid,1,True)
    corrupt=deepcopy(block.evidence);corrupt['after']['neurons'][str(nid)]['S']+=.1
    with pytest.raises(ValueError,match='Undeclared release'):check_release(corrupt,nid,1,True)


def test_observer_restores_method_and_existing_trace_on_error(tmp_path):
    a,_,g=networks(tmp_path);original=Neuron.tick
    sentinel=lambda frame,event,arg:None
    sys.settrace(sentinel)
    try:
        with pytest.raises(RuntimeError,match='existing debugger'):
            with ThresholdObserver(a.network.neurons[g['vision'][0]]).observe():a.run_tick()
        assert sys.gettrace() is sentinel and Neuron.tick is original
    finally:sys.settrace(None)


def test_independent_threshold_audit_checks_integration_and_cooldown(tmp_path):
    net,_,g=networks(tmp_path);nid=g['vision'][0];initial=json.loads(dynamic_snapshot(net))
    observer=ThresholdObserver(net.network.neurons[nid]);states=[]
    with observer.observe():
        for t in range(10):
            net.set_external_input(nid,0,1.);net.run_tick();states.append(cellular(list(net.network.neurons.values())))
    states=np.array(states);trial=dict(start=0)
    margins=check_thresholds(observer.rows,states,nid,initial,trial)
    assert len(margins)==10
    altered=deepcopy(observer.rows);altered[2]['pre_S']+=.01
    with pytest.raises(ValueError,match='integration'):check_thresholds(altered,states,nid,initial,trial)
    altered=deepcopy(observer.rows);altered[2]['active_threshold']+=.01
    with pytest.raises(ValueError,match='active threshold'):check_thresholds(altered,states,nid,initial,trial)
    altered=deepcopy(observer.rows);altered[2]['dt']=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):check_thresholds(altered,states,nid,initial,trial)


def test_empty_current_retains_native_weak_scalar_precision():
    row=dict(tick=8,neuron=1,old_S=0.4189007878303528,pre_S=0.3141756057739258,
        I_t=0.,dS=-0.1047251969575882,active_threshold=.65,r=.65,b=.9,c=3,
        dt=1.,lambda_param=4.,since_last=7.,will_fire=False,S_dtype='float32',I_dtype='float64')
    cells=np.zeros((1,1,8));cells[0,0,0]=row['pre_S'];cells[0,0,5]=.65
    initial={'neurons':{'1':{'S':row['old_S'],'t_last_fire':1}}}
    assert check_thresholds([row],cells,1,initial,{'start':8})[0]==pytest.approx(row['pre_S']-.65)
