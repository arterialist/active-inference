from copy import deepcopy
import numpy as np
import pytest
from simulations.active_inference.experiments.evidence_consumer import consumer_config,record
from simulations.active_inference.experiments.composition_probe import encode,snapshot
from simulations.active_inference.experiments.multimodal_pairing_probe import fresh
from neuron.extensions.graded import GradedNeuron


def test_constructor_changes_only_declared_time_constant():
    slow=consumer_config('evidence_slow');fast=consumer_config('evidence_fast')
    assert slow['connections']==fast['connections'] and slow['synaptic_points']==fast['synaptic_points']
    assert all(n['params']['eta_post']>0 and n['params']['eta_retro']>0 for n in slow['neurons'])
    assert {e['target_neuron'] for e in slow['external_inputs']}==set(range(1,33))
    assert len(slow['neurons'])==65


def test_recorded_source_ports_and_observer_are_exact(tmp_path):
    p=tmp_path/'config.json';p.write_text(encode(consumer_config('evidence_slow')))
    net,*_=fresh(p,11,GradedNeuron);other=deepcopy(net)
    tape=np.zeros((64,192),bool);tape[0,:6]=True
    data=record(net,tape)
    for t in range(64):
        if t==1:
            for i in range(6):other.set_external_input(1,i,1.)
        other.run_tick()
    assert encode(snapshot(other))==encode(snapshot(net))
    assert data['incoming'][1,:6].tolist()==[1.]*6
    assert not data['incoming'][0].any()
    assert data['states'][1,0,1]>0
    assert not data['states'][:3,33:,1].any()
    assert data['states'][3,33,1]>0
    with pytest.raises(ValueError):record(net,np.zeros((3,32)))


def test_short_common_mode_does_not_select_a_coordinate(tmp_path):
    p=tmp_path/'config.json';p.write_text(encode(consumer_config('evidence_slow')))
    net,*_=fresh(p,11,GradedNeuron)
    tape=np.zeros((64,192),bool);tape[[0,8,16,24],:]=True
    data=record(net,tape)
    assert data['states'][:,33:,1].sum()==0
