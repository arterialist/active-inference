from copy import deepcopy

import numpy as np

from test_predictive_receptor import load
from simulations.active_inference.experiments.active_sweep_probe import configure
from simulations.active_inference.components.arbitration.ventilation_feedback import (
    append_ventilation_feedback,install_ventilation_feedback,MODES)
from simulations.active_inference.core.runtime_checkpoint import check_buffer_aliases
from neuron.extensions.experimental.cascade_eligibility import CascadeEligibilityNeuron


def test_modes_keep_matching_graphs_and_preserve_old_runtime(tmp_path):
    original,g,_=configure(11,width=8); untouched=deepcopy(original)
    variants=[append_ventilation_feedback(original,g,mode=m)[0] for m in MODES]
    assert original==untouched
    for key in ('neurons','connections','external_inputs'):
        for cfg in variants[1:]: assert variants[0][key]==cfg[key]
    net=load(tmp_path,original,CascadeEligibilityNeuron)
    for _ in range(12):
        net.set_external_input(g['context'][0],0,1.);net.run_tick()
    before=deepcopy(net);old=dict(net.network.neurons)
    cfg=variants[0];fresh=load(tmp_path,cfg,CascadeEligibilityNeuron)
    install_ventilation_feedback(net,fresh,original,cfg);check_buffer_aliases(net)
    assert net.current_tick==before.current_tick
    assert len(net.network.neurons)==len(old)+7
    for nid,n in old.items():
        assert net.network.neurons[nid] is n
        previous=before.network.neurons[nid]
        assert n.S==previous.S and n.O==previous.O and n.propagation_queue==previous.propagation_queue
        np.testing.assert_array_equal(n.prediction_context,previous.prediction_context)
        np.testing.assert_array_equal(n.credit_states,previous.credit_states)
        for sid,p in previous.postsynaptic_points.items():
            assert n.postsynaptic_points[sid].u_i.info==p.u_i.info
        assert n.upper_t_ref_bound==n.params.c*n.params.num_inputs
    assert all(n.params.eta_post>0 and n.params.eta_retro>0 for n in net.network.neurons.values())


def test_phase_relay_requires_phase_and_reports_one_tick_dendritic_delay(tmp_path):
    original,g,_=configure(11,width=8);cfg,m=append_ventilation_feedback(original,g)
    net=load(tmp_path,cfg,CascadeEligibilityNeuron)
    relay=net.network.neurons[m['relays'][0]];out=[]
    for t in range(16):
        relay.input_buffer[0,0]=1. if t==8 else 0.
        relay.input_buffer[1,0]=.6
        relay.tick({},t);out.append(relay.O)
    assert np.flatnonzero(out).tolist()==[9]
    assert abs(out[9]-3*(.99*1.6-1))<2e-5
    assert relay.params.eta_post>0 and relay.bounded_updates>0


def test_observer_does_not_insert_leaf_cache_keys_and_replay_is_exact(tmp_path):
    from simulations.active_inference.experiments import context_organization as base
    from simulations.active_inference.experiments.active_sweep_credit import configure as kernel,record_credit
    from simulations.active_inference.experiments.active_sweep_probe import PhysicalDelay
    from simulations.active_inference.experiments.ventilation_regulation import (
        CoupledHinge,ResourceInputs,OrganDelay,organ_afferents,record)
    from simulations.active_inference.experiments.ventilation_regulation_replay import restore,exact_prefix
    from simulations.active_inference.experiments.ventilation_regulation_audit import audit
    from neuron.network import TravelingSignal
    original,g,_=configure(11,width=8);original=kernel(original,g,'matched_cascade')
    cfg,m=append_ventilation_feedback(original,g)
    net=load(tmp_path,cfg,CascadeEligibilityNeuron);body=CoupledHinge(net,g)
    delay=PhysicalDelay();od=OrganDelay(np.tile(organ_afferents(body),(64,1)))
    leaf,term=next((n.id,tid) for n in net.network.neurons.values() for tid in n.presynaptic_points
                  if (n.id,tid) not in net.network.connection_cache)
    net.presynaptic_wheel[0].append(TravelingSignal((leaf,term,.123),0))
    keys=set(net.network.connection_cache)
    features=dict(visual=np.full((300,96),.2),auditory=np.full((300,96),.1))
    base.save_checkpoint(net,tmp_path/'start.paula',sources=[])
    np.savez_compressed(tmp_path/'body.npz',state=body.state(),delay=delay.state(),
                        organ=body.organs.state(),organ_delay=od.state(),gate=[0,1])
    data=record(net,body,delay,od,features,g,m,256)
    assert set(net.network.connection_cache)==keys
    check_buffer_aliases(net);assert audit(data,cfg,g,features)<3e-6
    copy,b,d,o=restore(tmp_path/'start.paula',tmp_path/'body.npz',g)
    prefix=record(copy,b,d,o,features,g,m,64);exact_prefix(data,prefix)
    plain,b,d,o=restore(tmp_path/'start.paula',tmp_path/'body.npz',g)
    reference=record_credit(ResourceInputs(plain,b,m,o),b,d,features,g,256)
    for k,v in reference.items(): np.testing.assert_array_equal(data[k],v,err_msg=k)
    for key in ('reg_inputs','reg_q_after','reg_scheduled','organs','organ_drive','reg_returns','cpg_inputs'):
        bad={k:v.copy() for k,v in data.items()};bad[key].flat[-1]+=.01
        try: audit(bad,cfg,g,features)
        except (ValueError,AssertionError): pass
        else: raise AssertionError('Audit accepted corruption: '+key)


def test_install_does_not_reinject_consumed_birth_pulse(tmp_path):
    original,g,_=configure(11,width=8)
    cfg,_=append_ventilation_feedback(original,g)
    net=load(tmp_path,original,CascadeEligibilityNeuron)
    birth=(g['cpg'][0],0)
    net.set_external_input(*birth,5.);net.run_tick()
    topo=net.network;row=topo._ext_vec['row_of'][birth]
    assert topo.external_inputs[birth]['info']==5.
    assert topo._ext_vec['info'][row]==0.
    baseline=deepcopy(net)
    fresh=load(tmp_path,cfg,CascadeEligibilityNeuron)
    install_ventilation_feedback(net,fresh,original,cfg)
    assert topo.external_inputs[birth]['info']==0.
    for _ in range(96):
        net.run_tick();baseline.run_tick()
        for nid in g['cpg']:
            a,b=topo.neurons[nid],baseline.network.neurons[nid]
            assert (a.S,a.O,a.t_last_fire)==(b.S,b.O,b.t_last_fire)


def test_external_sync_rejects_pending_or_mismatched_cache_without_mutation(tmp_path):
    import pytest
    from simulations.active_inference.core.external_input_state import synchronize_quiescent_external_inputs
    original,g,_=configure(11,width=8)
    for field in ('info','plast','mod'):
        net=load(tmp_path,original,CascadeEligibilityNeuron);topo=net.network
        vec=topo._ensure_ext_vectorized();vec[field].flat[0]=.125
        before=deepcopy(topo.external_inputs)
        with pytest.raises(ValueError,match='pending'): synchronize_quiescent_external_inputs(topo)
        for k in before:
            for f in before[k]: np.testing.assert_array_equal(before[k][f],topo.external_inputs[k][f])
        assert vec[field].flat[0]==.125
    vec['info'].fill(0);vec['plast'].fill(0);vec['mod'].fill(0)
    vec['row_of']={}
    with pytest.raises(ValueError,match='interface'): synchronize_quiescent_external_inputs(topo)
