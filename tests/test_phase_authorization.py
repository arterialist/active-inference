from copy import deepcopy

import numpy as np
import pytest

from test_predictive_receptor import load
from simulations.active_inference.experiments.active_sweep_probe import configure,PhysicalDelay
from simulations.active_inference.experiments.active_sweep_credit import configure as kernel
from simulations.active_inference.components.arbitration.ventilation_feedback import append_ventilation_feedback
from simulations.active_inference.experiments.ventilation_phase_composition import condition_config,install,audit_gate,CONDITIONS
from simulations.active_inference.experiments.ventilation_regulation import CoupledHinge,OrganDelay,record
from simulations.active_inference.experiments.ventilation_changing_air import ChangingAirOrgans,audit as audit_body
from neuron.extensions.experimental.cascade_eligibility import CascadeEligibilityNeuron


def original():
    c,g,_ = configure(11,width=8)
    c,_ = append_ventilation_feedback(kernel(c,g,'matched_cascade'),g)
    return c,g


def test_conditions_preserve_graph_and_acquired_state(tmp_path):
    c,g = original(); unchanged = deepcopy(c)
    variants = [condition_config(c,g,v) for v in CONDITIONS]
    assert c==unchanged
    for cfg,_,_ in variants:
        for k in ('neurons','connections','external_inputs'):
            assert cfg[k]==variants[0][0][k]
    net = load(tmp_path,c,CascadeEligibilityNeuron)
    net.set_external_input(g['cpg'][0],0,5.); net.run_tick()
    old = dict(net.network.neurons); before = deepcopy(net)
    cfg,meta,selected = variants[1]; fresh = load(tmp_path,cfg,CascadeEligibilityNeuron)
    changes = install(net,fresh,c,cfg,selected)
    changed = {(n,s) for n,s,_,_ in changes}
    assert len(net.network.neurons)==len(old)+5
    for nid,n in old.items():
        previous = before.network.neurons[nid]
        assert net.network.neurons[nid] is n
        assert (n.S,n.O,n.t_last_fire)==(previous.S,previous.O,previous.t_last_fire)
        assert n.propagation_queue==previous.propagation_queue
        np.testing.assert_array_equal(n.credit_states,previous.credit_states)
        for sid,p in previous.postsynaptic_points.items():
            if (nid,sid) not in changed: assert n.postsynaptic_points[sid].u_i.info==p.u_i.info
    assert net.network.external_inputs[g['cpg'][0],0]['info']==0.
    assert all(n.params.eta_post>0 and n.params.eta_retro>0 for n in net.network.neurons.values())


@pytest.mark.parametrize('drive',[.05,.6,1.5,4.])
def test_neural_cancellation_stays_phase_selective_above_old_gate_range(tmp_path,drive):
    c,g = original(); cfg,meta,_ = condition_config(c,g,'authorized')
    net = load(tmp_path,cfg,CascadeEligibilityNeuron); p = meta['phase_authorization']; observed = []
    for t in range(64):
        # Isolated local-input fixture, not an embodied control policy.
        net.set_external_input(p['matched_drive'],0,drive)
        for j,nid in enumerate(p['inhibitors']):
            net.set_external_input(nid,0,drive)
            net.set_external_input(nid,1,1. if t==8+24*j else 0.)
        net.run_tick(); observed.append([net.network.neurons[n].O for n in p['outputs']])
    a = np.asarray(observed)
    for j,t in enumerate((11,35)):
        assert a[t,j]>.1*min(drive,1.)
        other = np.delete(a[:,j],t)
        assert np.max(other)<1e-3
        assert abs(a[t,j]-3*.99**2*min(drive,1.))<1e-3


def test_embodied_record_and_gate_audit_detect_corruption(tmp_path):
    c,g = original();cfg,meta,_ = condition_config(c,g,'authorized')
    net = load(tmp_path,cfg,CascadeEligibilityNeuron);body = CoupledHinge(net,g)
    body.organs = ChangingAirOrgans(body.organs.state())
    od = OrganDelay(np.tile([.75,.25,.5],(64,1)))
    features = dict(visual=np.full((300,96),.2),auditory=np.full((300,96),.1))
    z = record(net,body,PhysicalDelay(),od,features,g,meta,640)
    z['air_fraction'] = np.asarray(body.organs.air)
    audit_body(z,cfg,g,features,None); assert audit_gate(z,cfg,meta)<3e-6
    for k in ('reg_inputs','reg_scheduled','cells'):
        bad = {key:v.copy() for key,v in z.items()}
        if k=='cells': bad[k][20,-1,0]+=.01
        else: bad[k][20,list(z['reg_ids']).index(meta['phase_authorization']['outputs'][-1]),0]+=.01
        with pytest.raises((ValueError,AssertionError)): audit_gate(bad,cfg,meta)
