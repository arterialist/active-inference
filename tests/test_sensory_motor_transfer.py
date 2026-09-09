from copy import deepcopy

import numpy as np
import pytest

from test_predictive_receptor import load
from test_phase_authorization import original
from simulations.active_inference.experiments.sensory_motor_transfer import select_media, audit, BACKGROUNDS
from simulations.active_inference.experiments.ventilation_phase_composition import condition_config
from simulations.active_inference.experiments.ventilation_regulation import CoupledHinge, OrganDelay, record
from simulations.active_inference.experiments.active_sweep_probe import PhysicalDelay
from simulations.active_inference.experiments.active_sweep_memory import reset_selected
from neuron.extensions.experimental.cascade_eligibility import CascadeEligibilityNeuron


def test_sensory_factorial_selects_actual_streams_without_dose_normalization():
    f = [dict(visual=np.full((300,96),.1), auditory=np.full((300,96),.2)),
         dict(visual=np.full((300,96),.7), auditory=np.full((300,96),.8))]
    for key, (v,a) in BACKGROUNDS.items():
        selected = select_media(f, key)
        assert selected['visual'] is f[v]['visual']
        assert selected['auditory'] is f[a]['auditory']
    f[1]['auditory'][0,0] = np.nan
    with pytest.raises(ValueError): select_media(f, 'both-changed')


def test_expanded_fixed_world_audit_and_local_reset_scope(tmp_path):
    c,g = original(); cfg,meta,_ = condition_config(c,g,'authorized')
    net = load(tmp_path,cfg,CascadeEligibilityNeuron); body = CoupledHinge(net,g)
    od = OrganDelay(np.tile([.75,.25,.5],(64,1)))
    features = dict(visual=np.full((300,96),.2), auditory=np.full((300,96),.1))
    before = deepcopy(cfg)
    z = record(net,body,PhysicalDelay(),od,features,g,meta,256)
    assert audit(z,cfg,g,meta,features) < 3e-6
    assert cfg == before
    saved = deepcopy(net); changed = reset_selected(net,g)
    ports = {(int(n),int(s)) for n,s,_ in changed}
    for nid,n in net.network.neurons.items():
        old = saved.network.neurons[nid]
        assert n.params.eta_post>0 and n.params.eta_retro>0
        assert (n.S,n.O,n.t_last_fire) == (old.S,old.O,old.t_last_fire)
        assert n.propagation_queue == old.propagation_queue
        np.testing.assert_array_equal(n.credit_states,old.credit_states)
        for sid,p in n.postsynaptic_points.items():
            assert p.u_i.info == (0. if (nid,sid) in ports else old.postsynaptic_points[sid].u_i.info)
    for field in ('drive','reg_inputs','reg_q_after','organs'):
        bad = {k:v.copy() for k,v in z.items()}; bad[field].flat[-1] += .01
        with pytest.raises((ValueError,AssertionError)): audit(bad,cfg,g,meta,features)
