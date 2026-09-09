import numpy as np
import pytest

from simulations.active_inference.experiments.ventilation_changing_air import ChangingAirOrgans, audit, prefix
from simulations.active_inference.components.body.ventilation import VentilationOrgans


def test_world_transition_changes_only_oxygen_exchange_and_continues_state():
    ordinary = VentilationOrgans()
    changed = ChangingAirOrgans(ordinary.state(),511)
    a = ordinary.advance(-.01,.01,[.2,.1])
    b = changed.advance(-.01,.01,[.2,.1])
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(ordinary.state(),changed.state())
    a = ordinary.advance(-.01,.01,[.2,.1])
    b = changed.advance(-.01,.01,[.2,.1])
    assert b[3] == a[3]/2
    np.testing.assert_array_equal(a[[0,1,2,4,7,8]],b[[0,1,2,4,7,8]])
    np.testing.assert_array_equal(ordinary.energy.state(),changed.energy.state())
    restored = ChangingAirOrgans(changed.state(),changed.world_tick)
    np.testing.assert_array_equal(changed.advance(.01,-.01,[.1,.2]),restored.advance(.01,-.01,[.1,.2]))
    recovered = ChangingAirOrgans(changed.state(),1280)
    assert recovered.advance(-.01,.01,[.2,.1])[3] == a[3]
    with pytest.raises(ValueError): ChangingAirOrgans(changed.state(),2048).advance(0.,0.,[0.,0.])


def test_embodied_course_audit_and_exact_prefix_detect_corruption(tmp_path):
    from test_predictive_receptor import load
    from simulations.active_inference.experiments.active_sweep_probe import configure,PhysicalDelay
    from simulations.active_inference.experiments.active_sweep_credit import configure as kernel
    from simulations.active_inference.components.arbitration.ventilation_feedback import append_ventilation_feedback
    from simulations.active_inference.experiments.ventilation_regulation import CoupledHinge,record
    from simulations.active_inference.experiments.ventilation_input_clamp import ClampedOrganDelay
    from simulations.active_inference.experiments.ventilation_regulation_replay import exact_prefix
    from neuron.extensions.experimental.cascade_eligibility import CascadeEligibilityNeuron
    original,g,_ = configure(11,width=8)
    cfg,meta = append_ventilation_feedback(kernel(original,g,'matched_cascade'),g)
    features = dict(visual=np.full((300,96),.2),auditory=np.full((300,96),.1))
    net = load(tmp_path,cfg,CascadeEligibilityNeuron); body = CoupledHinge(net,g)
    body.organs = ChangingAirOrgans(body.organs.state())
    z = record(net,body,PhysicalDelay(),ClampedOrganDelay(np.tile([.75,.25,.5],(64,1)),.5),features,g,meta,640)
    z['air_fraction'] = np.asarray(body.organs.air)
    audit(z,cfg,g,features,.5)
    exact_prefix({k:v for k,v in z.items() if k!='air_fraction'},prefix(z,512))
    for k in ('air_fraction','exchange','organs','organ_drive','muscles','weights'):
        bad = {key:v.copy() for key,v in z.items()}; bad[k].flat[-1] += .01
        with pytest.raises((ValueError,AssertionError)):
            audit(bad,cfg,g,features,.5)
