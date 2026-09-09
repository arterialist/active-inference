import numpy as np
import pytest

from simulations.active_inference.experiments.ventilation_input_clamp import ClampedOrganDelay,audit


def test_clamp_changes_only_delivered_oxygen_and_preserves_real_queue():
    initial=np.tile([.75,.25,.5],(64,1))
    native=ClampedOrganDelay(initial,None);clamped=ClampedOrganDelay(initial,.75)
    for t in range(130):
        raw=np.array([.75,.25,t/130]);a=native.step(raw);b=clamped.step(raw)
        np.testing.assert_array_equal(a[:2],b[:2]);assert b[2]==.75
        np.testing.assert_array_equal(native.state(),clamped.state())
    with pytest.raises(ValueError): ClampedOrganDelay(initial,float('nan'))


def test_actual_organ_boundary_and_neural_input_corruption_are_detected(tmp_path):
    from test_predictive_receptor import load
    from simulations.active_inference.experiments.active_sweep_probe import configure,PhysicalDelay
    from simulations.active_inference.experiments.active_sweep_credit import configure as kernel
    from simulations.active_inference.components.arbitration.ventilation_feedback import append_ventilation_feedback
    from simulations.active_inference.experiments.ventilation_regulation import CoupledHinge,record
    from neuron.extensions.experimental.cascade_eligibility import CascadeEligibilityNeuron
    original,g,_=configure(11,width=8);cfg,meta=append_ventilation_feedback(kernel(original,g,'matched_cascade'),g)
    net=load(tmp_path,cfg,CascadeEligibilityNeuron);body=CoupledHinge(net,g)
    od=ClampedOrganDelay(np.tile([.75,.25,.5],(64,1)),.75)
    features=dict(visual=np.full((300,96),.2),auditory=np.full((300,96),.1))
    z=record(net,body,PhysicalDelay(),od,features,g,meta,96)
    assert audit(z,cfg,g,features,.75) is None
    for key in ('organ_before','organ_raw','organ_drive','organ_delay_final','muscles'):
        bad={k:v.copy() for k,v in z.items()};bad[key].flat[-1]+=.01
        with pytest.raises((ValueError,AssertionError)): audit(bad,cfg,g,features,.75)
    bad={k:v.copy() for k,v in z.items()}
    bad['reg_inputs'][0,list(z['reg_ids']).index(meta['oxygen']),0,0]+=.01
    with pytest.raises(AssertionError): audit(bad,cfg,g,features,.75)
