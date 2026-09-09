import numpy as np
import pytest

from simulations.active_inference.components.body.ventilation import (
    VentilationParameters, VentilationOrgans, VentilatedHinge)
from simulations.active_inference.components.body.loaded_hinge import LoadedHinge


def test_chamber_is_bounded_and_only_inspiration_replenishes():
    p = VentilationParameters()
    assert p.volume(-100) == 1 and p.volume(100) == 11
    o = VentilationOrgans()
    row = o.advance(-.01, .01, [1, 0])
    assert row[2] == pytest.approx(10*np.tanh(.2))
    assert row[3] == pytest.approx(.21*row[2])
    row = o.advance(.01, -.01, [0, 1])
    assert row[2] == row[3] == 0


def test_conservation_survives_overflow_and_starvation():
    o = VentilationOrgans(); initial = o.state()
    for t in range(400):
        a, b = (-.1, .1) if t < 100 else (0, 0)
        o.advance(a, b, [3., 2.], dt=.01)
        s = o.state()
        assert s[0]+s[3]-s[4]+s[5] == pytest.approx(initial[0]+s[2])
        assert s[6]+s[7]+s[9]-s[10]+s[11] == pytest.approx(initial[6]+initial[7]+s[12])
    assert o.oxygen_spill_ml > 0 and o.oxygen_unmet_ml > 0 and o.energy.unmet_j > 0


def test_silence_and_cocontraction_cannot_bypass_costs():
    still, active = VentilationOrgans(), VentilationOrgans()
    for _ in range(200):
        still.advance(0, 0, [0, 0]); active.advance(0, 0, [2, 2])
    assert still.oxygen_unmet_ml == pytest.approx(.3*.8-.15)
    assert active.oxygen_unmet_ml == still.oxygen_unmet_ml
    assert active.energy.demand_j-still.energy.demand_j == pytest.approx(.1*8*.8)


def test_positive_work_is_not_net_work_and_cut_still_costs_activation():
    a, b = VentilationOrgans(), VentilationOrgans()
    x = a.advance(0, .01, [2, 2]); y = b.advance(0, .01, [2, 2], transmission=0, exchange=0)
    assert x[7] == pytest.approx(.2*2*.01)
    assert y[7] == y[3] == 0 and y[8] == x[8] > 0
    assert y[2] == x[2] > 0  # The exchange lesion does not erase actual volume motion.


def test_body_preserves_mechanics_and_all_resource_state_on_reload():
    a, plain = VentilatedHinge(), LoadedHinge(.8)
    for t in range(150):
        m = np.array([1., .2]) if t % 40 < 20 else np.array([.2, 1.])
        assert a.step_muscles(m) == plain.step(float(m[0]-m[1]))
        np.testing.assert_array_equal(a.state(), plain.state())
    b = VentilatedHinge()
    b.restore(a.state(), next_gate=a.next_gate, crossings=a.crossings)
    b.organs.restore(a.organs.state())
    for _ in range(50):
        a.step_muscles([.5, .2]); b.step_muscles([.5, .2])
        np.testing.assert_array_equal(a.state(), b.state())
        np.testing.assert_array_equal(a.organs.state(), b.organs.state())
        np.testing.assert_array_equal(a.last_exchange, b.last_exchange)


@pytest.mark.parametrize('args', [{'oxygen_fraction':2}, {'chamber_swing_ml':6}, {'oxygen_demand_ml_s':0}])
def test_invalid_parameters(args):
    with pytest.raises(ValueError): VentilationParameters(**args)


def test_reject_invalid_input_without_stepping_body():
    b = VentilatedHinge(); initial = b.state()
    for m in ([1,-1], [float('nan'),0], [1]):
        with pytest.raises(ValueError): b.step_muscles(m)
        np.testing.assert_array_equal(initial, b.state())
    with pytest.raises(ValueError): b.organs.restore(np.zeros(2))
    with pytest.raises(TypeError): b.step(1.)
    np.testing.assert_array_equal(initial,b.state())


def test_independent_audit_detects_hidden_resource_and_actuator_changes():
    from simulations.active_inference.experiments.ventilation_screen import replay, audit
    from simulations.active_inference.experiments.context_organization import FIELDS
    b = LoadedHinge(.8)
    z = dict(body_initial=b.state(),gate_initial=np.array([0,1]),
             physical_parameters=np.array([.8,.15,.008,1.]),neuron_ids=np.array([1,2]))
    cells=np.zeros((160,2,len(FIELDS))); body=[]; states=[]
    for t in range(160):
        m=[1.,.1] if t%60<30 else [.1,1.]
        cells[t,:,FIELDS.index('O')]=m
        command=m[0]-m[1]; force,_=b.step(command)
        body.append([b.data.time,b.data.qpos[0],b.data.qvel[0],command,force]); states.append(b.state())
    z.update(cells=cells,body=np.asarray(body),physical_states=np.asarray(states))
    data=replay(z,{'muscle':[1,2]})
    assert audit(data)==0
    for key,col in [('organs',4),('exchange',3),('muscles',0),('body',3),('gate',0)]:
        bad={k:v.copy() for k,v in data.items()}; bad[key][80,col]+=1. if key=='gate' else .01
        with pytest.raises(AssertionError): audit(bad)
    for key in ('muscles','organs','physical_states'):
        bad={k:v.copy() for k,v in data.items()}; bad[key]=bad[key][:0]
        with pytest.raises(ValueError): audit(bad)
    bad={k:v.copy() for k,v in data.items()}; bad['organ_initial'][0]+=.01
    with pytest.raises(AssertionError): audit(bad)


def test_final_positive_reserve_does_not_erase_earlier_failure():
    o=VentilationOrgans()
    for _ in range(200): o.advance(0,0,[0,0])
    debt=o.oxygen_unmet_ml
    assert debt>0
    o.advance(-.1,.1,[1,0])
    assert o.oxygen_ml==o.params.oxygen_capacity_ml
    assert o.oxygen_unmet_ml==debt
