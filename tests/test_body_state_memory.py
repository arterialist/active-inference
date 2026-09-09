import numpy as np
import pytest

from simulations.active_inference.experiments import context_organization as base
from simulations.active_inference.experiments.body_state_memory_analysis import mechanics,intervals
from simulations.active_inference.experiments.body_state_memory_probe import check_prefix


def test_actual_mechanics_includes_damping_and_discretization():
    arm=base.Arm();arm.data.qpos[0]=.5;arm.data.qvel[0]=-.8
    initial=arm.state();rows=[];states=[]
    for t in range(32):
        command=.7*np.sin(t/5);load=.2
        arm.step(command,load);states.append(arm.state())
        rows.append([arm.data.time,arm.data.qpos[0],arm.data.qvel[0],command,load])
    data=dict(body=np.array(rows),physical_states=np.array(states),body_initial=initial)
    out=mechanics(data)
    assert np.any(out[:,2]) and np.all(out[:,-1]>=0)
    np.testing.assert_allclose(out[:,0]+out[:,1]+out[:,2],out[:,3],atol=1e-15)
    data['body'][0,2]+=.01
    with pytest.raises(ValueError,match='balance'):mechanics(data)


def test_intervals_preserve_disjoint_events_and_empty_case():
    assert intervals([0,1,1,0,1])==[[1,3],[4,5]]
    assert intervals([])==[]


def test_prefix_uses_declared_boundary_not_new_endpoint(tmp_path):
    original=dict(neuron_ids=np.array([8]),body=np.zeros((4,5)),delay_final=np.zeros((2,4)),
        retrograde_offsets=np.array([0,1,1,2,2]),retrograde_events=np.zeros((2,7)))
    p=tmp_path/'original.npz';np.savez(p,**original)
    new=dict(neuron_ids=np.array([8]),body=np.zeros((8,5)),raw_afferents=np.zeros((8,4)),
        delay_initial=np.zeros((2,4)),retrograde_offsets=np.array([0,1,1,2,2,3,3,3,3]),
        retrograde_events=np.zeros((3,7)))
    new['raw_afferents'][7]=1.
    with np.load(p) as z:
        check_prefix(new,z,4)
        new['body'][3,1]=1.
        with pytest.raises(ValueError,match='body'):check_prefix(new,z,4)
