import numpy as np
import pytest

from simulations.drosophila.orn_train import pulse_course
from simulations.drosophila.orn_train_analysis import audit_sources,audit_resources
from simulations.drosophila.electrophysiology import CurrentElectrode
from simulations.drosophila.paula import Neuron
from neuron.neuron import NeuronParameters,PresynapticPoint,PresynapticOutputVector,PostsynapticPoint,PostsynapticInputVector
from neuron.extensions.experimental.release_depression import DepressingReleaseNeuron


def test_population_pulses_are_staggered_and_recovery_is_not_omitted():
    cmd,epochs=pulse_course(42)
    np.testing.assert_array_equal(np.count_nonzero(cmd[200:1200],axis=0),10)
    np.testing.assert_array_equal(np.count_nonzero(cmd[2200:3200],axis=0),50)
    assert not cmd[:200].any() and not cmd[1200:2200].any() and not cmd[3200:].any()
    assert cmd[200,0]==40 and cmd[297,41]==40
    assert len(epochs)==5 and epochs[-1]["stop"]==4200
    with pytest.raises(ValueError):pulse_course(0)


def test_source_and_resource_audits_reject_forged_state():
    cell=DepressingReleaseNeuron(2,NeuronParameters(num_inputs=1),log_level="CRITICAL")
    cell.postsynaptic_points[0]=PostsynapticPoint(PostsynapticInputVector(1.,0.,np.zeros(2)))
    cell.distances[0]=0
    cell.presynaptic_points[0]=PresynapticPoint(PresynapticOutputVector(1.,np.zeros(2)),1.)
    cell.configure_release_depression([0],.22,893.)
    cmd=np.zeros((400,1));cmd[::20]=40
    before=np.zeros_like(cmd);soma=np.zeros((400,1,3));intrinsic=np.zeros((400,1,4))
    resource={k:np.zeros_like(cmd) for k in ("available","used","native","effective")}
    with CurrentElectrode(cell,cmd[:,0]) as instrument:
        for t in range(400):
            before[t,0]=cell.S
            cell.tick({},t)
            soma[t,0]=[cell.S,cell.O,cell.F_avg]
            intrinsic[t,0]=[cell.t_ref,cell.r,cell.b,cell.params.lambda_param]
            for key,attr in (("available","release_available"),("used","release_used_fraction"),
                             ("native","release_native_amplitude"),("effective","release_effective_amplitude")):
                resource[key][t]=getattr(cell,attr)
    args=dict(first_tick=0,last_fire=np.array([-np.inf]),previous_S=np.zeros(1))
    result=audit_sources(before,instrument.native_current[:,None],cmd,soma,intrinsic,**args)
    assert result[2:]==(0,0.,0)
    audit_resources(*resource.values(),soma[:,:,1],np.array([0,1]),.22,np.ones(1))
    false=soma.copy();false[14,0,0]+=.01
    with pytest.raises(AssertionError):
        audit_sources(before,instrument.native_current[:,None],cmd,false,intrinsic,**args)
    false=soma.copy();false[1,0,1]=1
    with pytest.raises(AssertionError,match="refractory"):
        audit_sources(before,instrument.native_current[:,None],cmd,false,intrinsic,**args)
    resource["available"][60,0]+=.01
    with pytest.raises(AssertionError):
        audit_resources(*resource.values(),soma[:,:,1],np.array([0,1]),.22,np.ones(1))
