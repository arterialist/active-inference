from copy import deepcopy

import numpy as np
import pytest

from test_predictive_receptor import fixture_config, load
from simulations.active_inference.components.body.research_rower import ResearchRower
from simulations.active_inference.components.motor.proprioceptive_rower import append_proprioceptive_rower
from simulations.active_inference.experiments.proprioceptive_loop_probe import PhysicalLoop, record_loop, audit_physical
from simulations.active_inference.experiments.predictive_bridge_probe import audit_record
from simulations.active_inference.experiments.composition_probe import encode, snapshot


def test_builder_keeps_original_and_positive_learning(tmp_path):
    old = fixture_config(); before = deepcopy(old)
    old['metadata']['predictive_bridge'] = {'original': True}
    before = deepcopy(old)
    cfg,motor,bridge,_ = append_proprioceptive_rower(old,[1,2])
    assert old == before
    assert cfg['metadata']['predictive_bridge'] == {'original': True}
    assert len(cfg['neurons']) == len(old['neurons'])+32
    assert len(set(n['id'] for n in cfg['neurons'])) == len(cfg['neurons'])
    assert len(cfg['metadata']['proprioceptive_rower']['ascending']) == 4
    net = load(tmp_path,cfg)
    for n in net.network.neurons.values():
        assert n.params.eta_post > 0 and n.params.eta_retro > 0
        assert n.lower_t_ref_bound <= n.upper_t_ref_bound
        assert len(n.postsynaptic_points) == n.params.num_inputs
    assert set(motor['muscles']).isdisjoint(bridge['prediction'])


def test_compiled_joint_units_and_opponent_sensor():
    b = ResearchRower()
    # Historical XML omits compiler angle: -1.8..1.8 means DEGREES.
    np.testing.assert_allclose(b.model.jnt_range[b.joint_ids],np.tile(np.deg2rad([-1.8,1.8]),(2,1)))
    b.data.qpos[b.positions] = np.deg2rad([.9,-.45])
    np.testing.assert_allclose(b.sense(),[.5,0,0,.25])


def test_radian_variant_is_explicit_and_replays(tmp_path):
    from simulations.active_inference.components.body.radian_research_rower import RadianResearchRower
    from simulations.active_inference.experiments.proprioceptive_body_comparison import audit_body_replay
    b=RadianResearchRower()
    np.testing.assert_allclose(b.model.jnt_range[b.joint_ids],np.tile([-1.8,1.8],(2,1)))
    cfg,motor,bridge,_=append_proprioceptive_rower(fixture_config(),[1,2])
    data=record_loop(load(tmp_path,cfg),b,motor,bridge,128)
    assert audit_body_replay(data,RadianResearchRower(),gain=8.) == 0
    assert np.any((data['joint_input']>0)&(data['joint_input']<1))
    with pytest.raises(AssertionError): audit_body_replay(data,ResearchRower(),gain=8.)


def test_physical_checkpoint_exact_with_active_controls():
    b = ResearchRower()
    for t in range(40): b.step([.1*(t%3),0,.1,0])
    restored = ResearchRower(); restored.restore(b.state())
    for t in range(40):
        u=[0,.1,.05*(t%2),0]
        b.step(u); restored.step(u)
        np.testing.assert_array_equal(b.state(),restored.state())
        np.testing.assert_array_equal(b.sense(),restored.sense())


@pytest.mark.parametrize('gain',[0.,8.])
def test_recorder_is_passive_and_both_audits_are_exact(tmp_path,gain):
    cfg,motor,bridge,_ = append_proprioceptive_rower(fixture_config(),[1,2])
    net=load(tmp_path,cfg); control=deepcopy(net)
    body=ResearchRower(); other=PhysicalLoop(control,ResearchRower(),motor,gain=gain)
    data=record_loop(net,body,motor,bridge,128,gain=gain)
    for _ in range(128): other.run_tick()
    assert encode(snapshot(net)) == encode(snapshot(control))
    np.testing.assert_array_equal(body.state(),other.body.state())
    assert audit_record(data,cfg,bridge) < 2e-12
    assert audit_physical(data,gain=gain) == 0
    assert np.max(data['muscle_state']) > 0
    if gain == 0:
        assert not data['joint_input'].any()
        assert not data['actuator_ctrl'].any()
    else:
        assert data['joint_input'].any()
    altered={k:v.copy() for k,v in data.items()}; altered['joint_input'][70,0]+=.01
    with pytest.raises(AssertionError): audit_physical(altered,gain=gain)


def test_whole_loop_checkpoint_continuation(tmp_path):
    import random
    from simulations.active_inference.core.runtime_checkpoint import save_checkpoint,load_checkpoint
    from simulations.active_inference.experiments.proprioceptive_learning_probe import BranchNetwork
    cfg,motor,bridge,_=append_proprioceptive_rower(fixture_config(),[1,2])
    net=load(tmp_path,cfg);body=ResearchRower()
    record_loop(net,body,motor,bridge,96)
    file=tmp_path/'moving.neural-checkpoint';save_checkpoint(net,file)
    branch=load_checkpoint(file,trusted=True)
    restored=ResearchRower();restored.restore(body.state())
    a=record_loop(net,body,motor,bridge,64)
    ambient=random.getstate()
    b=record_loop(BranchNetwork(branch),restored,motor,bridge,64)
    assert random.getstate()==ambient
    for key in a: np.testing.assert_array_equal(a[key],b[key])
    assert encode(snapshot(net))==encode(snapshot(branch.network))


def test_resumed_float32_muscle_audit_keeps_exactness(tmp_path):
    from simulations.active_inference.components.body.radian_research_rower import RadianResearchRower
    from simulations.active_inference.experiments.proprioceptive_learning_audit import verify_physics
    cfg,motor,bridge,_=append_proprioceptive_rower(fixture_config(),[1,2])
    net=load(tmp_path,cfg);body=RadianResearchRower()
    record_loop(net,body,motor,bridge,192,gain=.08)
    data=record_loop(net,body,motor,bridge,64,gain=.08)
    assert data['muscle_state'].dtype==np.float32
    assert verify_physics(data,.08)==0
    data['actuator_ctrl'][0,0]=np.nextafter(data['actuator_ctrl'][0,0],np.inf)
    with pytest.raises(AssertionError):verify_physics(data,.08)


def test_conditional_capacity_envelope_contains_bounded_native_arithmetic():
    from simulations.active_inference.experiments.proprioceptive_capacity_audit import envelope
    rng=np.random.default_rng(11)
    a=rng.uniform(0,2,(256,3,8)).astype(np.float32);a[80:180]=0
    lam=np.array([1.,4.,64.]);decay=np.array([.99,.95,1.]);cap=np.ones(3)
    upper=envelope(a,lam,decay,cap)
    s=np.array([0.,300.,999.],dtype=np.float32)
    for t in range(1,len(a)):
        q=rng.uniform(0,1,(3,8)).astype(np.float32)
        current=np.zeros(3,dtype=np.float32)
        for j in rng.permutation(8):current+=a[t-1,:,j]*q[:,j]*decay.astype(np.float32)
        s+=(1/lam).astype(np.float32)*(-s+current)
        assert np.all(s<=upper[t])
    with pytest.raises(ValueError):envelope(-a,lam,decay,cap)
