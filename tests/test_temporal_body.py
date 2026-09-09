from copy import deepcopy

import numpy as np
import pytest

from test_predictive_receptor import fixture_config,load
from simulations.active_inference.components.learning.temporal_basis import append_temporal_basis
from simulations.active_inference.components.learning.predictive_bridge import append_predictive_bridge
from simulations.active_inference.components.motor.proprioceptive_rower import append_proprioceptive_rower
from simulations.active_inference.components.body.radian_research_rower import RadianResearchRower
from simulations.active_inference.experiments.temporal_body_probe import record_history,audit_history
from simulations.active_inference.experiments.proprioceptive_loop_probe import PhysicalLoop
from simulations.active_inference.experiments.proprioceptive_learning_audit import verify_physics
from simulations.active_inference.experiments.predictive_bridge_probe import audit_record
from simulations.active_inference.experiments.composition_probe import encode,snapshot


def test_matched_size_wiring_weights_and_only_timescale_difference():
    original=fixture_config();before=deepcopy(original)
    a,ids=append_temporal_basis(original,[1,2],mode='short')
    b,other=append_temporal_basis(original,[1,2],mode='multiscale')
    assert original==before and ids==other and len(ids)==32
    for key in ('synaptic_points','connections','external_inputs'):assert a[key]==b[key]
    for x,y in zip(a['neurons'],b['neurons']):
        x=deepcopy(x);y=deepcopy(y)
        x['params'].pop('lambda_param');y['params'].pop('lambda_param')
        assert x==y
    assert len(set(n['id'] for n in a['neurons']))==len(a['neurons'])


def test_impulse_history_survives_after_fast_bank_fades(tmp_path):
    traces={}
    for mode in ('short','multiscale'):
        cfg,ids=append_temporal_basis(fixture_config(),[1],mode=mode)
        net=load(tmp_path,cfg);rows=[]
        for t in range(192):
            if t==0:net.set_external_input(1,0,1.)
            net.run_tick();rows.append([net.network.neurons[n].O for n in ids])
        traces[mode]=np.array(rows)
        for n in ids:
            cell=net.network.neurons[n]
            assert cell.params.eta_post>0 and cell.params.eta_retro>0
            j=ids.index(n);first=np.flatnonzero(traces[mode][:,j]>0)[0]
            assert first==2+cell.distances[0]
    assert traces['short'][128:].max()<1e-20
    assert traces['multiscale'][128:].max()>.001


def test_effect_summary_keeps_unfavorable_intervals_and_channels():
    from simulations.active_inference.experiments.temporal_body_audit import describe
    effect=np.ones((512,4));effect[120:180]=-1;effect[:,0]-=2
    result=describe(effect)
    assert result['negative_mean_intervals']==[[120,179]]
    assert result['negative_ticks_per_channel'].tolist()==[512,60,60,60]
    assert result['last164_mean']>0


@pytest.mark.parametrize('mode',['short','multiscale'])
def test_observer_passivity_and_full_tick_equations(tmp_path,mode):
    cfg,motor,_,_=append_proprioceptive_rower(fixture_config(),[1,2])
    cfg,basis=append_temporal_basis(cfg,motor['cpg']+motor['muscles'],mode=mode)
    cfg,bridge,_,_=append_predictive_bridge(cfg,basis,motor['joint_position'],fanin=len(basis),consumers=8)
    net=load(tmp_path,cfg);control=deepcopy(net)
    body=RadianResearchRower();other=PhysicalLoop(control,RadianResearchRower(),motor,gain=.08)
    data=record_history(net,body,motor,bridge,basis,192)
    for _ in range(192):other.run_tick()
    assert encode(snapshot(net))==encode(snapshot(control))
    np.testing.assert_array_equal(body.state(),other.body.state())
    assert audit_history(data,cfg,basis)<2e-12
    assert audit_record(data,cfg,bridge)<2e-12
    assert verify_physics(data,.08)==0
    continuation=record_history(net,body,motor,bridge,basis,64)
    assert audit_history(continuation,cfg,basis)<2e-12
    for key in ('history_arrivals','history_scheduled','history_weights'):
        changed={k:v.copy() for k,v in data.items()};changed[key][100,0]+=.01
        with pytest.raises(ValueError):audit_history(changed,cfg,basis)
