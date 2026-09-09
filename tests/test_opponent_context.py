from copy import deepcopy

import numpy as np
import pytest

from simulations.active_inference.experiments import context_organization as base
from simulations.active_inference.experiments.opponent_context import (
    AfferentDelay, course, verify_afferents,
)
from simulations.active_inference.components.learning.opponent_prediction import couple_opponent_predictions


def setup(tmp_path, opponent=False):
    cfg,g,s=base.configure(width=8)
    if opponent:
        cfg=couple_opponent_predictions(cfg,g,g['force'])
    path=tmp_path/'config.json';path.write_text(base.encode(cfg))
    net,_,_,_=base.fresh(path,11,base.PredictiveReceptorNeuron)
    rng=np.random.default_rng(829)
    f=[dict(visual=rng.random((300,96)),auditory=rng.random((300,96))) for _ in range(2)]
    return net,g,s,f


def test_opposing_comparators_only_add_declared_wiring():
    original,g,s=base.configure(width=8)
    before=deepcopy(original)
    cfg=couple_opponent_predictions(original,g,g['force'])
    assert original==before
    assert len(cfg['neurons'])==len(original['neurons'])
    assert cfg['connections'][:-8]==original['connections']
    points={(p['neuron_id'],p['synapse_id']):p for p in cfg['synaptic_points'] if p['type']=='postsynaptic'}
    for ch in (0,1):
        for role,sign in [('error_positive',1),('error_negative',-1)]:
            target=g[role][ch]
            actual={c['source_neuron']:points[target,c['target_synapse']]['u_i']['info']
                    for c in cfg['connections'] if c['target_neuron']==target}
            assert actual=={g['force'][ch]:sign,g['prediction'][ch]:-sign,
                            g['force'][1-ch]:-sign,g['prediction'][1-ch]:sign}


def test_zero_delay_exactly_preserves_original_recorder(tmp_path):
    net,g,s,f=setup(tmp_path)
    cp=tmp_path/'state.paula';base.save_checkpoint(net,cp)
    other=base.load_checkpoint(cp,trusted=True).network
    old=base.course(net,base.Arm(),f,g,s,0,0,ticks=48)
    new=course(other,base.Arm(),f,g,s,0,0,ticks=48)
    assert all(np.array_equal(old[k],new[k]) for k in old)
    assert verify_afferents(new)==0


def test_delayed_loop_replays_with_positive_learning(tmp_path):
    net,g,s,f=setup(tmp_path,opponent=True)
    arm=base.Arm();delay=AfferentDelay(7)
    data=course(net,arm,f,g,s,0,0,ticks=48,delay=delay)
    assert np.all(data['drive'][:7,194:198]==0)
    assert np.array_equal(data['drive'][7:,194:198],data['raw_afferents'][:-7])
    assert base.verify_physics(data)==base.verify_learning(data)==verify_afferents(data)==0
    assert np.all(data['eta']>0)
    cp=tmp_path/'state.paula';base.save_checkpoint(net,cp)
    other=base.load_checkpoint(cp,trusted=True).network
    body=base.Arm();body.restore(arm.state())
    delay2=AfferentDelay(7,delay.state())
    a=course(net,arm,f,g,s,1,1,ticks=48,delay=delay)
    b=course(other,body,f,g,s,1,1,ticks=48,delay=delay2)
    assert all(np.array_equal(a[k],b[k]) for k in a)
    bad=deepcopy(a);bad['drive'][9,194]+=.1
    with pytest.raises(ValueError,match='Delayed'):verify_afferents(bad)
    bad=deepcopy(a);bad['raw_afferents'][9,2]+=.1
    with pytest.raises(ValueError,match='Raw'):verify_afferents(bad)
    bad=deepcopy(a);bad['weights'][9,0,0]+=.1
    with pytest.raises(ValueError,match='Learning'):base.verify_learning(bad)


def test_auditor_rejects_neural_motor_trace_mismatch(tmp_path):
    from simulations.active_inference.experiments.opponent_context_analysis import read_record
    net,g,s,f=setup(tmp_path,opponent=True)
    data=course(net,base.Arm(),f,g,s,0,0,ticks=24)
    path=tmp_path/'record.npz';np.savez_compressed(path,**data)
    row=dict(file=path.name,sha256=base.digest(path))
    assert np.array_equal(read_record(tmp_path,row,dict(groups=g))['body'],data['body'])
    col=list(data['neuron_ids']).index(g['muscle'][0])
    data['cells'][12,col,base.FIELDS.index('O')]+=.1
    np.savez_compressed(path,**data);row['sha256']=base.digest(path)
    with pytest.raises(ValueError,match='Physical command'):
        read_record(tmp_path,row,dict(groups=g))


def test_feedback_intervention_preserves_selected_memory_and_positive_rates(tmp_path):
    from simulations.active_inference.experiments.opponent_feedback_probe import attenuate_comparators
    net,g,s,f=setup(tmp_path,opponent=True)
    arm=base.Arm();course(net,arm,f,g,s,0,0,ticks=32)
    nodes=net.network.neurons
    q={(n,sid):nodes[n].postsynaptic_points[sid].u_i.info for n,sid,_ in s}
    changes=attenuate_comparators(net,g)
    assert len(changes)==16
    assert all(after==before*.5 for _,_,before,after in changes)
    assert all(nodes[n].postsynaptic_points[sid].u_i.info==value for (n,sid),value in q.items())
    assert all(n.params.eta_post>0 and n.params.eta_retro>0 for n in nodes.values())
    data=course(net,arm,f,g,s,0,0,ticks=32)
    assert base.verify_learning(data)==base.verify_physics(data)==verify_afferents(data)==0
