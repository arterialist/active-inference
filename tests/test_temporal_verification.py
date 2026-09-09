from copy import deepcopy

import numpy as np
import pytest

from simulations.active_inference.experiments import context_organization as base
from simulations.active_inference.experiments.opponent_context import AfferentDelay,course,verify_afferents
from simulations.active_inference.experiments.temporal_verification import configure,verify_learning


def network(tmp_path,aligned=True):
    cfg,g,s=configure(width=8,aligned=aligned)
    path=tmp_path/f'{aligned}.json';path.write_text(base.encode(cfg))
    net,_,_,_=base.fresh(path,11,base.PredictiveReceptorNeuron)
    return net,g,s,cfg


def test_only_verification_timing_differs_between_paired_configs():
    a,g,s=configure(width=8,aligned=False);b,gb,sb=configure(width=8,aligned=True)
    assert g==gb and s==sb and a['neurons']==b['neurons'] and a['connections']==b['connections']
    changed=[]
    for old,new in zip(a['synaptic_points'],b['synaptic_points']):
        if old!=new:
            assert new['distance_to_hillock']==old['distance_to_hillock']+64
            assert new['u_i']['info']==pytest.approx(old['u_i']['info']*.99**-64)
            changed.append(new)
    assert len(changed)==8
    assert not any(p['neuron_id'] in g['muscle'] for p in changed)


def test_prediction_copy_impulse_is_delayed_not_removed(tmp_path):
    traces=[]
    for aligned in (False,True):
        net,g,_,cfg=network(tmp_path,aligned)
        nid=g['error_negative'][0]
        sid=next(c['target_synapse'] for c in cfg['connections'] if
                 c['target_neuron']==nid and c['source_neuron']==g['prediction'][0])
        n=net.network.neurons[nid];trace=[]
        for tick in range(80):
            if tick==0:n.input_buffer[sid,0]=1.
            n.tick({},tick);trace.append(n.S)
        traces.append(np.array(trace))
    assert np.flatnonzero(traces[0])[0]==1
    assert np.flatnonzero(traces[1])[0]==65
    np.testing.assert_allclose(traces[0][:16],traces[1][64:80],atol=1e-14,rtol=0)


def test_full_state_continuation_with_pending_delayed_predictions(tmp_path):
    net,g,s,_=network(tmp_path)
    rng=np.random.default_rng(702)
    f=[dict(visual=rng.random((300,96)),auditory=rng.random((300,96))) for _ in range(2)]
    arm=base.Arm();delay=AfferentDelay(64)
    data=course(net,arm,f,g,s,0,0,ticks=144,delay=delay)
    assert verify_learning(data)==base.verify_physics(data)==verify_afferents(data)==0
    assert np.any(data['weights'][-1]>0) and np.all(data['eta']>0)
    with pytest.raises(ValueError,match='Learning'):base.verify_learning(data)
    checkpoint=tmp_path/'state.paula';base.save_checkpoint(net,checkpoint)
    other=base.load_checkpoint(checkpoint,trusted=True).network
    body=base.Arm();body.restore(arm.state());other_delay=AfferentDelay(64,delay.state())
    a=course(net,arm,f,g,s,1,1,ticks=80,delay=delay)
    b=course(other,body,f,g,s,1,1,ticks=80,delay=other_delay)
    assert all(np.array_equal(a[k],b[k]) for k in a)
    bad=deepcopy(a);bad['weights'][10,0,0]+=.01
    with pytest.raises(ValueError,match='Temporal learning'):verify_learning(bad)


def test_transplant_changes_only_declared_information_weights(tmp_path):
    from simulations.active_inference.experiments.temporal_memory_transplant import transplant_selected
    net,g,s,_=network(tmp_path)
    cp=tmp_path/'birth.paula';base.save_checkpoint(net,cp)
    other=base.load_checkpoint(cp,trusted=True).network
    selected={(nid,sid) for nid,sid,_ in s}
    before={(nid,sid):float(p.u_i.info) for nid,n in net.network.neurons.items()
            for sid,p in n.postsynaptic_points.items()}
    for nid,sid in selected:other.network.neurons[nid].postsynaptic_points[sid].u_i.info=.12
    transplant_selected(net,other,s)
    for nid,n in net.network.neurons.items():
        assert n.S==0 and n.O==0 and not n.propagation_queue
        for sid,p in n.postsynaptic_points.items():
            assert p.u_i.info==(.12 if (nid,sid) in selected else before[nid,sid])
        assert n.params.eta_post>0 and n.params.eta_retro>0


def test_analyses_reject_empty_evidence(tmp_path):
    from simulations.active_inference.experiments.temporal_verification_analysis import compare,compare_transplants
    for analyze in (compare,compare_transplants):
        with pytest.raises(ValueError,match='No experimental conditions'):
            analyze([],tmp_path/'empty')
    assert not (tmp_path/'empty').exists()
