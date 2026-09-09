import numpy as np
import pytest

from simulations.active_inference.experiments import context_organization as base
from simulations.active_inference.components.learning.feedback_competition import append_feedback_competition
from neuron.extensions.experimental.magnitude_retrograde import MagnitudeRetrogradeNeuron


def small():
    cfg=dict(metadata={},global_params=dict(num_inputs=2,num_neuromodulators=2),
        simulation_params=dict(max_history=1),neurons=[],synaptic_points=[],connections=[],external_inputs=[])
    for nid in range(1,33):
        n=base.k.neuron(nid,lam=2,c=3,eta_post=1e-7,eta_retro=1e-7,delta_decay=.99,
            meta=dict(graded_gain=1.,bounded_plasticity=True,retrograde_magnitude_error=True))
        n['params']['num_inputs']=2;cfg['neurons'].append(n)
        cfg['synaptic_points'].extend([base.k.term(nid),base.k.syn(nid,0,1.,adapt=[0.,0.]),
                                       base.k.syn(nid,1,0.,adapt=[0.,0.])])
        cfg['external_inputs'].append(base.k.ext(nid,0))
    return cfg


def test_disabled_exact_and_conductance_not_multiplied():
    cfg=small();copy,groups=append_feedback_competition(cfg,{'core':list(range(1,33))})
    assert copy==cfg and groups=={} and copy is not cfg
    new,groups=append_feedback_competition(cfg,{'core':list(range(1,33))},enabled=True)
    assert len(groups['competition_core'])==4 and len(new['neurons'])==36
    assert len(cfg['neurons'])==32
    for row in new['metadata']['feedback_competition']['territories']:
        assert len(row['members'])*row['input_weight']==1.
        assert row['output_weight']==-1.
    destinations=[c['target_neuron'] for c in new['connections'] if c['source_neuron']>=33]
    assert sorted(destinations)==list(range(1,33))
    with pytest.raises(ValueError,match='disjoint'):
        append_feedback_competition(cfg,{'a':list(range(1,17)),'b':list(range(1,17))},enabled=True)


def test_running_local_loop_suppresses_without_silencing(tmp_path):
    cfg,g=append_feedback_competition(small(),{'core':list(range(1,33))},enabled=True)
    p=tmp_path/'network.json';p.write_text(base.encode(cfg))
    net=base.fresh(p,11,MagnitudeRetrogradeNeuron)[0]
    record=[]
    for t in range(256):
        for nid in range(1,33):net.set_external_input(nid,0,1.)
        net.run_tick();record.append([n.O for n in net.network.neurons.values()])
    record=np.array(record)
    assert np.isfinite(record).all()
    assert np.all((record[192:,:32]>.45)&(record[192:,:32]<.6))
    assert np.all(record[192:,32:]>0)
    assert all(n.params.eta_post>0 and n.params.eta_retro>0 for n in net.network.neurons.values())
    # Frozen-coefficient, all-active common-mode delay calculation. This is
    # checked separately from the adaptive full circuit, not a stability proof.
    roots=np.roots([1.,-1.,.25,0.,.25*.99**2])
    assert abs(roots).max()<1.
