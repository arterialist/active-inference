from copy import deepcopy
import numpy as np

from simulations.active_inference.experiments.association_evidence_population import evidence_config, record, test_schedules as schedules
from simulations.active_inference.experiments.association_balance_probe import record_trial
from simulations.active_inference.experiments.eligibility_association_probe import protocol, dynamic_snapshot
from simulations.active_inference.experiments.composition_probe import encode
from simulations.active_inference.experiments.multimodal_pairing_probe import fresh
from neuron.extensions.experimental.port_modulation import PortModulationNeuron


def test_population_scale_and_heterogeneity_do_not_wire_stimulus_assignments():
    diverse, g = evidence_config(11, 'diverse'); homogeneous, h = evidence_config(11, 'homogeneous')
    assert g == h and len(diverse['neurons']) == 368
    assert diverse['connections'] == homogeneous['connections']
    assert diverse['synaptic_points'] == homogeneous['synaptic_points']
    assert len({n['id'] for n in diverse['neurons']}) == 368
    for c in (diverse, homogeneous):
        assert all(n['params']['eta_post'] > 0 and n['params']['eta_retro'] > 0 for n in c['neurons'])
        assert {e['target_neuron'] for e in c['external_inputs']} == set(g['vision']+g['audio'])
        old = {(e['source_neuron'],e['source_terminal']) for e in c['connections'] if e['target_neuron'] <= 176}
        new = {(e['source_neuron'],e['source_terminal']) for e in c['connections'] if e['target_neuron'] > 176}
        assert not old & new
    for ns in g['evidence_coordinates']:
        assert len({diverse['neurons'][i-1]['params']['r_base'] for i in ns}) == 6
        assert len({homogeneous['neurons'][i-1]['params']['r_base'] for i in ns}) == 1


def test_evidence_observer_is_passive_and_learning_is_recorded(tmp_path):
    cfg,g=evidence_config(11);path=tmp_path/'config.json';path.write_text(encode(cfg))
    net,*_=fresh(path,11,PortModulationNeuron);control=deepcopy(net)
    masks,trials=protocol(g,11,'paired',1)
    a=record(net,g,masks,trials[0]);b=record_trial(control,g,masks,trials[0])
    assert all(np.array_equal(a[k],b[k]) for k in b)
    assert dynamic_snapshot(net)==dynamic_snapshot(control)
    assert a['evidence_before'].shape == (96,192,67)
    assert np.any(a['evidence_before'] != a['evidence_after'])


def test_event_transitions_are_external_drive_without_neural_boundary_flags():
    _,g=evidence_config(11);masks,_=protocol(g,11,'paired',1)
    all_schedules=schedules(masks,11)
    assert len(all_schedules)==54
    for first in (0,1):
        for gap in (0,8,24):
            a=all_schedules[f'transition-first{first}-gap{gap}']
            assert a.shape==(96,32) and a.sum()==128
            assert not a[32:32+gap].any()
            assert a[32+gap,np.array(masks['vision'][1-first])-1].all()
