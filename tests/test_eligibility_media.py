from copy import deepcopy

import numpy as np
import pytest

from simulations.active_inference.experiments.eligibility_association_probe import config, protocol, run_trial, dynamic_snapshot
from simulations.active_inference.experiments.eligibility_media_probe import EligibilityRecorder, configure
from simulations.active_inference.experiments.eligibility_media_audit import verify_ledger, check_config, check_branch_start
from simulations.active_inference.experiments.multimodal_pairing_probe import fresh
from simulations.active_inference.experiments.population_hierarchy import cellular
from simulations.active_inference.experiments.composition_probe import encode
from neuron.extensions.experimental.eligibility_trace import EligibilityTraceNeuron


def test_generic_ledger_is_passive_and_equation_detects_corruption(tmp_path):
    cfg, groups = config()
    path = tmp_path/'config.json'; path.write_text(encode(cfg))
    a, _, members, _ = fresh(path, 11, EligibilityTraceNeuron)
    b = deepcopy(a)
    masks, trials = protocol(groups, 11, 'paired', 1)
    trial = trials[0]
    ports = [(n, sid, src) for n in groups['auditory'] for sid, src in enumerate(groups['vision'])]
    recorder = EligibilityRecorder(a, ports, 96)
    cells = []
    with recorder.observe():
        for t in range(96):
            if t < 32 and t % 8 == 0:
                for role, key in (('vision', 'cue'), ('audio', 'sound')):
                    for nid in masks[role][trial[key]]: a.set_external_input(nid, 0, 1.)
            a.run_tick(); recorder(); cells.append(cellular(members))
    data = dict(cells=np.array(cells), **recorder.finish())
    old = run_trial(b, groups, masks, trial)
    np.testing.assert_array_equal(data['cells'], old['states'])
    assert dynamic_snapshot(a) == dynamic_snapshot(b)
    points = {(p['neuron_id'], p['synapse_id']): p for p in cfg['synaptic_points'] if p['type'] == 'postsynaptic'}
    initial = np.array([points[n, sid]['u_i']['info'] for n, sid, _ in ports])
    args = (cfg, ports, initial, np.zeros(len(ports)), np.zeros(len(ports)), np.zeros(len(members), bool))
    assert verify_ledger(data, *args)[4] < 2e-12
    for field, message in (('weights', 'eligibility equation'), ('arrivals', 'arrival masks'), ('eta', 'rate ledger')):
        corrupt = {k: v.copy() for k, v in data.items()}; corrupt[field][0, 0] += .1
        with pytest.raises(ValueError, match=message): verify_ledger(corrupt, *args)


def test_intervention_changes_only_declared_metadata():
    cfg, groups = config()
    for n in cfg['neurons']:
        for key in list(n['metadata']):
            if key.startswith('eligibility_'): del n['metadata'][key]
    manifest = {'groups': {'visual_core': groups['vision'], 'tactile_core': groups['auditory']},
                'edges': [(c['source_neuron'], c['target_neuron'], c['target_synapse'], 'crossmodal', True)
                          for c in cfg['connections']]}
    for condition in ('control', 'eligibility'):
        changed, ports = configure(cfg, manifest, condition)
        declared = dict(manifest, selected_ports=ports, condition=condition)
        assert check_config(changed, cfg, declared) == ports
        altered = deepcopy(changed); altered['connections'][0]['source_neuron'] += 1
        with pytest.raises(ValueError, match='Undeclared config'): check_config(altered, cfg, declared)


def test_recorder_restores_class_on_exception(tmp_path):
    cfg, groups = config()
    path = tmp_path/'config.json'; path.write_text(encode(cfg))
    net, _, _, _ = fresh(path, 11, EligibilityTraceNeuron)
    original = EligibilityTraceNeuron.tick
    with pytest.raises(RuntimeError):
        with EligibilityRecorder(net, [(groups['auditory'][0], 0, groups['vision'][0])], 1).observe():
            raise RuntimeError('test')
    assert EligibilityTraceNeuron.tick is original


def test_branch_validator_rejects_hidden_state_changes(tmp_path):
    import json
    cfg, groups = config()
    path = tmp_path/'config.json'; path.write_text(encode(cfg))
    net, _, _, _ = fresh(path, 11, EligibilityTraceNeuron)
    parent = json.loads(dynamic_snapshot(net)); opposite = deepcopy(parent)
    nid = groups['auditory'][0]; ports = [(nid, 0, groups['vision'][0])]
    original = {(nid, 0): parent['neurons'][str(nid)]['synapses']['0'][0]}
    opposite['neurons'][str(nid)]['synapses']['0'][0] = .2
    check_branch_start(opposite, parent, opposite, original, ports, 'opposite_selected')
    changed = deepcopy(opposite); changed['eligibility'][str(nid)]['post'] = .1
    with pytest.raises(ValueError, match='Undeclared branch state'):
        check_branch_start(changed, parent, opposite, original, ports, 'opposite_selected')
    check_branch_start(parent, parent, opposite, original, ports, 'reset_selected')


def test_assignment_preserves_each_distribution_and_independent_audit():
    from simulations.active_inference.experiments.eligibility_media_state import assignment_values
    ports = [(9, s, s+20) for s in range(4)]
    parent = {'neurons': {'9': {'synapses': {str(s): [v, 0., [0.,0.], 0.] for s,v in enumerate([.4,.1,.3,.2])}}},
              'eligibility': {'9': {'pre': [1.,2.,3.,4.], 'post': .7}}, 'tick': 100}
    other = deepcopy(parent)
    for s,v in enumerate([.11,.44,.33,.22]): other['neurons']['9']['synapses'][str(s)][0] = v
    rotations = []
    for condition in ('rotate_1','rotate_2','rotate_3','opposite_rank'):
        assigned = assignment_values(parent, other, ports, condition)
        start = deepcopy(parent)
        for (nid,sid),v in assigned.items(): start['neurons'][str(nid)]['synapses'][str(sid)][0] = v
        check_branch_start(start, parent, other, {}, ports, condition)
        assert sorted(assigned.values()) == [.1,.2,.3,.4]
        if condition.startswith('rotate'): rotations.append([assigned[9,s] for s in range(4)])
        else: assert [assigned[9,s] for s in range(4)] == [.1,.4,.3,.2]
        broken = deepcopy(start); broken['eligibility']['9']['pre'][0] += .1
        with pytest.raises(ValueError, match='Undeclared branch state'): check_branch_start(broken,parent,other,{},ports,condition)
        broken = deepcopy(start); broken['neurons']['9']['synapses']['0'][0] += .01
        with pytest.raises(ValueError, match='distribution'): check_branch_start(broken,parent,other,{},ports,condition)
    for j,v in enumerate([.4,.1,.3,.2]):
        assert set(r[j] for r in rotations) == {.1,.2,.3,.4}-{v}


def test_congruence_schedule_and_factorial_control():
    from simulations.active_inference.experiments.eligibility_media_state import probe_schedule
    from simulations.active_inference.experiments.audiovisual_expectation_audit import interaction
    schedule=probe_schedule('congruence',300,3168)
    assert len(schedule)==12
    for condition in ('initial','unchanged','reset_selected'):
        rows=[s for s in schedule if s['condition']==condition]
        assert {(s['trial']['visual_clip'],s['trial']['audio_clip']) for s in rows}=={(0,0),(0,1),(1,0),(1,1)}
        assert {s['trial']['start'] for s in rows}=={0 if condition=='initial' else 3168}
    assert len(probe_schedule('exchange',300,3168))==6
    assert len(probe_schedule('assignment',300,3168))==10
    rng=np.random.default_rng(1)
    vision=rng.normal(size=(2,12,5,8));audio=rng.normal(size=(2,12,5,8))
    values={(v,a):vision[v]+audio[a] for v in (0,1) for a in (0,1)}
    np.testing.assert_allclose(interaction(values),0,atol=1e-15)
    for v,a in values:
        if v!=a: values[v,a]+=2.
    np.testing.assert_allclose(interaction(values),2.,atol=1e-15)
    with pytest.raises(ValueError): interaction({(0,0):values[0,0]})


def test_weight_component_decomposition_is_orthogonal_not_a_memory_score():
    from simulations.active_inference.experiments.audiovisual_expectation_audit import weight_components
    delta=np.array([[1.,2.,3.,4.],[-2.,0.,2.,0.]])
    parts,fractions=weight_components(delta)
    np.testing.assert_allclose(sum(parts.values()),delta)
    assert sum(fractions.values())==pytest.approx(1.)
    keys=list(parts)
    for i,k in enumerate(keys):
        for other in keys[i+1:]: assert float((parts[k]*parts[other]).sum())==pytest.approx(0.,abs=1e-15)
    assert all(v is None for v in weight_components(np.zeros((2,4)))[1].values())


def test_factor_transplants_separate_target_mean_and_input_residual():
    from simulations.active_inference.experiments.eligibility_media_state import assignment_values, probe_schedule
    ports=[(9,s,s+20) for s in range(4)]
    parent={'neurons': {'9': {'synapses': {str(s): [v,0.,[0.,0.],0.] for s,v in enumerate([.2,.3,.4,.5])}}},
            'eligibility': {'9': {'pre':[1.,2.,3.,4.], 'post':.7}}, 'tick':100}
    other=deepcopy(parent)
    for s,v in enumerate([.45,.55,.25,.35]): other['neurons']['9']['synapses'][str(s)][0]=v
    values={}
    for condition in ('opposite_mean','opposite_residual'):
        moved=assignment_values(parent,other,ports,condition); values[condition]=np.array(list(moved.values()))
        start=deepcopy(parent)
        for (n,s),v in moved.items(): start['neurons'][str(n)]['synapses'][str(s)][0]=v
        check_branch_start(start,parent,other,{},ports,condition)
        broken=deepcopy(start); broken['eligibility']['9']['post']+=.01
        with pytest.raises(ValueError,match='Undeclared branch state'): check_branch_start(broken,parent,other,{},ports,condition)
        broken=deepcopy(start); broken['neurons']['9']['synapses']['0'][0]+=.01
        with pytest.raises(ValueError,match='factor differs'): check_branch_start(broken,parent,other,{},ports,condition)
    own=np.array([.2,.3,.4,.5]); donor=np.array([.45,.55,.25,.35])
    np.testing.assert_allclose(values['opposite_mean']-values['opposite_mean'].mean(),own-own.mean())
    assert values['opposite_residual'].mean()==pytest.approx(own.mean())
    np.testing.assert_allclose(values['opposite_mean']+values['opposite_residual'],own+donor)
    assert len(probe_schedule('factors',300,3168))==8
    for s in range(4): other['neurons']['9']['synapses'][str(s)][0]=.99
    with pytest.raises(ValueError,match='bounds'): assignment_values(parent,other,ports,'opposite_mean')


def test_response_factorial_distinguishes_additivity_from_a_memory_claim():
    from simulations.active_inference.experiments.eligibility_media_audit import factor_response
    baseline=np.arange(12,dtype=float).reshape(3,4)
    samples=dict(unchanged=baseline,opposite_mean=baseline+2,opposite_residual=baseline-3,opposite_selected=baseline-1)
    np.testing.assert_array_equal(factor_response(samples),np.zeros((3,4)))
    samples['opposite_selected'][1,2]+=.5
    expected=np.zeros((3,4));expected[1,2]=.5
    np.testing.assert_array_equal(factor_response(samples),expected)
    with pytest.raises(ValueError,match='four'): factor_response({'unchanged':baseline})


def test_conditional_effect_can_be_silent_in_spikes_but_not_membrane():
    from simulations.active_inference.experiments.eligibility_media_audit import conditional_factor_effects
    baseline=np.zeros((5,2,2));residual=baseline.copy();residual[2,0,1]=1.
    both=baseline.copy();both[1:,0,0]=.2
    result=conditional_factor_effects(dict(unchanged=baseline,opposite_mean=baseline.copy(),
        opposite_residual=residual,opposite_selected=both),['S','O'])
    assert result['residual_on_own_mean']['O']['first_tick']==2
    assert result['residual_on_opposite_mean']['O']['different_cell_ticks']==0
    assert result['residual_on_opposite_mean']['S']['first_tick']==1
    assert result['residual_on_opposite_mean']['S']['different_cells_per_tick']==[0,1,1,1,1]
