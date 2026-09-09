from copy import deepcopy

import numpy as np
import pytest

from simulations.active_inference.components.learning.inhibitory_capacity import match_inhibitory_capacity
from simulations.active_inference.experiments.population_hierarchy import make_config
from simulations.active_inference.experiments.inhibitory_capacity_audit import check_capacity


def current_capacity(config, target):
    ns = {n['id']: n for n in config['neurons']}
    ps = {(p['neuron_id'], p['synapse_id']): p for p in config['synaptic_points'] if p['type'] == 'postsynaptic'}
    ts = {(p['neuron_id'], p['terminal_id']): p for p in config['synaptic_points'] if p['type'] == 'presynaptic'}
    result = []
    for c in config['connections']:
        if c['target_neuron'] == target:
            p = ps[target, c['target_synapse']]
            result.append(p['u_i']['info'] * ns[target]['params']['delta_decay'] ** p['distance_to_hillock'] *
                          ts[c['source_neuron'], c['source_terminal']]['u_o']['info'] / ns[c['source_neuron']]['params']['c'])
    return np.array(result)


def test_capacity_balancing_changes_only_selected_incoming_inhibition():
    original, groups, _ = make_config(1152, 11)
    before = deepcopy(original)
    changed, report = match_inhibitory_capacity(original, groups['upper_core'])
    assert original == before
    assert changed['neurons'] == original['neurons']
    assert changed['connections'] == original['connections']
    assert changed['external_inputs'] == original['external_inputs']
    ports = {(r['target'], p[0]) for r in report['targets'] for p in r['changed_ports']}
    assert len(ports) == 4*len(groups['upper_core'])
    for a, b in zip(original['synaptic_points'], changed['synaptic_points']):
        if (a['neuron_id'], a.get('synapse_id')) not in ports:
            assert a == b
        else:
            assert b['u_i']['info'] < a['u_i']['info'] < 0
            restored = deepcopy(b)
            restored['u_i']['info'] = a['u_i']['info']
            assert restored == a
    for target in groups['upper_core']:
        v = current_capacity(changed, target)
        assert abs(v.sum()) < 1e-14
        assert v[v > 0].sum() > 0
    assert check_capacity(changed, original, groups['upper_core'], 'balanced') == len(ports)
    assert check_capacity(original, original, groups['upper_core'], 'control') == 0
    corrupt = deepcopy(changed)
    corrupt['neurons'][0]['params']['eta_post'] = 0.
    with pytest.raises(ValueError, match='Unexpected'):
        check_capacity(corrupt, original, groups['upper_core'], 'balanced')
    for p in corrupt['synaptic_points']:
        if (p['neuron_id'], p.get('synapse_id')) in ports:
            p['u_i']['info'] *= .5
            break
    corrupt['neurons'] = deepcopy(changed['neurons'])
    with pytest.raises(ValueError, match='balance'):
        check_capacity(corrupt, original, groups['upper_core'], 'balanced')


def test_capacity_rejects_unspecified_external_drive_and_excess_bound():
    cfg, groups, _ = make_config(288)
    with pytest.raises(ValueError, match='External'):
        match_inhibitory_capacity(cfg, groups['vision'])
    for n in cfg['neurons']:
        n['metadata']['plasticity_magnitude_cap'] = .1
    with pytest.raises(ValueError, match='bound'):
        match_inhibitory_capacity(cfg, groups['upper_core'])


def test_capacity_uses_actual_terminal_release_and_source_cooldown():
    cfg, groups, _ = make_config(288)
    for n in cfg['neurons']:
        if n['id'] in groups['connector']:
            n['params']['c'] = 6
    for p in cfg['synaptic_points']:
        if p['type'] == 'presynaptic' and p['neuron_id'] in groups['connector']:
            p['u_o']['info'] = .75
    changed, report = match_inhibitory_capacity(cfg, groups['upper_core'])
    assert len(report['targets']) == len(groups['upper_core'])
    for target in groups['upper_core']:
        assert abs(current_capacity(changed, target).sum()) < 1e-14
