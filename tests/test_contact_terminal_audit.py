from copy import deepcopy

import pytest

from simulations.active_inference.components.learning.projection_terminals import contact_terminals, mean_pooled_return_rates
from simulations.active_inference.experiments.contact_terminal_audit import wiring_checks, initial_state_checks
from simulations.active_inference.experiments.population_hierarchy import make_config


@pytest.mark.parametrize('mode,builder', [('contact', contact_terminals), ('mean_pooled', mean_pooled_return_rates)])
def test_independent_configuration_check_and_undeclared_fields(mode, builder):
    original, groups, _ = make_config(288)
    selected = groups['visual_core']
    candidate, _ = builder(original, source_ids=selected)
    assert all(wiring_checks(original, candidate, selected, mode).values())
    changed = deepcopy(candidate)
    changed['neurons'][0]['params']['eta_retro'] *= 2
    assert not all(wiring_checks(original, changed, selected, mode).values())
    changed = deepcopy(candidate)
    changed['hidden_gain'] = 2
    assert not wiring_checks(original, changed, selected, mode)['all_other_configuration_exact']
    changed = deepcopy(candidate)
    next(p for p in changed['synaptic_points'] if p['type'] == 'presynaptic')['u_o']['info'] += .1
    assert not wiring_checks(original, changed, selected, mode)['same_initial_per_edge_release']


def test_contact_check_rejects_repooling_and_missing_edges():
    original, groups, _ = make_config(288)
    selected = groups['visual_core']
    candidate, _ = contact_terminals(original, source_ids=selected)
    edges = [c for c in candidate['connections'] if c['source_neuron'] == selected[0]]
    edges[1]['source_terminal'] = edges[0]['source_terminal']
    assert not wiring_checks(original, candidate, selected, 'contact')['selected_contact_fanout']
    candidate['connections'].pop()
    assert not wiring_checks(original, candidate, selected, 'contact')['same_edge_count']


def test_initial_state_audit_rejects_other_cell_or_queue_changes():
    a = dict(tick=0, presynaptic_wheel=[], neurons={'1': dict(S=0., terminals={'900': [1., [0., 0.], 1.]})})
    b = deepcopy(a); b['neurons']['1']['terminals']['901'] = [1., [0., 0.], 1.]
    cfg = dict(connections=[dict(source_neuron=1, source_terminal=900)], synaptic_points=[dict(type='presynaptic', neuron_id=1, terminal_id=900)])
    split = deepcopy(cfg); split['connections'][0]['source_terminal'] = 901
    split['synaptic_points'].append(dict(type='presynaptic', neuron_id=1, terminal_id=901))
    assert all(initial_state_checks(a, b, cfg, split).values())
    b['neurons']['1']['S'] = .001
    assert not initial_state_checks(a, b, cfg, split)['all_other_recorded_initial_state_exact']
    b['neurons']['1']['S'] = 0.
    b['presynaptic_wheel'].append([1, 900, .1])
    assert not initial_state_checks(a, b, cfg, split)['all_other_recorded_initial_state_exact']
