from copy import deepcopy
import json

import numpy as np
import pytest

from simulations.active_inference.experiments.composition_probe import encode
from simulations.active_inference.experiments.eligibility_association_probe import config, dynamic_snapshot
from simulations.active_inference.experiments.graded_recall_factors import intervene
from simulations.active_inference.experiments.multimodal_pairing_probe import fresh
from neuron.extensions.experimental.graded_eligibility import GradedEligibilityNeuron


def test_intervention_preserves_parent_and_unselected_state(tmp_path):
    cfg, groups = config()
    for n in cfg['neurons']:
        if n['id'] in groups['vision']:
            n['metadata'].update(graded_gain=.25, graded_S0=0., graded_max=0.)
    path = tmp_path/'config.json'; path.write_text(encode(cfg))
    parent, _, _, _ = fresh(path, 11, GradedEligibilityNeuron)
    parent.set_external_input(groups['vision'][0], 0, 1.)
    parent.run_tick()
    saved = dynamic_snapshot(parent)
    branch = deepcopy(parent)
    nid = groups['auditory'][0]
    ports = [(nid, 0, groups['vision'][0])]
    before, after = intervene(branch, ports, groups['vision'], np.array([.123]), 'spiking')
    expected = json.loads(saved)
    expected['neurons'][str(nid)]['synapses']['0'][0] = .123
    assert before == json.loads(saved)
    assert after == expected == json.loads(dynamic_snapshot(branch))
    assert dynamic_snapshot(parent) == saved
    assert all(parent.network.neurons[n]._gg == .25 for n in groups['vision'])
    assert all(branch.network.neurons[n]._gg == 0 for n in groups['vision'])
    assert all(n.params.eta_post > 0 and n.params.eta_retro > 0 for n in branch.network.neurons.values())
    for bad in (np.array([np.nan]), np.array([])):
        with pytest.raises(ValueError, match='Malformed'):
            intervene(branch, ports, groups['vision'], bad, 'spiking')


def test_invalid_expression_rejected_before_mutation():
    with pytest.raises(ValueError, match='Unknown'):
        intervene(None, [], [], np.array([]), 'automatic')
