from copy import deepcopy

import numpy as np
import pytest

from simulations.active_inference.experiments.eligibility_association_probe import config, protocol, run_trial, dynamic_snapshot
from simulations.active_inference.experiments.composition_probe import encode
from simulations.active_inference.experiments.multimodal_pairing_probe import fresh
from simulations.active_inference.experiments.population_hierarchy import cellular
from neuron.extensions.experimental.eligibility_trace import EligibilityTraceNeuron
from simulations.active_inference.experiments.eligibility_association_audit import verify_trace


def test_observer_does_not_change_dynamics_or_hidden_traces(tmp_path):
    cfg, groups = config()
    masks, trials = protocol(groups, 11, 'paired', 1)
    path = tmp_path/'config.json'; path.write_text(encode(cfg))
    a, _, _, _ = fresh(path, 11, EligibilityTraceNeuron)
    b = deepcopy(a)
    observed = run_trial(a, groups, masks, trials[0])
    rows = []
    for t in range(96):
        if t < 32 and t % 8 == 0:
            for role, key in (('vision', 'cue'), ('audio', 'sound')):
                for n in masks[role][trials[0][key]]: b.set_external_input(n, 0, 1.)
        b.run_tick()
        rows.append(cellular(list(b.network.neurons.values())))
    np.testing.assert_array_equal(rows, observed['states'])
    assert dynamic_snapshot(a) == dynamic_snapshot(b)
    class Record(dict):
        @property
        def files(self): return list(self)
    raw = Record(observed)
    points = {(p['neuron_id'], p['synapse_id']): p for p in cfg['synaptic_points'] if p['type'] == 'postsynaptic'}
    initial = np.array([[points[n, i]['u_i']['info'] for i in range(32)] for n in groups['auditory']])
    args = (initial, np.zeros((32,32)), np.zeros(32), np.zeros(32, dtype=bool), cfg, groups, True)
    assert verify_trace(raw, *args)[-1] < 2e-12
    broken = Record({k: v.copy() for k,v in observed.items()})
    broken['weights'][5,0,0] += .1
    with pytest.raises(ValueError, match='eligibility equation'):
        verify_trace(broken, *args)
    broken = Record({k: v.copy() for k,v in observed.items()})
    broken['arrivals'][0,0,0] = 1.
    with pytest.raises(ValueError, match='arrival masks'):
        verify_trace(broken, *args)


def test_pairing_enters_only_as_stimulus_contingency():
    cfg, groups = config()
    a, paired = protocol(groups, 11, 'paired', 16)
    b, swapped = protocol(groups, 11, 'swapped', 16)
    c, separated = protocol(groups, 11, 'separated', 16)
    assert a == b == c
    assert [t['cue'] for t in paired] == [t['cue'] for t in swapped] == [t['cue'] for t in separated]
    assert [t['sound'] for t in swapped] == [1-t['sound'] for t in paired]
    assert sorted(t['sound'] for t in paired) == sorted(t['sound'] for t in swapped)
    assert all(t['audio_offset'] == 32 for t in separated)
    assert set(e['target_neuron'] for e in cfg['external_inputs']) == set(groups['vision']+groups['audio'])
