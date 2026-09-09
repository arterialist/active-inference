import random

import numpy as np
import pytest

from test_hierarchical_body_perturbation import preparation
from test_predictive_receptor import load
from simulations.active_inference.core.runtime_checkpoint import RuntimeBranch
from simulations.active_inference.experiments.sensory_motor_placement import temporal_permutation, intervene, payload_hash, reference_prefix
from simulations.active_inference.experiments.predictive_weight_transplant import install_weights


@pytest.mark.parametrize('seed', [23, 44, 77, 101])
def test_distribution_and_anatomical_group_preservation(seed):
    cfg, m = preparation()
    nodes = {n['id']: n for n in cfg['neurons']}
    edges = {(e['target_neuron'], e['target_synapse']): e['source_neuron'] for e in cfg['connections']}
    rows = m['bridge']['prediction']
    q = np.tile(np.linspace(.01, .9, 128), (len(rows), 1))
    out, mapping = temporal_permutation(cfg, m['bridge'], q, seed)
    assert not np.array_equal(q, out)
    np.testing.assert_array_equal(np.sort(q, axis=1), np.sort(out, axis=1))
    for row, nid in enumerate(rows):
        ports = nodes[nid]['metadata']['prediction_ports']
        for col, from_col in enumerate(mapping[row]):
            a = nodes[edges[nid, ports[col]]]
            b = nodes[edges[nid, ports[from_col]]]
            assert a['metadata']['history_source'] == b['metadata']['history_source']
            assert a['params']['lambda_param'] == b['params']['lambda_param']
            assert out[row, col] == q[row, from_col]
    again, _ = temporal_permutation(cfg, m['bridge'], q, seed)
    np.testing.assert_array_equal(out, again)


def test_weight_only_intervention_recovers_complete_runtime(tmp_path):
    cfg, m = preparation(); net = load(tmp_path, cfg)
    # Distinct weights make the permutation observable without a training run.
    q = np.tile(np.linspace(.01, .9, 128), (len(m['bridge']['prediction']), 1))
    install_weights(net, m['bridge'], q)
    branch = RuntimeBranch(net, random.getstate(), np.random.get_state(), {})
    expected = payload_hash(branch)
    before, after, mapping, original = intervene(branch, cfg, m['bridge'], 23)
    assert expected == original
    assert payload_hash(branch) != expected
    np.testing.assert_array_equal(before, q)
    np.testing.assert_array_equal(after, np.take_along_axis(before, mapping, axis=1))
    assert all(n.params.eta_post > 0 and n.params.eta_retro > 0 for n in net.network.neurons.values())


def test_prefix_keeps_recorded_column_identity_whole():
    a = np.arange(768)
    for name in ('feedback_sources', 'feedback_targets', 'feedback_ports',
                 'neuron_ids', 'history_neuron_ids', 'start_weights'):
        np.testing.assert_array_equal(reference_prefix(name, a, 32), a)
    np.testing.assert_array_equal(reference_prefix('ticks', a, 32), a[:32])
