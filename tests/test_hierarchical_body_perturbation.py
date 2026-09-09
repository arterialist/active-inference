from copy import deepcopy

import numpy as np
import pytest

from test_predictive_receptor import fixture_config, load
from simulations.active_inference.experiments.composition_probe import k, encode, snapshot
from simulations.active_inference.components.motor.proprioceptive_rower import append_proprioceptive_rower
from simulations.active_inference.components.learning.temporal_basis import append_temporal_basis
from simulations.active_inference.components.learning.predictive_bridge import append_predictive_bridge
from simulations.active_inference.experiments.temporal_body_probe import record_history, audit_history
from simulations.active_inference.experiments.predictive_bridge_probe import audit_record
from simulations.active_inference.experiments.hierarchical_body_perturbation import (
    DisturbedRower, feedback_edges, record_feedback, audit_intervention)


def preparation():
    cfg = fixture_config()
    for n, role in zip(cfg['neurons'], ('upper_core', 'upper_core', 'visual_core', 'tactile_core')):
        n['metadata']['role'] = role
    for source, target in ((1, 3), (2, 4)):
        point = next(p for p in cfg['synaptic_points'] if p['type'] == 'postsynaptic'
                     and p['neuron_id'] == target and p['synapse_id'] == 1)
        point['u_i']['info'] = .7
        cfg['connections'].append(k.conn(source, target, 1))
    cfg, motor, _, _ = append_proprioceptive_rower(cfg, [1, 2])
    cfg, basis = append_temporal_basis(cfg, motor['cpg']+motor['muscles'])
    cfg, bridge, _, _ = append_predictive_bridge(cfg, basis, motor['joint_position'], fanin=len(basis), consumers=8)
    return cfg, dict(motor=motor, basis=basis, bridge=bridge, gain=.08)


def test_identity_ports_and_reject_missing_feedback():
    cfg, _ = preparation()
    assert [(e['source_neuron'], e['target_neuron'], e['target_synapse'])
            for e in feedback_edges(cfg)] == [(1, 3, 1), (2, 4, 1)]
    with pytest.raises(ValueError):
        feedback_edges(fixture_config())


def test_intact_observer_exact_and_cut_nontrivial(tmp_path):
    cfg, m = preparation()
    net = load(tmp_path, cfg); control = deepcopy(net); lesion = deepcopy(net)
    body = DisturbedRower()
    data = record_feedback(net, body, m, cfg, 192)
    other_body = DisturbedRower()
    plain = record_history(control, other_body, m['motor'], m['bridge'], m['basis'], 192, gain=.08)
    for key in plain:
        np.testing.assert_array_equal(data[key], plain[key])
    assert encode(snapshot(net)) == encode(snapshot(control))
    np.testing.assert_array_equal(body.state(), other_body.state())
    cut = record_feedback(lesion, DisturbedRower(), m, cfg, 192, cut=True)
    assert cut['feedback_before'][:, :, 0].max() > 0
    assert np.all(cut['feedback_after'][:, :, 0] == 0)
    assert np.any(data['cells'][:, [2, 3], 1] != cut['cells'][:, [2, 3], 1])
    for d, flag in ((data, False), (cut, True)):
        assert audit_intervention(d, cfg, cut=flag, torque=0., gain=.08) == 0
        assert audit_history(d, cfg, m['basis']) == 0
        assert audit_record(d, cfg, m['bridge']) == 0


def test_force_physics_and_deliberate_corruption(tmp_path):
    cfg, m = preparation(); net = load(tmp_path, cfg)
    baseline = record_feedback(deepcopy(net), DisturbedRower(), m, cfg, 192)
    data = record_feedback(net, DisturbedRower(.5), m, cfg, 192)
    np.testing.assert_array_equal(data['physical_after'][:64], baseline['physical_after'][:64])
    assert np.any(data['physical_after'][64:] != baseline['physical_after'][64:])
    assert np.count_nonzero(data['applied_force']) == 48
    assert audit_intervention(data, cfg, cut=False, torque=.5, gain=.08) == 0
    assert audit_history(data, cfg, m['basis']) == 0
    assert audit_record(data, cfg, m['bridge']) == 0
    for key in ('applied_force', 'feedback_before', 'feedback_after', 'physical_after'):
        bad = {k: v.copy() for k, v in data.items()}
        bad[key][70].flat[0] += .01
        with pytest.raises(AssertionError):
            audit_intervention(bad, cfg, cut=False, torque=.5, gain=.08)


def test_forward_path_claim_does_not_invent_return_paths():
    from simulations.active_inference.experiments.hierarchical_body_audit import pathway_summary
    cfg, m = preparation()
    result = pathway_summary(cfg, m['bridge'])
    assert result['motor_neurons_forward_reachable_from_upper'] == []
    assert result['upper_neurons_forward_reachable_from_new_consumers'] == []
    # Reachability must change when an actual forward motor route is installed.
    cfg['connections'].append(k.conn(3, m['motor']['cpg'][0], 1))
    assert set(pathway_summary(cfg, m['bridge'])['motor_neurons_forward_reachable_from_upper']) == set(
        m['motor']['cpg']+m['motor']['muscles'])


def test_causal_summary_keeps_signs_and_exact_null():
    from simulations.active_inference.experiments.hierarchical_body_audit import describe
    x = np.zeros((192, 2)); x[12:16, 0] = -1; x[150, 1] = 1e-12
    result = describe(x)
    assert result['first_nonzero'] == 12
    assert result['nonzero_intervals'] == [[12, 15], [150, 150]]
    assert result['negative_cell_ticks'] == 4 and result['positive_cell_ticks'] == 1
    assert result['last64_maximum_absolute'] == 1e-12
    assert describe(np.zeros_like(x))['first_nonzero'] is None


def test_observability_does_not_equate_inputs_with_complete_state():
    from simulations.active_inference.experiments.hierarchical_body_audit import observability
    fields = ('history_arrivals', 'history_weights', 'history_scheduled', 'arrivals',
              'weights', 'eta', 'error_arrival', 'joint_input')
    a = {key: np.zeros((192, 2, 2) if key == 'arrivals' else (192, 2)) for key in fields}
    b = deepcopy(a)
    b['joint_input'][65:, 0] = 1.
    b['weights'][70:, 0] = .1
    b['arrivals'][100, 0, 0] = .1
    result = observability(a, b)
    assert result['different_sensation_with_identical_context_prefix'] == [[65, 99]]
    assert result['weights']['first_nonzero'] == 70
    assert result['equal_context_and_different_sensation_ticks'] == 126
