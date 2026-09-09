from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from test_hierarchical_body_perturbation import preparation
from test_predictive_receptor import load
from simulations.active_inference.components.motor.sensory_correction import append_sensory_correction, install_on_runtime
from simulations.active_inference.experiments.hierarchical_body_perturbation import DisturbedRower, audit_intervention
from simulations.active_inference.experiments.sensory_motor_probe import record_correction, audit_motor, body_measures
from simulations.active_inference.experiments.temporal_body_probe import record_history, audit_history
from simulations.active_inference.experiments.predictive_bridge_probe import audit_record
from simulations.active_inference.experiments.composition_probe import encode, snapshot
from simulations.active_inference.core.runtime_checkpoint import check_buffer_aliases


def test_modes_match_edges_cells_and_only_new_motor_weights_differ():
    old, m = preparation(); before = deepcopy(old)
    variants = [append_sensory_correction(old, m['motor'], m['bridge'], mode=mode)[0]
                for mode in ('none', 'position', 'expectation')]
    assert old == before
    for a in variants:
        assert len(a['neurons']) == len(old['neurons'])+4
        assert a['connections'] == variants[0]['connections']
        assert a['neurons'] == variants[0]['neurons']
        for x, y in zip(a['synaptic_points'], variants[0]['synaptic_points']):
            if x != y:
                assert x['neuron_id'] in m['motor']['muscles'] and x['synapse_id'] >= 2
    for mode, cfg in zip(('none', 'position', 'expectation'), variants):
        weights = [p[3] for p in cfg['metadata']['sensory_correction']['ports']]
        assert np.count_nonzero(weights) == {'none': 0, 'position': 8, 'expectation': 16}[mode]


@pytest.mark.parametrize('mode', ['none', 'position', 'expectation'])
def test_acquired_state_installation_and_motor_causality(tmp_path, mode):
    old, m = preparation(); net = load(tmp_path, old); body = DisturbedRower()
    record_history(net, body, m['motor'], m['bridge'], m['basis'], 96, gain=.08)
    old_cells = dict(net.network.neurons)
    old_points = {(nid, sid): p for nid, n in old_cells.items() for sid, p in n.postsynaptic_points.items()}
    old_state = {n: (cell.S, cell.O, cell.F_avg, cell.M_vector.copy(), deepcopy(cell.propagation_queue))
                 for n, cell in old_cells.items()}
    cfg, _ = append_sensory_correction(old, m['motor'], m['bridge'], mode=mode)
    configured = load(tmp_path, cfg)
    install_on_runtime(SimpleNamespace(network=net), configured, old, cfg)
    check_buffer_aliases(net)
    for nid, cell in old_cells.items():
        assert net.network.neurons[nid] is cell
        s, o, f, mod, queue = old_state[nid]
        assert (cell.S, cell.O, cell.F_avg) == (s, o, f)
        np.testing.assert_array_equal(cell.M_vector, mod)
        assert cell.propagation_queue == queue
        assert cell.params.eta_post > 0 and cell.params.eta_retro > 0
    for (nid, sid), p in old_points.items():
        assert net.network.neurons[nid].postsynaptic_points[sid] is p
    plain = deepcopy(net)
    disturbed = DisturbedRower(.5); disturbed.restore(body.state())
    plain_body = DisturbedRower(.5); plain_body.restore(body.state())
    data = record_correction(net, disturbed, m, cfg, 192)
    other = record_history(plain, plain_body, m['motor'], m['bridge'], m['basis'], 192, gain=.08)
    for key in other:
        np.testing.assert_array_equal(data[key], other[key])
    assert encode(snapshot(net)) == encode(snapshot(plain))
    assert audit_motor(data, cfg, m['motor']) == 0
    assert audit_intervention(data, cfg, cut=False, torque=.5, gain=.08) == 0
    assert audit_history(data, cfg, m['basis']) == 0
    assert audit_record(data, cfg, m['bridge']) == 0
    currents = data['motor_scheduled'][:, :, 2:]
    assert bool(np.any(currents)) == (mode != 'none')
    metrics = body_measures(data)
    assert np.isfinite(metrics['sampled_positive_work']) and metrics['sampled_positive_work'] >= 0
    bad = {k: v.copy() for k, v in data.items()}; bad['motor_scheduled'][100, 0, 2] += .01
    with pytest.raises(ValueError):
        audit_motor(bad, cfg, m['motor'])
