from copy import deepcopy
import json
import numpy as np
import pytest

from simulations.active_inference.experiments.media_weight_identity import cycle_weights, set_selected_weights
from simulations.active_inference.experiments.eligibility_association_probe import config, dynamic_snapshot
from simulations.active_inference.experiments.multimodal_pairing_probe import fresh
from neuron.extensions.experimental.eligibility_trace import EligibilityTraceNeuron


def test_cycle_preserves_each_target_multiset_and_covers_all_other_sources():
    ports = [(n, s, 100+s) for n in (10, 11) for s in range(4)]
    q = np.arange(8.)/10
    alternatives = []
    for shift in (1, 2, 3):
        other, perm = cycle_weights(ports, q, shift)
        assert (perm != np.arange(8)).all()
        for start in (0, 4):
            assert np.array_equal(np.sort(other[start:start+4]), q[start:start+4])
        alternatives.append(perm)
    for i in range(8):
        assert set(np.array(alternatives)[:, i]) == set(range(i//4*4, i//4*4+4))-{i}
    with pytest.raises(ValueError):
        cycle_weights(ports[:-1], q[:-1], 1)


def test_weight_intervention_preserves_other_state_and_types(tmp_path):
    cfg, groups = config(); path = tmp_path/'config.json'; path.write_text(json.dumps(cfg))
    net, _, _, _ = fresh(path, 11, EligibilityTraceNeuron)
    ports = [(n.id, sid, n.synapse_sources[sid][0]) for n in net.network.neurons.values()
             for sid in n.eligibility_ports if sid in n.synapse_sources]
    assert ports
    before = json.loads(dynamic_snapshot(net))
    q = np.full(len(ports), .3)
    expected = deepcopy(before)
    types = [type(net.network.neurons[n].postsynaptic_points[s].u_i.info) for n, s, _ in ports]
    for n, s, _ in ports:
        expected['neurons'][str(n)]['synapses'][str(s)][0] = .3
    assert set_selected_weights(net, ports, q) == expected
    assert types == [type(net.network.neurons[n].postsynaptic_points[s].u_i.info) for n, s, _ in ports]
    stable = dynamic_snapshot(net)
    with pytest.raises(ValueError):
        set_selected_weights(net, ports, q+2)
    assert dynamic_snapshot(net) == stable
