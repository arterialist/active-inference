from copy import deepcopy
import json
import random

import numpy as np
import pytest

pytest.importorskip('cloudpickle')
from simulations.active_inference.core.runtime_checkpoint import save_checkpoint, load_checkpoint, check_buffer_aliases
from simulations.active_inference.experiments.eligibility_association_probe import config, dynamic_snapshot
from simulations.active_inference.experiments.multimodal_pairing_probe import fresh
from neuron.extensions.experimental.eligibility_trace import EligibilityTraceNeuron


def test_exact_active_checkpoint_preserves_types_queues_aliases_and_future(tmp_path):
    cfg, groups = config()
    p = tmp_path/'config.json'; p.write_text(json.dumps(cfg))
    original, _, _, _ = fresh(p, 11, EligibilityTraceNeuron)
    for t in range(18):
        if t % 4 == 0:
            for nid in groups['vision'][:12]+groups['audio'][:12]:
                original.set_external_input(nid, 0, 1.)
        original.run_tick()
    # Pending external input and in-flight events must survive serialization.
    original.set_external_input(groups['vision'][15], 0, .7)
    assert sum(map(len, original.presynaptic_wheel))+sum(map(len, original.retrograde_wheel)) > 0
    file = tmp_path/'neural.checkpoint'
    save_checkpoint(original, file, sources=[p])
    with pytest.raises(ValueError, match='trusted=True'):
        load_checkpoint(file)
    restored = load_checkpoint(file, trusted=True)
    assert dynamic_snapshot(restored.network) == dynamic_snapshot(original)
    for nid, neuron in original.network.neurons.items():
        other = restored.network.network.neurons[nid]
        assert type(neuron.S) is type(other.S)
        for sid, syn in neuron.postsynaptic_points.items():
            assert type(syn.u_i.info) is type(other.postsynaptic_points[sid].u_i.info)
    for t in range(20):
        for net in (original, restored.network):
            if t % 5 == 0:
                net.set_external_input(groups['audio'][t % 32], 0, .9)
        original.run_tick()
        ambient = random.getstate(); numpy_state = deepcopy(np.random.get_state())
        restored.step()
        assert random.getstate() == ambient
        assert np.array_equal(np.random.get_state()[1], numpy_state[1])
        assert dynamic_snapshot(restored.network) == dynamic_snapshot(original)
    n = next(iter(restored.network.network.neurons.values()))
    assert n is not next(iter(original.network.neurons.values()))
    with pytest.raises(FileExistsError):
        save_checkpoint(original, file)
    p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError, match='source mismatch'):
        load_checkpoint(file, trusted=True)


def test_buffer_audit_detects_equal_but_disconnected_array(tmp_path):
    cfg, _ = config(); p = tmp_path/'cfg.json'; p.write_text(json.dumps(cfg))
    net, _, _, _ = fresh(p, 11, EligibilityTraceNeuron)
    check_buffer_aliases(net)
    key = next(iter(net.network.fast_connection_cache))
    buffer, sid = net.network.fast_connection_cache[key][0]
    net.network.fast_connection_cache[key][0] = (buffer.copy(), sid)
    with pytest.raises(ValueError, match='alias'):
        check_buffer_aliases(net)
