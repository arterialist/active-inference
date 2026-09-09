from copy import deepcopy

import numpy as np
import pytest

from simulations.active_inference.experiments.composition_probe import encode, snapshot
from simulations.active_inference.experiments.population_hierarchy import make_config, cellular
from simulations.active_inference.experiments.multimodal_pairing_probe import fresh
from simulations.active_inference.experiments.upper_signal_replay import PortReplay, upper_ports, replacement
from neuron.extensions.experimental.bounded_plasticity import BoundedPlasticityNeuron
from simulations.active_inference.experiments.upper_signal_audit import verify_replacement


def test_capture_and_exact_clamp_preserve_complete_neural_state(tmp_path):
    config, groups, _ = make_config(288)
    for n in config["neurons"]:
        n["metadata"]["bounded_plasticity"] = True
    path = tmp_path/"config.json"
    path.write_text(encode(config))
    net, _, _, _ = fresh(path, 11, BoundedPlasticityNeuron)
    birth = deepcopy(net)
    ports = upper_ports(net, groups)

    def drive(branch):
        cells = []
        for t in range(64):
            for nid in groups["vision"]+groups["touch"]:
                if t % 4 == nid % 4:
                    branch.set_external_input(nid, 0, 2. if t < 48 else .5)
            branch.run_tick()
            cells.append(cellular(list(branch.network.neurons.values())))
        return np.stack(cells)

    original = BoundedPlasticityNeuron.tick
    baseline = drive(net)
    observed = deepcopy(birth)
    recorder = PortReplay(observed, ports, 64)
    with recorder.installed():
        captured = drive(observed)
    recorder.verify()
    assert BoundedPlasticityNeuron.tick is original
    assert recorder.data[:, :, 0].any()
    assert np.array_equal(baseline, captured)
    assert encode(snapshot(observed)) == encode(snapshot(net))
    replay = deepcopy(birth)
    clamp = PortReplay(replay, ports, 64, recorder.data)
    with clamp.installed():
        repeated = drive(replay)
    clamp.verify()
    assert np.array_equal(clamp.data, recorder.data)
    assert np.array_equal(baseline, repeated)
    assert encode(snapshot(replay)) == encode(snapshot(net))
    with pytest.raises(RuntimeError):
        with clamp.installed():
            raise RuntimeError("test cleanup")
    assert BoundedPlasticityNeuron.tick is original


def test_control_invariants_and_nontrivial_changes():
    ports = [(1, 0, 10, 900), (1, 1, 11, 900), (2, 0, 12, 900), (2, 1, 13, 900)]
    rng = np.random.default_rng(2)
    data = rng.random((80, 4, 4)).astype(np.float32)
    other = rng.random(data.shape).astype(np.float32)
    exact, _ = replacement(data, other, ports, "exact", 11)
    assert np.array_equal(exact, data) and exact is not data
    shuffled, index = replacement(data, other, ports, "time_shuffle", 11)
    assert not np.array_equal(shuffled, data)
    for start in range(0, 80, 32):
        assert np.array_equal(np.sort(shuffled[start:start+32], axis=0), np.sort(data[start:start+32], axis=0))
        assert sorted(index["time_index"][start:start+32]) == list(range(start, min(start+32, 80)))
    swapped, _ = replacement(data, other, ports, "port_swap", 11)
    assert np.array_equal(swapped[:, [0, 2]]+swapped[:, [1, 3]], data[:, [0, 2]]+data[:, [1, 3]])
    assert not np.array_equal(swapped, data)
    foreign, _ = replacement(data, other, ports, "other_video", 11)
    assert np.array_equal(foreign, other)
    for mode in ("exact", "time_shuffle", "port_swap", "other_video"):
        delivered, transform = replacement(data, other, ports, mode, 11)
        verify_replacement(data, other, delivered, ports, mode, transform)
        corrupt = delivered.copy()
        corrupt[10, 0, 0] += .125
        with pytest.raises(ValueError):
            verify_replacement(data, other, corrupt, ports, mode, transform)
    with pytest.raises(ValueError):
        replacement(data, other, ports, "unknown", 11)
    with pytest.raises(ValueError, match="exactly two"):
        replacement(data[:, :3], other[:, :3], ports[:3], "port_swap", 11)
