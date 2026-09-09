from copy import deepcopy
import json

import numpy as np
import pytest

from simulations.active_inference.experiments.composition_probe import encode, snapshot
from simulations.active_inference.experiments.population_hierarchy import make_config
from simulations.active_inference.experiments.multimodal_pairing_probe import fresh
from simulations.active_inference.experiments.population_state_branch import branch_network, TickDriver
from neuron.extensions.experimental.bounded_plasticity import BoundedPlasticityNeuron


def preparation(tmp_path):
    config, groups, _ = make_config(288)
    for n in config["neurons"]:
        n["metadata"]["bounded_plasticity"] = True
    path = tmp_path/"config.json"
    path.write_text(encode(config))
    net, core, _, _ = fresh(path, 11, BoundedPlasticityNeuron)
    for t in range(24):
        for nid in groups["vision"]+groups["touch"]:
            if t % 4 == nid % 4:
                net.set_external_input(nid, 0, 2.)
        assert "error" not in core.do_tick()
    return path, net, groups


def test_continuation_preserves_queues_and_next_ticks_exactly(tmp_path):
    path, net, groups = preparation(tmp_path)
    assert any(net.presynaptic_wheel) and any(net.retrograde_wheel)
    assert any(n.propagation_queue for n in net.network.neurons.values())
    clone = branch_network(net, "continuation", path, 11)
    assert encode(snapshot(clone)) == encode(snapshot(net))
    for t in range(32):
        for branch in (clone, net):
            for nid in groups["vision"]:
                if t % 4 == nid % 4:
                    branch.set_external_input(nid, 0, .8)
            TickDriver(branch).do_tick()
        assert encode(snapshot(clone)) == encode(snapshot(net))


def test_branch_buffers_and_weights_cannot_mutate_parent(tmp_path):
    path, net, _ = preparation(tmp_path)
    before = encode(snapshot(net))
    clone = branch_network(net, "continuation", path, 11)
    original = net.network.neurons
    for key, connections in clone.network.fast_connection_cache.items():
        for buffer, sid in connections:
            assert not any(buffer is n.input_buffer for n in original.values())
    next(iter(clone.network.neurons.values())).postsynaptic_points[0].u_i.info = .125
    clone.run_tick()
    assert encode(snapshot(net)) == before


def test_all_synaptic_preserves_local_parameters_but_resets_activity(tmp_path):
    path, net, _ = preparation(tmp_path)
    clone = branch_network(net, "all_synaptic", path, 11)
    assert clone.current_tick == 0
    assert not any(clone.presynaptic_wheel) and not any(clone.retrograde_wheel)
    assert any(n.O > 0 for n in net.network.neurons.values())
    for nid, n in clone.network.neurons.items():
        old = net.network.neurons[nid]
        assert n.S == n.O == n.F_avg == 0
        assert np.all(n.M_vector == 0) and n.t_last_fire == -np.inf
        assert not n.propagation_queue and not n.input_buffer.any()
        assert encode(n.params) == encode(old.params)
        assert n.params.eta_post > 0 and n.params.eta_retro > 0
        for sid, point in n.postsynaptic_points.items():
            assert encode(point.u_i) == encode(old.postsynaptic_points[sid].u_i)
            assert point.u_i is not old.postsynaptic_points[sid].u_i
            assert point.potential == 0
        for tid, point in n.presynaptic_points.items():
            assert encode(point.u_o) == encode(old.presynaptic_points[tid].u_o)
            assert type(point.u_o.info) is type(old.presynaptic_points[tid].u_o.info)


def test_unknown_partition_and_stochastic_delay_are_rejected(tmp_path, monkeypatch):
    from simulations.active_inference.experiments.population_state_branch import network_module
    path, net, _ = preparation(tmp_path)
    with pytest.raises(ValueError):
        branch_network(net, "unknown", path, 11)
    monkeypatch.setattr(network_module, "MAX_CONNECTION_SIGNAL_TRAVEL_TICKS", 2)
    with pytest.raises(ValueError, match="RNG"):
        branch_network(net, "continuation", path, 11)


def test_independent_partition_audit_rejects_hidden_fast_state(tmp_path):
    from simulations.active_inference.experiments.population_state_audit import partition_valid
    path, net, _ = preparation(tmp_path)
    parameters = {str(n["id"]): n["params"] for n in json.loads(path.read_text())["neurons"]}
    trained = json.loads(encode(snapshot(net)))
    fresh_state = json.loads(encode(snapshot(branch_network(net, "all_synaptic", path, 11))))
    assert partition_valid(fresh_state, trained, "all_synaptic", parameters)
    assert partition_valid(trained, trained, "continuation")
    for field, value in (("r", 123.), ("O", 1.), ("t_ref", -1.), ("M", [0., 1.])):
        wrong = deepcopy(fresh_state)
        wrong["neurons"]["1"][field] = value
        assert not partition_valid(wrong, trained, "all_synaptic", parameters)
    wrong = deepcopy(fresh_state)
    wrong["neurons"]["1"]["terminals"]["900"][0] += .01
    assert not partition_valid(wrong, trained, "all_synaptic", parameters)


def test_drift_analysis_can_distinguish_rotation_from_lost_separation():
    from simulations.active_inference.experiments.population_state_audit import reference_comparison
    old = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
    new = np.array([[0., 0., 1., 0.], [0., 0., 0., 1.]])
    result = reference_comparison(new, old, new)
    assert result["original_reference_contrast"] == 0
    assert result["current_reference_contrast"] == 1
    assert result["reference_axis_cosine"] == 0
    assert result["reference_norm_ratio"] == 1
    silent = reference_comparison(new, old, np.zeros_like(new))
    assert silent["current_reference_contrast"] is None
    assert silent["reference_axis_cosine"] is None


def test_tick_projection_integrates_to_declared_readout():
    from simulations.active_inference.experiments.population_state_audit import tick_projection
    from simulations.active_inference.experiments.multimodal_pairing_audit import readout, contrast
    rng = np.random.default_rng(81)
    cells = [(rng.random((160, 8, 8)) > .8).astype(float) for _ in range(2)]
    ids = list(range(1, 9))
    refs = rng.random((2, 8)); refs -= refs.mean(axis=1, keepdims=True)
    curve = tick_projection(cells, ids, refs)
    assert curve.shape == (160,)
    values = np.stack([readout(c, ids, "population_centered_rate") for c in cells])
    assert curve[32:].mean() == pytest.approx(contrast(values, refs))
    assert tick_projection(cells, ids, np.zeros((2, 8))) is None
