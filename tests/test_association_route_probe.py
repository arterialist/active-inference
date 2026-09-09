from copy import deepcopy
import gzip
import json

import numpy as np
import pytest

from simulations.active_inference.experiments.composition_probe import encode, snapshot
from simulations.active_inference.experiments.population_hierarchy import make_config, weight_values, FIELDS
from simulations.active_inference.experiments.multimodal_pairing_probe import fresh, episode, pairing_protocol, WeightObserver
from simulations.active_inference.experiments.association_route_probe import cut_routes, run
from simulations.active_inference.experiments.association_route_audit import audit, spike_effect, pathway_envelope
from neuron.extensions.experimental.bounded_plasticity import BoundedPlasticityNeuron


def preparation(tmp_path):
    config, groups, edges = make_config(288)
    for n in config["neurons"]:
        n["metadata"]["bounded_plasticity"] = True
    path = tmp_path/"config.json"
    path.write_text(encode(config))
    net, core, neurons, synapses = fresh(path, 11, BoundedPlasticityNeuron)
    return net, core, neurons, synapses, groups, edges


def test_cut_changes_only_selected_forward_and_retrograde_routes(tmp_path):
    net, _, _, _, groups, _ = preparation(tmp_path)
    original = deepcopy(net)
    before = encode(snapshot(net))
    removed = cut_routes(net, groups, "direct_cut")
    assert len(removed) == len(groups["tactile_core"])*4
    assert encode(snapshot(net)) == before
    for src, term, tgt, sid in removed:
        assert src in groups["visual_core"] and tgt in groups["tactile_core"]
        assert sid in net.network.neurons[tgt].postsynaptic_points
        assert sid not in net.network.neurons[tgt].synapse_sources
        assert (tgt, sid) not in net.network.connection_cache[src, term]
        assert not any(buf is net.network.neurons[tgt].input_buffer and syn == sid
                       for buf, syn in net.network.fast_connection_cache[src, term])
    assert len(net.network.connections)+len(removed) == len(original.network.connections)
    for src, term, tgt, sid in net.network.connections:
        assert net.network.neurons[tgt].synapse_sources[sid] == (src, term)
        assert sum(buf is net.network.neurons[tgt].input_buffer and syn == sid
                   for buf, syn in net.network.fast_connection_cache[src, term]) == 1
    assert all(n.params.eta_post > 0 and n.params.eta_retro > 0 for n in net.network.neurons.values())


def test_impulse_reaches_intact_port_but_not_removed_route(tmp_path):
    intact, _, _, _, groups, _ = preparation(tmp_path)
    cut = deepcopy(intact)
    src, term, tgt, sid = cut_routes(cut, groups, "direct_cut")[0]
    for net in (intact, cut):
        net.add_signal(src, 0, term, signal_strength=1., travel_time=1)
        net.run_tick()
        assert net.network.neurons[tgt].postsynaptic_points[sid].potential == 0
        net.run_tick()
    assert intact.network.neurons[tgt].postsynaptic_points[sid].potential > 0
    assert cut.network.neurons[tgt].postsynaptic_points[sid].potential == 0


def test_rejects_queued_history_and_unknown_intervention(tmp_path):
    net, _, _, _, groups, _ = preparation(tmp_path)
    with pytest.raises(ValueError):
        cut_routes(net, groups, "unknown")
    net.add_signal(groups["vision"][0], 0, 900)
    with pytest.raises(ValueError, match="empty event queues"):
        cut_routes(net, groups, "both_cut")


def test_replay_and_all_factorial_branches(tmp_path):
    source, output = tmp_path/"source", tmp_path/"output"
    source.mkdir()
    net, core, neurons, synapses, groups, edges = preparation(source)
    rng = np.random.default_rng(9)
    features = []
    for i in (0, 1):
        f = {"ticks": 8, "visual": rng.random((8, len(groups["vision"]))),
             "auditory": rng.random((8, len(groups["touch"])))}
        features.append(f)
        np.savez_compressed(source/f"sensory-{i}.npz", **f)
    trials = pairing_protocol(8, 1, 11, "paired")
    manifest = {"seed": 11, "mapping": "paired", "weight_dynamics": "bounded", "source_hashes": {},
                "groups": groups, "edges": edges, "trials": trials, "clip_ticks": 8, "fields": FIELDS}
    (source/"manifest.json").write_text(encode(manifest))
    initial = weight_values(synapses)
    observer = WeightObserver(neurons, synapses)
    for i, trial in enumerate(trials):
        observer.rows = []
        cells = episode(net, core, neurons, features, groups, trial, observer)
        np.savez_compressed(source/f"experience-{i:03d}.npz", cells=cells,
                            incoming_info_after=weight_values(synapses), weight_health=np.array(observer.rows))
    np.savez_compressed(source/"parameters.npz", initial_info=initial, learned_info=weight_values(synapses))
    with gzip.open(source/"training-final-state.json.gz", "wt") as f:
        f.write(encode(snapshot(net)))
    result = run(source, output)
    assert result["training_exact_episodes"] == list(range(4))
    assert result["full_training_snapshot_exact"]
    assert len(result["probes"]) == 16
    assert result["ticks"] == 208+128
    for state in ("initial", "trained"):
        rows = [p for p in result["probes"] if p["state"] == state]
        assert len({p["start_local_state_sha256"] for p in rows}) == 1
        for p in rows:
            assert p["parent_unchanged"]
            with np.load(output/p["file"]) as raw:
                assert raw["cells"].shape == (8, 288, 8)


def test_tick_effect_preserves_latency_and_opposite_cell_changes():
    before = np.array([[0, 0], [1, 0], [1, 0]], dtype=bool)
    after = np.array([[0, 0], [1, 0], [0, 1]], dtype=bool)
    changed, signed, first = spike_effect(before, after)
    assert list(changed) == [0, 0, 2]
    assert list(signed) == [0, 0, 0]
    assert first == 2


def test_fixed_coefficient_envelope_matches_maximal_periodic_drive():
    config = {"neurons": [{"id": 1, "params": {"c": 3}},
                          {"id": 2, "params": {"lambda_param": 4., "delta_decay": .99, "r_base": .65}}],
        "connections": [{"source_neuron": 1, "source_terminal": 900, "target_neuron": 2, "target_synapse": 0}],
        "synaptic_points": [{"neuron_id": 1, "terminal_id": 900, "type": "presynaptic", "u_o": {"info": 1.}},
                            {"neuron_id": 2, "synapse_id": 0, "type": "postsynaptic", "distance_to_hillock": 2,
                             "u_i": {"info": .25, "plast": 0.}}]}
    result = pathway_envelope(config, {"upper_core": [1], "tactile_core": [2]}, "upper_core")
    s, peak = 0., 0.
    for t in range(300):
        current = .25*.99**2 if t % 3 == 0 else 0.
        s += (-s+current)/4.
        peak = max(peak, s)
        assert s <= result["maximum"]+1e-14
    assert peak == pytest.approx(result["maximum"])
    assert result["count_at_or_above_base_r"] == 0


def test_independent_route_audit_and_corruption_rejection(tmp_path):
    paths = []
    for mapping in ("paired", "swapped"):
        source, output = tmp_path/mapping, tmp_path/(mapping+"-routes")
        source.mkdir()
        net, core, neurons, synapses, groups, edges = preparation(source)
        features = []
        rng = np.random.default_rng(9)
        for clip in (0, 1):
            f = {"ticks": 40, "visual": rng.random((40, len(groups["vision"]))),
                 "auditory": rng.random((40, len(groups["touch"])))}
            features.append(f)
            np.savez_compressed(source/f"sensory-{clip}.npz", **f)
        for sense in ("visual", "audio"):
            for clip in (0, 1):
                branch, driver, members, syns = fresh(source/"config.json", 11, BoundedPlasticityNeuron)
                trial = {"start": 0, "stop": 40,
                         "visual_clip": clip if sense == "visual" else None,
                         "audio_clip": clip if sense == "audio" else None}
                cells = episode(branch, driver, members, features, groups, trial)
                np.savez_compressed(source/f"probe-initial-{sense}-{clip}.npz", cells=cells,
                                    incoming_info_after=weight_values(syns))
        trials = pairing_protocol(40, 1, 11, mapping)
        manifest = {"seed": 11, "mapping": mapping, "weight_dynamics": "bounded", "source_hashes": {},
                    "groups": groups, "edges": edges, "trials": trials, "clip_ticks": 40, "fields": FIELDS}
        (source/"manifest.json").write_text(encode(manifest))
        initial = weight_values(synapses)
        observer = WeightObserver(neurons, synapses)
        for i, trial in enumerate(trials):
            observer.rows = []
            cells = episode(net, core, neurons, features, groups, trial, observer)
            np.savez_compressed(source/f"experience-{i:03d}.npz", cells=cells,
                                incoming_info_after=weight_values(synapses), weight_health=np.array(observer.rows))
        np.savez_compressed(source/"parameters.npz", initial_info=initial, learned_info=weight_values(synapses))
        with gzip.open(source/"training-final-state.json.gz", "wt") as f:
            f.write(encode(snapshot(net)))
        run(source, output)
        paths.append(output)
    result = audit(*paths, tmp_path/"audit")
    assert result["structurally_valid"]
    with np.load(tmp_path/"audit"/"tick-effects.npz") as raw:
        assert all(v.shape == (40,) for v in raw.values())
    summary_path = paths[0]/"summary.json"
    summary = json.loads(summary_path.read_text())
    next(p for p in summary["probes"] if p["condition"] == "direct_cut")["removed_edges"].pop()
    summary_path.write_text(encode(summary))
    with pytest.raises(ValueError, match="anatomical route"):
        audit(*paths, tmp_path/"invalid-audit")
