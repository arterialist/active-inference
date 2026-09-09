"""Post-experience pathway interventions, not a new recall controller.

Replay a bounded population recording exactly. Branch its complete trained
state, or the matched initial state, and remove direct visual->auditory and/or
upper->auditory connections. Present the same silent video to every branch.
Keep all neurons, postsynaptic ports, parameters and positive learning rates.
The intervention removes forward delivery and its corresponding retrograde
route, not weights. Other recurrent and modulatory paths remain functional.

This measures pathway dependence during cue-driven activity. It does not by
itself identify a stored association, prove hierarchical content, or require
persistence after cue withdrawal. Each branch has its own ongoing plasticity.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .composition_probe import encode, snapshot
from .multimodal_pairing_probe import fresh, episode, WeightObserver
from .population_hierarchy import weight_values
from .population_state_branch import branch_network, TickDriver
from neuron.extensions.experimental.bounded_plasticity import BoundedPlasticityNeuron

CONDITIONS = ("intact", "direct_cut", "descending_cut", "both_cut")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cut_routes(net, groups, condition):
    """Surgical edge intervention at a quiescent event boundary.

    Reject in-flight signals instead of quietly discarding pre-intervention
    history. Rebuild BOTH connection caches and remove matching retrograde
    source mappings. Keep unconnected input ports and all their local state.
    """
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if (any(net.presynaptic_wheel) or any(net.retrograde_wheel) or
            any(n.propagation_queue or n.input_buffer.any()
                for n in net.network.neurons.values())):
        raise ValueError("Route intervention requires empty event queues and buffers")
    sources = set()
    if condition in ("direct_cut", "both_cut"):
        sources.update(groups["visual_core"])
    if condition in ("descending_cut", "both_cut"):
        sources.update(groups["upper_core"])
    targets = set(groups["tactile_core"])
    topo = net.network
    removed = [tuple(e) for e in topo.connections if e[0] in sources and e[2] in targets]
    if condition != "intact" and not removed:
        raise ValueError("Requested route is absent")
    if not removed:
        return []
    before = encode(snapshot(net))
    removed_set = set(removed)
    topo.connections = [e for e in topo.connections if tuple(e) not in removed_set]
    topo.connection_cache = defaultdict(list)
    topo.fast_connection_cache = defaultdict(list)
    for src, term, tgt, sid in removed:
        if topo.neurons[tgt].synapse_sources.pop(sid) != (src, term):
            raise AssertionError("Retrograde source does not match removed edge")
    for src, term, tgt, sid in topo.connections:
        topo.connection_cache[src, term].append((tgt, sid))
        topo.fast_connection_cache[src, term].append((topo.neurons[tgt].input_buffer, sid))
    # snapshot excludes topology. All recorded local state must stay identical.
    if encode(snapshot(net)) != before:
        raise AssertionError("Route intervention changed local neural state")
    return removed


def run(recording, output):
    recording, output = Path(recording).resolve(), Path(output).resolve()
    manifest = json.loads((recording/"manifest.json").read_text())
    if manifest.get("weight_dynamics") != "bounded":
        raise ValueError("Use a bounded population source recording")
    if "terminal_partition" in manifest:
        raise ValueError("This experiment uses the original shared-terminal control")
    if shutil.disk_usage(output.parent).free < 650*1024**2:
        raise OSError("Less than 650 MiB free; do not start another recording")
    sources = dict(manifest["source_hashes"])
    for module in (Path(__file__), Path(__file__).with_name("population_state_branch.py")):
        sources[str(module.resolve())] = digest(module)
    for name, expected in sources.items():
        if digest(name) != expected:
            raise ValueError(f"Runtime changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    input_files = ["manifest.json", "config.json", "parameters.npz", "training-final-state.json.gz",
                   "sensory-0.npz", "sensory-1.npz"]
    input_files += [f"experience-{i:03d}.npz" for i in range(len(manifest["trials"]))]
    source_files = {name: digest(recording/name) for name in input_files}
    features = []
    for clip in (0, 1):
        with np.load(recording/f"sensory-{clip}.npz") as raw:
            features.append({key: raw[key] for key in raw.files})
    config, seed = recording/"config.json", manifest["seed"]
    net, core, neurons, synapses = fresh(config, seed, BoundedPlasticityNeuron)
    initial = branch_network(net, "continuation", config, seed)
    observer = WeightObserver(neurons, synapses)
    started, checks = time.perf_counter(), []
    for i, trial in enumerate(manifest["trials"]):
        observer.rows = []
        cells = episode(net, core, neurons, features, manifest["groups"], trial, observer)
        with np.load(recording/f"experience-{i:03d}.npz") as reference:
            exact = (np.array_equal(cells, reference["cells"]) and
                     np.array_equal(weight_values(synapses), reference["incoming_info_after"]) and
                     np.array_equal(observer.rows, reference["weight_health"]))
        if not exact:
            raise AssertionError(f"Training replay diverged at episode {i}")
        checks.append(i)
        print(encode({"stage": "verified_training", "episode": i,
                      "seconds": round(time.perf_counter()-started, 2)}), flush=True)
    with gzip.open(recording/"training-final-state.json.gz", "rt") as stream:
        if encode(snapshot(net)) != encode(json.load(stream)):
            raise AssertionError("Full training state differs")
    rows = []
    for state, parent in (("initial", initial), ("trained", net)):
        parent_state = encode(snapshot(parent))
        parent_hash = hashlib.sha256(parent_state.encode()).hexdigest()
        for condition in CONDITIONS:
            for clip in (0, 1):
                branch = branch_network(parent, "continuation", config, seed)
                removed = cut_routes(branch, manifest["groups"], condition)
                members = list(branch.network.neurons.values())
                syns = [p for n in members for p in n.postsynaptic_points.values()]
                before = weight_values(syns)
                health = WeightObserver(members, syns)
                begin = branch.current_tick
                trial = {"start": begin, "stop": begin+manifest["clip_ticks"],
                         "visual_clip": clip, "audio_clip": None}
                cells = episode(branch, TickDriver(branch), members, features,
                                manifest["groups"], trial, health)
                name = f"{state}-{condition}-visual-{clip}.npz"
                np.savez_compressed(output/name, cells=cells, incoming_info_before=before,
                    incoming_info_after=weight_values(syns), weight_health=np.array(health.rows))
                if encode(snapshot(parent)) != parent_state:
                    raise AssertionError("Probe mutated parent")
                row = {"state": state, "condition": condition, "clip": clip,
                       "start_tick": begin, "start_local_state_sha256": parent_hash,
                       "parent_unchanged": True, "removed_edges": removed,
                       "file": name, "sha256": digest(output/name)}
                rows.append(row)
                print(encode({"stage": "probe", "state": state, "condition": condition,
                              "clip": clip, "seconds": round(time.perf_counter()-started, 2)}), flush=True)
                del branch, members, syns, health, cells
    for name, expected in sources.items():
        if digest(name) != expected:
            raise ValueError(f"Runtime changed during experiment: {name}")
    for name, expected in source_files.items():
        if digest(recording/name) != expected:
            raise ValueError(f"Recording changed during experiment: {name}")
    result = {"source_recording": str(recording), "seed": seed, "mapping": manifest["mapping"],
              "source_hashes": sources, "source_files_sha256": source_files,
              "training_exact_episodes": checks, "full_training_snapshot_exact": True,
              "probes": rows, "seconds": time.perf_counter()-started,
              "ticks": manifest["trials"][-1]["stop"]+16*manifest["clip_ticks"],
              "limits": "Route dependence, not a memory verdict. Cuts happen after learning, preserve local state and ports, and remove forward connections plus their retrograde registration. All surviving adaptation remains active. Other feedback and modulation remain. Full video cue stays present. No isolated upper-content test, spontaneous replay, embodiment or category generalization."}
    (output/"summary.json").write_text(encode(result)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.recording, args.output)
