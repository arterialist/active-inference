"""Counterfactual recall from a trained PAULA network, without a decoder.

Reconstruct the live training network by exact replay rather than treating a
JSON snapshot as a complete portable checkpoint. Compare intact continuation
with a fresh activity state carrying ALL learned synaptic/terminal variables.
The existing incoming-info-only probes remain a third, separately named case.
Every branch stays plastic. No semantic input, policy or clock is added.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .composition_probe import encode, snapshot, network_module
from .multimodal_pairing_probe import fresh, episode, WeightObserver
from .population_hierarchy import weight_values
from neuron.extensions.experimental.bounded_plasticity import BoundedPlasticityNeuron


def branch_network(trained, condition, config_path, seed):
    """Only these known deterministic cell classes are supported by this probe.

    Deepcopy preserves numeric scalar types and shared buffer references inside
    a branch. Config-based fresh construction supplies all constructor defaults.
    The clone never shares mutable neural state with its parent. RNG is not a
    neural state here: cleft delay is fixed and these ticks draw no other noise.
    This is an intervention tool, not part of the agent's runtime controller.
    """
    if condition not in ("continuation", "all_synaptic"):
        raise ValueError(condition)
    if not (network_module.MIN_CONNECTION_SIGNAL_TRAVEL_TICKS ==
            network_module.MAX_CONNECTION_SIGNAL_TRAVEL_TICKS == 1):
        raise ValueError("Stochastic delays need explicit per-branch RNG state")
    if any(type(n) is not BoundedPlasticityNeuron for n in trained.network.neurons.values()):
        raise ValueError("This state partition has only been specified for the bounded population model")
    if condition == "continuation":
        return deepcopy(trained)
    result, _, neurons, _ = fresh(config_path, seed, BoundedPlasticityNeuron)
    for n in neurons:
        source = trained.network.neurons[n.id]
        if set(n.postsynaptic_points) != set(source.postsynaptic_points) or set(n.presynaptic_points) != set(source.presynaptic_points):
            raise ValueError("Synapse/terminal indexing differs")
        for sid, point in n.postsynaptic_points.items():
            point.u_i = deepcopy(source.postsynaptic_points[sid].u_i)
        for tid, point in n.presynaptic_points.items():
            point.u_o = deepcopy(source.presynaptic_points[tid].u_o)
            point.u_i_retro = deepcopy(source.presynaptic_points[tid].u_i_retro)
    return result


class TickDriver:
    """Use the ordinary network tick, with no GUI thread or control policy."""
    def __init__(self, net):
        self.net = net

    def do_tick(self):
        return self.net.run_tick()


def run(recording, output):
    recording, output = Path(recording).resolve(), Path(output).resolve()
    manifest = json.loads((recording/"manifest.json").read_text())
    if manifest.get("weight_dynamics") != "bounded":
        raise ValueError("Use a bounded population recording")
    for name, expected in manifest["source_hashes"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Runtime source changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    features = []
    for i in (0, 1):
        with np.load(recording/f"sensory-{i}.npz") as raw:
            features.append({key: raw[key] for key in raw.files})
    config = recording/"config.json"
    net, core, neurons, synapses = fresh(config, manifest["seed"], BoundedPlasticityNeuron)
    observer = WeightObserver(neurons, synapses)
    started = time.perf_counter()
    checks = []
    # Reconstruct actual numeric types, all ongoing state and queued events.
    for i, trial in enumerate(manifest["trials"]):
        observer.rows = []
        cells = episode(net, core, neurons, features, manifest["groups"], trial, observer)
        with np.load(recording/f"experience-{i:03d}.npz") as reference:
            exact = (np.array_equal(cells, reference["cells"]) and
                     np.array_equal(weight_values(synapses), reference["incoming_info_after"]) and
                     np.array_equal(observer.rows, reference["weight_health"]))
        if not exact:
            raise AssertionError(f"Training replay diverged in episode {i}")
        checks.append({"episode": i, "every_cell_weight_endpoint_and_health_exact": True})
        print(encode({"stage": "verified_training", "episode": i,
                      "seconds": round(time.perf_counter()-started, 2)}), flush=True)
    with gzip.open(recording/"training-final-state.json.gz", "rt") as stream:
        saved = json.load(stream)
    parent_state = encode(snapshot(net))
    if parent_state != encode(saved):
        raise AssertionError("Final recorded full snapshot differs")
    parent_digest = hashlib.sha256(parent_state.encode()).hexdigest()
    branch_rows = []
    for condition in ("continuation", "all_synaptic"):
        for sense in ("visual", "audio"):
            for clip in (0, 1):
                branch = branch_network(net, condition, config, manifest["seed"])
                before_state = snapshot(branch)
                if condition == "continuation" and encode(before_state) != parent_state:
                    raise AssertionError("Continuation does not start at the trained state")
                with gzip.open(output/f"{condition}-{sense}-{clip}-start.json.gz", "wt") as stream:
                    stream.write(encode(before_state)+"\n")
                members = list(branch.network.neurons.values())
                syns = [p for n in members for p in n.postsynaptic_points.values()]
                health = WeightObserver(members, syns)
                begin = branch.current_tick
                trial = {"start": begin, "stop": begin+manifest["clip_ticks"],
                         "visual_clip": clip if sense == "visual" else None,
                         "audio_clip": clip if sense == "audio" else None}
                before = weight_values(syns)
                cells = episode(branch, TickDriver(branch), members, features,
                                manifest["groups"], trial, health)
                np.savez_compressed(output/f"{condition}-{sense}-{clip}.npz", cells=cells,
                    weight_health=np.array(health.rows), incoming_info_before=before,
                    incoming_info_after=weight_values(syns))
                if encode(snapshot(net)) != parent_state:
                    raise AssertionError("A branch mutated its trained parent")
                branch_rows.append({"condition": condition, "sense": sense, "clip": clip,
                                    "start_tick": begin, "parent_unchanged": True})
                print(encode({"stage": "probe", **branch_rows[-1],
                              "seconds": round(time.perf_counter()-started, 2)}), flush=True)
                del branch, members, syns, health, cells
    sources = {str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for name, expected in manifest["source_hashes"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Runtime source changed during replay: {name}")
    files = ("manifest.json", "config.json", "parameters.npz", "training-final-state.json.gz")
    result = {"source_recording": str(recording), "seed": manifest["seed"], "mapping": manifest["mapping"],
              "source_hashes": sources,
              "source_files_sha256": {name: hashlib.sha256((recording/name).read_bytes()).hexdigest() for name in files},
              "training_checks": checks, "full_training_snapshot_exact": True,
              "trained_state_sha256": parent_digest, "branches": branch_rows,
              "seconds": time.perf_counter()-started,
              "ticks": manifest["trials"][-1]["stop"]+8*manifest["clip_ticks"],
              "state_partition": {"continuation": "All actual trained neural state, including queues and numeric types.",
                  "all_synaptic": "All incoming u_i vectors and outgoing u_o/u_i_retro variables, with fresh activity, modulation, queues and synaptic potential.",
                  "incoming_info": "Existing source probes: only incoming u_i.info, fresh everything else."},
              "limits": "A separate branch for each probe, not sequential exposure. Training ends after a 96-tick withdrawal. No body, retention-duration sweep, semantic generalization or consciousness claim. Basal and neurally amplified plasticity remain active in every branch."}
    (output/"summary.json").write_text(encode(result)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.recording, args.output)
