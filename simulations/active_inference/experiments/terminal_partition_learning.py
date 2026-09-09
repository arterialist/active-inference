"""Controlled terminal-sharing experiment using the recorded audiovisual task.

Reuse immutable transduction and trial order from a bounded learning recording.
Only the build-time terminal partition differs. Initial, incoming-info-only and
intact trained-state probes are separately named. All branches stay plastic.
The old runner remains unchanged so its source-checked controls can be replayed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from ..components.learning import projection_terminals as terminals_module
from .composition_probe import encode, snapshot
from . import population_state_branch as branching
from .multimodal_pairing_probe import fresh, episode, WeightObserver
from .population_hierarchy import weight_values
from neuron.extensions.experimental.bounded_plasticity import BoundedPlasticityNeuron


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(source, output, mode="family"):
    source, output = Path(source).resolve(), Path(output).resolve()
    source_manifest = json.loads((source/"manifest.json").read_text())
    if source_manifest.get("weight_dynamics") != "bounded" or source_manifest.get("architecture") != "regional":
        raise ValueError("Use the bounded regional-control recording as the source")
    if shutil.disk_usage(output.parent).free < 600*1024**2:
        raise OSError("Less than 600 MiB free: do not start another recording")
    for name, expected in source_manifest["source_hashes"].items():
        if digest(name) != expected:
            raise ValueError(f"Control runtime changed: {name}")
    config = json.loads((source/"config.json").read_text())
    config, terminal_report = terminals_module.projection_terminals(
        config, source_manifest["edges"], mode=mode, seed=source_manifest["seed"])
    output.mkdir(parents=True, exist_ok=False)
    config_path = output/"config.json"
    config_path.write_text(encode(config)+"\n")
    features = []
    for i in (0, 1):
        with np.load(source/f"sensory-{i}.npz") as raw:
            features.append({k: raw[k] for k in raw.files})
    sources = dict(source_manifest["source_hashes"])
    for p in (Path(__file__).resolve(), Path(terminals_module.__file__), Path(branching.__file__)):
        sources[str(p)] = digest(p)
    manifest = {**source_manifest, "source_recording": str(source), "source_hashes": sources,
        "terminal_partition": terminal_report,
        "source_files_sha256": {name: digest(source/name) for name in
                                ("manifest.json", "config.json", "sensory-0.npz", "sensory-1.npz")},
        "probe": "Initial and incoming_info probes have fresh activity. Continuation retains the complete trained neural state. All rates stay positive. Incoming_info probes retain only incoming information weights; never label them full learned state.",
        "recording": "All eight cellular fields every tick as float64, per-tick weight health, incoming weights at episode boundaries, complete final training snapshot. Sensory records referenced by path/hash, not duplicated."}
    (output/"manifest.json").write_text(encode(manifest)+"\n")
    net, core, neurons, synapses = fresh(config_path, manifest["seed"], BoundedPlasticityNeuron)
    initial = weight_values(synapses)
    observer = WeightObserver(neurons, synapses)
    started = time.perf_counter()
    for i, trial in enumerate(manifest["trials"]):
        before = weight_values(synapses)
        observer.rows = []
        cells = episode(net, core, neurons, features, manifest["groups"], trial, observer)
        np.savez_compressed(output/f"experience-{i:03d}.npz", cells=cells,
                            incoming_info_before=before, incoming_info_after=weight_values(synapses),
                            weight_health=np.array(observer.rows))
        print(encode({"stage": "training", "episode": i, "seconds": round(time.perf_counter()-started, 2)}), flush=True)
    learned = weight_values(synapses)
    np.savez_compressed(output/"parameters.npz", initial_info=initial, learned_info=learned)
    trained_state = encode(snapshot(net))
    with gzip.open(output/"training-final-state.json.gz", "wt") as stream:
        stream.write(trained_state+"\n")
    probes = []
    # Full-state readout is primary. The incoming-only visual probes retain the
    # previous memory intervention, but are not a replacement for that readout.
    for condition, senses in (("continuation", ("visual", "audio")),
                              ("initial", ("visual", "audio")),
                              ("incoming_info", ("visual",))):
        for sense in senses:
            for clip in (0, 1):
                if condition == "continuation":
                    branch = branching.branch_network(net, condition, config_path, manifest["seed"])
                    members = list(branch.network.neurons.values())
                    syns = [p for n in members for p in n.postsynaptic_points.values()]
                    if encode(snapshot(branch)) != trained_state:
                        raise AssertionError("Bad continuation branch")
                else:
                    branch, _, members, syns = fresh(config_path, manifest["seed"], BoundedPlasticityNeuron)
                    if condition == "incoming_info":
                        for point, value in zip(syns, learned):
                            point.u_i.info = float(value)
                begin = branch.current_tick
                trial = {"start": begin, "stop": begin+manifest["clip_ticks"],
                         "visual_clip": clip if sense == "visual" else None,
                         "audio_clip": clip if sense == "audio" else None}
                health = WeightObserver(members, syns)
                before = weight_values(syns)
                start_digest = hashlib.sha256(encode(snapshot(branch)).encode()).hexdigest()
                cells = episode(branch, branching.TickDriver(branch), members, features, manifest["groups"], trial, health)
                np.savez_compressed(output/f"probe-{condition}-{sense}-{clip}.npz", cells=cells,
                                    incoming_info_before=before, incoming_info_after=weight_values(syns),
                                    weight_health=np.array(health.rows))
                if encode(snapshot(net)) != trained_state:
                    raise AssertionError("A probe mutated the trained parent")
                probes.append({"condition": condition, "sense": sense, "clip": clip,
                               "start_tick": begin, "start_state_sha256": start_digest,
                               "parent_unchanged": True})
                print(encode({"stage": "probe", **probes[-1], "seconds": round(time.perf_counter()-started, 2)}), flush=True)
                del branch, members, syns, health, cells
    if any(digest(p) != expected for p, expected in sources.items()):
        raise RuntimeError("Runtime changed during the experiment")
    result = {"seed": manifest["seed"], "mapping": manifest["mapping"], "mode": mode,
              "probes": probes, "trained_state_sha256": hashlib.sha256(trained_state.encode()).hexdigest(),
              "ticks": manifest["trials"][-1]["stop"]+10*manifest["clip_ticks"],
              "seconds": time.perf_counter()-started,
              "limits": "A terminal-sharing intervention, not an embodied acceptance test. Two clips, no category generalization. Inspect full-state recall across assignments, controls, seeds and time windows; bounds and readout magnitude are insufficient."}
    (output/"summary.json").write_text(encode(result)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("shared", "family", "shuffled"), default="family")
    args = parser.parse_args()
    run(args.source, args.output, args.mode)
