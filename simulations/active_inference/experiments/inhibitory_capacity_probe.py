"""Test one declared inhibitory-capacity intervention on real audiovisual input.

No parameter search. Train from the original config or its upper-inhibition
capacity-matched copy. Keep complete-state branches, positive plasticity and
the source sensory inputs/trial order. Record active learning credit online.
The control must exactly reproduce existing training records and final state.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
import time

import numpy as np

from ..components.learning import inhibitory_capacity as capacity
from . import association_credit_probe as credit
from . import population_state_branch as branching
from .association_route_probe import digest
from .composition_probe import encode, snapshot
from .multimodal_pairing_probe import fresh, episode, WeightObserver
from .population_hierarchy import weight_values
from neuron.extensions.experimental.bounded_plasticity import BoundedPlasticityNeuron


def run(source, output, condition="balanced"):
    if condition not in ("control", "balanced"):
        raise ValueError(condition)
    source, output = Path(source).resolve(), Path(output).resolve()
    m = json.loads((source/"manifest.json").read_text())
    if m.get("weight_dynamics") != "bounded" or m.get("architecture") != "regional":
        raise ValueError("Requires the bounded regional recording")
    if shutil.disk_usage(output.parent).free < 1500*1024**2:
        raise OSError("Need 1.5 GiB free before starting")
    hashes = dict(m["source_hashes"])
    for p in (Path(__file__), Path(capacity.__file__), Path(credit.__file__), Path(branching.__file__)):
        hashes[str(p.resolve())] = digest(p)
    if any(digest(p) != value for p, value in hashes.items()):
        raise ValueError("Runtime source changed")
    config = json.loads((source/"config.json").read_text())
    report = {"condition": "control", "targets": []}
    if condition == "balanced":
        config, report = capacity.match_inhibitory_capacity(config, m["groups"]["upper_core"])
    output.mkdir(parents=True, exist_ok=False)
    cfg = output/"config.json"
    cfg.write_text(encode(config)+"\n")
    ports = credit.selected_ports(m)
    manifest = {**m, "condition": condition, "source_recording": str(source),
        "source_hashes": hashes, "capacity_intervention": report, "credit_ports": ports,
        "source_files_sha256": {name: digest(source/name) for name in
                                ("manifest.json", "config.json", "sensory-0.npz", "sensory-1.npz", "training-final-state.json.gz")},
        "probe": "Initial fresh-state and complete trained-state branches. All learning remains positive. No incoming-only substitution.",
        "scope": "A declared capacity intervention, not a memory acceptance test. Credit and distinct neural responses are necessary diagnostics, not sufficient proof of recall."}
    (output/"manifest.json").write_text(encode(manifest)+"\n")
    features = []
    for clip in (0, 1):
        with np.load(source/f"sensory-{clip}.npz") as z:
            features.append({key: z[key] for key in z.files})
    net, core, neurons, syns = fresh(cfg, m["seed"], BoundedPlasticityNeuron)
    health = WeightObserver(neurons, syns)
    initial = weight_values(syns)
    started, episodes = time.perf_counter(), []
    for i, trial in enumerate(m["trials"]):
        before = weight_values(syns)
        health.rows = []
        recorder = credit.CreditRecorder(net, ports, trial["stop"]-trial["start"])
        with recorder.observe():
            cells = episode(net, core, neurons, features, m["groups"], trial, health)
        after = weight_values(syns)
        if condition == "control":
            with np.load(source/f"experience-{i:03d}.npz") as z:
                if not (np.array_equal(cells, z["cells"]) and np.array_equal(after, z["incoming_info_after"]) and
                        np.array_equal(health.rows, z["weight_health"])):
                    raise AssertionError("Control replay differs")
        name = f"experience-{i:03d}.npz"
        np.savez_compressed(output/name, cells=cells, incoming_info_before=before,
                            incoming_info_after=after, weight_health=np.array(health.rows))
        event_name = f"credit-{i:03d}.npz"
        np.savez_compressed(output/event_name, events=recorder.events())
        episodes.append({"episode": i, "events": recorder.count, "file": name,
                         "sha256": digest(output/name), "credit_file": event_name,
                         "credit_sha256": digest(output/event_name)})
        print(encode({"episode": i, "events": recorder.count, "seconds": round(time.perf_counter()-started, 2)}), flush=True)
        del cells, recorder
    learned = weight_values(syns)
    np.savez_compressed(output/"parameters.npz", initial_info=initial, learned_info=learned)
    parent = encode(snapshot(net))
    if condition == "control":
        with gzip.open(source/"training-final-state.json.gz", "rt") as stream:
            if parent != encode(json.load(stream)):
                raise AssertionError("Control final state differs")
    with gzip.open(output/"training-final-state.json.gz", "wt") as stream:
        stream.write(parent+"\n")
    probes = []
    for state in ("initial", "continuation"):
        for sense in ("visual", "audio"):
            for clip in (0, 1):
                if state == "initial":
                    branch, _, members, points = fresh(cfg, m["seed"], BoundedPlasticityNeuron)
                else:
                    branch = branching.branch_network(net, "continuation", cfg, m["seed"])
                    members = list(branch.network.neurons.values())
                    points = [p for n in members for p in n.postsynaptic_points.values()]
                    if encode(snapshot(branch)) != parent:
                        raise AssertionError("Continuation differs from parent")
                t = branch.current_tick
                trial = {"start": t, "stop": t+m["clip_ticks"], "visual_clip": clip if sense == "visual" else None,
                         "audio_clip": clip if sense == "audio" else None}
                observer = WeightObserver(members, points)
                before = weight_values(points)
                cells = episode(branch, branching.TickDriver(branch), members, features, m["groups"], trial, observer)
                name = f"probe-{state}-{sense}-{clip}.npz"
                np.savez_compressed(output/name, cells=cells, incoming_info_before=before,
                                    incoming_info_after=weight_values(points), weight_health=np.array(observer.rows))
                if encode(snapshot(net)) != parent:
                    raise AssertionError("Probe mutated trained parent")
                probes.append({"condition": state, "sense": sense, "clip": clip, "start_tick": t,
                               "file": name, "sha256": digest(output/name), "parent_unchanged": True})
                print(encode({"probe": name, "seconds": round(time.perf_counter()-started, 2)}), flush=True)
                del branch, members, points, observer, cells
    if any(digest(p) != value for p, value in hashes.items()):
        raise ValueError("Runtime changed during experiment")
    result = {"condition": condition, "seed": m["seed"], "mapping": m["mapping"],
              "episodes": episodes, "probes": probes, "seconds": time.perf_counter()-started,
              "control_replay_exact": True if condition == "control" else None,
              "ticks": m["trials"][-1]["stop"]+8*m["clip_ticks"]}
    (output/"summary.json").write_text(encode(result)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--condition", choices=("control", "balanced"), default="balanced")
    args = p.parse_args()
    run(args.source, args.output, args.condition)
