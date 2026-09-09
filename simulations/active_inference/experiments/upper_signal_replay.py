"""Counterfactual upper->auditory delivery, separate from the cognitive model.

Capture native float32 input rows after network delivery and before the bounded
neuron's local computation. Replacing those rows with the unchanged recording
must reproduce the intact trajectory and complete final snapshot exactly.
Other replacements break the natural feedback loop at these named ports only.
The topology, neuron rules, local plasticity and all other inputs stay live.
No target, label, decoded content or optimizer enters the neural model.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .association_route_probe import digest
from .composition_probe import encode, snapshot
from .multimodal_pairing_probe import fresh, episode, WeightObserver
from .population_hierarchy import weight_values
from .population_state_branch import branch_network, TickDriver
from neuron.extensions.experimental.bounded_plasticity import BoundedPlasticityNeuron

MODES = ("exact", "time_shuffle", "port_swap", "other_video")
WINDOW = 32


def upper_ports(net, groups):
    sources, targets = set(groups["upper_core"]), set(groups["tactile_core"])
    result = sorted((tgt, sid, src, term) for src, term, tgt, sid in net.network.connections
                    if src in sources and tgt in targets)
    if not result or len({r[:2] for r in result}) != len(result):
        raise ValueError("Require unique upper-to-auditory input ports")
    return result


class PortReplay:
    """Single-process diagnostic hook. Never install in a live shared server."""
    def __init__(self, net, ports, ticks, replacement=None):
        self.net, self.ports, self.begin = net, ports, net.current_tick
        self.data = np.zeros((ticks, len(ports), 4), dtype=np.float32)
        self.seen = np.zeros((ticks, len(ports)), dtype=bool)
        self.replacement = replacement
        if replacement is not None and (replacement.shape != self.data.shape or replacement.dtype != self.data.dtype
                                         or not np.isfinite(replacement).all()):
            raise ValueError("Replacement must match finite native input rows")
        self.by_neuron = {}
        for i, (nid, sid, _, _) in enumerate(ports):
            self.by_neuron.setdefault(nid, []).append((i, sid))

    @contextmanager
    def installed(self):
        original = BoundedPlasticityNeuron.tick
        probe = self

        def tick(neuron, external_inputs, current_tick, dt=1.):
            rows = probe.by_neuron.get(neuron.id)
            if rows and probe.net.network.neurons[neuron.id] is neuron:
                t = current_tick-probe.begin
                if not 0 <= t < len(probe.data):
                    raise ValueError("Delivery probe outside declared interval")
                for index, sid in rows:
                    if sid in external_inputs or probe.seen[t, index]:
                        raise AssertionError("External overlap or duplicate port observation")
                    probe.data[t, index] = neuron.input_buffer[sid]
                    probe.seen[t, index] = True
                    if probe.replacement is not None:
                        neuron.input_buffer[sid] = probe.replacement[t, index]
            return original(neuron, external_inputs, current_tick, dt)

        BoundedPlasticityNeuron.tick = tick
        try:
            yield self
        finally:
            BoundedPlasticityNeuron.tick = original

    def verify(self):
        if not self.seen.all():
            raise AssertionError("Missing port/tick observations")


def replacement(stream, other, ports, mode, seed, window=WINDOW):
    """Return the manipulated raw input stream and its explicit index maps.

    Time shuffle applies one time permutation to all ports within each window:
    per-port sample multisets and instantaneous joint input patterns survive.
    Port swap exchanges the two upper afferents of each target: raw summed
    input per target/tick survives, but weighted/delayed currents need not.
    Other-video substitution does not preserve dose and is labelled as such.
    """
    times, columns = np.arange(len(stream)), np.arange(len(ports))
    if mode == "time_shuffle":
        rng = np.random.default_rng(seed+9701)
        for start in range(0, len(times), window):
            times[start:start+window] = rng.permutation(times[start:start+window])
    elif mode == "port_swap":
        for nid in sorted({p[0] for p in ports}):
            indexes = [i for i, p in enumerate(ports) if p[0] == nid]
            if len(indexes) != 2:
                raise ValueError("Port-swap control requires exactly two upper afferents per target")
            columns[indexes] = indexes[::-1]
    elif mode not in ("exact", "other_video"):
        raise ValueError(mode)
    origin = other if mode == "other_video" else stream
    result = origin[times][:, columns].copy()
    return result, {"time_index": times.tolist(), "port_index": columns.tolist(), "window": window}


def run(routes, output):
    routes, output = Path(routes).resolve(), Path(output).resolve()
    route_summary = json.loads((routes/"summary.json").read_text())
    source = Path(route_summary["source_recording"])
    manifest = json.loads((source/"manifest.json").read_text())
    if manifest.get("weight_dynamics") != "bounded" or not route_summary["full_training_snapshot_exact"]:
        raise ValueError("Use verified bounded route-probe results")
    if shutil.disk_usage(output.parent).free < 750*1024**2:
        raise OSError("Less than 750 MiB free; do not start another recording")
    hashes = dict(route_summary["source_hashes"])
    hashes[str(Path(__file__).resolve())] = digest(__file__)
    for name, expected in hashes.items():
        if digest(name) != expected:
            raise ValueError(f"Runtime changed: {name}")
    for name, expected in route_summary["source_files_sha256"].items():
        if digest(source/name) != expected:
            raise ValueError(f"Source record changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    features = []
    for clip in (0, 1):
        with np.load(source/f"sensory-{clip}.npz") as raw:
            features.append({k: raw[k] for k in raw.files})
    config, seed, groups = source/"config.json", manifest["seed"], manifest["groups"]
    net, core, members, synapses = fresh(config, seed, BoundedPlasticityNeuron)
    initial = branch_network(net, "continuation", config, seed)
    health = WeightObserver(members, synapses)
    started = time.perf_counter()
    for i, trial in enumerate(manifest["trials"]):
        health.rows = []
        cells = episode(net, core, members, features, groups, trial, health)
        with np.load(source/f"experience-{i:03d}.npz") as ref:
            if not (np.array_equal(cells, ref["cells"]) and np.array_equal(weight_values(synapses), ref["incoming_info_after"])
                    and np.array_equal(health.rows, ref["weight_health"])):
                raise AssertionError(f"Training replay diverged in episode {i}")
        print(encode({"stage": "verified_training", "episode": i, "seconds": round(time.perf_counter()-started, 2)}), flush=True)
    with gzip.open(source/"training-final-state.json.gz", "rt") as stream:
        if encode(snapshot(net)) != encode(json.load(stream)):
            raise AssertionError("Trained snapshot differs")
    ports, probes = upper_ports(net, groups), []
    for state, parent in (("initial", initial), ("trained", net)):
        parent_state = encode(snapshot(parent))
        native, final = {}, {}
        # First observe native delivery, verifying it against the earlier route
        # experiment. Baseline cellular records are referenced, not duplicated.
        for clip in (0, 1):
            branch = branch_network(parent, "continuation", config, seed)
            neurons = list(branch.network.neurons.values())
            syns = [p for n in neurons for p in n.postsynaptic_points.values()]
            trial = {"start": branch.current_tick, "stop": branch.current_tick+manifest["clip_ticks"], "visual_clip": clip, "audio_clip": None}
            observer = PortReplay(branch, ports, manifest["clip_ticks"])
            with observer.installed():
                cells = episode(branch, TickDriver(branch), neurons, features, groups, trial)
            observer.verify()
            ref_path = routes/f"{state}-intact-visual-{clip}.npz"
            row = next(r for r in route_summary["probes"] if r["state"] == state and r["condition"] == "intact" and r["clip"] == clip)
            if digest(ref_path) != row["sha256"]:
                raise ValueError("Intact reference changed")
            with np.load(ref_path) as ref:
                if not np.array_equal(cells, ref["cells"]) or not np.array_equal(weight_values(syns), ref["incoming_info_after"]):
                    raise AssertionError("Native capture changes the original trajectory")
            native[clip], final[clip] = observer.data, encode(snapshot(branch))
            np.savez_compressed(output/f"{state}-native-{clip}.npz", delivered=observer.data)
            del branch, neurons, syns, observer, cells
        for mode in MODES:
            for clip in (0, 1):
                branch = branch_network(parent, "continuation", config, seed)
                neurons = list(branch.network.neurons.values())
                syns = [p for n in neurons for p in n.postsynaptic_points.values()]
                before = weight_values(syns)
                signals, transform = replacement(native[clip], native[1-clip], ports, mode, seed)
                observer = PortReplay(branch, ports, manifest["clip_ticks"], signals)
                health = WeightObserver(neurons, syns)
                begin = branch.current_tick
                trial = {"start": begin, "stop": begin+manifest["clip_ticks"], "visual_clip": clip, "audio_clip": None}
                with observer.installed():
                    cells = episode(branch, TickDriver(branch), neurons, features, groups, trial, health)
                observer.verify()
                exact = None
                if mode == "exact":
                    with np.load(routes/f"{state}-intact-visual-{clip}.npz") as ref:
                        exact = np.array_equal(cells, ref["cells"]) and np.array_equal(weight_values(syns), ref["incoming_info_after"])
                    exact = exact and encode(snapshot(branch)) == final[clip]
                    if not exact:
                        raise AssertionError("Unchanged input replacement does not preserve complete final state")
                if encode(snapshot(parent)) != parent_state:
                    raise AssertionError("Branch changed parent")
                name = f"{state}-{mode}-{clip}.npz"
                np.savez_compressed(output/name, cells=cells, original_delivery=observer.data,
                    replacement_delivery=signals, incoming_info_before=before,
                    incoming_info_after=weight_values(syns), weight_health=np.array(health.rows))
                probes.append({"state": state, "mode": mode, "clip": clip, "transform": transform,
                    "exact_replacement_verified": exact, "parent_unchanged": True,
                    "file": name, "sha256": digest(output/name), "start_tick": begin})
                print(encode({"stage": "probe", "state": state, "mode": mode, "clip": clip,
                              "seconds": round(time.perf_counter()-started, 2)}), flush=True)
                del branch, neurons, syns, observer, health, cells
    if any(digest(name) != value for name, value in hashes.items()):
        raise ValueError("Runtime changed during experiment")
    result = {"routes": str(routes), "source_recording": str(source), "seed": seed,
        "mapping": manifest["mapping"], "ports": ports, "source_hashes": hashes,
        "route_summary_sha256": digest(routes/"summary.json"), "probes": probes,
        "training_replay_exact": True, "native_capture_exact": True,
        "ticks": manifest["trials"][-1]["stop"]+20*manifest["clip_ticks"],
        "seconds": time.perf_counter()-started,
        "limits": "Input-clamp interventions, not neural control. Local learning and all other paths remain live; upper-to-auditory natural feedback is replaced. Time shuffle preserves per-port samples within 32-tick windows, not timing; port swap preserves each target's raw summed input, not weighted/delayed current; other-video substitution does not preserve dose. Sensitivity to these changes alone is not semantic memory or consciousness."}
    (output/"summary.json").write_text(encode(result)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.routes, args.output)
