"""Read-only event-level plasticity audit of a recorded PAULA experience.

Capture active inputs before the bounded neuron runs and final local weights
after it returns. This avoids confusing provisional inherited updates with the
effective extension rule. Exact replay against the original cellular records,
all incoming-weight endpoints, health records and final snapshot is mandatory.
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
from neuron.extensions.experimental.bounded_plasticity import BoundedPlasticityNeuron

EVENT = np.dtype([("tick", "<i4"), ("port", "<i4"), ("input", "<f4", (4,)),
                  ("before", "<f8"), ("plast", "<f8"), ("error", "<f8"),
                  ("eta", "<f8"), ("age", "<i4"), ("t_ref", "<f8"),
                  ("direction", "i1"), ("after", "<f8"), ("spike", "u1")])


def selected_ports(manifest):
    auditory, upper = set(manifest["groups"]["tactile_core"]), set(manifest["groups"]["upper_core"])
    return sorted((tgt, sid, src, family) for src, tgt, sid, family, present in manifest["edges"]
        if present and ((tgt in auditory and family in ("crossmodal", "descending")) or
                        (tgt in upper and family == "ascending")))


class CreditRecorder:
    def __init__(self, net, ports, ticks):
        self.net, self.ports = net, ports
        self.rows = np.empty(ticks*len(ports), dtype=EVENT)
        self.count = 0
        self.by_neuron = {}
        for i, (nid, sid, _, _) in enumerate(ports):
            self.by_neuron.setdefault(nid, []).append((i, sid))

    @contextmanager
    def observe(self):
        original = BoundedPlasticityNeuron.tick
        recorder = self

        def tick(n, external_inputs, current_tick, dt=1.):
            ports = recorder.by_neuron.get(n.id)
            if not ports or recorder.net.network.neurons[n.id] is not n:
                return original(n, external_inputs, current_tick, dt)
            if not n._bounded_enabled or n.params.weight_decay_tau != 0:
                raise ValueError("Recorder requires active bounded rule without passive decay")
            active = []
            eta = n.params.eta_post*n.rate_multiplier()
            for index, sid in ports:
                row = n.input_buffer[sid]
                if row[0] <= 0:
                    continue
                p = n.postsynaptic_points[sid]
                error = float(np.linalg.norm(np.array([row[0]-p.u_i.info, row[1]-p.u_i.plast, *row[2:]])))
                active.append((index, sid, row.copy(), float(p.u_i.info), float(p.u_i.plast), error))
            result = original(n, external_inputs, current_tick, dt)
            age = int(current_tick-n.t_last_fire) if np.isfinite(n.t_last_fire) else -1
            direction = 1 if age >= 0 and age <= n.t_ref else -1
            for index, sid, row, before, plast, error in active:
                if recorder.count >= len(recorder.rows):
                    raise AssertionError("Credit recorder capacity exceeded")
                recorder.rows[recorder.count] = (current_tick, index, row, before, plast, error,
                    eta, age, n.t_ref, direction, n.postsynaptic_points[sid].u_i.info, int(n.O > 0))
                recorder.count += 1
            return result

        BoundedPlasticityNeuron.tick = tick
        try:
            yield self
        finally:
            BoundedPlasticityNeuron.tick = original

    def events(self):
        return self.rows[:self.count]


def run(recording, output):
    source, output = Path(recording).resolve(), Path(output).resolve()
    manifest = json.loads((source/"manifest.json").read_text())
    if manifest.get("weight_dynamics") != "bounded":
        raise ValueError("Use a bounded population recording")
    if shutil.disk_usage(output.parent).free < 750*1024**2:
        raise OSError("Less than 750 MiB free")
    hashes = dict(manifest["source_hashes"])
    hashes[str(Path(__file__).resolve())] = digest(__file__)
    for name, expected in hashes.items():
        if digest(name) != expected:
            raise ValueError(f"Runtime changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    ports = selected_ports(manifest)
    features = []
    for clip in (0, 1):
        with np.load(source/f"sensory-{clip}.npz") as raw:
            features.append({key: raw[key] for key in raw.files})
    net, core, neurons, synapses = fresh(source/"config.json", manifest["seed"], BoundedPlasticityNeuron)
    health = WeightObserver(neurons, synapses)
    started, rows = time.perf_counter(), []
    for i, trial in enumerate(manifest["trials"]):
        recorder = CreditRecorder(net, ports, trial["stop"]-trial["start"])
        health.rows = []
        with recorder.observe():
            cells = episode(net, core, neurons, features, manifest["groups"], trial, health)
        reference_path = source/f"experience-{i:03d}.npz"
        with np.load(reference_path) as reference:
            if not (np.array_equal(cells, reference["cells"]) and
                    np.array_equal(weight_values(synapses), reference["incoming_info_after"]) and
                    np.array_equal(health.rows, reference["weight_health"])):
                raise AssertionError(f"Instrumentation changed replay at episode {i}")
        filename = f"credit-{i:03d}.npz"
        np.savez_compressed(output/filename, events=recorder.events())
        rows.append({"episode": i, "file": filename, "events": recorder.count,
                     "sha256": digest(output/filename), "reference_sha256": digest(reference_path)})
        print(encode({"episode": i, "events": recorder.count, "exact_replay": True,
                      "seconds": round(time.perf_counter()-started, 2)}), flush=True)
        del recorder, cells
    with gzip.open(source/"training-final-state.json.gz", "rt") as stream:
        if encode(snapshot(net)) != encode(json.load(stream)):
            raise AssertionError("Full final snapshot differs")
    if any(digest(name) != expected for name, expected in hashes.items()):
        raise ValueError("Runtime changed during replay")
    result = {"source_recording": str(source), "seed": manifest["seed"], "mapping": manifest["mapping"],
              "ports": ports, "source_hashes": hashes, "episodes": rows,
              "full_final_snapshot_exact": True, "seconds": time.perf_counter()-started,
              "ticks": manifest["trials"][-1]["stop"],
              "source_files_sha256": {name: digest(source/name) for name in ("manifest.json", "config.json", "parameters.npz", "training-final-state.json.gz")},
              "limits": "Active postsynaptic updates on three selected routes only. Signed timing eligibility is neuron-wide, but recorded events require an actual positive input. Does not claim synapse-specific causal responsibility for a spike, record every retrograde update, or establish memory. No neural variables were changed by this observer."}
    (output/"summary.json").write_text(encode(result)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.recording, args.output)
