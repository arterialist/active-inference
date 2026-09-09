"""Sparse PAULA population preparation, motivated by the author's notebook.

This is a falsifiable first wiring, not a claim of learned hierarchy. There is
no host decoder in the neural loop. External inputs enter two sensory sheets;
the upper population receives only neural connector outputs. All basal learning
rates are positive. The sole subclass is the existing local rate receptor.

Raw chunks contain every completed tick's cellular observables, all incoming
information weights and outgoing information amplitudes at float64 precision.
They are NOT complete restart snapshots: queues, plastic/adapt components and
retrograde fields are not recorded each tick. Initial config, explicit stimuli,
source hashes and final full snapshot support investigation and exact reruns.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from .composition_probe import encode, fingerprint, k, snapshot
from neuron.extensions.experimental.plasticity_rate import PlasticityRateNeuron


SIZES = {"vision": 96, "touch": 96, "visual_core": 192, "tactile_core": 192,
         "visual_inhibition": 48, "tactile_inhibition": 48, "connector": 64,
         "upper_core": 256, "upper_inhibition": 64, "mismatch_candidate": 64,
         "activity_regulator": 32}
LABELS = {"vision": "Visual input", "touch": "Tactile input",
          "visual_core": "Visual population", "tactile_core": "Tactile population",
          "visual_inhibition": "Local inhibition", "tactile_inhibition": "Local inhibition",
          "connector": "Neural interface · LLGC candidate", "upper_core": "Upper population",
          "upper_inhibition": "Upper inhibition", "mismatch_candidate": "Mismatch candidate",
          "activity_regulator": "Activity regulator"}
FIELDS = ("S", "O", "F_avg", "M0", "M1", "r", "t_ref", "rate_multiplier")
MODES = ("intact", "recurrence_cut", "ascending_cut", "descending_cut", "receptor_cut")
TRIAL_TICKS = 96


def make_config(size=1152, seed=11, mode="intact"):
    if size not in (288, 576, 1152) or mode not in MODES:
        raise ValueError("Use 288, 576 or 1152 neurons and a declared condition")
    rng = np.random.default_rng(seed)
    groups, neurons, synapses, connections, external, edges = {}, [], [], [], [], []
    cursor = 1
    for role, count in SIZES.items():
        groups[role] = list(range(cursor, cursor + count * size // 1152))
        cursor += len(groups[role])
    ports = {nid: 0 for ids in groups.values() for nid in ids}
    for role, ids in groups.items():
        core = role in ("visual_core", "tactile_core", "upper_core")
        sensory = role in ("vision", "touch")
        regulator = role in ("mismatch_candidate", "activity_regulator")
        for nid in ids:
            r = .65 if core else .6
            lam = 1 if sensory else 4 if core else 3
            if role == "activity_regulator":
                r, lam = 1.1, 8
            params = k.neuron(nid, r=r, b=r+.25, c=3, lam=lam,
                eta_post=1e-5 if core else 1e-6, eta_retro=1e-7,
                w_r=[0., .6] if core else None,
                w_b=[0., .6] if core else None,
                w_tref=[-100., 0.] if core else None,
                delta_decay=.99,
                meta={"role": role, "plasticity_rate_boost": 499. if core and mode != "receptor_cut" else 0.,
                      "plasticity_rate_index": 0, "plasticity_rate_half_saturation": .1})
            neurons.append(params)
            terminal = k.term(nid, mod=[.25, 0.] if role == "mismatch_candidate" else
                              [0., .25] if role == "activity_regulator" else [0., 0.])
            if regulator:
                terminal["u_o"]["info"] = 0.
            synapses.append(terminal)
            if sensory:
                synapses.extend([k.syn(nid, 0, 2., adapt=[0., 0.]), k.syn(nid, 1, 0., adapt=[0., 0.])])
                external.append(k.ext(nid, 0))
                ports[nid] = 2

    def project(source, target, fanin, weight, family, adapt=None):
        # Fixed fan-in and delay distribution across scales. No stimulus labels
        # or pattern membership are available to this graph constructor.
        for tgt in groups[target]:
            candidates = [i for i in groups[source] if i != tgt]
            for src in rng.choice(candidates, size=min(fanin, len(candidates)), replace=False):
                src = int(src)
                sid = ports[tgt]
                ports[tgt] += 1
                w = weight * float(rng.uniform(.85, 1.15))
                dist = int(rng.integers(1, 5))
                synapses.append(k.syn(tgt, sid, w, dist, adapt=adapt or [0., 0.]))
                cut = (mode == "recurrence_cut" and family in ("recurrent", "crossmodal")) or \
                      (mode == "ascending_cut" and family == "ascending") or \
                      (mode == "descending_cut" and family == "descending")
                if not cut:
                    connections.append(k.conn(src, tgt, sid))
                edges.append([src, tgt, sid, family, not cut])

    for sensor, own, other, inhib in (("vision", "visual_core", "tactile_core", "visual_inhibition"),
                                     ("touch", "tactile_core", "visual_core", "tactile_inhibition")):
        project(sensor, own, 8, 1.8, "sensory")
        project(own, own, 8, .32, "recurrent")
        project(other, own, 4, .30, "crossmodal")
        project(own, inhib, 8, .8, "inhibitory_drive")
        project(inhib, own, 4, -.8, "inhibition")
        project(own, "connector", 4, 1.0, "interface")
        project("upper_core", own, 2, .25, "descending")
        project(sensor, "mismatch_candidate", 4, .8, "observed_sensation")
        project(own, "mismatch_candidate", 4, -.65, "predicted_suppression")
        project(own, "activity_regulator", 4, .7, "observed_activity")
    project("connector", "upper_core", 12, .9, "ascending")
    project("upper_core", "upper_core", 8, .30, "recurrent")
    project("upper_core", "upper_inhibition", 8, .8, "inhibitory_drive")
    project("upper_inhibition", "upper_core", 4, -.8, "inhibition")
    for target in ("visual_core", "tactile_core", "upper_core"):
        project("mismatch_candidate", target, 2, 0., "plasticity_modulation", [1., 0.])
        project("activity_regulator", target, 2, 0., "excitability_modulation", [0., 1.])
    for n in neurons:
        n["params"]["num_inputs"] = ports[n["id"]]
    config = {"metadata": {"preparation": "population-hierarchy-v0", "seed": seed, "mode": mode},
              "global_params": {"num_neuromodulators": 2, "num_inputs": 1},
              "simulation_params": {"max_history": 1}, "neurons": neurons,
              "synaptic_points": synapses, "connections": connections, "external_inputs": external}
    return config, groups, edges


def protocol(groups, seed, repetitions=6):
    rng = np.random.default_rng(seed + 7919)
    patterns = {}
    # Equal-sized binary feature masks, no semantic labels in neural wiring.
    for role in ("vision", "touch"):
        shuffled = rng.permutation(groups[role])
        n = len(shuffled)//4
        patterns[role] = [sorted(map(int, shuffled[:n])), sorted(map(int, shuffled[n:2*n]))]
    trials = []
    def add(phase, a=None, b=None, partial=False):
        inputs = ([] if a is None else patterns["vision"][a][::2] if partial else patterns["vision"][a]) + \
                 ([] if b is None else patterns["touch"][b])
        trials.append({"index": len(trials), "phase": phase, "visual": a, "tactile": b,
                       "partial": partial, "active_ids": inputs, "start": len(trials)*TRIAL_TICKS,
                       "stop": (len(trials)+1)*TRIAL_TICKS})
    for a in (0, 1):
        add("before_learning", a, partial=True)
    for a in (0, 1):
        add("before_familiar_pair", a, a)
        add("before_conflicting_pair", a, 1-a)
    for _ in range(repetitions):
        for a in rng.permutation(2):
            add("paired_experience", int(a), int(a))
    add("silent_interval")
    add("silent_interval")
    for a in (0, 1):
        add("partial_recall", a, partial=True)
    for a in (0, 1):
        add("familiar_pair", a, a)
        add("conflicting_pair", a, 1-a)
    return patterns, trials


def input_ids(trial, tick):
    rel = tick-trial["start"]
    return trial["active_ids"] if 8 <= rel < 56 and rel % 4 == 0 else []


def cellular(neurons):
    return np.array([[n.S, n.O, n.F_avg, *n.M_vector, n.r, n.t_ref, n.last_tick_rate_multiplier]
                     for n in neurons], dtype=np.float64)


def weight_values(synapses):
    return np.fromiter((p.u_i.info for p in synapses), dtype=np.float64, count=len(synapses))


def finite_cosine(a, b):
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a, b)/norm) if norm > 1e-15 else None


def analyze(cell, trials, groups, weights_before, weights_after):
    spikes = cell[:, :, 1] > 0
    responses = []
    for tr in trials:
        on = spikes[tr["start"]+12:tr["start"]+64].mean(axis=0)
        off = spikes[tr["start"]+72:tr["stop"]].mean(axis=0)
        responses.append({"index": tr["index"], "phase": tr["phase"],
            "visual": tr["visual"], "tactile": tr["tactile"],
            "rates": {g: float(on[np.array(ids)-1].mean()) for g, ids in groups.items()},
            "late_withdrawal_rates": {g: float(off[np.array(ids)-1].mean()) for g, ids in groups.items()}})
    templates = {}
    for a in (0, 1):
        examples = [tr for tr in trials if tr["phase"] == "paired_experience" and tr["visual"] == a]
        templates[a] = np.mean([spikes[tr["start"]+12:tr["start"]+64].mean(axis=0) for tr in examples[-2:]], axis=0)
    recalls = []
    for tr in trials:
        if tr["phase"] not in ("before_learning", "partial_recall"):
            continue
        response = spikes[tr["start"]+12:tr["start"]+64].mean(axis=0)
        for group in ("tactile_core", "upper_core"):
            ids = np.array(groups[group])-1
            a = tr["visual"]
            yes, no = finite_cosine(response[ids], templates[a][ids]), finite_cosine(response[ids], templates[1-a][ids])
            recalls.append({"trial": tr["index"], "phase": tr["phase"], "group": group,
                            "correct_similarity": yes, "other_similarity": no,
                            "specificity_margin": None if yes is None or no is None else yes-no,
                            "active_neurons": int(np.count_nonzero(response[ids]))})
    def contrast(role, prefix=""):
        familiar = [r["rates"][role] for r in responses if r["phase"] == prefix+"familiar_pair"]
        conflict = [r["rates"][role] for r in responses if r["phase"] == prefix+"conflicting_pair"]
        return float(np.mean(conflict)-np.mean(familiar))
    return {"trial_responses": responses, "recall": recalls,
        "mismatch_conflict_minus_familiar": contrast("mismatch_candidate"),
        "mismatch_contrast_before_learning": contrast("mismatch_candidate", "before_"),
        "changed_information_weights": int(np.count_nonzero(weights_after != weights_before)),
        "information_weight_change_l1": float(np.abs(weights_after-weights_before).sum()),
        "templates": {str(a): templates[a].tolist() for a in (0, 1)},
        "interpretation": "Cosine margins are offline measurements, never neural inputs. A changed weight is not by itself memory; a positive margin is not by itself learned recall."}


def run(destination, size=1152, seed=11, mode="intact", repetitions=6, record_weights=True, preparation=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    config, groups, edges = make_config(size, seed, mode) if preparation is None else preparation["network"]
    (destination/"config.json").write_text(encode(config)+"\n")
    random.seed(seed)
    np.random.seed(seed)
    net, core = k.load(str(destination/"config.json"), neuron_class=PlasticityRateNeuron)
    net.record_history = False
    neurons = list(net.network.neurons.values())
    assert [n.id for n in neurons] == list(range(1, size+1))
    assert all(n.params.eta_post > 0 and n.params.eta_retro > 0 for n in neurons)
    syn_index = [(n.id, sid) for n in neurons for sid in n.postsynaptic_points]
    synapses = [n.postsynaptic_points[sid] for n in neurons for sid in n.postsynaptic_points]
    terminals = [n.presynaptic_points[900] for n in neurons]
    initial_w = weight_values(synapses)
    patterns, trials = protocol(groups, seed, repetitions) if preparation is None else preparation["protocol"]
    sources = fingerprint()
    sources[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for source in [] if preparation is None else preparation.get("source_files", []):
        sources[str(Path(source).resolve())] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    manifest = {"schema": "paula-population-v1", "size": size, "seed": seed, "mode": mode,
        "groups": groups, "labels": LABELS, "fields": FIELDS, "synapse_index": syn_index,
        "edges": edges, "patterns": patterns, "trials": trials, "source_hashes": sources,
        "record_weights_every_tick": record_weights,
        "recording_limits": "Every completed tick: listed cellular fields and terminal info. Optional every-tick incoming info weights. Not every queue/retrograde/adapt field. Final full snapshot and config are retained.",
        "scope": "Isolated recursive population prototype; no physical body, action-learning, established hierarchy, or consciousness result."}
    if preparation is not None:
        manifest.update(preparation.get("metadata", {}))
    (destination/"manifest.json").write_text(encode(manifest)+"\n")
    all_cells, chunk_cells, chunk_weights, chunk_outputs = [], [], [], []
    started = time.perf_counter()
    for trial in trials:
        for t in range(trial["start"], trial["stop"]):
            inputs = [(nid, 2.) for nid in input_ids(trial, t)] if preparation is None else preparation["inputs"](trial, t)
            for nid, amplitude in inputs:
                net.set_external_input(nid, 0, amplitude)
            result = core.do_tick()
            if "error" in result or net.current_tick != t+1:
                raise RuntimeError(result)
            state = cellular(neurons)
            if not np.isfinite(state).all():
                raise FloatingPointError(f"Nonfinite state at tick {t}")
            chunk_cells.append(state)
            if record_weights:
                chunk_weights.append(weight_values(synapses))
            chunk_outputs.append([p.u_o.info for p in terminals])
        chunk = np.stack(chunk_cells)
        payload = {"cells": chunk, "terminal_info": np.array(chunk_outputs, dtype=np.float64)}
        if record_weights:
            # Lossless XOR coding, not a floating-point difference. Nearby
            # unchanged weights become zeros; every float64 bit is recoverable.
            bits = np.stack(chunk_weights).view(np.uint64)
            payload["incoming_info_xor"] = np.concatenate((bits[:1], np.bitwise_xor(bits[1:], bits[:-1])))
        np.savez_compressed(destination/f"ticks-{trial['start']:06d}.npz", **payload)
        all_cells.append(chunk)
        chunk_cells, chunk_weights, chunk_outputs = [], [], []
        counts = (chunk[:, :, 1] > 0).sum(axis=0)
        print(encode({"trial": trial["index"], "phase": trial["phase"], "tick": net.current_tick,
            "seconds": round(time.perf_counter()-started, 2),
            "spikes": {g: int(counts[np.array(ids)-1].sum()) for g, ids in groups.items()}}), flush=True)
    cell = np.concatenate(all_cells)
    final_w = weight_values(synapses)
    analysis = analyze if preparation is None else preparation["analyze"]
    report = analysis(cell, trials, groups, initial_w, final_w)
    report["elapsed_seconds"] = time.perf_counter()-started
    report["ticks"] = net.current_tick
    report["nonpositive_terminal_info"] = sum(p.u_o.info <= 0 for n, p in zip(neurons, terminals)
        if n.metadata["role"] not in ("mismatch_candidate", "activity_regulator"))
    (destination/"summary.json").write_text(encode(report)+"\n")
    np.savez_compressed(destination/"parameters.npz", initial_info=initial_w, learned_info=final_w)
    with gzip.open(destination/"final-state.json.gz", "wt") as stream:
        stream.write(encode(snapshot(net)))
    return report


def read_information_weights(chunk):
    if "incoming_info" in chunk:
        return chunk["incoming_info"]
    return np.bitwise_xor.accumulate(chunk["incoming_info_xor"], axis=0).view(np.float64)


def transplant_probe(directory):
    """Diagnostic intervention: change ONLY incoming info weights at birth.

    Two fresh copies share all fast-state initial conditions. One receives the
    final learned incoming weights; the other retains initial weights. Neither
    copy freezes adaptation. This tests weight-carried effects separately from
    residual activity and other plastic fields; it is not a deployment reset.
    """
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    with np.load(directory/"parameters.npz") as parameters:
        learned = parameters["learned_info"].copy()
    results = []
    for condition in ("initial_weights", "learned_weights"):
        for cue in (0, 1):
            random.seed(manifest["seed"])
            np.random.seed(manifest["seed"])
            net, core = k.load(str(directory/"config.json"), neuron_class=PlasticityRateNeuron)
            net.record_history = False
            neurons = list(net.network.neurons.values())
            synapses = [n.postsynaptic_points[sid] for n in neurons for sid in n.postsynaptic_points]
            if condition == "learned_weights":
                for point, weight in zip(synapses, learned):
                    point.u_i.info = float(weight)
            tr = {"start": 0, "active_ids": manifest["patterns"]["vision"][cue][::2]}
            recorded = []
            for t in range(TRIAL_TICKS):
                for nid in input_ids(tr, t):
                    net.set_external_input(nid, 0, 2.)
                output = core.do_tick()
                if "error" in output or net.current_tick != t+1:
                    raise RuntimeError(output)
                recorded.append(cellular(neurons))
            cells = np.stack(recorded)
            np.savez_compressed(directory/f"transplant-{condition}-{cue}.npz", cells=cells)
            response = (cells[12:64, :, 1] > 0).mean(axis=0)
            results.append({"condition": condition, "cue": cue,
                "rates": {g: float(response[np.array(ids)-1].mean()) for g, ids in manifest["groups"].items()},
                "response": response.tolist()})
    report = {"intervention": "Incoming info weights only; identical fresh fast state, all basal adaptation positive", "results": results}
    (directory/"transplant.json").write_text(encode(report)+"\n")
    return report


def export(directory, destination):
    """Compact read-only display projection; raw files remain the authority."""
    directory, destination = Path(directory), Path(destination)
    manifest = json.loads((directory/"manifest.json").read_text())
    report = json.loads((directory/"summary.json").read_text())
    chunks = []
    for path in sorted(directory.glob("ticks-*.npz")):
        with np.load(path) as chunk:
            chunks.append(chunk["cells"])
    cell = np.concatenate(chunks)
    groups = manifest["groups"]
    rates = {g: (cell[:, np.array(ids)-1, 1] > 0).mean(axis=1).tolist() for g, ids in groups.items()}
    # Every tick is present. Binary display data avoids millions of JS array
    # objects. This float32 presentation is not the float64 research record.
    displayed = cell[:, :, [0, 1, 2, 3, 4]].astype("<f4")
    meta = {**manifest, "summary": report, "raw_directory": str(directory.resolve()),
            "display_fields": ["S", "O", "F_avg", "M0", "M1"],
            "display_precision": "Little-endian float32 presentation; raw float64 arrays retained separately",
            "rates": rates, "shape": list(displayed.shape)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    packed = base64.b64encode(gzip.compress(displayed.tobytes(), compresslevel=6)).decode()
    destination.write_text('window.POPULATION_META = '+encode(meta)+';\nwindow.POPULATION_RECORDING = "'+packed+'";\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--size", type=int, choices=(288, 576, 1152), default=1152)
    runner.add_argument("--seed", type=int, default=11)
    runner.add_argument("--mode", choices=MODES, default="intact")
    runner.add_argument("--repetitions", type=int, default=6)
    runner.add_argument("--observables-only", action="store_true")
    exporter = sub.add_parser("export")
    exporter.add_argument("--input", type=Path, required=True)
    exporter.add_argument("--output", type=Path, required=True)
    probe = sub.add_parser("transplant")
    probe.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        export(args.input, args.output)
    elif args.command == "transplant":
        print(encode(transplant_probe(args.input)))
    else:
        if args.repetitions < 1:
            parser.error("repetitions must be positive")
        run(args.output, args.size, args.seed, args.mode, args.repetitions, not args.observables_only)


if __name__ == "__main__":
    main()
