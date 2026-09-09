"""Content-swapped audiovisual learning, with weight-only causal probes.

Two real clips supply raw pixels and sound measurements. Clip indices select
physical stimuli only; no labels, loss, decoded error or phase instruction enter
the brain. The paired and swapped conditions have identical sensory marginals.
Evaluation uses fresh activity states and positive basal plasticity throughout.
This is clip association, not animal recognition or a consciousness assay.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from .audiovisual_population import decode_media
from .composition_probe import encode, fingerprint, k, snapshot
from .population_hierarchy import cellular, make_config, weight_values, FIELDS
from neuron.extensions.experimental.plasticity_rate import PlasticityRateNeuron


def pairing_protocol(length, repeats, seed, mapping):
    if mapping not in ("paired", "swapped") or length < 4 or length % 4 or repeats < 1:
        raise ValueError("Need paired/swapped mapping, positive repeats and four-tick length")
    rng = np.random.default_rng(seed+6113)
    result = []
    start = 0
    for _ in range(repeats):
        for visual in rng.permutation(2):
            visual = int(visual)
            audio = visual if mapping == "paired" else 1-visual
            result.append({"start": start, "stop": start+length, "visual_clip": visual,
                           "audio_clip": audio, "phase": "experience"})
            start += length
            result.append({"start": start, "stop": start+96, "visual_clip": None,
                           "audio_clip": None, "phase": "withdrawal"})
            start += 96
    return result


def inputs(features, groups, trial, tick):
    rel = tick-trial["start"]
    # Each channel is physically selected before the existing transducer. No
    # clip ID is passed to a neuron or used to choose a projection/weight.
    output = []
    for key, role, field in (("visual_clip", "vision", "visual"),
                             ("audio_clip", "touch", "auditory")):
        clip = trial[key]
        if clip is None:
            continue
        values = features[clip][field][rel % features[clip]["ticks"]]
        output.extend((nid, float(2*value)) for nid, value in zip(groups[role], values)
                      if rel % 4 == nid % 4 and value > 0)
    return output


def fresh(config_path, seed, neuron_class=PlasticityRateNeuron):
    random.seed(seed)
    np.random.seed(seed)
    net, core = k.load(str(config_path), neuron_class=neuron_class)
    net.record_history = False
    neurons = list(net.network.neurons.values())
    assert all(n.params.eta_post > 0 and n.params.eta_retro > 0 for n in neurons)
    synapses = [p for n in neurons for p in n.postsynaptic_points.values()]
    return net, core, neurons, synapses


def episode(net, core, neurons, features, groups, trial, observer=None):
    states = []
    for t in range(trial["start"], trial["stop"]):
        for nid, value in inputs(features, groups, trial, t):
            net.set_external_input(nid, 0, value)
        result = core.do_tick()
        if "error" in result:
            raise RuntimeError(result)
        if observer is not None:
            observer()
        states.append(cellular(neurons))
    states = np.stack(states)
    if not np.isfinite(states).all():
        raise FloatingPointError("Nonfinite cell observable")
    return states


class WeightObserver:
    """Read-only per-tick health accounting; no data feeds back into the brain."""
    fields = ("changed_weights", "sign_changes_from_initial", "sign_changes_this_tick",
              "max_abs_weight", "zeroed_nonzero_weights", "at_native_bound",
              "bounded_update_total", "bounded_underflow_total",
              "terminal_info_min", "terminal_info_max", "negative_terminal_count")

    def __init__(self, neurons, synapses):
        self.neurons, self.synapses = neurons, synapses
        self.initial = weight_values(synapses)
        self.previous = self.initial.copy()
        self.terminals = [p for n in neurons for p in n.presynaptic_points.values()]
        self.rows = []

    def __call__(self):
        weights = weight_values(self.synapses)
        terminal = np.array([p.u_o.info for p in self.terminals])
        if not np.isfinite(weights).all() or not np.isfinite(terminal).all():
            raise FloatingPointError("Nonfinite information weight/terminal")
        self.rows.append([np.count_nonzero(weights != self.previous),
            np.count_nonzero(weights*self.initial < 0),
            np.count_nonzero(weights*self.previous < 0), float(np.max(abs(weights))),
            np.count_nonzero((self.initial != 0) & (weights == 0)), np.count_nonzero(abs(weights) >= 100),
            sum(getattr(n, "bounded_updates", 0) for n in self.neurons),
            sum(getattr(n, "bounded_underflows", 0) for n in self.neurons),
            float(terminal.min()), float(terminal.max()), np.count_nonzero(terminal < 0)])
        self.previous = weights


def content_projection(visual_rates, audio_rates):
    """Offline phase-insensitive population contrast, with no fitted classifier.

    Audio references define a neuron-wise contrast axis. Projection is in units
    of their separation; +.5/- .5 are the two references. This does not prove
    generalization or sound reconstruction. Zero reference separation is invalid.
    """
    axis = audio_rates[0]-audio_rates[1]
    squared = float(axis@axis)
    if squared <= 1e-16:
        return {"defined": False, "reason": "Auditory references are indistinguishable"}
    middle = (audio_rates[0]+audio_rates[1])/2
    projections = (visual_rates-middle)@axis/squared
    return {"defined": True, "reference_distance": float(np.sqrt(squared)),
            "visual_projections": projections.tolist(),
            "visual_content_contrast": float(projections[0]-projections[1])}


def run(source_paths, destination, mapping="paired", repeats=4, seed=11, architecture="baseline", weight_dynamics="native"):
    if architecture not in ("baseline", "regional", "shuffled"):
        raise ValueError(architecture)
    if weight_dynamics not in ("native", "bounded"):
        raise ValueError(weight_dynamics)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    features = [decode_media(path) for path in source_paths]
    length = min(f["ticks"] for f in features)//4*4
    for index, feature in enumerate(features):
        for key in ("visual", "auditory", "pixels", "audio_rms", "band_db"):
            feature[key] = feature[key][:length]
        feature["ticks"] = length
        np.savez_compressed(destination/f"sensory-{index}.npz", **feature)
    config, groups, edges = make_config(1152, seed)
    architecture_metadata = {"routing": "baseline"}
    if architecture != "baseline":
        from ..components.learning.regional_regulation import regional_regulation
        config, edges, architecture_metadata = regional_regulation(config, groups, edges, seed=seed, routing=architecture)
    neuron_class = PlasticityRateNeuron
    if weight_dynamics == "bounded":
        from neuron.extensions.experimental.bounded_plasticity import BoundedPlasticityNeuron
        neuron_class = BoundedPlasticityNeuron
        for neuron in config["neurons"]:
            neuron["metadata"].update(bounded_plasticity=True, plasticity_magnitude_cap=10., plasticity_magnitude_decay=.02)
    config_path = destination/"config.json"
    config_path.write_text(encode(config)+"\n")
    trials = pairing_protocol(length, repeats, seed, mapping)
    sources = fingerprint()
    for path in (Path(__file__), Path(__file__).with_name("audiovisual_population.py"),
                 Path(__file__).with_name("population_hierarchy.py")):
        sources[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    if architecture != "baseline":
        from ..components.learning import regional_regulation as regional_module
        path = Path(regional_module.__file__)
        sources[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    if weight_dynamics == "bounded":
        from neuron.extensions.experimental import bounded_plasticity
        path = Path(bounded_plasticity.__file__)
        sources[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"seed": seed, "mapping": mapping, "repetitions": repeats, "clip_ticks": length,
        "architecture": architecture, "architecture_metadata": architecture_metadata,
        "weight_dynamics": weight_dynamics, "weight_health_fields": WeightObserver.fields,
        "groups": groups, "edges": edges, "fields": FIELDS, "trials": trials,
        "source_hashes": sources,
        "media": [{"path": str(Path(p).resolve()), "sha256": f["source_sha256"]}
                  for p, f in zip(source_paths, features)],
        "probe": "Each probe starts from original fast state with either original or learned incoming info weights only. Every basal adaptation rate stays positive. Outgoing terminal changes, queues, M, membrane, t_ref and firing histories are not transplanted.",
        "recording": "Every tick's cellular fields; incoming info weights before/after each episode; training final full snapshot. Not every intracellular field at every tick.",
        "scope": "Two recordings, one initialization per run, no held-out category generalization. Audio contains background and speech. Neural hierarchy and supervisor function not assumed."}
    (destination/"manifest.json").write_text(encode(manifest)+"\n")
    started = time.perf_counter()
    net, core, neurons, synapses = fresh(config_path, seed, neuron_class)
    initial_weights = weight_values(synapses)
    observer = WeightObserver(neurons, synapses)
    rows = []
    for index, trial in enumerate(trials):
        before = weight_values(synapses)
        observer.rows = []
        cells = episode(net, core, neurons, features, groups, trial, observer)
        after = weight_values(synapses)
        np.savez_compressed(destination/f"experience-{index:03d}.npz", cells=cells,
                            incoming_info_before=before, incoming_info_after=after,
                            weight_health=np.array(observer.rows))
        counts = (cells[:, :, 1] > 0).sum(axis=0)
        row = {"index": index, **trial, "spikes": {g: int(counts[np.array(ids)-1].sum()) for g, ids in groups.items()}}
        rows.append(row)
        print(encode({"mapping": mapping, "stage": "experience", "episode": index,
                      "seconds": round(time.perf_counter()-started, 2)}), flush=True)
    learned_weights = weight_values(synapses)
    np.savez_compressed(destination/"parameters.npz", initial_info=initial_weights, learned_info=learned_weights)
    with gzip.open(destination/"training-final-state.json.gz", "wt") as stream:
        stream.write(encode(snapshot(net))+"\n")
    # Do not retain the training network while allocating fresh probes.
    del net, core, neurons, synapses, observer
    responses = {}
    for stage, weights in (("initial", initial_weights), ("learned_info", learned_weights)):
        responses[stage] = {}
        for sense in ("visual", "audio"):
            for clip in (0, 1):
                net, core, neurons, synapses = fresh(config_path, seed, neuron_class)
                for synapse, value in zip(synapses, weights):
                    synapse.u_i.info = float(value)
                trial = {"start": 0, "stop": length,
                         "visual_clip": clip if sense == "visual" else None,
                         "audio_clip": clip if sense == "audio" else None}
                observer = WeightObserver(neurons, synapses)
                cells = episode(net, core, neurons, features, groups, trial, observer)
                np.savez_compressed(destination/f"probe-{stage}-{sense}-{clip}.npz", cells=cells,
                                    incoming_info_before=weights, incoming_info_after=weight_values(synapses),
                                    weight_health=np.array(observer.rows))
                spikes = cells[:, :, 1] > 0
                # Discard only declared initial transport transient, not quiet
                # periods selected after inspecting the result.
                rates = spikes[32:].mean(axis=0)
                responses[stage][f"{sense}_{clip}"] = {g: rates[np.array(ids)-1].tolist() for g, ids in groups.items()}
                print(encode({"mapping": mapping, "stage": stage, "sense": sense, "clip": clip,
                              "seconds": round(time.perf_counter()-started, 2)}), flush=True)
                del net, core, neurons, synapses, observer
    contrasts = {}
    for group in ("tactile_core", "upper_core", "visual_core"):
        contrasts[group] = {}
        for reference_stage in ("initial", "learned_info"):
            reference = np.array([responses[reference_stage][f"audio_{i}"][group] for i in (0, 1)])
            contrasts[group][reference_stage+"_audio_axis"] = {
                stage: content_projection(np.array([responses[stage][f"visual_{i}"][group] for i in (0, 1)]), reference)
                for stage in ("initial", "learned_info")}
    result = {"mapping": mapping, "seed": seed, "ticks": trials[-1]["stop"]+8*length,
        "seconds": time.perf_counter()-started, "trials": rows, "responses": responses,
        "contrasts": contrasts, "changed_information_weights": int(np.count_nonzero(initial_weights != learned_weights)),
        "interpretation": "Compare paired versus swapped change in content contrast against the SAME initial audio axis. A weight change or positive raw contrast alone is not learned association. Phase-invariant rate readout does not assess all temporal codes."}
    (destination/"summary.json").write_text(encode(result)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs=2, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping", choices=("paired", "swapped"), default="paired")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--architecture", choices=("baseline", "regional", "shuffled"), default="baseline")
    parser.add_argument("--weight-dynamics", choices=("native", "bounded"), default="native")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 12:
        parser.error("Use 1-12 repeats for this bounded experiment")
    run(args.sources, args.output, args.mapping, args.repeats, args.seed, args.architecture, args.weight_dynamics)
