"""Locate weight-carried audiovisual effects without removing signal pathways.

This is a 2x2 intervention on stored incoming information weights. One factor
is the learned weights on visual/upper projections into the auditory core;
the other is all remaining incoming information weights. Initial and fully
learned reference responses already exist in the source experiment. We replay
the fully learned condition exactly before interpreting the two hybrid states.
Every probe starts fresh and remains plastic. No cue label enters the graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .composition_probe import encode
from .multimodal_pairing_probe import episode, fresh
from .population_hierarchy import weight_values
from neuron.extensions.experimental.plasticity_rate import PlasticityRateNeuron


def selected_ports(edges, targets, families=("crossmodal", "descending")):
    return {(tgt, sid) for _, tgt, sid, family, present in edges
            if present and tgt in targets and family in families}


def hybrid_weights(initial, learned, index, selected, condition):
    if len(initial) != len(learned) or len(initial) != len(index):
        raise ValueError("Weight arrays and synapse index must match")
    if len(set(index)) != len(index) or not selected.issubset(set(index)):
        raise ValueError("Selected ports must exist in a unique synapse index")
    mask = np.array([port in selected for port in index])
    if condition == "learned_all":
        return learned.copy()
    if condition == "learned_selected_only":
        return np.where(mask, learned, initial)
    if condition == "learned_remainder_only":
        return np.where(mask, initial, learned)
    raise ValueError(condition)


def run(recording, output, donor=None):
    recording, output = Path(recording).resolve(), Path(output).resolve()
    manifest = json.loads((recording/"manifest.json").read_text())
    # The original experiment hashed its runtime and transduction sources.
    # Refuse to call a different implementation an exact replay.
    for name, expected in manifest["source_hashes"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed since original recording: {name}")
    with np.load(recording/"parameters.npz") as raw:
        initial, learned = raw["initial_info"], raw["learned_info"]
    if not np.isfinite(initial).all() or not np.isfinite(learned).all():
        raise ValueError("Nonfinite weights")
    donor_weights, donor_metadata = None, None
    if donor is not None:
        donor = Path(donor).resolve()
        dm = json.loads((donor/"manifest.json").read_text())
        if ((donor/"config.json").read_bytes() != (recording/"config.json").read_bytes() or
                dm["seed"] != manifest["seed"] or dm["mapping"] == manifest["mapping"] or
                dm["media"] != manifest["media"] or dm["source_hashes"] != manifest["source_hashes"]):
            raise ValueError("Donor must be the matched opposite-assignment experiment")
        with np.load(donor/"parameters.npz") as raw:
            if not np.array_equal(initial, raw["initial_info"]):
                raise ValueError("Donor initial weights differ")
            donor_weights = raw["learned_info"]
        if not np.isfinite(donor_weights).all():
            raise ValueError("Nonfinite donor weights")
        donor_metadata = {"recording": str(donor), "mapping": dm["mapping"],
                          "parameters_sha256": hashlib.sha256((donor/"parameters.npz").read_bytes()).hexdigest()}
    features = []
    for clip in (0, 1):
        with np.load(recording/f"sensory-{clip}.npz") as raw:
            features.append({key: raw[key] for key in raw.files})
    groups, seed = manifest["groups"], manifest["seed"]
    neuron_class = PlasticityRateNeuron
    if manifest.get("weight_dynamics", "native") == "bounded":
        from neuron.extensions.experimental.bounded_plasticity import BoundedPlasticityNeuron
        neuron_class = BoundedPlasticityNeuron
    families = ("crossmodal",) if donor is not None else ("crossmodal", "descending")
    selected = selected_ports(manifest["edges"], set(groups["tactile_core"]), families)
    output.mkdir(parents=True, exist_ok=False)
    source_checksums = {name: hashlib.sha256((recording/name).read_bytes()).hexdigest()
                        for name in ("manifest.json", "config.json", "parameters.npz",
                                     "sensory-0.npz", "sensory-1.npz")}
    start, counts, index = time.perf_counter(), {}, None
    conditions = ("learned_all", "selected_from_donor", "remainder_from_donor") if donor is not None else (
        "learned_all", "learned_selected_only", "learned_remainder_only")
    for condition in conditions:
        counts[condition] = {}
        for clip in (0, 1):
            net, core, neurons, synapses = fresh(recording/"config.json", seed, neuron_class)
            current_index = [(n.id, sid) for n in neurons for sid in n.postsynaptic_points]
            if index is None:
                index = current_index
            if index != current_index or not np.array_equal(weight_values(synapses), initial):
                raise AssertionError("Fresh network does not match original parameter indexing")
            if condition == "selected_from_donor":
                weights = hybrid_weights(learned, donor_weights, index, selected, "learned_selected_only")
            elif condition == "remainder_from_donor":
                weights = hybrid_weights(learned, donor_weights, index, selected, "learned_remainder_only")
            else:
                weights = hybrid_weights(initial, learned, index, selected, condition)
            for synapse, value in zip(synapses, weights):
                synapse.u_i.info = float(value)
            trial = {"start": 0, "stop": manifest["clip_ticks"],
                     "visual_clip": clip, "audio_clip": None}
            cells = episode(net, core, neurons, features, groups, trial)
            exact = None
            if condition == "learned_all":
                with np.load(recording/f"probe-learned_info-visual-{clip}.npz") as reference:
                    exact = (np.array_equal(cells, reference["cells"]) and
                             np.array_equal(weight_values(synapses), reference["incoming_info_after"]))
                if not exact:
                    raise AssertionError("Fully learned replay differs from original")
            np.savez_compressed(output/f"{condition}-visual-{clip}.npz", cells=cells,
                                incoming_info_before=weights,
                                incoming_info_after=weight_values(synapses))
            spikes = cells[:, :, 1] > 0
            counts[condition][str(clip)] = {
                "exact_cells_and_weights_replay": exact,
                "spikes": {role: int(spikes[:, np.array(ids)-1].sum()) for role, ids in groups.items()}}
            print(encode({"condition": condition, "clip": clip,
                          "seconds": round(time.perf_counter()-start, 2),
                          **counts[condition][str(clip)]}), flush=True)
            del net, core, neurons, synapses
    report = {"source_recording": str(recording), "source_files_sha256": source_checksums,
              "observer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "mapping": manifest["mapping"], "seed": seed,
              "donor": donor_metadata, "selected_families": families,
              "selected_ports": sorted(selected), "synapse_index": index,
              "conditions": counts, "ticks": 6*manifest["clip_ticks"],
              "seconds": time.perf_counter()-start,
              "intervention": "Only incoming information weights differ at birth. No edges, delays, neural parameters or adaptation rates change. See selected_families for inputs to auditory core. With a donor, exchange learned weights between matched opposite assignments; otherwise compare learned and initial weights.",
              "limits": "Probes remain plastic. Selective weight transplants measure effects of stored weights in a fresh network, not necessity of a pathway during natural recall, retention duration, learned terminal memory, or semantic recognition."}
    (output/"summary.json").write_text(encode(report)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--donor", type=Path)
    args = parser.parse_args()
    run(args.recording, args.output, args.donor)
