"""A four-condition test of existing neural feedback control authority.

No new neuron rule. Independently vary the activity regulator's firing threshold
and the postsynaptic excitability receptor gain. Positive basal learning remains
active. This tests desaturation, not acquired multimodal recall.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .composition_probe import encode, fingerprint
from .population_hierarchy import FIELDS, weight_values
from .multimodal_pairing_probe import fresh, episode


def regulation_config(config, condition):
    if condition not in ("original", "sensitivity_only", "receptor_only", "combined"):
        raise ValueError(condition)
    result = deepcopy(config)
    for n in result["neurons"]:
        role = n["metadata"]["role"]
        if role == "activity_regulator" and condition in ("sensitivity_only", "combined"):
            # Existing detector gets eight inputs of nominal gain .7. Its old
            # r=1.1 requires high aggregate activity. Lower its activation point
            # without a host rate target or time-dependent parameter schedule.
            n["params"]["r_base"] = .2
            n["params"]["b_base"] = .45
        if role in ("visual_core", "tactile_core", "upper_core") and condition in ("receptor_only", "combined"):
            # At measured M1 ~.05 the old gain .6 moves r by only ~.03.
            # Gain 12 allows a threshold shift of ~.6 at the same concentration.
            # This is a declared coarse intervention, not an optimized value.
            n["params"]["w_r"][1] = 12.
            n["params"]["w_b"][1] = 12.
    return result


def run(recording, output):
    recording, output = Path(recording), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((recording/"manifest.json").read_text())
    config = json.loads((recording/"config.json").read_text())
    with np.load(recording/"sensory-features.npz") as f:
        feature = {key: f[key] for key in f.files}
    groups = manifest["groups"]
    trials = [
        {"start": 0, "stop": 192, "phase": "silent_movie", "visual_clip": 0, "audio_clip": None},
        {"start": 192, "stop": 384, "phase": "audiovisual", "visual_clip": 0, "audio_clip": 0},
        {"start": 384, "stop": 480, "phase": "withdrawal", "visual_clip": None, "audio_clip": None},
    ]
    results = {}
    for condition in ("original", "sensitivity_only", "receptor_only", "combined"):
        started = time.perf_counter()
        directory = output/condition
        directory.mkdir()
        modified = regulation_config(config, condition)
        config_path = directory/"config.json"
        config_path.write_text(encode(modified)+"\n")
        net, core, neurons, synapses = fresh(config_path, manifest["seed"])
        initial_weights = weight_values(synapses)
        reports = []
        for trial in trials:
            before = weight_values(synapses)
            cells = episode(net, core, neurons, [feature], groups, trial)
            np.savez_compressed(directory/f"{trial['phase']}.npz", cells=cells, fields=FIELDS,
                                incoming_info_before=before, incoming_info_after=weight_values(synapses))
            active = cells[32:, :, 1] > 0
            reports.append({"phase": trial["phase"], "groups": {
                g: {"spikes": int((cells[:, np.array(ids)-1, 1] > 0).sum()),
                    "mean_rate_after_transient": float(active[:, np.array(ids)-1].mean()),
                    "fraction_near_ceiling": float((active[:, np.array(ids)-1].mean(axis=0) >= .30).mean()),
                    "r_range": [float(cells[:, np.array(ids)-1, 5].min()), float(cells[:, np.array(ids)-1, 5].max())],
                    "M1_max": float(cells[:, np.array(ids)-1, 4].max())}
                for g, ids in groups.items()}})
        results[condition] = {"trials": reports, "seconds": time.perf_counter()-started,
            "weight_change_l1": float(np.abs(weight_values(synapses)-initial_weights).sum())}
        print(encode({"condition": condition, **results[condition]}), flush=True)
    sources = fingerprint()
    sources[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = {"source_recording": str(recording.resolve()), "source_hashes": sources,
        "conditions": results, "trials": trials,
        "interpretation": "Both factor levels selected before running. Neither silence nor reduced firing is success by itself. Inspect retained sensory responsiveness and recall separately. Current regulator pools modalities; this does not test localized supervisory organization or energy constraints."}
    (output/"summary.json").write_text(encode(result)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.recording, args.output)
