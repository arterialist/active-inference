"""Data-only state-partition and representation-drift audit.

Compare incoming-info-only recall, all-synaptic recall, and intact continuation.
Keep both original and contemporaneous sound references. Neither a favorable
axis nor a favorable time window is selected as a replacement acceptance test.
Raw cellular records remain the evidence; exported projections are observers.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .multimodal_pairing_audit import readout, contrast

KINDS = ("mean_rate", "population_centered_rate", "state_covariance", "population_centered_covariance")
CONDITIONS = ("incoming_info", "all_synaptic", "continuation")


def canonical(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def reference_comparison(visual, original, current):
    old, new = original[0]-original[1], current[0]-current[1]
    a, b = float(np.linalg.norm(old)), float(np.linalg.norm(new))
    return {"original_reference_contrast": contrast(visual, original),
            "current_reference_contrast": contrast(visual, current),
            "original_reference_distance": a, "current_reference_distance": b,
            "reference_axis_cosine": float(old@new/(a*b)) if a*b > 1e-16 else None,
            "reference_norm_ratio": b/a if a > 1e-16 else None}


def tick_projection(samples, ids, refs):
    """Instantaneous spike-pattern contrast in the fixed centered-rate basis.

    Its average after tick 32 is exactly the centered-rate readout contrast.
    Individual values can be noisy and need not follow a zero-lag waveform.
    """
    axis = refs[0]-refs[1]
    norm2 = float(axis@axis)
    if norm2 <= 1e-16:
        return None
    signals = [(c[:, np.array(ids)-1, 1] > 0).astype(float) for c in samples]
    diff = signals[0]-signals[1]
    diff -= diff.mean(axis=1, keepdims=True)
    return diff@axis/norm2


def partition_valid(start, trained, condition, parameters=None):
    if condition == "continuation":
        return canonical(start) == canonical(trained)
    if condition != "all_synaptic":
        raise ValueError(condition)
    if parameters is None:
        raise ValueError("Fresh-state validation requires the configured parameters")
    if (start["tick"] != 0 or start["presynaptic_wheel"] or start["retrograde_wheel"] or
            start["neurons"].keys() != trained["neurons"].keys()):
        return False
    for nid, n in start["neurons"].items():
        old = trained["neurons"][nid]
        p = parameters[nid]
        if (n["S"] != 0 or n["O"] != 0 or n["F_avg"] != 0 or any(n["M"]) or
                n["t_last_fire"] is not None or n["dendritic_queue"] or
                n["r"] != p["r_base"] or n["b"] != p["b_base"] or
                n["t_ref"] != p["c"]*p["num_inputs"]):
            return False
        if n["synapses"].keys() != old["synapses"].keys() or n["terminals"] != old["terminals"]:
            return False
        for sid, point in n["synapses"].items():
            if point[:3] != old["synapses"][sid][:3] or point[3] != 0:
                return False
    return True


def audit(paired_path, swapped_path, output):
    paths, output = [Path(paired_path), Path(swapped_path)], Path(output)
    summaries = [json.loads((p/"summary.json").read_text()) for p in paths]
    if [s["mapping"] for s in summaries] != ["paired", "swapped"]:
        raise ValueError("Supply paired and swapped in that order")
    recordings = [Path(s["source_recording"]) for s in summaries]
    manifests = [json.loads((p/"manifest.json").read_text()) for p in recordings]
    checks = {"same_seed": summaries[0]["seed"] == summaries[1]["seed"],
              "same_config": (recordings[0]/"config.json").read_bytes() == (recordings[1]/"config.json").read_bytes(),
              "same_sources": manifests[0]["source_hashes"] == manifests[1]["source_hashes"],
              "same_media": manifests[0]["media"] == manifests[1]["media"],
              "same_probe_source": summaries[0]["source_hashes"] == summaries[1]["source_hashes"],
              "source_records_unchanged": True, "declared_partition": True,
              "finite_cells_and_weights": True, "incoming_weights_match": True,
              "initial_references_identical": True, "all_branches_present": True}
    results, trajectories, parameter_partitions, state_comparisons = {}, {}, {}, {}
    first_refs = None
    for path, summary, recording, manifest in zip(paths, summaries, recordings, manifests):
        mapping = summary["mapping"]
        for name, expected in summary["source_files_sha256"].items():
            checks["source_records_unchanged"] &= hashlib.sha256((recording/name).read_bytes()).hexdigest() == expected
        with gzip.open(recording/"training-final-state.json.gz", "rt") as stream:
            trained = json.load(stream)
        with np.load(recording/"parameters.npz") as raw:
            learned = raw["learned_info"]
        config = json.loads((recording/"config.json").read_text())
        parameters = {str(n["id"]): n["params"] for n in config["neurons"]}
        parameter_partitions[mapping] = {key: 0 for key in
            ("incoming_info", "incoming_plast", "incoming_adapt", "terminal_info", "terminal_mod", "terminal_retro")}
        for point in config["synaptic_points"]:
            n = trained["neurons"][str(point["neuron_id"])]
            if point["type"] == "postsynaptic":
                actual = n["synapses"][str(point["synapse_id"])]
                for i, key in enumerate(("info", "plast", "adapt")):
                    parameter_partitions[mapping]["incoming_"+key] += int(actual[i] != point["u_i"][key])
            else:
                actual = n["terminals"][str(point["terminal_id"])]
                for i, key in enumerate(("info", "mod")):
                    parameter_partitions[mapping]["terminal_"+key] += int(actual[i] != point["u_o"][key])
                parameter_partitions[mapping]["terminal_retro"] += int(actual[2] != point["u_i_retro"])
        expected_keys = {(c, s, i) for c in CONDITIONS[1:] for s in ("visual", "audio") for i in (0, 1)}
        got_keys = [(b["condition"], b["sense"], b["clip"]) for b in summary["branches"]]
        checks["all_branches_present"] &= set(got_keys) == expected_keys and len(got_keys) == len(expected_keys)
        original = []
        for clip in (0, 1):
            with np.load(recording/f"probe-initial-audio-{clip}.npz") as raw:
                original.append(raw["cells"])
        if first_refs is None:
            first_refs = original
        else:
            checks["initial_references_identical"] &= all(np.array_equal(a, b) for a, b in zip(first_refs, original))
        results[mapping] = {}
        partition_cells, partition_weights = {}, {}
        for condition in CONDITIONS:
            samples = {}
            for sense in ("visual", "audio"):
                samples[sense] = []
                for clip in (0, 1):
                    filename = recording/f"probe-learned_info-{sense}-{clip}.npz" if condition == "incoming_info" else path/f"{condition}-{sense}-{clip}.npz"
                    if condition != "incoming_info":
                        with gzip.open(path/f"{condition}-{sense}-{clip}-start.json.gz", "rt") as stream:
                            checks["declared_partition"] &= partition_valid(json.load(stream), trained, condition, parameters)
                    with np.load(filename) as raw:
                        cells = raw["cells"]
                        expected_shape = (manifest["clip_ticks"], len(trained["neurons"]), len(manifest["fields"]))
                        checks["finite_cells_and_weights"] &= (cells.shape == expected_shape and np.isfinite(cells).all() and np.isfinite(raw["incoming_info_after"]).all())
                        checks["incoming_weights_match"] &= np.array_equal(raw["incoming_info_before"], learned)
                        samples[sense].append(cells)
                        if condition != "incoming_info":
                            partition_cells[condition, sense, clip] = cells
                            partition_weights[condition, sense, clip] = raw["incoming_info_after"]
            results[mapping][condition] = {}
            for role in ("tactile_core", "upper_core"):
                ids = manifest["groups"][role]
                role_result = {}
                for kind in KINDS:
                    old, new, visual = [np.stack([readout(c, ids, kind) for c in group])
                                        for group in (original, samples["audio"], samples["visual"])]
                    role_result[kind] = reference_comparison(visual, old, new)
                    if kind == "population_centered_rate":
                        for label, ref in (("original", old), ("current", new)):
                            curve = tick_projection(samples["visual"], ids, ref)
                            if curve is None:
                                continue
                            key = f"{mapping}/{condition}/{role}/{label}"
                            trajectories[key] = curve
                            role_result[kind][f"{label}_reference_windows"] = [
                                {"start": a, "stop": min(b, len(curve)), "contrast": float(curve[a:min(b, len(curve))].mean())}
                                for a, b in ((0, 32), (32, 96), (96, 160), (160, 224), (224, len(curve))) if a < len(curve)]
                results[mapping][condition][role] = role_result
        state_comparisons[mapping] = []
        for sense in ("visual", "audio"):
            for clip in (0, 1):
                a, b = [partition_cells[c, sense, clip] for c in ("continuation", "all_synaptic")]
                different = a[:, :, 1] != b[:, :, 1]
                times = np.flatnonzero(different.any(axis=1))
                wa, wb = [partition_weights[c, sense, clip] for c in ("continuation", "all_synaptic")]
                state_comparisons[mapping].append({"sense": sense, "clip": clip,
                    "different_spike_entries": int(different.sum()),
                    "first_spike_difference": int(times[0]) if times.size else None,
                    "maximum_absolute_state_difference_by_field": dict(zip(manifest["fields"], np.max(abs(a-b), axis=(0, 1)).tolist())),
                    "final_incoming_weight_difference_linf": float(np.max(abs(wa-wb)))})
    checks = {k: bool(v) for k, v in checks.items()}
    if not all(checks.values()):
        raise AssertionError(checks)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"tick-projections.npz", **trajectories)
    result = {"structural_checks": checks, "structurally_valid": True, "results": results,
              "changed_parameter_counts": parameter_partitions,
              "continuation_vs_all_synaptic": state_comparisons,
              "recordings": [str(p.resolve()) for p in paths],
              "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limits": "Exploratory paired/swapped comparison, not a new pass criterion. Report every reference/readout and all declared windows. Tick projections are offline linear observers, not neural consumers or evidence of waveform reinstatement. Training-replay claims come from the separately tested probe; this data-only audit independently checks state partitions. No embodied or consciousness conclusion."}
    (output/"summary.json").write_text(json.dumps(result, allow_nan=False, indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--swapped", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.paired, args.swapped, args.output)
    print(json.dumps({"structural_checks": result["structural_checks"], "output": str(args.output)}, indent=2))
