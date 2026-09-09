"""Independent, data-only audit of two content-swapped PAULA experiments.

Does not import PAULA, run a simulation, or modify any recording. Reports
neuron-wise mean firing and temporal-state covariance. Population-centered
versions were added after the first audit to exclude uniform activation level.
None demands zero-lag waveform replay. No automated consciousness inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def readout(cells, ids, kind):
    values = cells[32:, np.array(ids)-1]
    if kind in ("mean_rate", "population_centered_rate"):
        rates = (values[:, :, 1] > 0).mean(axis=0)
        return rates-rates.mean() if kind == "population_centered_rate" else rates
    if kind in ("state_covariance", "population_centered_covariance"):
        # F_avg is native low-pass activity, not smoothed by a decoder. This
        # captures pairwise cofluctuation, invariant to a common time shift or
        # sample reordering. It can miss ordered sequences and higher moments.
        x = values[:, :, 2]
        if kind == "population_centered_covariance":
            x = x-x.mean(axis=1, keepdims=True)
        covariance = np.cov(x, rowvar=False, ddof=0)
        return covariance[np.triu_indices(len(ids))]
    raise ValueError(kind)


def contrast(visual, reference):
    axis = reference[0]-reference[1]
    norm2 = float(axis@axis)
    if norm2 <= 1e-16:
        return None
    return float((visual[0]-visual[1])@axis/norm2)


def sensory_marginals(path, manifest):
    """Independently reconstruct actual receptor doses, including staggering."""
    features = []
    for i in (0, 1):
        with np.load(path/f"sensory-{i}.npz") as f:
            features.append({key: f[key] for key in ("visual", "auditory")})
    doses = {nid: [] for g in ("vision", "touch") for nid in manifest["groups"][g]}
    for tr in manifest["trials"]:
        for role, field, key in (("vision", "visual", "visual_clip"), ("touch", "auditory", "audio_clip")):
            clip = tr[key]
            if clip is None:
                continue
            data = features[clip][field]
            for j, nid in enumerate(manifest["groups"][role]):
                values = [float(2*data[t % len(data), j]) for t in range(tr["stop"]-tr["start"])
                          if t % 4 == nid % 4 and data[t % len(data), j] > 0]
                doses[nid].extend(values)
    return {nid: sorted(values) for nid, values in doses.items()}


def weight_health(path, manifest):
    """Independently verify episode endpoints and inspect per-tick observer data.

    Full synaptic arrays exist only at episode boundaries in this protocol.
    Interior-tick sign counts come from the separately tested read-only observer;
    this function cannot independently reconstruct every interior weight.
    """
    path = Path(path)
    with np.load(path/"parameters.npz") as raw:
        initial, learned = raw["initial_info"], raw["learned_info"]
    rows, previous = [], initial
    for index, trial in enumerate(manifest["trials"]):
        with np.load(path/f"experience-{index:03d}.npz") as raw:
            before, after = raw["incoming_info_before"], raw["incoming_info_after"]
            if not np.array_equal(before, previous) or not np.isfinite(after).all():
                raise ValueError("Weight continuity or finite-value check failed")
            metrics = {"episode": index, "stop": trial["stop"],
                       "changed_weights": int(np.count_nonzero(before != after)),
                       "max_abs_weight": float(abs(after).max()),
                       "sign_changes_from_initial": int(np.count_nonzero(after*initial < 0)),
                       "at_native_bound": int(np.count_nonzero(abs(after) >= 100)),
                       "zeroed_nonzero_weights": int(np.count_nonzero((initial != 0) & (after == 0)))}
            if "weight_health" in raw:
                health = raw["weight_health"]
                fields = manifest["weight_health_fields"]
                if health.shape != (trial["stop"]-trial["start"], len(fields)) or not np.isfinite(health).all():
                    raise ValueError("Malformed tick-level weight health")
                for key in ("max_abs_weight", "sign_changes_from_initial", "at_native_bound", "zeroed_nonzero_weights"):
                    if health[-1, fields.index(key)] != metrics[key]:
                        raise ValueError(f"Tick observer disagrees with saved endpoint: {key}")
                metrics["per_tick_observer"] = {key: {"minimum": float(health[:, i].min()),
                                                      "maximum": float(health[:, i].max()),
                                                      "last": float(health[-1, i])}
                                                  for i, key in enumerate(fields)}
            rows.append(metrics)
            previous = after
    if not np.array_equal(previous, learned):
        raise ValueError("Final episode does not match learned parameter artifact")
    return {"episodes": rows, "endpoints_verified": True,
            "per_tick_available": all("per_tick_observer" in row for row in rows),
            "limits": "Every saved endpoint checked independently. Interior-tick health is recorded by a read-only observer; complete interior weight arrays are not stored by this protocol."}


def weight_assignment_geometry(paths, manifests, probes):
    """Exploratory weight-space measurement, distinct from successful recall.

    Project learned direct visual-to-auditory weight changes onto the product
    of independently evoked, population-centered source/target rate contrasts.
    No decoder is fitted or inserted in the neural loop. This can miss temporal
    codes and is not by itself a memory criterion.
    """
    config = json.loads((paths[0]/"config.json").read_text())
    ordered = {n["id"]: [] for n in config["neurons"]}
    for syn in config["synaptic_points"]:
        if syn["type"] == "postsynaptic":
            ordered[syn["neuron_id"]].append((syn["synapse_id"], syn["u_i"]["info"]))
    index = {(nid, sid): i for i, (nid, sid) in enumerate(
        (nid, sid) for nid, rows in ordered.items() for sid, _ in rows)}
    expected = np.array([weight for rows in ordered.values() for _, weight in rows])
    visual_ids, auditory_ids = [manifests[0]["groups"][g] for g in ("visual_core", "tactile_core")]
    visual = [readout(probes[0][f"initial-visual-{i}"], visual_ids, "population_centered_rate") for i in (0, 1)]
    auditory = [readout(probes[0][f"initial-audio-{i}"], auditory_ids, "population_centered_rate") for i in (0, 1)]
    v = dict(zip(visual_ids, visual[0]-visual[1]))
    a = dict(zip(auditory_ids, auditory[0]-auditory[1]))
    selected = [(index[(tgt, sid)], v[src]*a[tgt]) for src, tgt, sid, family, present in manifests[0]["edges"]
                if present and family == "crossmodal" and src in v and tgt in a]
    positions = np.array([i for i, _ in selected], dtype=int)
    pattern = np.array([value for _, value in selected])
    norm2 = float(pattern@pattern)
    results = {}
    for path, manifest in zip(paths, manifests):
        with np.load(path/"parameters.npz") as raw:
            initial, learned = raw["initial_info"], raw["learned_info"]
        if not np.array_equal(expected, initial):
            raise ValueError("Cannot establish saved weight ordering from configuration")
        delta = (learned-initial)[positions]
        results[manifest["mapping"]] = {"contrast_product_coefficient": float(delta@pattern/norm2) if norm2 > 1e-16 else None,
            "selected_weight_change_l1": float(abs(delta).sum()),
            "selected_changed_count": int(np.count_nonzero(delta))}
    return {"selected_synapses": len(selected), "contrast_product_norm": float(np.sqrt(norm2)),
            "conditions": results,
            "limits": "Exploratory analysis added after initial learning results. Weight alignment is not functional retrieval, and mean-rate contrasts can miss temporal associations. Input order remains a possible contributor to assignment differences."}


def audit(paired_path, swapped_path):
    paths = [Path(paired_path), Path(swapped_path)]
    manifests = [json.loads((p/"manifest.json").read_text()) for p in paths]
    if [m["mapping"] for m in manifests] != ["paired", "swapped"]:
        raise ValueError("Supply paired recording first, swapped second")
    structural = {
        "same_config": (paths[0]/"config.json").read_bytes() == (paths[1]/"config.json").read_bytes(),
        "same_sources": manifests[0]["source_hashes"] == manifests[1]["source_hashes"],
        "same_media": manifests[0]["media"] == manifests[1]["media"],
        "matched_per_receptor_training_marginals": sensory_marginals(paths[0], manifests[0]) == sensory_marginals(paths[1], manifests[1]),
    }
    probes = []
    for path in paths:
        current = {}
        for stage in ("initial", "learned_info"):
            for sense in ("visual", "audio"):
                for clip in (0, 1):
                    name = f"{stage}-{sense}-{clip}"
                    with np.load(path/f"probe-{name}.npz") as f:
                        current[name] = f["cells"]
                        assert np.isfinite(current[name]).all()
        probes.append(current)
    structural["initial_probes_bit_identical"] = all(
        np.array_equal(probes[0][key], probes[1][key]) for key in probes[0] if key.startswith("initial-"))
    results = {}
    for role in ("tactile_core", "upper_core", "visual_core"):
        ids = manifests[0]["groups"][role]
        results[role] = {}
        for kind in ("mean_rate", "population_centered_rate", "state_covariance", "population_centered_covariance"):
            refs = np.array([readout(probes[0][f"initial-audio-{i}"], ids, kind) for i in (0, 1)])
            before = contrast(np.array([readout(probes[0][f"initial-visual-{i}"], ids, kind) for i in (0, 1)]), refs)
            after = [contrast(np.array([readout(p[f"learned_info-visual-{i}"], ids, kind) for i in (0, 1)]), refs) for p in probes]
            valid = all(x is not None for x in [before, *after])
            results[role][kind] = {"reference_distance": float(np.linalg.norm(refs[0]-refs[1])),
                "before_contrast": before, "paired_after_contrast": after[0], "swapped_after_contrast": after[1],
                "paired_change": after[0]-before if valid else None,
                "swapped_change": after[1]-before if valid else None,
                "pairing_dependent_difference": after[0]-after[1] if valid else None,
                "both_changes_follow_assignment": bool(after[0] > before and after[1] < before) if valid else None}
    return {"structural_checks": structural, "structurally_valid": all(structural.values()),
        "weight_health": {m["mapping"]: weight_health(path, m) for path, m in zip(paths, manifests)},
        "weight_assignment_geometry": weight_assignment_geometry(paths, manifests, probes),
        "content_results": results,
        "limits": "Readouts are descriptive, not statistical replication. One seed and two clips do not establish generalization. Covariance misses temporal ordering. A positive pairing-dependent difference alone is weaker than changes following both assignments. All probes remain plastic; results include adaptation to the probe itself."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--swapped", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.paired, args.swapped)
    text = json.dumps(report, allow_nan=False, indent=2)+"\n"
    if args.output:
        with args.output.open("x") as f:
            f.write(text)
    print(text)
