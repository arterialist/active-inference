"""Read-only population evidence audit. Does not import or run PAULA.

Useful to agents that cannot infer behavior from a movie. Reconstructs counts
from tick arrays, checks explicit sensory exclusion and diagnoses the most
obvious false positive: apparent recall already present before experience.
No success threshold is inferred from the mere existence of these files.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def audit_regional(directory, control):
    """Audit one matched seed without importing the graph builder or PAULA."""
    if control is None:
        raise ValueError("Regional audit needs the shuffled control")
    paths = [Path(directory), Path(control)]
    manifests = [json.loads((p/"manifest.json").read_text()) for p in paths]
    configs = [json.loads((p/"config.json").read_text()) for p in paths]
    summaries = [json.loads((p/"summary.json").read_text()) for p in paths]
    a, b = configs
    structural = {
        "same_neurons": a["neurons"] == b["neurons"],
        "same_synapses": a["synaptic_points"] == b["synaptic_points"],
        "same_external_ports": a["external_inputs"] == b["external_inputs"],
        "same_source_outdegrees": Counter(c["source_neuron"] for c in a["connections"]) == Counter(c["source_neuron"] for c in b["connections"]),
        "same_target_indegrees": Counter(c["target_neuron"] for c in a["connections"]) == Counter(c["target_neuron"] for c in b["connections"]),
        "same_source_hashes": manifests[0]["source_hashes"] == manifests[1]["source_hashes"],
        "same_seed": manifests[0]["seed"] == manifests[1]["seed"],
    }
    changed = [(x, y) for x, y in zip(manifests[0]["edges"], manifests[1]["edges"]) if x != y]
    structural["only_modulatory_sources_changed"] = bool(changed) and all(x[1:] == y[1:] and x[3] == "excitability_modulation" for x, y in changed)
    with np.load(paths[0]/"sensory.npz") as x, np.load(paths[1]/"sensory.npz") as y:
        structural["same_raw_sensory_measurements"] = all(np.array_equal(x[k], y[k]) for k in ("visual", "auditory", "ticks", "source_sha256"))
    errors = [k for k, value in structural.items() if not value]
    results = {}
    for path, manifest, summary in zip(paths, manifests, summaries):
        groups, length = manifest["groups"], manifest["stimulus_ticks"]
        arrays = {}
        for name in ("vision", "audio", "both"):
            with np.load(path/f"{name}.npz") as f:
                cells = f["cells"]
            if cells.shape != (length+96, manifest["size"], len(manifest["fields"])) or not np.isfinite(cells).all():
                errors.append(f"Invalid cell recording {path.name}/{name}")
            arrays[name] = cells
            counts = {g: int((cells[:length, np.array(ids)-1, 1] > 0).sum()) for g, ids in groups.items()}
            if counts != {g: v["spikes"] for g, v in summary["results"][name]["groups"].items()}:
                errors.append(f"Summary differs from raw ticks {path.name}/{name}")
            absent = "touch" if name == "vision" else "vision" if name == "audio" else None
            if absent and np.any(cells[:, np.array(groups[absent])-1, 1]):
                errors.append(f"Absent sensory receptors spiked {name}/{absent}")
        ids = np.array(groups["tactile_core"])-1
        rates = {name: float((cells[32:length, ids, 1] > 0).mean()) for name, cells in arrays.items()}
        reference = arrays["audio"][32:length, ids, 2]
        difference = arrays["both"][32:length, ids, 2]-arrays["vision"][32:length, ids, 2]
        norm2 = float(np.square(reference).sum())
        windows = []
        for start in range(0, length, 48):
            end = min(start+48, length)
            counts = {name: int((cells[start:end, ids, 1] > 0).sum()) for name, cells in arrays.items()}
            windows.append({"start": start, "stop": end, **counts})
        results[manifest["routing"]] = {"auditory_rates": rates,
            "sound_increment_with_vision": rates["both"]-rates["vision"],
            "increment_over_audio_alone": (rates["both"]-rates["vision"])/rates["audio"] if rates["audio"] else None,
            "audio_trace_projection": float((difference*reference).sum()/norm2) if norm2 else None,
            "auditory_windows": windows,
            "upper_near_ceiling_fraction": float(((arrays["both"][32:length, np.array(groups["upper_core"])-1, 1] > 0).mean(axis=0) >= .30).mean())}
    return {"valid_recording": not errors, "errors": errors, "structural_checks": structural,
        "seed": manifests[0]["seed"], "results": results,
        "limits": "Acute sensory responsiveness and suppression, not association, content recognition, recovered performance after injury, or consciousness. Ratios above one can reflect amplification, not better coding. Full windows are retained to expose transients."}


def audit(directory, control=None):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    summary = json.loads((directory/"summary.json").read_text())
    errors, trials, totals = [], [], {}
    groups = manifest["groups"]
    for tr in manifest["trials"]:
        with np.load(directory/f"ticks-{tr['start']:06d}.npz") as raw:
            cells = raw["cells"]
            expected = (tr["stop"]-tr["start"], manifest["size"], len(manifest["fields"]))
            if cells.shape != expected:
                errors.append(f"Shape mismatch at trial {tr['index']}")
                continue
            if not np.isfinite(cells).all() or not np.isfinite(raw["terminal_info"]).all():
                errors.append(f"Nonfinite state at trial {tr['index']}")
            spiking = cells[:, :, 1] > 0
            counts = {g: int(spiking[:, np.array(ids)-1].sum()) for g, ids in groups.items()}
            if "spikes" in summary["trial_responses"][tr["index"]] and counts != summary["trial_responses"][tr["index"]]["spikes"]:
                errors.append(f"Summary counts disagree with ticks at trial {tr['index']}")
            if manifest["record_weights_every_tick"]:
                if "incoming_info_xor" in raw:
                    weights = np.bitwise_xor.accumulate(raw["incoming_info_xor"], axis=0).view(np.float64)
                else:
                    weights = raw["incoming_info"]
                if weights.shape != (expected[0], len(manifest["synapse_index"])) or not np.isfinite(weights).all():
                    errors.append(f"Invalid weights at trial {tr['index']}")
            if control and tr["phase"] in ("silent_video_before", "audio_reference_before"):
                with np.load(Path(control)/f"ticks-{tr['start']:06d}.npz") as baseline:
                    if not np.array_equal(cells, baseline["cells"]):
                        errors.append(f"Pre-intervention states differ at trial {tr['index']}")
            # Late half of silence separates queue-draining activity from a
            # continuing regime. Full counts remain above, not discarded.
            late = {g: int(spiking[len(spiking)//2:, np.array(ids)-1].sum()) for g, ids in groups.items()}
            trials.append({"phase": tr["phase"], "start": tr["start"], "stop": tr["stop"],
                           "spikes": counts, "late_half_spikes": late})
            for g, count in counts.items():
                totals[g] = totals.get(g, 0)+count
    findings = []
    media = manifest.get("media")
    if media:
        before = next(r for r in trials if r["phase"] == "silent_video_before")
        after = next(r for r in trials if r["phase"] == "silent_video_after")
        if before["spikes"]["tactile_core"]:
            findings.append({"kind": "preexisting_cross_activation", "before": before["spikes"]["tactile_core"],
                "after": after["spikes"]["tactile_core"],
                "conclusion": "Auditory firing during silent video is not sufficient evidence of learning."})
        silent = next(r for r in trials if r["phase"] == "media_silence")
        findings.append({"kind": "withdrawal", "all_spikes": sum(silent["spikes"].values()),
                         "late_half_spikes": sum(silent["late_half_spikes"].values())})
        experience = next(r for r in trials if r["phase"] == "audiovisual_experience")
        findings.append({"kind": "sensory_operating_point", "vision_spikes": experience["spikes"]["vision"],
            "audio_spikes": experience["spikes"]["touch"],
            "vision_to_audio_ratio": experience["spikes"]["vision"]/max(1, experience["spikes"]["touch"]),
            "conclusion": "Equal receptor counts do not imply equal delivered neural drive. This ratio is descriptive, not a requirement that senses have equal rates."})
        # Reconstruct the declared external event schedules without neural code.
        auditory = np.array(media["auditory_features"])
        dose = {str(nid): [] for nid in groups["touch"]}
        for tr in manifest["trials"]:
            if not tr["audio_enabled"]:
                continue
            if tr["phase"] == "audiovisual_experience":
                for rel in range(tr["stop"]-tr["start"]):
                    source = (rel+tr["audio_shift"]) % media["clip_ticks"]
                    for index, nid in enumerate(groups["touch"]):
                        if rel % 4 == nid % 4 and auditory[source, index] > 0:
                            dose[str(nid)].append(float(auditory[source, index]))
        if control:
            cm = json.loads((Path(control)/"manifest.json").read_text())
            if cm["media"]["source_sha256"] != media["source_sha256"]:
                errors.append("Controls use different source media")
            other_dose = {str(nid): [] for nid in groups["touch"]}
            for tr in cm["trials"]:
                if tr["phase"] == "audiovisual_experience":
                    for rel in range(tr["stop"]-tr["start"]):
                        source = (rel+tr["audio_shift"]) % media["clip_ticks"]
                        for index, nid in enumerate(groups["touch"]):
                            if rel % 4 == nid % 4 and auditory[source, index] > 0:
                                other_dose[str(nid)].append(float(auditory[source, index]))
            if any(sorted(dose[k]) != sorted(other_dose[k]) for k in dose):
                errors.append("Audio amplitude multiset is not preserved by the control")
    return {"valid_recording": not errors, "errors": errors, "ticks": summary["ticks"],
            "neurons": manifest["size"], "findings": findings, "totals": totals, "trials": trials,
            "limits": "Structural audit and descriptive causal warnings, not acceptance of hierarchical learning. Exclusion follows the explicit stimulus schedule; queues can deliver earlier events after a condition changes."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--regional", action="store_true")
    args = parser.parse_args()
    result = audit_regional(args.recording, args.control) if args.regional else audit(args.recording, args.control)
    text = json.dumps(result, allow_nan=False, indent=2)
    if args.output:
        args.output.write_text(text+"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("trials", "totals")}))
    if not result["valid_recording"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
