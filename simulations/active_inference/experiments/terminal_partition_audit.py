"""Independent data-only audit of the terminal-sharing learning intervention."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .multimodal_pairing_audit import readout, contrast, sensory_marginals, weight_health
from .population_state_audit import KINDS, reference_comparison, tick_projection


def wiring_checks(config, original, edges, mode):
    """Check actual edges and release vectors without invoking the builder."""
    old_terms = {(p["neuron_id"], p["terminal_id"]): p for p in original["synaptic_points"] if p["type"] == "presynaptic"}
    new_terms = {(p["neuron_id"], p["terminal_id"]): p for p in config["synaptic_points"] if p["type"] == "presynaptic"}
    family = {(s, t, i): f for s, t, i, f, present in edges if present}
    checks = {"same_neurons": config["neurons"] == original["neurons"],
              "same_external_inputs": config["external_inputs"] == original["external_inputs"],
              "same_incoming_points": [p for p in config["synaptic_points"] if p["type"] == "postsynaptic"] == [p for p in original["synaptic_points"] if p["type"] == "postsynaptic"],
              "only_terminals_rewired": len(config["connections"]) == len(original["connections"]),
              "same_initial_edge_release": True, "terminal_ids_unique": True,
              "correct_family_partition": True}
    ids = [(p["neuron_id"], p.get("synapse_id", p.get("terminal_id"))) for p in config["synaptic_points"]]
    checks["terminal_ids_unique"] = len(ids) == len(set(ids))
    assigned = defaultdict(set)
    for old, new in zip(original["connections"], config["connections"]):
        checks["only_terminals_rewired"] &= {k: v for k, v in old.items() if k != "source_terminal"} == {k: v for k, v in new.items() if k != "source_terminal"}
        a = old_terms[old["source_neuron"], old["source_terminal"]]
        b = new_terms[new["source_neuron"], new["source_terminal"]]
        checks["same_initial_edge_release"] &= {k: v for k, v in a.items() if k != "terminal_id"} == {k: v for k, v in b.items() if k != "terminal_id"}
        assigned[new["source_neuron"], new["source_terminal"]].add(family[new["source_neuron"], new["target_neuron"], new["target_synapse"]])
    if mode == "family":
        checks["correct_family_partition"] = all(len(v) == 1 for v in assigned.values())
    elif mode == "shared":
        checks["correct_family_partition"] = config == original
    elif mode == "shuffled":
        checks["correct_family_partition"] = any(len(v) > 1 for v in assigned.values())
    else:
        raise ValueError(mode)
    return checks


def matched_control_checks(config, control):
    fanout = lambda cfg: Counter((c["source_neuron"], c["source_terminal"]) for c in cfg["connections"])
    stripped = lambda cfg: [{k: v for k, v in c.items() if k != "source_terminal"} for c in cfg["connections"]]
    return {"control_same_neurons": config["neurons"] == control["neurons"],
            "control_same_synaptic_points": config["synaptic_points"] == control["synaptic_points"],
            "control_same_external_inputs": config["external_inputs"] == control["external_inputs"],
            "control_same_terminal_fanout": fanout(config) == fanout(control),
            "control_same_target_wiring": stripped(config) == stripped(control)}


def audit(paired, swapped, output, matched_control=None):
    paths, output = [Path(paired), Path(swapped)], Path(output)
    manifests = [json.loads((p/"manifest.json").read_text()) for p in paths]
    summaries = [json.loads((p/"summary.json").read_text()) for p in paths]
    if [m["mapping"] for m in manifests] != ["paired", "swapped"]:
        raise ValueError("Supply paired then swapped")
    sources = [Path(m["source_recording"]) for m in manifests]
    checks = {"same_seed": manifests[0]["seed"] == manifests[1]["seed"],
              "same_partition": summaries[0]["mode"] == summaries[1]["mode"],
              "same_config": (paths[0]/"config.json").read_bytes() == (paths[1]/"config.json").read_bytes(),
              "same_runtime_sources": manifests[0]["source_hashes"] == manifests[1]["source_hashes"],
              "matched_sensory_marginals": sensory_marginals(sources[0], manifests[0]) == sensory_marginals(sources[1], manifests[1]),
              "source_records_unchanged": True, "declared_probes": True,
              "finite_and_correct_shape": True, "correct_starting_weights": True,
              "continuation_parent_identity": True, "initial_probes_identical": True}
    if matched_control is not None:
        control = Path(matched_control)
        cm = json.loads((control/"manifest.json").read_text())
        checks["control_same_seed"] = cm["seed"] == manifests[0]["seed"]
        checks["control_complementary_partition"] = {cm["terminal_partition"]["mode"], summaries[0]["mode"]} == {"family", "shuffled"}
        checks.update(matched_control_checks(json.loads((paths[0]/"config.json").read_text()),
                                             json.loads((control/"config.json").read_text())))
    output_values, trajectories, healths = {}, {}, {}
    first_initial = None
    for path, manifest, summary, source in zip(paths, manifests, summaries, sources):
        for name, expected in manifest["source_files_sha256"].items():
            checks["source_records_unchanged"] &= hashlib.sha256((source/name).read_bytes()).hexdigest() == expected
        cfg, original = [json.loads(p.read_text()) for p in (path/"config.json", source/"config.json")]
        for key, value in wiring_checks(cfg, original, manifest["edges"], summary["mode"]).items():
            checks[key] = checks.get(key, True) and value
        healths[manifest["mapping"]] = weight_health(path, manifest)
        with np.load(path/"parameters.npz") as raw:
            initial, learned = raw["initial_info"], raw["learned_info"]
        records = {}
        expected = {(c, s, i) for c, senses in (("initial", ("visual", "audio")),
                    ("continuation", ("visual", "audio")), ("incoming_info", ("visual",))) for s in senses for i in (0, 1)}
        got = [(p["condition"], p["sense"], p["clip"]) for p in summary["probes"]]
        checks["declared_probes"] &= set(got) == expected and len(got) == len(expected)
        for p in summary["probes"]:
            c, s, i = p["condition"], p["sense"], p["clip"]
            if c == "continuation":
                checks["continuation_parent_identity"] &= p["parent_unchanged"] and p["start_state_sha256"] == summary["trained_state_sha256"] and p["start_tick"] == manifest["trials"][-1]["stop"]
            with np.load(path/f"probe-{c}-{s}-{i}.npz") as raw:
                cells = raw["cells"]
                checks["finite_and_correct_shape"] &= cells.shape == (manifest["clip_ticks"], len(cfg["neurons"]), len(manifest["fields"])) and np.isfinite(cells).all() and np.isfinite(raw["incoming_info_after"]).all()
                checks["correct_starting_weights"] &= np.array_equal(raw["incoming_info_before"], initial if c == "initial" else learned)
                records[c, s, i] = cells
        if first_initial is None:
            first_initial = {k: v for k, v in records.items() if k[0] == "initial"}
        else:
            checks["initial_probes_identical"] &= all(np.array_equal(records[k], v) for k, v in first_initial.items())
        result = {}
        for role in ("tactile_core", "upper_core"):
            ids = manifest["groups"][role]
            result[role] = {}
            for kind in KINDS:
                def group(condition, sense):
                    return np.stack([readout(records[condition, sense, i], ids, kind) for i in (0, 1)])
                original_audio, trained_audio = group("initial", "audio"), group("continuation", "audio")
                before = contrast(group("initial", "visual"), original_audio)
                row = {"before_contrast": before}
                for condition in ("continuation", "incoming_info"):
                    value = reference_comparison(group(condition, "visual"), original_audio, trained_audio)
                    value["change_on_original_reference"] = value["original_reference_contrast"]-before if before is not None and value["original_reference_contrast"] is not None else None
                    if kind == "population_centered_rate":
                        for name, ref in (("original", original_audio), ("trained_continuation", trained_audio)):
                            curve = tick_projection([records[condition, "visual", i] for i in (0, 1)], ids, ref)
                            if curve is not None:
                                trajectories[f'{manifest["mapping"]}/{role}/{condition}/{name}'] = curve
                                value[name+"_windows"] = [{"start": a, "stop": min(b, len(curve)), "contrast": float(curve[a:min(b,len(curve))].mean())} for a, b in ((0,32),(32,96),(96,160),(160,224),(224,len(curve))) if a < len(curve)]
                    row[condition] = value
                result[role][kind] = row
            output_values[manifest["mapping"]] = result
    checks = {k: bool(v) for k, v in checks.items()}
    if not all(checks.values()):
        raise AssertionError(checks)
    contrasts = {}
    for role in ("tactile_core", "upper_core"):
        contrasts[role] = {}
        for kind in KINDS:
            a, b = [output_values[m][role][kind]["continuation"]["change_on_original_reference"] for m in ("paired", "swapped")]
            contrasts[role][kind] = {"paired_change": a, "swapped_change": b,
                "both_change_directions_follow_assignment": bool(a > 0 and b < 0) if a is not None and b is not None else None}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"tick-projections.npz", **trajectories)
    result = {"structural_checks": checks, "structurally_valid": True,
              "seed": manifests[0]["seed"], "mode": summaries[0]["mode"],
              "full_state_directional_readouts": contrasts, "results": output_values,
              "weight_health": healths, "recordings": [str(p.resolve()) for p in paths],
              "matched_control": str(Path(matched_control).resolve()) if matched_control is not None else None,
              "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limits": "Descriptive directions are not acceptance, statistical significance or semantic recognition. Report all readouts and windows, compare the shuffled control and replicate. Current sound references are from intact continuation even when projecting incoming-only probes. This audit verifies records and topology, not every hidden starting variable independently."}
    (output/"summary.json").write_text(json.dumps(result, allow_nan=False, indent=2)+"\n")
    print(json.dumps({"structurally_valid": True, "full_state_directional_readouts": contrasts}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--swapped", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matched-control", type=Path)
    args = parser.parse_args()
    audit(args.paired, args.swapped, args.output, args.matched_control)
