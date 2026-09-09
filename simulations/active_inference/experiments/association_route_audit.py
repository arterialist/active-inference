"""Data-only route audit. Never feeds a score back into a neural population.

Export tick-resolved intervention effects and fixed-reference projections.
Spike changes establish causal influence, not remembered content. The linear
reference is an observer, not a neural consumer; sequence codes can be missed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .multimodal_pairing_audit import readout, sensory_marginals

CONDITIONS = ("intact", "direct_cut", "descending_cut", "both_cut")


def pathway_envelope(config, groups, source_role, trained=None):
    """Upper bound for a fixed-coefficient, pathway-only linear membrane.

    One source with cooldown c can release no more often than once per c
    ticks. For a=1-1/lambda and effective pulse W, its maximum steady peak is
    (W/lambda)/(1-a**c). Sum individually phase-aligned positive contributions.
    Dendritic decay is included. Inhibition and firing resets can only lower
    this envelope when lambda>=1. Start at rest or below the envelope.

    This does NOT bound the full plastic network: weights/release can change,
    other pathways supply current, and modulators can change thresholds. It
    is an operating-range calculation, not a frozen-learning simulation.
    """
    params = {n["id"]: n["params"] for n in config["neurons"]}
    posts = {(p["neuron_id"], p["synapse_id"]): p for p in config["synaptic_points"] if p["type"] == "postsynaptic"}
    pres = {(p["neuron_id"], p["terminal_id"]): p for p in config["synaptic_points"] if p["type"] == "presynaptic"}
    targets, sources = set(groups["tactile_core"]), set(groups[source_role])
    envelope = {nid: 0. for nid in sorted(targets)}
    for e in config["connections"]:
        src, tgt, sid, term = e["source_neuron"], e["target_neuron"], e["target_synapse"], e["source_terminal"]
        if src not in sources or tgt not in targets:
            continue
        p = posts[tgt, sid]
        if trained is None:
            weight = p["u_i"]["info"]+p["u_i"].get("plast", 0.)
            release = pres[src, term]["u_o"]["info"]
        else:
            v = trained["neurons"][str(tgt)]["synapses"][str(sid)]
            weight = v[0]+v[1]
            release = trained["neurons"][str(src)]["terminals"][str(term)][0]
        lam, c = params[tgt]["lambda_param"], params[src]["c"]
        if lam < 1 or c < 1 or c != int(c):
            raise ValueError("Envelope requires lambda>=1 and positive integer cooldown")
        # Native input processing ignores non-positive presynaptic info.
        pulse = max(0., release)*max(0., weight)*params[tgt]["delta_decay"]**p["distance_to_hillock"]
        envelope[tgt] += (pulse/lam)/(1-(1-1/lam)**c)
    return {"neuron_ids": list(envelope), "upper_bounds": list(envelope.values()),
            "minimum": min(envelope.values()), "maximum": max(envelope.values()),
            "count_at_or_above_base_r": sum(v >= params[n]["r_base"] for n, v in envelope.items()),
            "scope": "Fixed coefficients, pathway alone, maximal-rate aligned release, rest/bounded starting S. Not a bound on changing weights or the full network."}


def spike_effect(intact, intervened):
    """Per-tick changed cells, signed spike change and first divergence."""
    if intact.shape != intervened.shape or intact.ndim != 2:
        raise ValueError("Use same-shaped tick-by-cell rasters")
    changed = np.count_nonzero(intact != intervened, axis=1)
    signed = intervened.astype(int).sum(axis=1)-intact.astype(int).sum(axis=1)
    times = np.flatnonzero(changed)
    return changed, signed, int(times[0]) if len(times) else None


def audit(paired, swapped, output):
    paths, output = [Path(paired).resolve(), Path(swapped).resolve()], Path(output).resolve()
    summaries = [json.loads((p/"summary.json").read_text()) for p in paths]
    sources = [Path(s["source_recording"]) for s in summaries]
    manifests = [json.loads((p/"manifest.json").read_text()) for p in sources]
    if [s["mapping"] for s in summaries] != ["paired", "swapped"]:
        raise ValueError("Supply paired then swapped")
    if (summaries[0]["seed"] != summaries[1]["seed"] or
            (sources[0]/"config.json").read_bytes() != (sources[1]/"config.json").read_bytes() or
            manifests[0]["source_hashes"] != manifests[1]["source_hashes"] or
            sensory_marginals(sources[0], manifests[0]) != sensory_marginals(sources[1], manifests[1])):
        raise ValueError("Pairing controls differ beyond assignment")
    expected = {(state, c, i) for state in ("initial", "trained") for c in CONDITIONS for i in (0, 1)}
    curves, results, initial_control, references, envelopes = {}, {}, {}, {}, {}
    for path, summary, source, manifest in zip(paths, summaries, sources, manifests):
        if not summary["full_training_snapshot_exact"] or summary["training_exact_episodes"] != list(range(len(manifest["trials"]))):
            raise ValueError("Training replay was not completely verified")
        for name, digest in summary["source_files_sha256"].items():
            if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source record changed: {name}")
        rows = summary["probes"]
        if len(rows) != 16 or {(r["state"], r["condition"], r["clip"]) for r in rows} != expected:
            raise ValueError("Incomplete or duplicated factorial probes")
        config = json.loads((source/"config.json").read_text())
        groups = manifest["groups"]
        with gzip.open(source/"training-final-state.json.gz", "rt") as stream:
            trained = json.load(stream)
        envelopes[summary["mapping"]] = {state: {role: pathway_envelope(config, groups, role, snapshot)
            for role in ("visual_core", "upper_core")}
            for state, snapshot in (("initial", None), ("trained", trained))}
        references[summary["mapping"]] = {str(source/f"probe-initial-{sense}-{clip}.npz"):
            hashlib.sha256((source/f"probe-initial-{sense}-{clip}.npz").read_bytes()).hexdigest()
            for sense in ("audio", "visual") for clip in (0, 1)}
        records = {}
        with np.load(source/"parameters.npz") as raw:
            weights = {"initial": raw["initial_info"], "trained": raw["learned_info"]}
        for r in rows:
            state, condition, clip = r["state"], r["condition"], r["clip"]
            if not r["parent_unchanged"] or r["start_tick"] != (0 if state == "initial" else manifest["trials"][-1]["stop"]):
                raise ValueError("Starting state or parent mutation mismatch")
            src_ids = set(groups["visual_core"] if condition in ("direct_cut", "both_cut") else [])
            src_ids.update(groups["upper_core"] if condition in ("descending_cut", "both_cut") else [])
            targets = set(groups["tactile_core"])
            expected_edges = {(c["source_neuron"], c["source_terminal"], c["target_neuron"], c["target_synapse"])
                              for c in config["connections"] if c["source_neuron"] in src_ids and c["target_neuron"] in targets}
            if set(map(tuple, r["removed_edges"])) != expected_edges or len(r["removed_edges"]) != len(expected_edges):
                raise ValueError("Removed edges differ from declared anatomical route")
            filename = path/r["file"]
            if hashlib.sha256(filename.read_bytes()).hexdigest() != r["sha256"]:
                raise ValueError("Probe record changed")
            with np.load(filename) as raw:
                cells = raw["cells"]
                if (cells.shape != (manifest["clip_ticks"], len(config["neurons"]), len(manifest["fields"])) or
                        not np.isfinite(cells).all() or not np.array_equal(raw["incoming_info_before"], weights[state]) or
                        not np.isfinite(raw["incoming_info_after"]).all()):
                    raise ValueError("Probe shape, finiteness or initial weights disagree")
                if state == "initial" and condition == "intact":
                    with np.load(source/f"probe-initial-visual-{clip}.npz") as original:
                        if not np.array_equal(cells, original["cells"]) or not np.array_equal(raw["incoming_info_after"], original["incoming_info_after"]):
                            raise ValueError("Intact initial replay differs")
            if state == "initial":
                key = condition, clip
                if summary["mapping"] == "paired":
                    initial_control[key] = filename
                else:
                    with np.load(initial_control[key]) as original:
                        if not np.array_equal(cells, original["cells"]):
                            raise ValueError("Initial pairing controls differ")
            # Retain lossless O>0 rasters and the exact per-tick extrema used
            # below, not sixteen full float64 cellular tensors in RAM. Full
            # cellular records stay untouched on disk for other analyses.
            records[state, condition, clip] = {"spikes": cells[:, :, 1] > 0,
                "extrema": {role: (cells[:, np.array(groups[role])-1, 0].max(axis=1),
                                    cells[:, np.array(groups[role])-1, 5].min(axis=1))
                            for role in ("visual_core", "tactile_core", "upper_core")}}
            del cells
        for state in ("initial", "trained"):
            if len({r["start_local_state_sha256"] for r in rows if r["state"] == state}) != 1:
                raise ValueError("Interventions did not share a starting local state")
        mapping = summary["mapping"]
        results[mapping] = {}
        for role in ("visual_core", "tactile_core", "upper_core"):
            ids = groups[role]
            refs = []
            for clip in (0, 1):
                with np.load(source/f"probe-initial-audio-{clip}.npz") as raw:
                    refs.append(readout(raw["cells"], ids, "population_centered_rate"))
            refs = np.array(refs)
            results[mapping][role] = {}
            for state in ("initial", "trained"):
                state_rows = {}
                for condition in CONDITIONS:
                    row = {"clips": []}
                    for clip in (0, 1):
                        record = records[state, condition, clip]
                        spikes = record["spikes"][:, np.array(ids)-1]
                        intact = records[state, "intact", clip]["spikes"][:, np.array(ids)-1]
                        changed, signed, first = spike_effect(intact, spikes)
                        prefix = f"{mapping}-{role}-{state}-{condition}-{clip}"
                        curves[prefix+"-changed_cells"] = changed
                        curves[prefix+"-signed_spike_change"] = signed
                        curves[prefix+"-spikes"] = spikes.sum(axis=1)
                        # S is post-reset on firing ticks. Its peak is not a
                        # pre-threshold margin for cells which have fired.
                        curves[prefix+"-max_post_tick_S"], curves[prefix+"-min_r"] = record["extrema"][role]
                        times = np.flatnonzero(spikes.any(axis=1))
                        row["clips"].append({"clip": clip, "spikes": int(spikes.sum()),
                            "active_cells": int(spikes.any(axis=0).sum()),
                            "first_spike": int(times[0]) if len(times) else None,
                            "last_spike": int(times[-1]) if len(times) else None,
                            "first_spike_difference_from_intact": first,
                            "changed_spike_entries": int(changed.sum())})
                    axis = refs[0]-refs[1]
                    norm2 = float(axis@axis)
                    curve = None
                    if norm2 > 1e-16:
                        signals = [records[state, condition, i]["spikes"][:, np.array(ids)-1].astype(float)
                                   for i in (0, 1)]
                        diff = signals[0]-signals[1]
                        diff -= diff.mean(axis=1, keepdims=True)
                        curve = diff@axis/norm2
                    row["reference_defined"] = curve is not None
                    if curve is not None:
                        curves[f"{mapping}-{role}-{state}-{condition}-content_projection"] = curve
                        row["windows"] = [{"start": a, "stop": min(b, len(curve)),
                            "contrast": float(curve[a:min(b, len(curve))].mean())}
                            for a, b in ((0,32),(32,96),(96,160),(160,224),(224,len(curve))) if a < len(curve)]
                    state_rows[condition] = row
                results[mapping][role][state] = state_rows
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"tick-effects.npz", **curves)
    result = {"structurally_valid": True, "seed": summaries[0]["seed"], "results": results,
              "recordings": list(map(str, paths)), "reference_files_sha256": references,
              "fixed_coefficient_pathway_envelopes": envelopes,
              "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "checks": "Complete exact training replay reported; files unchanged; initial intact replay independently exact; identical initial factorial controls; same starting incoming weights and local-state hashes across cuts; independently selected anatomical edges; matched sensory marginals.",
              "limits": "Two clips and two assignments, not category recognition. Read raw tick effects before window summaries. Route necessity does not prove that route stores the association. Content axes are fixed pre-training auditory responses and can miss temporal codes. Complete hidden starting-state identity relies on the tested replay/branch instrument, not on these eight-field records alone."}
    (output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--swapped", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.paired, args.swapped, args.output)
