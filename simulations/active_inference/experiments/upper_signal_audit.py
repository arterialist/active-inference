"""Independent recorded-signal checks and tick-level consequences of replay.

No neuron imports, fitted classifier or feedback to the simulated network.
Observe all windows, including onset. Pattern sensitivity does not by itself
demonstrate content-specific learning; the other-video control changes dose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .association_route_audit import spike_effect
from .multimodal_pairing_audit import readout, sensory_marginals

MODES = ("exact", "time_shuffle", "port_swap", "other_video")


def verify_replacement(native, other, delivered, ports, mode, transform):
    if delivered.shape != native.shape or delivered.dtype != np.float32 or not np.isfinite(delivered).all():
        raise ValueError("Invalid replacement tensor")
    time_index, port_index = np.array(transform["time_index"]), np.array(transform["port_index"])
    identity_t, identity_p = np.arange(len(native)), np.arange(len(ports))
    if mode == "exact":
        expected = native
        valid = np.array_equal(time_index, identity_t) and np.array_equal(port_index, identity_p)
    elif mode == "other_video":
        expected = other
        valid = np.array_equal(time_index, identity_t) and np.array_equal(port_index, identity_p)
    elif mode == "time_shuffle":
        window = transform["window"]
        valid = window == 32 and np.array_equal(port_index, identity_p)
        for start in range(0, len(native), 32):
            stop = min(start+32, len(native))
            valid &= np.array_equal(np.sort(time_index[start:stop]), identity_t[start:stop])
            valid &= np.array_equal(np.sort(delivered[start:stop], axis=0), np.sort(native[start:stop], axis=0))
        if not valid:
            raise ValueError("Timing control violates per-port/window conservation")
        expected = native[time_index]
    elif mode == "port_swap":
        expected = native.copy()
        valid = np.array_equal(time_index, identity_t)
        target_count = {}
        for i, (nid, _, _, _) in enumerate(ports):
            target_count.setdefault(nid, []).append(i)
        for positions in target_count.values():
            if len(positions) != 2:
                raise ValueError("Require two upper afferents per target")
            a, b = positions
            expected[:, a], expected[:, b] = native[:, b], native[:, a]
            valid &= port_index[a] == b and port_index[b] == a
            valid &= np.array_equal(delivered[:, a]+delivered[:, b], native[:, a]+native[:, b])
    else:
        raise ValueError(mode)
    if not valid or not np.array_equal(delivered, expected):
        raise ValueError("Replacement does not implement the declared control")


def audit(paired, swapped, output):
    paths = [Path(paired).resolve(), Path(swapped).resolve()]
    summaries = [json.loads((p/"summary.json").read_text()) for p in paths]
    sources = [Path(s["source_recording"]) for s in summaries]
    manifests = [json.loads((p/"manifest.json").read_text()) for p in sources]
    if ([s["mapping"] for s in summaries] != ["paired", "swapped"] or
            summaries[0]["seed"] != summaries[1]["seed"] or
            summaries[0]["source_hashes"] != summaries[1]["source_hashes"] or
            (sources[0]/"config.json").read_bytes() != (sources[1]/"config.json").read_bytes() or
            sensory_marginals(sources[0], manifests[0]) != sensory_marginals(sources[1], manifests[1])):
        raise ValueError("Assignment controls do not match")
    results, curves, controls = {}, {}, {}
    for path, summary, source, manifest in zip(paths, summaries, sources, manifests):
        mapping = summary["mapping"]
        routes, ports = Path(summary["routes"]), summary["ports"]
        if hashlib.sha256((routes/"summary.json").read_bytes()).hexdigest() != summary["route_summary_sha256"]:
            raise ValueError("Original route manifest changed")
        config = json.loads((source/"config.json").read_text())
        groups = manifest["groups"]
        expected_ports = sorted((e["target_neuron"], e["target_synapse"], e["source_neuron"], e["source_terminal"])
                                for e in config["connections"] if e["source_neuron"] in groups["upper_core"] and e["target_neuron"] in groups["tactile_core"])
        if list(map(tuple, ports)) != expected_ports:
            raise ValueError("Recorded ports differ from upper-to-auditory anatomy")
        expected = {(s, m, c) for s in ("initial", "trained") for m in MODES for c in (0, 1)}
        if len(summary["probes"]) != 16 or {(p["state"], p["mode"], p["clip"]) for p in summary["probes"]} != expected:
            raise ValueError("Missing factorial probes")
        if not summary["training_replay_exact"] or not summary["native_capture_exact"]:
            raise ValueError("Replay verification failed")
        with np.load(source/"parameters.npz") as data:
            weights = {"initial": data["initial_info"], "trained": data["learned_info"]}
        results[mapping] = {}
        for state in ("initial", "trained"):
            native = []
            for clip in (0, 1):
                with np.load(path/f"{state}-native-{clip}.npz") as data:
                    native.append(data["delivered"])
            state_rows = {}
            for mode in MODES:
                mode_rows = []
                for clip in (0, 1):
                    row = next(p for p in summary["probes"] if (p["state"], p["mode"], p["clip"]) == (state, mode, clip))
                    file = path/row["file"]
                    if hashlib.sha256(file.read_bytes()).hexdigest() != row["sha256"] or not row["parent_unchanged"]:
                        raise ValueError("Changed record or parent mutation")
                    if row["start_tick"] != (0 if state == "initial" else manifest["trials"][-1]["stop"]):
                        raise ValueError("Probe starts at wrong tick")
                    with np.load(file) as data, np.load(routes/f"{state}-intact-visual-{clip}.npz") as baseline:
                        cells, original = data["cells"], baseline["cells"]
                        if cells.shape != original.shape or not np.isfinite(cells).all() or not np.array_equal(data["incoming_info_before"], weights[state]):
                            raise ValueError("Bad cellular record or starting weights")
                        verify_replacement(native[clip], native[1-clip], data["replacement_delivery"], ports, mode, row["transform"])
                        if mode == "exact" and (not row["exact_replacement_verified"] or not np.array_equal(cells, original) or
                                not np.array_equal(data["incoming_info_after"], baseline["incoming_info_after"]) or
                                not np.array_equal(data["original_delivery"], native[clip])):
                            raise ValueError("Unchanged replay differs from original")
                    if state == "initial":
                        if mapping == "paired":
                            controls[mode, clip] = file
                        else:
                            with np.load(controls[mode, clip]) as ref:
                                if not np.array_equal(cells, ref["cells"]):
                                    raise ValueError("Untrained assignment controls differ")
                    observations = {}
                    for role in ("visual_core", "tactile_core", "upper_core"):
                        ids = np.array(groups[role])-1
                        a, b = original[:, ids, 1] > 0, cells[:, ids, 1] > 0
                        changed, signed, first = spike_effect(a, b)
                        prefix = f"{mapping}-{state}-{mode}-{clip}-{role}"
                        curves[prefix+"-changed_cells"], curves[prefix+"-signed_spike_change"] = changed, signed
                        curves[prefix+"-spikes"] = b.sum(axis=1)
                        observations[role] = {"spikes": int(b.sum()), "first_difference": first,
                                              "changed_entries": int(changed.sum())}
                        refs = []
                        for sound in (0, 1):
                            with np.load(source/f"probe-initial-audio-{sound}.npz") as ref:
                                refs.append(readout(ref["cells"], groups[role], "population_centered_rate"))
                        axis = refs[0]-refs[1]
                        norm = float(axis@axis)
                        if norm > 1e-16:
                            diff = b.astype(float)-a.astype(float)
                            diff -= diff.mean(axis=1, keepdims=True)
                            projection = diff@axis/norm
                            curves[prefix+"-auditory_axis_change"] = projection
                            observations[role]["axis_change_windows"] = [{"start": lo, "stop": min(hi, len(projection)),
                                "change": float(projection[lo:min(hi,len(projection))].mean())}
                                for lo, hi in ((0,32),(32,96),(96,160),(160,224),(224,len(projection))) if lo < len(projection)]
                    mode_rows.append({"clip": clip, "populations": observations})
                    del cells, original
                state_rows[mode] = mode_rows
            results[mapping][state] = state_rows
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"tick-effects.npz", **curves)
    result = {"structurally_valid": True, "seed": summaries[0]["seed"], "results": results,
        "recordings": list(map(str, paths)), "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limits": "Changes in output do not identify remembered semantics. Time shuffle conserves per-port sample multisets in each 32-tick window. Port swap conserves raw target input sums but not weighted/delayed current. Other-video substitution changes dose as well as content. Fixed initial auditory axes can miss temporal codes. Full-final-state equality relies on the tested replay instrument; recorded rasters and endpoint weights are independently checked here."}
    (output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--swapped", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.paired, args.swapped, args.output)
