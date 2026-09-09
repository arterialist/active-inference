"""Independent event-ledger reconstruction of bounded local plasticity.

Recorded error magnitude is checked against native float32 input arithmetic;
the update is independently calculated from the scalar differential equation.
Per-port event chains must reproduce every saved training endpoint. Cellular
rasters independently determine spike ages, t_ref and effective rate gains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def expected_magnitude(q, error, direction, eta, cap, decay):
    """Vectorized scalar-flow solution, independent of the neuron extension."""
    a, b = error-decay, error/cap
    z = np.exp(-a*eta)
    quotient = np.empty_like(a)
    zero = a == 0
    quotient[zero] = eta[zero]
    quotient[~zero] = -np.expm1(-a[~zero]*eta[~zero])/a[~zero]
    positive = q/(z+b*q*quotient)
    negative = q*np.exp(-eta*(error+decay))
    return np.where(direction > 0, positive, negative)


def verify_events(events, ports, trial, cells, last_fire, previous, eta_base, cap, decay):
    """Return next per-port weights and quantitative reconstruction errors."""
    if len(events) and (events["tick"].min() < trial["start"] or events["tick"].max() >= trial["stop"] or
                       events["port"].min() < 0 or events["port"].max() >= len(ports)):
        raise ValueError("Event index outside recording")
    ages = np.empty(cells.shape[:2], dtype=np.int32)
    for j, row in enumerate(cells):
        t = trial["start"]+j
        last_fire[row[:, 1] > 0] = t
        ages[j] = np.where(last_fire >= 0, t-last_fire, -1)
    p, ticks = events["port"], events["tick"]-trial["start"]
    targets = np.array([port[0]-1 for port in ports])[p]
    if not ((events["input"][:, 0] > 0).all() and (events["eta"] > 0).all() and
            all(np.isfinite(events[name]).all() for name in ("input", "before", "plast", "error", "eta", "t_ref", "after"))):
        raise ValueError("Invalid active input, learning rate or finite state")
    if not (np.array_equal(events["age"], ages[ticks, targets]) and
            np.array_equal(events["spike"], cells[ticks, targets, 1] > 0) and
            np.array_equal(events["t_ref"], cells[ticks, targets, 6])):
        raise ValueError("Credit timing disagrees with cellular trace")
    direction = np.where((events["age"] >= 0) & (events["age"] <= events["t_ref"]), 1, -1)
    if not np.array_equal(direction, events["direction"]):
        raise ValueError("Incorrect timing direction")
    if not np.allclose(events["eta"], eta_base[p]*cells[ticks, targets, 7], rtol=1e-14, atol=0):
        raise ValueError("Effective learning rate disagrees with neural rate receptor")
    x = events["input"]
    components = np.column_stack((x[:, 0]-events["before"].astype(np.float32),
                                  x[:, 1]-events["plast"].astype(np.float32), x[:, 2:]))
    errors = np.sqrt(np.sum(components*components, axis=1)).astype(float)
    # Different float32 norm reduction orders can differ by a few ulps.
    if not np.allclose(errors, events["error"], rtol=3e-7, atol=3e-7):
        raise ValueError("Local error magnitude disagrees with recorded input")
    predicted = np.copysign(expected_magnitude(abs(events["before"]), events["error"], direction,
                                             events["eta"], cap[p], decay[p]), events["before"])
    residual = float(np.max(abs(predicted-events["after"]), initial=0))
    if residual > 5e-12:
        raise ValueError("Recorded weight update violates the declared local equation")
    next_weights = previous.copy()
    ordered = events[np.lexsort((events["tick"], events["port"]))]
    if len(ordered):
        same = ordered["port"][1:] == ordered["port"][:-1]
        if (np.any(ordered["tick"][1:][same] <= ordered["tick"][:-1][same]) or
                not np.array_equal(ordered["before"][1:][same], ordered["after"][:-1][same])):
            raise ValueError("Duplicate event or broken within-port weight chain")
        starts = np.r_[True, ~same]
        stops = np.r_[~same, True]
        if not np.array_equal(ordered["before"][starts], previous[ordered["port"][starts]]):
            raise ValueError("Weight chain does not start at previous endpoint")
        next_weights[ordered["port"][stops]] = ordered["after"][stops]
    return next_weights, residual, float(np.max(abs(errors-events["error"]), initial=0))


def audit(recording, output):
    recording, output = Path(recording).resolve(), Path(output).resolve()
    summary = json.loads((recording/"summary.json").read_text())
    source = Path(summary["source_recording"])
    for name, digest in summary["source_files_sha256"].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source record changed: {name}")
    manifest = json.loads((source/"manifest.json").read_text())
    config = json.loads((source/"config.json").read_text())
    ports = summary["ports"]
    auditory, upper = set(manifest["groups"]["tactile_core"]), set(manifest["groups"]["upper_core"])
    allowed = {(tgt, sid, src, family) for src, tgt, sid, family, present in manifest["edges"]
               if present and ((tgt in auditory and family in ("crossmodal", "descending")) or
                               (tgt in upper and family == "ascending"))}
    if len(set(map(tuple, ports))) != len(ports) or set(map(tuple, ports)) != allowed:
        raise ValueError("Port index does not match the recorded graph")
    params = {n["id"]: n for n in config["neurons"]}
    points = {(p["neuron_id"], p["synapse_id"]): p for p in config["synaptic_points"] if p["type"] == "postsynaptic"}
    weights = np.array([points[n, s]["u_i"]["info"] for n, s, _, _ in ports])
    eta = np.array([params[n]["params"]["eta_post"] for n, *_ in ports])
    cap = np.array([params[n]["metadata"].get("plasticity_magnitude_cap", 10.) for n, *_ in ports])
    decay = np.array([params[n]["metadata"].get("plasticity_magnitude_decay", .02) for n, *_ in ports])
    initial = weights.copy()
    families = sorted({p[3] for p in ports})
    family_index = np.array([families.index(p[3]) for p in ports])
    # Existing saved global weight arrays follow neuron/port insertion order.
    index = [(n["id"], p["synapse_id"]) for n in config["neurons"] for p in config["synaptic_points"]
             if p["type"] == "postsynaptic" and p["neuron_id"] == n["id"]]
    lookup = {pair: i for i, pair in enumerate(index)}
    take = [lookup[n, s] for n, s, *_ in ports]
    last = np.full(len(config["neurons"]), -1, dtype=np.int32)
    fields = ("events", "positive_direction", "negative_direction", "weight_increases", "weight_decreases",
              "weight_delta", "eta_sum", "positive_above_input", "positive_age_gt_16")
    ticks = np.zeros((manifest["trials"][-1]["stop"], len(families), len(fields)))
    rows = []
    if len(summary["episodes"]) != len(manifest["trials"]) or not summary["full_final_snapshot_exact"]:
        raise ValueError("Incomplete verified replay")
    for meta, trial in zip(summary["episodes"], manifest["trials"]):
        i = meta["episode"]
        path, reference_path = recording/meta["file"], source/f"experience-{i:03d}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"] or hashlib.sha256(reference_path.read_bytes()).hexdigest() != meta["reference_sha256"]:
            raise ValueError("Event or reference recording changed")
        with np.load(path) as raw, np.load(reference_path) as reference:
            events = raw["events"]
            if len(events) != meta["events"]:
                raise ValueError("Event count mismatch")
            weights, residual, norm_residual = verify_events(events, ports, trial, reference["cells"], last, weights, eta, cap, decay)
            if not np.array_equal(weights, reference["incoming_info_after"][take]):
                raise ValueError("Event chains do not reconstruct saved endpoint")
        f, t = family_index[events["port"]], events["tick"]
        delta = events["after"]-events["before"]
        values = (np.ones(len(events)), events["direction"] > 0, events["direction"] < 0, delta > 0, delta < 0,
                  delta, events["eta"], (events["direction"] > 0) & (events["before"] > events["input"][:, 0]),
                  (events["direction"] > 0) & (events["age"] > 16))
        for k, value in enumerate(values):
            np.add.at(ticks[:, :, k], (t, f), value)
        rows.append({"episode": i, "phase": trial["phase"], "audio_clip": trial["audio_clip"],
                     "visual_clip": trial["visual_clip"], "events": len(events),
                     "max_update_residual": residual, "max_float32_norm_residual": norm_residual})
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"tick-credit.npz", ticks=ticks, fields=fields, families=families,
                        initial_weights=initial, final_weights=weights, port_family=family_index)
    result = {"structurally_valid": True, "seed": summary["seed"], "mapping": summary["mapping"],
              "recording": str(recording), "episodes": rows, "fields": fields, "families": families,
              "family_totals": {name: dict(zip(fields, ticks[:, j].sum(axis=0).tolist())) for j, name in enumerate(families)},
              "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limits": "Endpoint reconstruction verifies cumulative observed updates, not independently every potentially zero-effect arrival. Direction is not proof a particular input caused the postsynaptic spike. Positive direction can still produce net decay. Above-input growth is a property of this error-magnitude rule, not by itself a coding bug or a demonstrated cause of failed recall."}
    (output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.recording, args.output)
