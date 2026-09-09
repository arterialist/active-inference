"""Streaming, equation-level checks of composition_probe records.

No PAULA import and no simulation writes. The output distinguishes persistence,
correct readout, and observer equivalence. It never calls silence success.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


def encode(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def rows(directory):
    with gzip.open(directory / "ticks.jsonl.gz", "rt") as source:
        yield from map(json.loads, source)


def source_projection(state):
    """Source-cell dynamics including recurrent terminal, excluding export 901.

    Separate-terminal intervention deliberately changes 901. Its zero-modulation
    feedback cannot change the recurrent cell state in this preparation.
    """
    return {i: dict(n, terminals={"900": n["terminals"]["900"]})
            for i, n in state["neurons"].items() if int(i) <= 6}


def first_difference(left, right, project=source_projection):
    a, b = rows(left), rows(right)
    for r, s in zip(a, b, strict=True):
        if project(r["state"]) != project(s["state"]):
            return r.get("executed_tick", -1)
    return None


def expected_ring_ticks(nid, phase, ticks):
    base, shift = (1, 0) if nid <= 3 else (4, phase)
    return list(range(4 + 7*((nid-base-shift) % 3), ticks, 21))


def analyze(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    iterator = rows(directory)
    previous = next(iterator)["state"]
    assert previous["tick"] == 0
    spikes = {str(i): [] for i in range(1, 9)}
    first_missing = {}
    expected = {str(i): set(expected_ring_ticks(i, manifest["phase"], manifest["ticks"]))
                for i in range(1, 7)}
    digest, source_digest = hashlib.sha256(), hashlib.sha256()
    max_retro_error = 0.
    retro_events = 0
    count = 0
    for count, record in enumerate(iterator, 1):
        t = count-1
        assert record["executed_tick"] == t
        state = record["state"]
        assert state["tick"] == t+1
        digest.update(encode(state).encode())
        source_digest.update(encode(source_projection(state)).encode())
        # Reconstruct terminal adaptation from events already queued last tick.
        q = {(i, j): v[0] for i, n in previous["neurons"].items()
             for j, v in n["terminals"].items()}
        for arrival, kind, event in previous["retrograde_wheel"]:
            if arrival != t:
                continue
            assert kind == "RetrogradeSignalEvent"
            key = str(event["target_neuron_id"]), str(event["target_terminal_id"])
            eta = manifest["resolved"][key[0]]["parameters"]["eta_retro"]
            q[key] = max(-100., min(100., q[key]+eta*event["error_vector"][0]))
            retro_events += 1
        for i, n in state["neurons"].items():
            assert math.isfinite(n["S"])
            if n["O"] > 0:
                spikes[i].append(t)
            for j, terminal in n["terminals"].items():
                max_retro_error = max(max_retro_error, abs(terminal[0]-q[i, j]))
            if i in expected and t in expected[i] and not n["O"] and i not in first_missing:
                old = previous["neurons"][i]
                arrivals = [a for a in old["dendritic_queue"] if a[0] == t]
                params = manifest["resolved"][i]["parameters"]
                # All recurrent synapses in this preparation have distance 6.
                drive = sum(a[2]*params["delta_decay"]**6 for a in arrivals)
                predicted_s = old["S"] + (-old["S"]+drive)/params["lambda_param"]
                first_missing[i] = {"tick": t, "S": n["S"], "r": n["r"],
                    "last_fire": n["t_last_fire"], "arrivals": arrivals,
                    "reconstructed_S": predicted_s,
                    "reconstruction_error": abs(predicted_s-n["S"]),
                    "cause": "subthreshold_arrival" if arrivals and n["S"] < n["r"]
                             else "no_arriving_pulse" if not arrivals else "inspect_trace"}
        previous = state
    assert count == manifest["ticks"], (directory, count, manifest["ticks"])
    assert digest.hexdigest() == summary["state_digest"], directory
    assert spikes == summary["spike_ticks"], directory
    # Floating-point array updates may round to float32 in the implementation.
    assert max_retro_error < 1e-5, (directory, max_retro_error)
    phase = manifest["phase"]
    winner, loser = str(7+phase), str(8-phase)
    ring_exact = all(spikes[i] == sorted(expected[i]) for i in expected)
    readout_exact = (manifest["attachment"] != "none"
                     and spikes[winner] == list(range(6, manifest["ticks"], 21))
                     and not spikes[loser])
    return {"run": directory.name, "ticks": count, "seed": manifest["seed"],
        "mode": manifest["mode"], "attachment": manifest["attachment"], "phase": phase,
        "ring_exact": ring_exact, "readout_exact": readout_exact,
        "first_missing": first_missing,
        "source_projection_digest": source_digest.hexdigest(),
        "retro_events_checked": retro_events, "max_retro_error": max_retro_error,
        "final_window_all_ring_cells_active": all(any(t >= count-105 for t in spikes[i])
                                                   for i in expected),
        "last_spike": {i: max(ts, default=None) for i, ts in spikes.items()},
        "trace_summary_verified": True}


def analyze_intervention(directory):
    """Check intervention histories without treating re-cueing as retention."""
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    digest, bare_digest = hashlib.sha256(), hashlib.sha256()
    spikes = {str(i): [] for i in range(1, 9)}
    actions, cues = [], []
    count = 0
    for record in rows(directory):
        if "executed_tick" not in record:
            continue
        t = record["executed_tick"]
        assert t == count
        count += 1
        state = record["state"]
        assert state["tick"] == count
        digest.update(encode(state).encode())
        bare_digest.update(encode({k: v for k, v in state.items() if k != "learning_rates"}).encode())
        if record["intervention"]:
            assert record["intervention"]["changes"]
            actions.append(t)
        if record["external"]:
            assert record["external"] == [[1, 1, 5.], [4+manifest["phase"], 1, 5.]]
            cues.append(t)
        for i, n in state["neurons"].items():
            if n["O"] > 0:
                spikes[i].append(t)
    assert count == manifest["ticks"]
    assert digest.hexdigest() == summary["state_digest"]
    assert spikes == summary["spike_ticks"]
    condition, intervention_tick = manifest["condition"], manifest["intervention_tick"]
    assert actions == ([] if condition == "none" else [intervention_tick])
    assert cues == ([3, intervention_tick] if condition == "restore_gains_and_reseed" else [3])
    winner, loser = str(7+manifest["phase"]), str(8-manifest["phase"])
    late = {i: [t for t in ts if t >= count-105] for i, ts in spikes.items()}
    late_active = all(late[str(i)] for i in range(1, 7))
    exact = all(spikes[str(i)] == expected_ring_ticks(i, manifest["phase"], count) for i in range(1, 7))
    reader_exact = spikes[winner] == list(range(6, count, 21)) and not spikes[loser]
    return {"run": directory.name, "seed": manifest["seed"], "phase": manifest["phase"],
        "ticks": count, "mode": f"{condition}@{intervention_tick}", "attachment": "shared",
        "ring_exact": exact, "readout_exact": reader_exact, "first_missing": {},
        "first_missing_not_analyzed": True, "final_window_all_ring_cells_active": late_active,
        "final_window_correct_reader_only": bool(late[winner]) and not late[loser],
        "post400_source_spikes": sum(sum(t>400 for t in spikes[str(i)]) for i in range(1, 7)),
        "post400_reader_spikes": sum(sum(t>400 for t in spikes[str(i)]) for i in (7, 8)),
        "state_digest_without_rates": bare_digest.hexdigest(),
        "intervention_ticks": actions, "external_cue_ticks": cues, "trace_summary_verified": True}


def analyze_gate(directory):
    """Audit local rate gating and both weight-update equations from raw traces."""
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    iterator = rows(directory)
    previous = next(iterator)["state"]
    initial = previous
    digest = hashlib.sha256()
    spikes = {str(i): [] for i in range(1, 10)}
    max_post_error = max_retro_error = max_rate_error = 0.
    post_events = retro_events = 0
    count = 0
    for count, record in enumerate(iterator, 1):
        t = count-1
        assert record["executed_tick"] == t
        state = record["state"]
        assert state["tick"] == count
        digest.update(encode(state).encode())
        gains = {}
        for i, resolved in manifest["resolved"].items():
            metadata = resolved["metadata"]
            concentration = max(0., previous["neurons"][i]["M"][metadata.get("plasticity_rate_index", 0)])
            half = metadata.get("plasticity_rate_half_saturation", .1)
            gains[i] = 1.+metadata.get("plasticity_rate_boost", 0.)*concentration/(half+concentration)
            rates = state["plasticity_rates"][i]
            assert rates["basal"] == resolved["parameters"]["eta_post"] == resolved["parameters"]["eta_retro"] == 1e-5
            max_rate_error = max(max_rate_error, abs(gains[i]-rates["used_multiplier"]))
        q = {(i, j): v[0] for i, n in previous["neurons"].items() for j, v in n["terminals"].items()}
        for arrival, kind, event in previous["retrograde_wheel"]:
            if arrival == t:
                i, j = str(event["target_neuron_id"]), str(event["target_terminal_id"])
                q[i, j] = max(-100., min(100., q[i, j]+1e-5*gains[i]*event["error_vector"][0]))
                retro_events += 1
        for i, n in state["neurons"].items():
            if n["O"] > 0:
                spikes[i].append(t)
            if int(i) <= 8:
                assert record["delivered_inputs"][i][2][0] == 0.
            for j, terminal in n["terminals"].items():
                max_retro_error = max(max_retro_error, abs(terminal[0]-q[i, j]))
            for j, synapse in n["synapses"].items():
                old = previous["neurons"][i]["synapses"][j]
                info, plast, *mods = record["delivered_inputs"][i][int(j)]
                predicted = old[0]
                if info > 0:
                    magnitude = math.sqrt((info-old[0])**2 + (plast-old[1])**2 + sum(x*x for x in mods))
                    direction = 1. if n["t_last_fire"] is not None and t-n["t_last_fire"] <= n["t_ref"] else -1.
                    delta = 1e-5*gains[i]*direction*magnitude*old[0]
                    if delta > 0:
                        delta *= max(0., 1.-old[0]/10.)
                    predicted = max(-100., min(100., old[0]+delta-1e-5*gains[i]*.02*old[0]))
                    post_events += 1
                max_post_error = max(max_post_error, abs(synapse[0]-predicted))
        previous = state
    assert count == manifest["ticks"]
    assert digest.hexdigest() == summary["state_digest"] and spikes == summary["spike_ticks"]
    assert max_rate_error < 1e-10 and max_post_error < 1e-5 and max_retro_error < 1e-5
    phase = manifest["phase"]
    exact = all(spikes[str(i)] == expected_ring_ticks(i, phase, count) for i in range(1, 7))
    reader_exact = spikes[str(7+phase)] == list(range(6, count, 21)) and not spikes[str(8-phase)]
    assert exact == summary["ring_exact"] and reader_exact == summary["readout_exact"]
    missing = {str(i): {"tick": min(set(expected_ring_ticks(i, phase, count))-set(spikes[str(i)]))}
               for i in range(1, 7) if set(expected_ring_ticks(i, phase, count))-set(spikes[str(i)])}
    return {"run": directory.name, "seed": manifest["seed"], "phase": phase, "ticks": count,
        "mode": manifest["mode"], "attachment": "shared+modulator", "ring_exact": exact,
        "readout_exact": reader_exact, "first_missing": missing,
        "post_events_checked": post_events, "retro_events_checked": retro_events,
        "max_post_error": max_post_error, "max_retro_error": max_retro_error, "max_rate_error": max_rate_error,
        "recurrent_weight_changes": {str(i): state["neurons"][str(i)]["synapses"]["0"][0]
             - initial["neurons"][str(i)]["synapses"]["0"][0] for i in range(1, 7)},
        "trace_summary_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = []
    for batch in args.directories:
        for file in sorted(batch.glob("*/summary.json")):
            manifest = json.loads(file.with_name("manifest.json").read_text())
            analyzer = analyze_gate if "modulator_stimulus" in manifest else analyze_intervention if "condition" in manifest else analyze
            result = analyzer(file.parent)
            result["batch"] = batch.name
            results.append(result)
    if not results:
        parser.error("no completed records found")
    groups = defaultdict(list)
    for r in results:
        groups[r["batch"], r["mode"], r["attachment"]].append(r)
    grouped = []
    for key, rs in groups.items():
        missing = [min(e["tick"] for e in r["first_missing"].values())
                   for r in rs if r["first_missing"]]
        g = dict(zip(("batch", "mode", "attachment"), key))
        g.update(n=len(rs), ring_exact=sum(r["ring_exact"] for r in rs),
                 readout_exact=sum(r["readout_exact"] for r in rs),
                 first_missing_range=[min(missing), max(missing)] if missing else None)
        grouped.append(g)
        print(encode(g), flush=True)
    with args.output.open("x") as target:
        target.write(encode({"groups": grouped, "runs": results})+"\n")


if __name__ == "__main__":
    main()
