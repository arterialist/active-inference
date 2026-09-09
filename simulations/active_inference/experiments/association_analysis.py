"""Streaming raw-trace audit for the adaptive-association preparation.

No PAULA/simulator import. Reconstructs rate gating, both adaptation directions,
modulator filtering, membrane integration and spikes from recorded inputs,
queues and parameters. Probe classifications are recomputed, not trusted from
the online success flag. All figures remain finite-protocol observations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from .composition_analysis import encode, rows


def audit(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if not manifest["observed"]:
        raise ValueError("Equation audit requires delivered-input recording")
    config = json.loads((directory / "config.json").read_text())
    distances = {(str(p["neuron_id"]), str(p["synapse_id"])): p["distance_to_hillock"]
                 for p in config["synaptic_points"] if p["type"] == "postsynaptic"}
    assert all(d >= 1 for d in distances.values()), "Zero-delay inputs require a different reconstruction"
    iterator = rows(directory)
    previous = next(iterator)["state"]
    assert previous["tick"] == 0
    initial = previous
    spikes = {i: [] for i in previous["neurons"]}
    trial_spikes = [Counter() for _ in manifest["trials"]]
    early_spikes = [Counter() for _ in manifest["trials"]]
    rate_integrals = [defaultdict(float) for _ in manifest["trials"]]
    errors = defaultdict(float)
    events = Counter()
    digest = hashlib.sha256()
    first_depression = None
    terminal_changes = Counter()
    terminal_ranges = {i: [previous["neurons"][i]["terminals"]["900"][0]]*2 for i in ("5", "6")}
    first_nonpositive_export = {}
    probe_margins = {}
    count = 0
    for count, row in enumerate(iterator, 1):
        t, state = count-1, row["state"]
        assert row["executed_tick"] == t and state["tick"] == count
        trial = manifest["trials"][row["trial"]]
        assert trial["start"] <= t < trial["stop"]
        digest.update(encode(state).encode())
        gains = {}
        for i, resolved in manifest["resolved"].items():
            p, meta = resolved["parameters"], resolved["metadata"]
            assert p["plasticity_mode"] == "legacy_multiplicative" and p["weight_decay_tau"] == 0.
            concentration = max(0., previous["neurons"][i]["M"][meta["plasticity_rate_index"]])
            gains[i] = 1.+meta["plasticity_rate_boost"]*concentration/(meta["plasticity_rate_half_saturation"]+concentration)
            recorded = state["plasticity_rates"][i]
            assert recorded["basal"] == p["eta_post"] == 1e-5 > 0.
            assert recorded.get("basal_retro", p["eta_retro"]) == p["eta_retro"] > 0.
            errors["rate"] = max(errors["rate"], abs(gains[i]-recorded["used_multiplier"]))
            if i in ("5", "6"):
                rate_integrals[row["trial"]][i] += p["eta_post"]*gains[i]
                assert row["delivered_inputs"][i][3][0] == row["delivered_inputs"][i][3][1] == 0.
        terminals = {(i, j): v[0] for i, n in previous["neurons"].items() for j, v in n["terminals"].items()}
        active_retro = set()
        for arrival, kind, event in previous["retrograde_wheel"]:
            if arrival == t:
                assert kind == "RetrogradeSignalEvent"
                i, j = str(event["target_neuron_id"]), str(event["target_terminal_id"])
                eta = manifest["resolved"][i]["parameters"]["eta_retro"]
                terminals[i, j] = max(-100., min(100., terminals[i, j]+eta*gains[i]*event["error_vector"][0]))
                events["retro_updates"] += 1
                active_retro.add(i)
        for i, n in state["neurons"].items():
            old = previous["neurons"][i]
            p = manifest["resolved"][i]["parameters"]
            delivered = row["delivered_inputs"][i]
            for channel in range(len(n["M"])):
                local = sum(port[channel+2]*old["synapses"][str(s)][2][channel] for s, port in enumerate(delivered))
                expected_m = p["gamma"][channel]*old["M"][channel]+(1-p["gamma"][channel])*p["nm_internal"]*local
                errors["modulator"] = max(errors["modulator"], abs(expected_m-n["M"][channel]))
            expected_f = p["beta_avg"]*old["F_avg"]+(1-p["beta_avg"])*old["O"]
            errors["firing_average"] = max(errors["firing_average"], abs(expected_f-n["F_avg"]))
            lower, upper = 2*p["c"], p["c"]*p["num_inputs"]
            expected_window = upper-(upper-lower)*max(0., min(1., expected_f*p["c"]))
            expected_window += sum(w*m for w, m in zip(p["w_tref"], n["M"], strict=True))
            expected_window = max(lower, min(upper, expected_window))
            errors["learning_window"] = max(errors["learning_window"], abs(expected_window-n["t_ref"]))
            drive = sum(a[2]*p["delta_decay"]**distances[i, str(a[3])] for a in old["dendritic_queue"] if a[0] <= t)
            expected_s = max(-1000., min(1000., old["S"]+(-old["S"]+drive)/p["lambda_param"]))
            elapsed = math.inf if old["t_last_fire"] is None else t-old["t_last_fire"]
            threshold = n["b"] if elapsed <= p["c"] else n["r"]
            if abs(expected_s) < .005:
                threshold = n["r"]
            expected_spike = expected_s >= threshold and elapsed >= p["c"]
            assert expected_spike == (n["O"] > 0.), (directory, t, i, expected_s, threshold)
            errors["membrane"] = max(errors["membrane"], abs(n["S"]-(0. if expected_spike else expected_s)))
            if i in ("5", "6") and trial["cue"] is not None and not trial["paired"]:
                key = trial["index"], i
                peak = probe_margins.setdefault(key, {"trial": trial["index"], "phase": trial["phase"],
                    "cue": trial["cue"], "neuron": i, "expected_prediction": i == str(5+trial["outcome"]),
                    "period": trial["cue_period"], "maximum_membrane_before_reset": -math.inf,
                    "maximum_threshold_margin": -math.inf})
                peak["maximum_membrane_before_reset"] = max(peak["maximum_membrane_before_reset"], expected_s)
                peak["maximum_threshold_margin"] = max(peak["maximum_threshold_margin"], expected_s-threshold)
            if expected_spike:
                spikes[i].append(t)
                trial_spikes[row["trial"]][i] += 1
                if t < trial["start"]+24:
                    early_spikes[row["trial"]][i] += 1
            for j, terminal in n["terminals"].items():
                errors["retro"] = max(errors["retro"], abs(terminal[0]-terminals[i, j]))
                if i in terminal_ranges and j == "900":
                    terminal_ranges[i][0] = min(terminal_ranges[i][0], terminal[0])
                    terminal_ranges[i][1] = max(terminal_ranges[i][1], terminal[0])
                    if terminal[0] <= 0 and i not in first_nonpositive_export:
                        first_nonpositive_export[i] = {"tick": t, "value": terminal[0]}
                    if i in active_retro:
                        regime = "basal" if gains[i] < 1.01 else "modulated"
                        terminal_changes[regime+"_update_ticks"] += 1
                        terminal_changes[regime+"_changed_ticks"] += terminal[0] != old["terminals"][j][0]
                        requested_change = terminals[i, j]-old["terminals"][j][0]
                        if requested_change != 0.:
                            terminal_changes[regime+"_nonzero_requested_ticks"] += 1
                            terminal_changes[regime+"_unrepresented_ticks"] += terminal[0] == old["terminals"][j][0]
            for j, synapse in n["synapses"].items():
                prior = old["synapses"][j]
                info, plast, *mods = delivered[int(j)]
                expected = prior[0]
                if info > 0:
                    magnitude = math.sqrt((info-prior[0])**2+(plast-prior[1])**2+sum(m*m for m in mods))
                    age = math.inf if n["t_last_fire"] is None else t-n["t_last_fire"]
                    direction = 1. if age <= n["t_ref"] else -1.
                    delta = 1e-5*gains[i]*direction*magnitude*prior[0]
                    if delta > 0:
                        delta *= max(0., 1.-prior[0]/10.)
                    expected = max(p["w_min"], min(p["w_max"], prior[0]+delta-1e-5*gains[i]*.02*prior[0]))
                    events["post_updates"] += 1
                    if (first_depression is None and trial["phase"] == "reversal_train"
                        and i == str(6-trial["outcome"]) and j == str(trial["cue"])
                        and direction < 0 and gains[i] > 1.1 and expected < prior[0]):
                        first_depression = {"tick": t, "trial": trial["index"], "neuron": i,
                            "synapse": j, "since_last_spike": age, "t_ref": n["t_ref"],
                            "rate_multiplier": gains[i], "weight_before": prior[0], "weight_after": synapse[0]}
                errors["post"] = max(errors["post"], abs(synapse[0]-expected))
        if t == trial["stop"]-1:
            online = summary["trials"][row["trial"]]
            assert {i: trial_spikes[row["trial"]][i] for i in spikes} == online["spikes"]
            assert {i: early_spikes[row["trial"]][i] for i in ("5", "6")} == online["pre_us_prediction_spikes"]
            for i in ("5", "6"):
                assert abs(rate_integrals[row["trial"]][i]-online["rate_integrals"][i]) < 1e-10
                assert [n[0] for j, n in sorted(state["neurons"][i]["synapses"].items()) if j in ("0", "1")] == online["cue_weights"][i]
        previous = state
    assert count == summary["ticks"] == manifest["trials"][-1]["stop"]
    assert digest.hexdigest() == summary["state_digest"] and spikes == summary["spike_ticks"]
    assert all(e < 1e-5 for e in errors.values()), (directory, dict(errors))
    phases = defaultdict(Counter)
    false_settlement = []
    probe_validity = defaultdict(list)
    for trial, counts, early in zip(manifest["trials"], trial_spikes, early_spikes, strict=True):
        outcome = trial["outcome"]
        if outcome is None:
            continue
        good, bad = counts[str(5+outcome)], counts[str(6-outcome)]
        category = "ambiguous" if good and bad else "correct_only" if good >= 2 and not bad else "wrong_only" if bad else "silent_or_insufficient"
        phases[trial["phase"]][category] += 1
        probe_validity[trial["phase"]].append(category == "correct_only")
        phases[trial["phase"]]["modulator_spikes"] += counts["7"]+counts["8"]
        if trial["paired"] and early[str(6-outcome)] and not counts["7"]+counts["8"]:
            false_settlement.append(trial["index"])
    initially_naive = all(not counts["5"] and not counts["6"] for tr, counts in
        zip(manifest["trials"], trial_spikes, strict=True) if tr["phase"] == "before")
    retention = probe_validity["after_second"] + probe_validity["retention"]
    retained = bool(retention) and all(retention)
    challenge_checks = [v for phase, values in probe_validity.items()
                        if phase == "reversal_probe" or phase.startswith("transfer") for v in values]
    challenge_passed = all(challenge_checks) if challenge_checks else None
    assert initially_naive == summary["initially_naive"]
    assert retained == summary["both_associations_retained"]
    if "challenge_passed" in summary:
        assert challenge_passed == summary["challenge_passed"]
    return {"run": str(directory), "seed": manifest["seed"], "mode": manifest["mode"],
        "variant": manifest.get("variant", "original"), "challenge": manifest.get("challenge", "standard"),
        "mapping": manifest["mapping"], "order": manifest["order"], "ticks": count,
        "phases": dict(phases), "false_settlement_trials": false_settlement,
        "initially_naive": initially_naive, "both_associations_retained": retained,
        "challenge_passed": challenge_passed,
        "first_modulated_depression_of_old_mapping": first_depression,
        "prediction_terminal_updates": dict(terminal_changes),
        "prediction_terminal_ranges": terminal_ranges,
        "first_nonpositive_prediction_export": first_nonpositive_export,
        "probe_response_margins": list(probe_margins.values()),
        "events_checked": dict(events), "max_equation_residuals": dict(errors),
        "trace_summary_verified": True,
        "auditor_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "total_rate_exposure": {i: sum(r[i] for r in rate_integrals) for i in ("5", "6")},
        "initial_cue_weights": {i: [initial["neurons"][i]["synapses"][j][0] for j in ("0", "1")] for i in ("5", "6")},
        "final_cue_weights": {i: [previous["neurons"][i]["synapses"][j][0] for j in ("0", "1")] for i in ("5", "6")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = []
    for directory in args.directories:
        runs = [directory] if (directory / "summary.json").is_file() else sorted(p.parent for p in directory.glob("*/summary.json"))
        for run in runs:
            result = audit(run)
            results.append(result)
            print(encode({k: result[k] for k in ("run", "phases", "false_settlement_trials")}), flush=True)
    with args.output.open("x") as target:
        target.write(encode({"runs": results})+"\n")


if __name__ == "__main__":
    main()
