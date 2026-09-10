"""Replay the complete target history and compare closed-loop step conditions.

Checks beyond counts: unchanged anatomy, exact target currents, somatic state,
postsynaptic adaptation and the first changed tick. Return events to target
terminals are not replayed; their weights are recorded, not asserted equal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .connectome import Subgraph
from .electrophysiology import CurrentElectrode
from .pn_current import dl5_cut
from .pn_current_steps import CONDITIONS, prepare, step_course, audit_trace
from .prisco import digest, dump_new


def first_true(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def replay_condition(graph, intrinsic, tail, directory):
    manifest = json.loads((directory/"analysis.json").read_text())
    if digest(directory/"structure.npz") != manifest["structure_sha256"]:
        raise ValueError("Changed structure record")
    for path, expected in manifest["source_hashes"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Changed experiment source: {path}")
    with np.load(directory/"structure.npz") as f:
        roots = f["roots"].tolist()
    root, cut = dl5_cut(graph)
    row = roots.index(root)
    kc_rows = [i for i, r in enumerate(roots) if graph.nodes[r]["annotation"]["cell_class"] == "Kenyon_Cell"]
    pn_rows = [i for i, r in enumerate(roots) if graph.nodes[r]["annotation"]["cell_class"] == "ALPN" and r != root]
    prep, target = prepare(cut, intrinsic, tail)
    np.testing.assert_array_equal([p.u_i.info for p in target.postsynaptic_points.values()], manifest["initial_post_weight"])
    command, epochs = step_course(manifest["protocol"]["current_levels_pa"])
    if epochs != manifest["epochs"]:
        raise ValueError("Declared epochs differ from implemented course")
    proposal = intrinsic["proposals"]["pooled_control"]
    trace = np.zeros((len(command), 11))
    apl = np.zeros((len(command), 7))
    cell_counts = np.zeros((len(epochs), len(roots)), dtype=np.int64)
    current_sum = np.zeros(target.params.num_inputs)
    first_input = np.full(target.params.num_inputs, -1, dtype=int)
    first_current = first_input.copy()
    terminal_last, post_last = None, None
    last_fire, previous_S, cursor = -np.inf, 0., 0
    checked_values = 0
    with CurrentElectrode(target, command/proposal["rheobase_pa"]) as electrode:
        for record in manifest["chunks"]:
            start, stop = record["start"], record["stop"]
            if start != cursor or stop <= start or stop > len(command):
                raise ValueError("Recording gap, overlap or extra ticks")
            path = directory/record["file"]
            if digest(path) != record["sha256"]:
                raise ValueError("Changed trace hash")
            with np.load(path) as a:
                if any(len(a[k]) != stop-start or not np.isfinite(a[k]).all() for k in a.files):
                    raise ValueError("Incomplete or nonfinite recording")
                observed = a["trace"]
                last_fire = audit_trace(observed, command[start:stop], proposal,
                    first_tick=start, last_fire=last_fire, initial_S=previous_S)
                previous_S = float(observed[-1, 5])
                np.testing.assert_array_equal(a["soma"][:, row], observed[:, 5:8])
                inputs, port_current, post = a["inputs"], a["port_current"], a["post_weight"]
                for tick in range(start, stop):
                    local = tick-start
                    target.input_buffer[:] = inputs[local]
                    target.tick({}, tick)
                    expected = [target.S, target.O, target.F_avg, target.t_ref, target.r, target.b]
                    np.testing.assert_array_equal(expected, observed[local, 5:])
                    np.testing.assert_array_equal(electrode.native_current[tick], observed[local, 2])
                    np.testing.assert_array_equal(electrode.total_current[tick], observed[local, 3])
                    np.testing.assert_array_equal(target.last_port_current, port_current[local])
                    np.testing.assert_array_equal([p.u_i.info for p in target.postsynaptic_points.values()], post[local])
                checked_values += (stop-start)*(8+2*target.params.num_inputs)
                for e, epoch in enumerate(epochs):
                    lo, hi = max(start, epoch["start"]), min(stop, epoch["stop"])
                    if lo < hi:
                        cell_counts[e] += (a["soma"][lo-start:hi-start, :, 1] > 0).sum(axis=0)
                for port in range(target.params.num_inputs):
                    if first_input[port] < 0:
                        t = first_true(inputs[:, port, 0] > 0)
                        if t is not None:
                            first_input[port] = start+t
                    if first_current[port] < 0:
                        t = first_true(port_current[:, port] != 0)
                        if t is not None:
                            first_current[port] = start+t
                current_sum += port_current.sum(axis=0)
                trace[start:stop] = observed
                apl[start:stop] = a["apl"]
                post_last, terminal_last = post[-1].copy(), a["terminal_weight"][-1].copy()
            cursor = stop
    if cursor != len(command):
        raise ValueError("Incomplete commanded course")
    epoch_reports = []
    for e, epoch in enumerate(epochs):
        segment = trace[epoch["start"]:epoch["stop"]]
        ticks = np.flatnonzero(segment[:, 6] > 0)+epoch["start"]
        native_charge = float(segment[:, 2].sum())
        injected_charge = float(np.sum(segment[:, 1]/proposal["rheobase_pa"]))
        recruited = [{"root": roots[i], "type": graph.nodes[roots[i]]["annotation"]["hemibrain_type"],
                      "spikes": int(cell_counts[e, i])} for i in pn_rows+kc_rows if cell_counts[e, i]]
        epoch_reports.append({**epoch, "target_spikes": len(ticks), "target_spike_ticks": ticks.tolist(),
            "target_rate_hz_nominal_clock": len(ticks)*1000/(epoch["stop"]-epoch["start"]),
            "native_current_sum_model_tick_units": native_charge,
            "injected_current_sum_model_tick_units": injected_charge,
            "abs_net_native_charge_over_injected_charge": abs(native_charge)/injected_charge if injected_charge else None,
            "kc_spikes": int(cell_counts[e, kc_rows].sum()), "other_pn_spikes": int(cell_counts[e, pn_rows].sum()),
            "recruited": recruited})
    inputs_report = []
    for port, meta in enumerate(manifest["target_inputs"]):
        if first_input[port] >= 0 or first_current[port] >= 0:
            inputs_report.append({**meta, "port": port, "first_receptor_tick": int(first_input[port]),
                "first_current_tick": int(first_current[port]), "signed_current_sum_model_tick_units": float(current_sum[port])})
    if first_input[-1] >= 0:
        raise ValueError("Electrical current was incorrectly injected into the experimental receptor")
    result = {"condition": manifest["condition"], "epochs": epoch_reports,
        "exact_replay_ticks": len(command), "exact_replay_scalar_values": checked_values,
        "exact_replay_scope": "all target currents, S/O/F_avg/t_ref/r/b and postsynaptic weights; not terminal return-event history",
        "active_target_inputs": inputs_report,
        "target_post_coefficients_changed": int(np.count_nonzero(post_last != manifest["initial_post_weight"])),
        "target_terminal_coefficients_changed": int(np.count_nonzero(terminal_last != manifest["initial_terminal_weight"])),
        "apl_first_activity_tick": first_true(apl[:, 1] > 0),
        "apl_max_compartment_voltage": float(apl[:, 2].max()), "apl_max_local_release": float(apl[:, 3].max()),
        "apl_attempted_forward_events": int(apl[:, 4].sum()), "apl_delivered_forward_events": int(apl[:, 5].sum()),
        "apl_returned_events": int(apl[:, 6].sum()),
        "record_manifest_sha256": digest(directory/"analysis.json")}
    return result, trace, manifest


def analyze(graph_path, intrinsic_path, tail_path, directory, output):
    if output.exists():
        raise FileExistsError(output)
    graph = Subgraph.load(graph_path)
    intrinsic = json.loads(intrinsic_path.read_text())
    tail = json.loads(tail_path.read_text())["fits"]["2"]["all_cells"]
    results, traces, manifests = {}, {}, {}
    for condition in CONDITIONS:
        results[condition], traces[condition], manifests[condition] = replay_condition(
            graph, intrinsic, tail, directory/condition)
    for condition in CONDITIONS:
        for key in ("root", "epochs", "target_inputs", "initial_post_weight", "initial_terminal_weight", "source_hashes"):
            if manifests[condition][key] != manifests["isolated"][key]:
                raise ValueError(f"Unmatched conditions: {key}")
    with np.load(directory/"intact/structure.npz") as intact, np.load(directory/"apl_release_block/structure.npz") as blocked:
        for key in intact.files:
            np.testing.assert_array_equal(intact[key], blocked[key])
    if (results["apl_release_block"]["apl_delivered_forward_events"] != 0 or
            results["intact"]["apl_delivered_forward_events"] != results["intact"]["apl_attempted_forward_events"]):
        raise ValueError("Wrong release intervention")
    comparisons = []
    for left, right in (("intact", "isolated"), ("apl_release_block", "isolated"), ("intact", "apl_release_block")):
        a, b = traces[left], traces[right]
        comparisons.append({"left": left, "right": right,
            "first_S_difference_tick": first_true(a[:, 5] != b[:, 5]),
            "first_spike_difference_tick": first_true(a[:, 6] != b[:, 6]),
            "spike_disagreement_ticks": np.flatnonzero(a[:, 6] != b[:, 6]).tolist(),
            "max_abs_S_difference": float(np.max(np.abs(a[:, 5]-b[:, 5]))),
            "epoch_spike_count_difference": [x["target_spikes"]-y["target_spikes"] for x,y in
                zip(results[left]["epochs"], results[right]["epochs"], strict=True)]})
    report = {"schema": 1, "conditions": results, "comparisons": comparisons,
        "intact_and_blocked_bindings_exact": True,
        "release_count_scope": "The raw field named delivered_forward_events counts tuples admitted through the intervention to the network router. It includes outgoing boundary terminals without simulated receivers, so it is not a count of receptor deliveries.",
        "analysis_source_sha256": digest(Path(__file__)),
        "source_hashes": {str(p.resolve()): digest(p) for p in (graph_path/"manifest.json", intrinsic_path, tail_path)},
        "claim": "Closed-loop feedback sensitivity with exact target replay; no source step-response or behavior acceptance"}
    dump_new(output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("graph", "intrinsic", "tail", "directory", "output"):
        p.add_argument(arg, type=Path)
    a = p.parse_args()
    analyze(a.graph, a.intrinsic, a.tail, a.directory, a.output)


if __name__ == "__main__":
    main()
