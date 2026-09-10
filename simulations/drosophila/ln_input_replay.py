"""Conditional input-path tests against recorded whole-network LN histories.

Source spike trains stay those of the original intact network. These are
open-loop receiving-history interventions, not predictions of a lesioned
closed-loop brain. Native receiving learning is active. Outgoing return
histories were not recorded and are not reconstructed.
"""
from __future__ import annotations

import argparse
import copy
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .connectome import Subgraph
from .electrical_analysis import prefix
from .orn_onset import WATCH
from .paula import Dynamics, build_paula, Neuron
from .prisco import digest, dump_new

PN = "720575940617207185"
CONDITIONS = ("all", "DL5_only", "without_DL5", "ALLN_only", "without_ALLN",
              "DL5_gain_quarter", "DL5_gain_half", "DL5_gain_double", "none")
FIELDS = ("S", "O", "F_avg", "t_ref", "r", "b", "current")


def cut_cells(graph, roots):
    if not roots or len(set(roots)) != len(roots) or not set(roots) <= set(graph.selected):
        raise ValueError("Cannot promote an incomplete boundary cell")
    selected = np.array(roots, dtype=np.int64)
    rows = graph.edges[np.isin(graph.edges[:, 0], selected) | np.isin(graph.edges[:, 1], selected)]
    endpoints = set(roots) | {str(x) for x in rows[:, :2].flat}
    result = Subgraph(tuple(roots), {r: graph.nodes[r] for r in endpoints}, rows.copy(),
        {"parent_provenance": graph.provenance, "cut": "conditional receiving replay; all incident pairs retained"})
    result.validate()
    return result


def port_groups(graph, root, pn_root=PN):
    edges = sorted(graph.edges[graph.edges[:, 1] == int(root)], key=lambda e: int(e[8]))
    pn = np.array([str(e[0]) == pn_root for e in edges] + [False])
    ln = np.array([graph.nodes[str(e[0])]["annotation"]["cell_class"] == "ALLN" for e in edges] + [False])
    return pn, ln, edges


def replay(cell, inputs, pn_ports, ln_ports, condition):
    if condition not in CONDITIONS or inputs.shape[1:] != cell.input_buffer.shape:
        raise ValueError("Unknown condition or incompatible receptor history")
    if cell.t_last_fire != -np.inf or cell.S != 0:
        raise ValueError("Use a fresh initial cell")
    keep = np.ones(cell.params.num_inputs, dtype=bool)
    if condition == "DL5_only": keep = pn_ports.copy()
    elif condition == "without_DL5": keep = ~pn_ports
    elif condition == "ALLN_only": keep = ln_ports.copy()
    elif condition == "without_ALLN": keep = ~ln_ports
    elif condition == "none": keep[:] = False
    gain = {"DL5_gain_quarter": .25, "DL5_gain_half": .5, "DL5_gain_double": 2.}.get(condition, 1.)
    for i in np.flatnonzero(pn_ports):
        cell.postsynaptic_points[int(i)].u_i.info *= gain
    initial = np.array([p.u_i.info for p in cell.postsynaptic_points.values()])
    trace = np.zeros((len(inputs), len(FIELDS)))
    weights = np.zeros((len(inputs), cell.params.num_inputs))
    native = Neuron._hillock_current

    def measure(c, tick, dt):
        current = native(c, tick, dt)
        if c is cell:
            trace[tick, 6] = current
        return current

    with patch.object(Neuron, "_hillock_current", measure):
        for t, values in enumerate(inputs):
            cell.input_buffer[:] = values
            cell.input_buffer[~keep] = 0
            cell.tick({}, t)
            trace[t, :6] = [cell.S, cell.O, cell.F_avg, cell.t_ref, cell.r, cell.b]
            weights[t] = [p.u_i.info for p in cell.postsynaptic_points.values()]
    if cell.params.eta_post <= 0 or cell.params.eta_retro <= 0 or cell._ablation:
        raise ValueError("Learning disabled")
    return {"trace": trace, "weights": weights, "initial_weights": initial,
            "kept_ports": keep, "gain": np.array(gain)}


def response_summary(data, epochs=((0, 200), (200, 334), (334, 1200), (1200, 1600))):
    a = data["trace"]; spikes = np.flatnonzero(a[:, 1] > 0)
    return {"first_spike": int(spikes[0]) if len(spikes) else None,
        "spikes": int(len(spikes)),
        "changed_receiving_weights": int(np.count_nonzero(data["initial_weights"] != data["weights"][-1])),
        "epochs": [{"start": lo, "stop": min(hi, len(a)),
            "spikes": int(a[lo:hi, 1].sum()), "peak_abs_current": float(np.max(np.abs(a[lo:hi, 6]), initial=0))}
            for lo, hi in epochs if lo < len(a)]}


def run(graph_path, reference, output):
    if output.exists(): raise FileExistsError(output)
    graph = Subgraph.load(graph_path)
    meta, arrays, structure = prefix(reference, 1600, current_source=False)
    if meta["condition"] != "intact" or meta["watch_roots"] != list(WATCH):
        raise ValueError("Need the original intact four-LN receiving record")
    core = Path(inspect.getfile(Neuron)).resolve()
    if meta["source_hashes"][str(core)] != digest(core):
        raise ValueError("Native neuron changed since the reference")
    if meta["source_hashes"][str((graph_path/"manifest.json").resolve())] != digest(graph_path/"manifest.json"):
        raise ValueError("Different anatomical graph")
    cut = cut_cells(graph, WATCH)
    prep = build_paula(cut, Dynamics(weight_per_count=.075))
    roots = structure["roots"].tolist()
    files = [Path(__file__), core, Path(__file__).with_name("paula.py"), graph_path/"manifest.json",
             reference/"analysis.json", reference/"structure.npz"]
    hashes = {str(p.resolve()): digest(p) for p in files}
    output.mkdir(parents=True)
    reports = []; exact = 0
    for col, root in enumerate(WATCH):
        lo, hi = structure["watch_offsets"][col:col+2]
        inputs = arrays["inputs"][:, lo:hi]
        original = prep.network.network.neurons[prep.root_to_id[root]]
        pn, ln, edges = port_groups(cut, root)
        if inputs.shape[1] != original.params.num_inputs or inputs[:, -1].any():
            raise ValueError("Incorrect ports or undeclared experimental current")
        initial = np.array([p.u_i.info for p in original.postsynaptic_points.values()])
        records = []
        for condition in CONDITIONS:
            data = replay(copy.deepcopy(original), inputs, pn, ln, condition)
            if condition == "all":
                expected = np.c_[arrays["soma"][:, roots.index(root)], arrays["intrinsic"][:, col, :3], arrays["current"][:, col]]
                np.testing.assert_array_equal(data["trace"], expected)
                np.testing.assert_array_equal(data["weights"], arrays["weights"][:, lo:hi])
                exact += expected.size+data["weights"].size
            name = f"{col}-{condition}.npz"
            with (output/name).open("xb") as f: np.savez_compressed(f, **data)
            records.append({"condition": condition, "file": name, "sha256": digest(output/name), **response_summary(data)})
        # Evaluate receiving impulses before the first recorded LN spike. These
        # weights exclude updates caused by the same arriving event.
        weights_before = np.vstack([initial, arrays["weights"][:-1, lo:hi]])
        arrivals = inputs[:, :, 0]*weights_before
        current = np.zeros_like(arrivals); current[2:, :-1] = arrivals[:-2, :-1]*.95**2
        residual = np.abs(current.sum(1)-arrays["current"][:, col])
        bound = 8*np.finfo(np.float32).eps*np.maximum(1, np.abs(current).sum(1))
        if np.any(residual > bound): raise ValueError("Port current decomposition exceeds rounding bound")
        first = records[0]["first_spike"]
        windows = []
        for a, b in ((200, 334), (334, 1200), (1200, 1600)):
            windows.append({"start": a, "stop": b, "signed_charge": {
                "DL5": float(current[a:b, pn].sum()), "ALLN": float(current[a:b, ln].sum()),
                "other": float(current[a:b, ~(pn|ln)].sum())}})
        reports.append({"root": root, "type": graph.nodes[root]["annotation"]["hemibrain_type"],
            "num_inputs": original.params.num_inputs, "first_recorded_spike": first,
            "DL5_ports": np.flatnonzero(pn).tolist(), "ALLN_ports": np.flatnonzero(ln).tolist(),
            "DL5_source_rows": [int(e[8]) for e in edges if str(e[0]) == PN],
            "conditions": records, "recorded_current_decomposition": windows,
            "max_current_decomposition_residual": float(residual.max())})
        print(root, [(r["condition"], r["first_spike"], r["spikes"]) for r in records], flush=True)
    if any(digest(Path(p)) != h for p, h in hashes.items()): raise ValueError("Source changed during experiment")
    result = {"schema": 1, "source_hashes": hashes, "reference_source_hashes": meta["source_hashes"],
        "full_receiving_replay_values_exact": exact, "neurons": reports,
        "claim": "Conditional receiving-path evidence with exact intact replay; not a closed-loop lesion or physiological calibration",
        "limits": ["Other source trains are fixed to their intact values even when a selected pathway is removed.",
            "Changed LN outputs cannot feed back into those recorded inputs.",
            "Only receiving weights and intrinsic state are replayed; outgoing return-event history is absent.",
            "Quarter, half and double DL5 gains are sensitivity probes, not measured parameters or selected repairs.",
            "No odor input or physiological sampling distribution is implied by these deterministic replays."]}
    dump_new(output/"analysis.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("graph", "reference", "output"): p.add_argument(name, type=Path)
    a = p.parse_args(); run(a.graph, a.reference, a.output)


if __name__ == "__main__": main()
