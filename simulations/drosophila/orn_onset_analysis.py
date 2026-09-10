"""Compare onset interventions and attribute first-spike voltage to input ports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .connectome import Subgraph
from .orn_onset import CONDITIONS, TICKS, WATCH
from .orn_train_analysis import audit_sources
from .prisco import digest, dump_new


def read_case(directory):
    m = json.loads((directory / "analysis.json").read_text())
    if m["ticks"] != TICKS:
        raise ValueError("Incomplete onset course")
    for p, h in m["source_hashes"].items():
        if digest(Path(p)) != h:
            raise ValueError(f"Changed source: {p}")
    if digest(directory / "structure.npz") != m["structure_sha256"]:
        raise ValueError("Changed structure")
    with np.load(directory / "structure.npz") as f:
        structure = {k: f[k] for k in f.files}
    arrays, cursor = {}, 0
    for c in m["chunks"]:
        if c["start"] != cursor or not cursor < c["stop"] <= TICKS:
            raise ValueError("Missing or duplicated ticks")
        path = directory / c["file"]
        if digest(path) != c["sha256"]:
            raise ValueError("Changed trace")
        with np.load(path) as f:
            for k in f.files:
                a = f[k]
                if len(a) != c["stop"]-cursor or not np.isfinite(a).all():
                    raise ValueError("Invalid trace array")
                if k not in arrays:
                    arrays[k] = np.empty((TICKS, *a.shape[1:]), dtype=a.dtype)
                arrays[k][cursor:c["stop"]] = a
        cursor = c["stop"]
    if cursor != TICKS:
        raise ValueError("Incomplete recording")
    roots = structure["roots"].tolist()
    watched = arrays["soma"][:, [roots.index(r) for r in WATCH]]
    audit_sources(np.vstack([np.zeros((1, len(WATCH))), watched[:-1, :, 0]]),
        arrays["current"], np.zeros_like(arrays["current"]), watched, arrays["intrinsic"],
        first_tick=0, last_fire=np.full(len(WATCH), -np.inf), previous_S=np.zeros(len(WATCH)))
    return m, structure, arrays


def input_witnesses(graph, m, s, a):
    """Linear contributions up to a cell's FIRST spike, before any soma reset.

    Coefficient values, not runtime scalar dtypes, are recorded. Explicitly
    bound float32 accumulation error; receiving replay in the runner is exact.
    """
    results = []
    roots = s["roots"].tolist()
    corrected = set(m["intervention"]["source_roots"]) if m["intervention"] else set()
    for col, root in enumerate(WATCH):
        lo, hi = s["watch_offsets"][col:col+2]
        inputs = a["inputs"][:, lo:hi, 0]
        weights = a["weights"][:, lo:hi]
        rows = sorted(graph.edges[graph.edges[:, 1] == int(root)], key=lambda e: int(e[8]))
        initial = np.array([(-1 if str(e[0]) in corrected else 1)*int(e[6])*.075 for e in rows] + [1.])
        prior = np.vstack([initial, weights[:-1]])
        local = inputs*prior
        current = np.zeros_like(local)
        current[2:, :-1] = local[:-2, :-1]*(.95**2)
        current[:, -1] = local[:, -1]
        if current[:, -1].any():
            raise ValueError("Unexpected experimental drive at a watched LN")
        residual = np.abs(current.sum(axis=1)-a["current"][:, col])
        bound = 8*np.finfo(np.float32).eps*np.maximum(1, np.abs(current).sum(axis=1))
        if np.any(residual > bound):
            raise ValueError("Per-port reconstruction exceeded float32 bound")
        spikes = np.flatnonzero(a["soma"][:, roots.index(root), 1] > 0)
        first = int(spikes[0]) if len(spikes) else None
        ports = []
        if first is not None:
            lam = a["intrinsic"][0, col, 3]
            if np.any(a["intrinsic"][:, col, 3] != lam):
                raise ValueError("First-spike attribution assumes constant membrane integration")
            factors = (1-1/lam)**np.arange(first, -1, -1)/lam
            contribution = factors @ current[:first+1]
            if contribution.sum() < a["intrinsic"][first, col, 1] - 1e-5:
                raise ValueError("Attributed first-spike voltage is below threshold")
            for port in np.argsort(-np.abs(contribution)):
                if port == len(rows) or contribution[port] == 0:
                    continue
                edge = rows[port]; source = str(edge[0]); info = graph.nodes[source]["annotation"]
                arrival = np.flatnonzero(inputs[:first+1, port] > 0)
                ports.append({"source_root": source, "source_type": info["hemibrain_type"],
                    "source_class": info["cell_class"], "source_row": int(edge[8]), "port": int(port),
                    "contacts": int(edge[4]), "source_model_sign": int(edge[5]),
                    "first_receiving_tick": int(arrival[0]), "receiving_events_before_first_spike": len(arrival),
                    "voltage_contribution_at_first_spike": float(contribution[port])})
        results.append({"root": root, "type": graph.nodes[root]["annotation"]["hemibrain_type"],
            "first_spike": first, "spikes": len(spikes), "ports": ports,
            "max_current_reconstruction_residual": float(residual.max()),
            "summed_first_spike_voltage": sum(p["voltage_contribution_at_first_spike"] for p in ports) if first is not None else None})
    return results


def analyze(graph_path, directory, output):
    if output.exists():
        raise FileExistsError(output)
    graph = Subgraph.load(graph_path)
    reports, states, structure = {}, {}, None
    for name in CONDITIONS:
        m, s, a = read_case(directory / name)
        if m["condition"] != name:
            raise ValueError("Condition mismatch")
        if structure is None:
            structure = s
        else:
            for k in structure:
                np.testing.assert_array_equal(s[k], structure[k])
        roots = s["roots"].tolist()
        phases = []
        for phase, start, stop in (("baseline", 0, 200), ("10Hz", 200, 1200), ("recovery", 1200, TICKS)):
            groups = {}
            for cls in ("olfactory", "ALLN", "ALPN", "Kenyon_Cell"):
                ids = [i for i, r in enumerate(roots) if graph.nodes[r]["annotation"]["cell_class"] == cls]
                counts = (a["soma"][start:stop, ids, 1] > 0).sum(axis=0)
                groups[cls] = {"spikes": int(counts.sum()), "cells_fired": int(np.count_nonzero(counts))}
            phases.append({"phase": phase, "start": start, "stop": stop, "populations": groups,
                "maximum_apl_local_release": float(a["apl_max"][start:stop].max())})
        if name == "intact":
            refpath = Path(m["reference"])
            ref = json.loads((refpath / "analysis.json").read_text())
            for c in ref["chunks"]:
                if c["start"] >= TICKS:
                    break
                p = refpath / c["file"]
                if digest(p) != c["sha256"]:
                    raise ValueError("Changed reference trace")
                with np.load(p) as f:
                    stop = min(c["stop"], TICKS)
                    np.testing.assert_array_equal(a["soma"][c["start"]:stop], f["soma"][:stop-c["start"]])
        reports[name] = {"epochs": phases, "input_witnesses": input_witnesses(graph, m, s, a),
            "block_events": a["block_events"].sum(axis=0).tolist(),
            "manifest_sha256": digest(directory / name / "analysis.json")}
        states[name] = a["soma"]
    comparisons = []
    for name in CONDITIONS[1:]:
        change = np.any(states[name] != states["intact"], axis=2)
        spike_change = states[name][:, :, 1] != states["intact"][:, :, 1]
        def earliest(mask):
            ticks = np.flatnonzero(mask.any(axis=1))
            if not len(ticks):
                return None
            tick = int(ticks[0])
            return {"tick": tick, "roots": [roots[i] for i in np.flatnonzero(mask[tick])]}
        comparisons.append({"against_intact": name, "first_state_difference": earliest(change),
            "first_spike_difference": earliest(spike_change), "spike_disagreement_entries": int(spike_change.sum())})
    report = {"schema": 1, "conditions": reports, "comparisons": comparisons,
        "original_intact_soma_values_exact": int(states["intact"].size), "bindings_exact": True,
        "source_sha256": digest(Path(__file__)),
        "claim": "Onset causality within this model; not stable olfactory coding or receptor validation"}
    dump_new(output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("graph", "directory", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    analyze(a.graph, a.directory, a.output)


if __name__ == "__main__":
    main()
