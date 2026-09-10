"""Offline replay and declared-current audits for paired and reunited recordings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .connectome import Subgraph
from .orn_train_analysis import audit_sources
from .paired_recording import simulate, PN, LN
from .pn_current import dl5_cut
from .pn_current_steps import prepare
from .prisco import digest, dump_new


def checked_manifest(directory, *, current_source=True):
    meta = json.loads((directory/"analysis.json").read_text())
    if current_source:
        for p, expected in meta["source_hashes"].items():
            if digest(Path(p)) != expected:
                raise ValueError(f"Changed source: {p}")
    return meta


def replay_pairs(directory, intrinsic_path, tail_path):
    meta = checked_manifest(directory)
    graph = Subgraph.load(directory/"graph")
    intrinsic = json.loads(intrinsic_path.read_text())
    tail = json.loads(tail_path.read_text())["fits"]["2"]["all_cells"]
    values = ticks = 0
    for r in meta["records"]:
        path = directory/r["file"]
        if digest(path) != r["sha256"]:
            raise ValueError("Changed paired record")
        data, _ = simulate(graph, intrinsic, tail, r["source"], r["epochs"][0]["level"],
                          r["g"], r["chemical_release_blocked"], **r["protocol"])
        with np.load(path) as saved:
            if set(saved.files) != set(data):
                raise ValueError("Changed fields")
            for key, value in data.items():
                np.testing.assert_array_equal(value, saved[key])
                values += value.size
        ticks += len(data["trace"])*2
    return {"courses": len(meta["records"]), "exact_replayed_cell_ticks": ticks,
            "exact_replayed_values": values, "manifest_sha256": digest(directory/"analysis.json")}


def prefix(directory, ticks, *, current_source=True):
    meta = checked_manifest(directory, current_source=current_source)
    if digest(directory/"structure.npz") != meta["structure_sha256"]:
        raise ValueError("Changed structure")
    collected = {}; cursor = 0
    for r in meta["chunks"]:
        if r["start"] >= ticks:
            break
        if r["start"] != cursor or r["stop"] <= cursor:
            raise ValueError("Discontinuous recording")
        path = directory/r["file"]
        if digest(path) != r["sha256"]:
            raise ValueError("Changed recording")
        with np.load(path) as a:
            for key in a.files:
                value = a[key]
                if len(value) != r["stop"]-r["start"] or not np.isfinite(value).all():
                    raise ValueError("Invalid tick array")
                collected.setdefault(key, []).append(value[:min(r["stop"], ticks)-cursor])
        cursor = min(r["stop"], ticks)
    if cursor != ticks:
        raise ValueError("Missing recorded ticks")
    with np.load(directory/"structure.npz") as a:
        structure = {k: a[k] for k in a.files}
    return meta, {k: np.concatenate(a) for k, a in collected.items()}, structure


def reunion(graph_path, intrinsic_path, tail_path, directory, reference):
    """Keep the finite 1,600-tick onset question separate from a complete course."""
    ticks = 1600
    graph = Subgraph.load(graph_path)
    intrinsic = json.loads(intrinsic_path.read_text())
    tail = json.loads(tail_path.read_text())["fits"]["2"]["all_cells"]
    old_meta, old, old_structure = prefix(reference, ticks, current_source=False)
    if old_meta["condition"] != "no_depression" or old_meta["ticks"] != 4200:
        raise ValueError("Need the original complete no-depression reference")
    results = {}; states = {}; parity = 0
    for name in ("zero_gap", "gap_003"):
        path = directory/name
        meta, data, structure = prefix(path, ticks)
        if meta["ticks"] != ticks or meta["complete_course"] or meta["condition"] != "no_depression":
            raise ValueError("Changed onset course")
        for key in old_structure:
            np.testing.assert_array_equal(old_structure[key], structure[key])
        roots = structure["roots"].tolist(); index = {r: i for i, r in enumerate(roots)}
        spec = meta["assumptions"]["electrical_junction"]
        if spec["roots"] != [PN, LN] or spec["g"] != (0. if name == "zero_gap" else .03):
            raise ValueError("Changed declared intervention")
        g = spec["g"]; e = data["electrical"]
        np.testing.assert_array_equal(e[:, :, 3], g*(e[:, ::-1, 0]-e[:, :, 0]))
        np.testing.assert_array_equal(e[:, :, 1], data["soma"][:, [index[PN], index[LN]], 0])
        for j, col in ((4, 1), (5, 2)):
            np.testing.assert_array_equal(e[:, :, j], data["soma"][:, [index[PN], index[LN]], col])
        lambdas = [intrinsic["proposals"]["pooled_control"]["lambda_ms"], 20.]
        fields = np.concatenate([e[:, :, [6, 7, 8]], np.broadcast_to(lambdas, (ticks, 2))[:, :, None]], axis=2)
        _, _, clips, residual, ambiguous = audit_sources(e[:, :, 0], e[:, :, 2]+e[:, :, 3],
            np.zeros((ticks, 2)), data["soma"][:, [index[PN], index[LN]]], fields,
            first_tick=0, last_fire=np.full(2, -np.inf), previous_S=np.zeros(2))
        _, cut = dl5_cut(graph)
        prep, pn = prepare(cut, intrinsic, tail, electrical_roots=(PN,))
        for t in range(ticks):
            pn.input_buffer[:] = data["pn_inputs"][t]
            pn.electrical_tick = t; pn.electrical_voltage = float(pn.S)
            pn.electrical_neighbors = ((e[t, 1, 0], g),)
            pn.tick({}, t)
            np.testing.assert_array_equal([pn.S, pn.O, pn.F_avg], data["soma"][t, index[PN]])
            np.testing.assert_array_equal([pn.t_ref, pn.r, pn.b, pn.total_current], data["pn_intrinsic"][t])
            np.testing.assert_array_equal(pn.last_port_current, data["pn_current"][t])
            np.testing.assert_array_equal([p.u_i.info for p in pn.postsynaptic_points.values()], data["pn_post_weight"][t])
            assert pn.electrical_native_current == e[t, 0, 2]
            assert pn.electrical_current == e[t, 0, 3]
        if name == "zero_gap":
            for key, a in old.items():
                np.testing.assert_array_equal(a, data[key]); parity += a.size
        epochs = []
        for lo, hi in ((200, 1200), (1200, 1600)):
            populations = {}
            for cls in ("olfactory", "ALLN", "ALPN", "Kenyon_Cell"):
                cols = [i for i, r in enumerate(roots) if graph.nodes[r]["annotation"]["cell_class"] == cls]
                spikes = data["soma"][lo:hi, cols, 1].sum(axis=0)
                populations[cls] = {"spikes": int(spikes.sum()), "cells_fired": int(np.count_nonzero(spikes))}
            epochs.append({"start": lo, "stop": hi, "populations": populations,
                "target_spikes": int(data["soma"][lo:hi, index[PN], 1].sum()),
                "ln_spikes": int(data["soma"][lo:hi, index[LN], 1].sum()),
                "maximum_apl_local_release": float(data["apl"][lo:hi, 2].max())})
        states[name] = data["soma"]
        results[name] = {"epochs": epochs, "exact_target_replay_ticks": ticks,
            "membrane_audit_cell_ticks": ticks*2, "clipping": clips,
            "maximum_membrane_audit_residual": residual, "ambiguous_spikes": ambiguous,
            "peak_abs_chemical_current": np.max(np.abs(e[:, :, 2]), axis=0).tolist(),
            "peak_abs_electrical_current": np.max(np.abs(e[:, :, 3]), axis=0).tolist(),
            "manifest_sha256": digest(path/"analysis.json")}
    a, b = states["zero_gap"], states["gap_003"]
    def first(mask):
        rows = np.flatnonzero(mask)
        return int(rows[0]) if len(rows) else None
    comparison = {"first_any_S_difference": first(np.any(a[:, :, 0] != b[:, :, 0], axis=1)),
        "first_any_spike_difference": first(np.any(a[:, :, 1] != b[:, :, 1], axis=1)),
        "first_PN_spike_difference": first(a[:, index[PN], 1] != b[:, index[PN], 1]),
        "first_LN_spike_difference": first(a[:, index[LN], 1] != b[:, index[LN], 1]),
        "cells_with_changed_spikes": int(np.any(a[:, :, 1] != b[:, :, 1], axis=0).sum()),
        "roots_with_changed_S": [roots[i] for i in np.flatnonzero(np.any(a[:, :, 0] != b[:, :, 0], axis=0))],
        "pair_max_abs_S_difference": np.max(np.abs(a[:, [index[PN], index[LN]], 0]-b[:, [index[PN], index[LN]], 0]), axis=0).tolist()}
    return {"conditions": results, "comparison": comparison, "old_record_exact_parity_values": parity,
        "reference_manifest_sha256": digest(reference/"analysis.json"),
        "claim": "Finite onset sensitivity to one hypothetical electrical contact; not physiological or behavioral acceptance",
        "limits": ["Only 400 recovery ticks after the first train; the 50-Hz challenge is not tested.",
            "PN receiving history is replayed exactly; LN chemical current is recorded, not independently replayed from its receptors.",
            "Native outgoing return histories and full network event queues are not replayed.",
            "The historical reference is checked by retained artifact hashes and direct array comparison, not claimed to share current source code."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    pair = sub.add_parser("pairs")
    for name in ("directory", "intrinsic", "tail", "output"):
        pair.add_argument(name, type=Path)
    connected = sub.add_parser("reunion")
    for name in ("graph", "intrinsic", "tail", "directory", "reference", "output"):
        connected.add_argument(name, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = replay_pairs(a.directory, a.intrinsic, a.tail) if a.mode == "pairs" else reunion(
        a.graph, a.intrinsic, a.tail, a.directory, a.reference)
    dump_new(a.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
