"""Read the intervention trajectories, verify their scope, attribute APL inputs.

This never decides biological acceptance from a count or a finite-value check.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from .connectome import sha256


def first_tick(mask):
    ticks = np.flatnonzero(mask)
    return int(ticks[0]) if len(ticks) else None


def inspect_record(directory: Path):
    m = json.loads((directory / "manifest.json").read_text())
    recorded = m["recording"]
    if recorded["soma_fields"] != ["S", "O", "F_avg", "r", "b", "t_ref", "t_last_fire"]:
        raise ValueError("Unknown soma fields")
    for name, field in (("columns.npz", "columns_sha256"), ("soma.npz", "soma_sha256")):
        if sha256(directory / name) != recorded[field]:
            raise ValueError(f"Changed recording {directory / name}")
    with np.load(directory / "columns.npz", allow_pickle=False) as c:
        ids, roots, ports, bindings, delay = (c[k] for k in ("cell_ids", "root_ids", "postsynaptic_ports", "edge_bindings", "input_delays"))
        n_terminals = len(c["terminals"])
    with np.load(directory / "soma.npz", allow_pickle=False) as c:
        soma = c["soma"]
    protocol = m["protocol"]
    n_ticks = protocol["ticks"]
    if soma.shape != (n_ticks, len(ids), 7):
        raise ValueError("Incomplete soma trajectory")
    if not np.isfinite(soma[..., :6]).all() or not (np.isfinite(soma[..., 6]) | np.isneginf(soma[..., 6])).all():
        raise ValueError("Nonfinite soma state other than the native never-fired sentinel")
    groups = {"KC": np.asarray(protocol["kc_rows"]), "PN": np.asarray(protocol["pn_rows"]),
              "APL": np.asarray([protocol["apl_row"]])}
    if sorted(np.concatenate(list(groups.values())).tolist()) != list(range(len(ids))):
        raise ValueError("Cell groups must partition recorded columns")
    if len(set(ids.tolist())) != len(ids) or len(set(roots.tolist())) != len(roots):
        raise ValueError("Duplicate cell identity")
    if delay.shape != (len(ports),) or np.any(delay < 0) or not np.equal(delay, np.floor(delay)).all():
        raise ValueError("Invalid input delays")
    row_for_id = {int(nid): i for i, nid in enumerate(ids)}
    port_for_pair = {tuple(map(int, pair)): i for i, pair in enumerate(ports)}
    if len(port_for_pair) != len(ports):
        raise ValueError("Duplicate receiving port identity")
    source_group = {int(ids[i]): name for name, rows in groups.items() for i in rows}
    apl_id = int(ids[protocol["apl_row"]])
    apl_ports = {name: [] for name in ("KC", "PN", "APL", "experimental", "boundary")}
    internal_posts = set()
    for _, pre, _, post, sid in bindings:
        pair = (int(post), int(sid))
        internal_posts.add(pair)
        if int(post) == apl_id:
            apl_ports[source_group[int(pre)]].append(port_for_pair[pair])
    # Boundary ports are declared in columns. Any remaining slot is experimental.
    with np.load(directory / "columns.npz", allow_pickle=False) as c:
        boundary_pairs = {(int(row[2]), int(row[3])) for row in c["incoming_boundary_ports"]}
    boundary_cols = np.asarray([port_for_pair[pair] for pair in boundary_pairs], dtype=int)
    experimental = {pair: col for pair, col in port_for_pair.items()
                    if pair not in internal_posts and pair not in boundary_pairs}
    experimental_by_cell = {pair[0]: col for pair, col in experimental.items()}
    if len(experimental) != len(ids) or set(experimental_by_cell) != set(ids):
        raise ValueError("Expected exactly one experimental port per cell")
    drive_cols = [experimental_by_cell[int(nid)] for nid in ids]
    for pair, col in port_for_pair.items():
        if pair[0] == apl_id:
            if pair in experimental:
                apl_ports["experimental"].append(col)
            elif pair in boundary_pairs:
                apl_ports["boundary"].append(col)
    currents = {key: np.zeros(n_ticks) for key in apl_ports}
    external = np.zeros((n_ticks, len(ids)))
    emitted = np.zeros((n_ticks, len(ids)), dtype=np.int64)
    blocked = emitted.copy()
    returned = emitted.copy()
    previous, expected_start = {}, 0
    positive_terminals = True
    changed_post = changed_pre = None
    initial_post = initial_pre = None
    for chunk in recorded["chunks"]:
        if chunk["start"] != expected_start or chunk["stop"] <= chunk["start"]:
            raise ValueError("Chunk gap, overlap or reversal")
        path = directory / chunk["file"]
        if path.name != chunk["file"] or sha256(path) != chunk["sha256"]:
            raise ValueError("Changed chunk or invalid chunk path")
        start, stop = chunk["start"], chunk["stop"]
        if stop > n_ticks:
            raise ValueError("Chunk extends beyond protocol")
        with np.load(path, allow_pickle=False) as data:
            for key in ("soma", "M", "post_weight", "terminal_info"):
                values = data[key]
                tail = {"soma": (len(ids), 7), "M": (len(ids), 2),
                        "post_weight": (len(ports),), "terminal_info": (n_terminals,)}[key]
                if values.shape != (stop - start + 1, *tail):
                    raise ValueError(f"Wrong state shape: {key}")
                if key == "M" and not np.isfinite(values).all():
                    raise ValueError("Nonfinite neuromodulator state")
                if key in previous and not np.array_equal(previous[key], values[0]):
                    raise ValueError(f"Discontinuous state at chunk boundary: {key}")
                previous[key] = values[-1].copy()
            if not np.array_equal(data["soma"][1:], soma[start:stop]):
                raise ValueError("Compact soma file disagrees with full trace")
            w, pre = data["post_weight"], data["terminal_info"]
            if not np.isfinite(w).all() or not np.isfinite(pre).all():
                raise ValueError("Nonfinite coefficients")
            if initial_post is None:
                initial_post, initial_pre = w[0].copy(), pre[0].copy()
                changed_post, changed_pre = np.zeros(w.shape[1], dtype=bool), np.zeros(pre.shape[1], dtype=bool)
            changed_post |= np.any(w != initial_post, axis=0)
            changed_pre |= np.any(pre != initial_pre, axis=0)
            positive_terminals &= bool((pre > 0).all())
            for key, target in (("external", external), ("emitted", emitted), ("blocked", blocked), ("returned", returned)):
                if data[key].shape != target[start:stop].shape:
                    raise ValueError(f"Wrong tick-array shape: {key}")
                if not np.isfinite(data[key]).all() or (data[key] < 0).any():
                    raise ValueError(f"Invalid tick values: {key}")
                if key != "external" and not np.equal(data[key], np.floor(data[key])).all():
                    raise ValueError(f"Noninteger event count: {key}")
                target[start:stop] = data[key]
            local = data["local_potential"]
            inputs = data["inputs"]
            if local.shape != (stop - start, len(ports)) or inputs.shape != (stop - start, len(ports), 4):
                raise ValueError("Wrong port-array shape")
            if not np.isfinite(local).all() or not np.isfinite(inputs).all():
                raise ValueError("Nonfinite port values")
            if inputs[:, boundary_cols].any() or local[:, boundary_cols].any():
                raise ValueError("The declared undriven boundary received input")
            if not np.array_equal(inputs[:, drive_cols, 0], external[start:stop].astype(np.float32)):
                raise ValueError("External drive did not arrive at its experimental ports")
            # Source-tagged creation potentials, shifted by their actual fixed
            # dendritic delay. Float64 sums are diagnostic currents, not a
            # bit-exact reimplementation of native heap accumulation order.
            for name, cols in apl_ports.items():
                cols = np.asarray(cols, dtype=int)
                for d in np.unique(delay[cols]):
                    chosen = cols[delay[cols] == d]
                    d = int(d)
                    end = min(stop + d, n_ticks)
                    if start + d < end:
                        currents[name][start + d:end] += local[:end - start - d, chosen].sum(axis=1) * m["assumptions"]["parameters"]["signal_decay"]**d
        expected_start = stop
    if expected_start != n_ticks:
        raise ValueError("Missing tail of recording")
    condition = m["condition"]
    if condition not in {"intact", "kc_release_block", "apl_release_block", "apl_activation"}:
        raise ValueError("Unknown intervention condition")
    expected_ids = ids[groups["KC"]].tolist() if condition == "kc_release_block" else ids[groups["APL"]].tolist() if condition == "apl_release_block" else []
    if sorted(expected_ids) != protocol["blocked_ids"]:
        raise ValueError("Condition and blockade targets disagree")
    expected_drive = np.zeros_like(external)
    epochs = []
    previous_stop = 0
    for epoch in protocol["epochs"]:
        start, stop = epoch["start"], epoch["stop"]
        if not previous_stop <= start < stop <= n_ticks:
            raise ValueError("Invalid experimental epoch")
        previous_stop = stop
        expected_drive[start:stop, groups["PN"]] = epoch["PN_drive"]
        kc = soma[start:stop, groups["KC"], 1] > 0
        epochs.append({**epoch, "kc_cells_fired": int(kc.any(axis=0).sum()),
                       "kc_spikes": int(kc.sum()),
                       "apl_output_min": float(soma[start:stop, groups["APL"], 1].min()),
                       "apl_output_max": float(soma[start:stop, groups["APL"], 1].max()),
                       "pn_spikes": int((soma[start:stop, groups["PN"], 1] > 0).sum())})
    if condition == "apl_activation":
        expected_drive[32:192, groups["APL"]] = 50.0
    if not np.array_equal(expected_drive, external):
        raise ValueError("Recorded input does not match the declared course")
    if epochs != m["epoch_summaries_not_acceptance"]:
        raise ValueError("Epoch summary disagrees with the full trajectory")
    blocked_rows = np.asarray([row_for_id[nid] for nid in protocol["blocked_ids"]], dtype=int)
    unblocked_rows = np.asarray([i for i in range(len(ids)) if i not in set(blocked_rows)], dtype=int)
    if not np.array_equal(blocked[:, blocked_rows], emitted[:, blocked_rows]) or blocked[:, unblocked_rows].any():
        raise ValueError("Release blockade did not match its declared targets")
    report = {
        "scope": m["claim"], "condition": m["condition"],
        "recording_checks_passed": True, "positive_terminal_coefficients": positive_terminals,
        "postsynaptic_coefficients_changed": int(changed_post.sum()),
        "terminal_coefficients_changed": int(changed_pre.sum()),
        "first_spike_or_release": {key: first_tick(np.any(soma[:, rows, 1] > 0, axis=1)) for key, rows in groups.items()},
        "kc_spikes_each_tick": (soma[:, groups["KC"], 1] > 0).sum(axis=1).tolist(),
        "apl_output_each_tick": soma[:, protocol["apl_row"], 1].tolist(),
        "apl_input_by_source_each_tick": {key: values.tolist() for key, values in currents.items()},
        "apl_release_cap_ticks": np.flatnonzero(soma[:, protocol["apl_row"], 0] * m["assumptions"]["parameters"]["apl_graded_gain"] >= m["assumptions"]["parameters"]["apl_release_max"]).tolist(),
        "blocked_forward_events": int(blocked.sum()),
        "native_return_events_from_blocked_cells": int(returned[:, blocked_rows].sum()),
        "epoch_summaries_not_acceptance": epochs,
        "manifest_sha256": sha256(directory / "manifest.json"),
        "source_hash_scope": "legacy hashes were taken at finish only" if "source_files" not in m else m["source_files"],
    }
    return report, soma, external, ids, roots, groups


def compare(reference: Path, variant: Path):
    ma, mb = (json.loads((directory / "manifest.json").read_text()) for directory in (reference, variant))
    for key in ("assumptions", "anatomical_provenance", "anatomy"):
        if ma[key] != mb[key]:
            raise ValueError(f"Unmatched {key}")
    for key in ("epochs", "ticks", "kc_rows", "pn_rows", "apl_row"):
        if ma["protocol"][key] != mb["protocol"][key]:
            raise ValueError(f"Unmatched protocol {key}")
    with np.load(reference / "columns.npz") as ca, np.load(variant / "columns.npz") as cb:
        if set(ca.files) != set(cb.files) or any(not np.array_equal(ca[key], cb[key]) for key in ca.files):
            raise ValueError("Unmatched anatomical port columns")
    with np.load(reference / ma["recording"]["chunks"][0]["file"]) as ca, np.load(variant / mb["recording"]["chunks"][0]["file"]) as cb:
        for key in ("soma", "M", "post_weight", "terminal_info"):
            if not np.array_equal(ca[key][0], cb[key][0]):
                raise ValueError(f"Unmatched initial {key}")
    a, sa, ea, ia, ra, groups = inspect_record(reference)
    b, sb, eb, ib, rb, _ = inspect_record(variant)
    if not np.array_equal(ia, ib) or not np.array_equal(ra, rb):
        raise ValueError("Neuron column identities differ")
    result = {"reference": str(reference), "variant": str(variant),
              "condition": b["condition"], "identical_PN_external_inputs": bool(np.array_equal(ea[:, groups["PN"]], eb[:, groups["PN"]])),
              "first_divergence": {}, "epoch_comparison": []}
    if not result["identical_PN_external_inputs"]:
        raise ValueError("Unmatched experimental PN drive")
    for name, rows in groups.items():
        result["first_divergence"][name] = {
            "S": first_tick(np.any(sa[:, rows, 0] != sb[:, rows, 0], axis=1)),
            "O": first_tick(np.any(sa[:, rows, 1] != sb[:, rows, 1], axis=1)),
        }
    for epoch in a["epoch_summaries_not_acceptance"]:
        start, stop = epoch["start"], epoch["stop"]
        ka = sa[start:stop, groups["KC"], 1] > 0
        kb = sb[start:stop, groups["KC"], 1] > 0
        result["epoch_comparison"].append({"start": start, "stop": stop,
            "reference_kc_cells": int(ka.any(axis=0).sum()), "variant_kc_cells": int(kb.any(axis=0).sum()),
            "reference_kc_spikes": int(ka.sum()), "variant_kc_spikes": int(kb.sum())})
    return result


def verify_unobserved(graph, directory: Path):
    """Replay an intact course without tick patches; compare every recorded state.

    This checks the installed implementation against the old recording. It does
    not retroactively prove which source bytes were loaded in that process.
    """
    from .execution_probe import SOMA_FIELDS
    from .paula import Dynamics, build_paula

    report, _, drive, ids, roots, _ = inspect_record(directory)
    m = json.loads((directory / "manifest.json").read_text())
    if m["condition"] != "intact":
        raise ValueError("Uninstrumented verification requires an intact record")
    if graph.provenance != m["anatomical_provenance"] or graph.summary() != m["anatomy"]:
        raise ValueError("Replay graph differs from recorded anatomy")
    p = build_paula(graph, Dynamics(**m["assumptions"]["parameters"]))
    if list(p.root_to_id) != roots.tolist() or list(p.root_to_id.values()) != ids.tolist():
        raise ValueError("Replay cell identities differ")
    cells = list(p.network.network.neurons.values())
    posts = [s for cell in cells for s in cell.postsynaptic_points.values()]
    terminals = [t for cell in cells for t in cell.presynaptic_points.values()]
    checked = {key: 0 for key in ("soma", "M", "post_weight", "terminal_info")}

    def check(data, row):
        states = {"soma": [[getattr(c, field) for field in SOMA_FIELDS] for c in cells],
                  "M": [c.M_vector for c in cells],
                  "post_weight": [s.u_i.info for s in posts],
                  "terminal_info": [t.u_o.info for t in terminals]}
        for key, values in states.items():
            if not np.array_equal(values, data[key][row]):
                raise ValueError(f"Unobserved replay mismatch: {key}, after {p.network.current_tick} ticks")
            checked[key] += np.asarray(values).size

    for chunk in m["recording"]["chunks"]:
        with np.load(directory / chunk["file"], allow_pickle=False) as compressed:
            data = {key: compressed[key] for key in checked}
        if chunk["start"] == 0:
            check(data, 0)
        for row, tick in enumerate(range(chunk["start"], chunk["stop"]), start=1):
            for i in np.flatnonzero(drive[tick]):
                p.stimulate(str(roots[i]), float(drive[tick, i]))
            p.network.run_tick()
            check(data, row)
    return {"scope": "exact recorded-state agreement without tick instrumentation; no biological acceptance",
            "ticks": p.network.current_tick, "equal_values": checked,
            "manifest_sha256": report["manifest_sha256"]}


def main(directory: Path, output: Path, reference: Path | None = None, source: Path | None = None):
    if source is not None and reference is not None:
        raise ValueError("Choose comparison or uninstrumented verification, not both")
    if source is not None:
        from .connectome import Subgraph
        report = verify_unobserved(Subgraph.load(source), directory)
    else:
        report = inspect_record(directory)[0] if reference is None else compare(reference, directory)
    with output.open("x") as out:
        json.dump(report, out, indent=2)
    print(json.dumps({k: v for k, v in report.items() if "each_tick" not in k}, indent=2))


if __name__ == "__main__":
    typer.run(main)
