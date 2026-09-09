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
    cable = m["assumptions"]["parameters"].get("apl_representation") == "local_cable"
    cable_report = {}
    if cable:
        from neuron.extensions.experimental.passive_cable import PassiveCable
        if "cable" not in recorded:
            raise ValueError("Local dynamics require full cable recordings")
        spatial = m["assumptions"]["spatial"]
        spatial_path = Path(spatial["analysis_path"])
        if sha256(spatial_path) != spatial["analysis_sha256"]:
            raise ValueError("Changed spatial analysis")
        geometry_path = spatial_path.parent / "anatomy.npz"
        if sha256(geometry_path) != spatial["anatomy_sha256"]:
            raise ValueError("Changed cable geometry")
        with np.load(geometry_path, allow_pickle=False) as geometry:
            node_ids = geometry["node_ids"]
            contacts, pair_rows = geometry["contacts"], geometry["pair_source_rows"]
            cable_operator = PassiveCable(geometry["parents"], geometry["xyz_nm"] / 1000,
                geometry["radius_nm"] / 1000, m["assumptions"]["parameters"]["apl_cable_rm_over_ra_um"])
        with np.load(directory / "columns.npz", allow_pickle=False) as columns:
            cable_nodes, capacities = columns["cable_nodes"], columns["cable_capacity"]
            cable_inputs, cable_terminals = columns["cable_input_ports"], columns["cable_terminals"]
            output_bindings = {int(row[0]): int(row[2]) for row in columns["edge_bindings"] if int(row[1]) == int(ids[protocol["apl_row"]])}
            output_bindings.update({int(row[0]): int(row[2]) for row in columns["outgoing_boundary_terminals"] if int(row[1]) == int(ids[protocol["apl_row"]])})
            input_bindings = {int(row[0]): int(row[4]) for row in columns["edge_bindings"] if int(row[3]) == int(ids[protocol["apl_row"]])}
            input_bindings.update({int(row[0]): int(row[3]) for row in columns["incoming_boundary_ports"] if int(row[2]) == int(ids[protocol["apl_row"]])})
        if (not np.array_equal(cable_nodes[:, 1], node_ids) or not np.all(cable_nodes[:, 0] == ids[protocol["apl_row"]])
                or not np.all(cable_inputs[:, 0] == ids[protocol["apl_row"]])
                or not np.array_equal(cable_inputs[:, 1], np.arange(len(cable_inputs)))
                or not np.all(cable_terminals[:, 0] == ids[protocol["apl_row"]])
                or not np.array_equal(cable_terminals[:, 1], np.arange(len(cable_terminals)))
                or not np.array_equal(capacities, cable_operator.capacity)):
            raise ValueError("Cable column identities or capacities disagree")
        output_mask = contacts[:, 1] == int(spatial["root"])
        output_nodes = contacts[output_mask, 5]
        linked_outputs = output_mask & (pair_rows >= 0)
        terminal_nodes = contacts[linked_outputs, 5]
        terminal_rows = np.array([output_bindings[int(row)] for row in pair_rows[linked_outputs]])
        terminal_counts = np.bincount(terminal_rows, minlength=len(cable_terminals))
        if np.any(terminal_counts == 0):
            raise ValueError("Cable terminal has no spatial contacts")
        input_mask = (contacts[:, 2] == int(spatial["root"])) & (pair_rows >= 0)
        input_nodes = contacts[input_mask, 6]
        input_rows = np.array([input_bindings[int(row)] for row in pair_rows[input_mask]])
        input_counts = np.bincount(input_rows, minlength=len(cable_inputs))
        uniform_ports = np.flatnonzero(input_counts == 0)
        if len(uniform_ports) != 1 or int(uniform_ports[0]) in input_bindings.values():
            raise ValueError("Expected one nonanatomical uniform cable drive port")
        cable_report = {key: [] for key in ("voltage_min_each_tick", "voltage_max_each_tick",
                        "saturated_nodes_each_tick", "terminal_release_min_each_tick", "terminal_release_max_each_tick",
                        "equation_max_residual_each_tick")}
        cable_report["checks_passed"] = False
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
            if cable:
                v, current, rel = data["cable_voltage"], data["cable_current"], data["cable_terminal_release"]
                arrived = data["cable_arrived_current"]
                if (v.shape != (stop-start+1, len(node_ids)) or current.shape != (stop-start, len(node_ids))
                        or rel.shape != (stop-start+1, len(cable_terminals))
                        or arrived.shape != (stop-start, len(cable_inputs))
                        or data["cable_checks"].shape != (stop-start, 1, 2)):
                    raise ValueError("Wrong cable-array shape")
                for key in ("cable_voltage", "cable_current", "cable_terminal_release", "cable_arrived_current", "cable_checks"):
                    if not np.isfinite(data[key]).all():
                        raise ValueError("Nonfinite cable trace")
                for key in ("cable_voltage", "cable_terminal_release"):
                    if key in previous and not np.array_equal(previous[key], data[key][0]):
                        raise ValueError(f"Discontinuous state at chunk boundary: {key}")
                    previous[key] = data[key][-1].copy()
                mean = v @ capacities
                alpha = 1 / m["assumptions"]["parameters"]["lambda_ticks"]
                if (not np.allclose(mean, data["soma"][:, protocol["apl_row"], 0], atol=1e-9, rtol=1e-9)
                        or not np.allclose(mean[1:], (1-alpha)*mean[:-1] + alpha*current.sum(axis=1), atol=1e-8, rtol=1e-8)
                        or not np.allclose(current.sum(axis=1), arrived.sum(axis=1), atol=1e-9, rtol=1e-9)):
                    raise ValueError("Cable current conservation disagrees with full trace")
                # Reconstruct the full equation by sparse multiplication, not
                # by running the solver again or trusting the recorded residual.
                left = v[1:] * capacities + alpha * (cable_operator.laplacian @ v[1:].T).T
                right = (1-alpha) * v[:-1] * capacities + alpha * current
                residual = np.abs(left - right)
                if np.any(residual > 1e-10 * np.maximum(1, np.abs(left) + np.abs(right))):
                    raise ValueError("Per-node passive cable equation disagrees with trace")
                cable_report["equation_max_residual_each_tick"].extend(residual.max(axis=1).tolist())
                # A conserved current and a satisfied PDE can both be wrong if
                # input was placed on the wrong branch. Check measured sites
                # independently of the builder's sparse projection matrix.
                for k, arriving in enumerate(arrived):
                    expected_current = np.bincount(input_nodes,
                        weights=arriving[input_rows] / input_counts[input_rows], minlength=len(node_ids))
                    expected_current += capacities * arriving[uniform_ports[0]]
                    if not np.allclose(expected_current, current[k], atol=1e-12, rtol=1e-12):
                        raise ValueError("Spatial input placement disagrees with measured contacts")
                gain = m["assumptions"]["parameters"]["apl_graded_gain"]
                cap = m["assumptions"]["parameters"]["apl_release_max"]
                for t, voltage in enumerate(v):
                    local_release = np.clip(voltage * gain, 0, cap)
                    expected = np.bincount(terminal_rows, weights=local_release[terminal_nodes], minlength=len(terminal_counts)) / terminal_counts
                    if (not np.allclose(expected, rel[t], atol=1e-12, rtol=1e-12)
                            or not np.isclose(local_release[output_nodes].mean(), data["soma"][t, protocol["apl_row"], 1], atol=1e-12, rtol=1e-12)):
                        raise ValueError("Local release disagrees with measured output sites")
                    if t:
                        cable_report["voltage_min_each_tick"].append(float(voltage.min()))
                        cable_report["voltage_max_each_tick"].append(float(voltage.max()))
                        cable_report["saturated_nodes_each_tick"].append(int((voltage*gain >= cap).sum()))
                        cable_report["terminal_release_min_each_tick"].append(float(rel[t].min()))
                        cable_report["terminal_release_max_each_tick"].append(float(rel[t].max()))
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
    if cable:
        cable_report["checks_passed"] = True
        report["cable"] = cable_report
        report["apl_release_cap_ticks"] = np.flatnonzero(cable_report["saturated_nodes_each_tick"]).tolist()
        report["apl_release_cap_scope"] = "at least one local node reached release cap; not inferred from mean voltage"
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
        keys = ("soma", "M", "post_weight", "terminal_info")
        if "cable_voltage" in ca.files:
            keys += ("cable_voltage", "cable_terminal_release")
        for key in keys:
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


def compare_apl_models(reference: Path, variant: Path):
    """Only the declared global-to-cable substitution may differ.

    Record every first field divergence for each population. In the output-
    blocked control this tests whether the replacement changes upstream cells
    through an unintended route. APL's scalar O has different semantics; actual
    receiving-port inputs and downstream state are the comparable quantities.
    """
    from dataclasses import asdict
    from .paula import Dynamics
    ma, mb = (json.loads((p / "manifest.json").read_text()) for p in (reference, variant))
    pa, pb = (asdict(Dynamics(**m["assumptions"]["parameters"])) for m in (ma, mb))
    if pa.pop("apl_representation") != "global_graded" or pb.pop("apl_representation") != "local_cable":
        raise ValueError("Model comparison requires global_graded then local_cable")
    if pa != pb:
        raise ValueError("Other dynamical parameters changed with APL representation")
    for key in ("anatomical_provenance", "anatomy", "protocol", "condition"):
        if ma[key] != mb[key]:
            raise ValueError(f"Unmatched model-comparison {key}")
    fields = ("soma", "M", "inputs", "local_potential", "post_weight", "terminal_info")
    state_fields = {"soma", "M", "post_weight", "terminal_info"}
    with np.load(reference / "columns.npz") as a, np.load(variant / "columns.npz") as b:
        base_columns = ("cell_ids", "root_ids", "postsynaptic_ports", "terminals", "input_delays",
                        "edge_bindings", "incoming_boundary_ports", "outgoing_boundary_terminals")
        if any(not np.array_equal(a[k], b[k]) for k in base_columns):
            raise ValueError("Model replacement changed neural wiring or column identities")
        ids, roots, ports, terminals = (a[k] for k in ("cell_ids", "root_ids", "postsynaptic_ports", "terminals"))
    inspected_a = inspect_record(reference)
    inspected_b = inspect_record(variant)
    groups = inspected_a[-1]
    first = {name: {field: None for field in fields} for name in groups}
    checked = {name: {field: 0 for field in fields} for name in groups}
    witnesses = {}
    ca, cb = ma["recording"]["chunks"], mb["recording"]["chunks"]
    if [(c["start"], c["stop"]) for c in ca] != [(c["start"], c["stop"]) for c in cb]:
        raise ValueError("Model comparison currently requires aligned recorded chunks")
    for chunk_a, chunk_b in zip(ca, cb, strict=True):
        with np.load(reference / chunk_a["file"]) as a, np.load(variant / chunk_b["file"]) as b:
            for field in fields:
                av, bv = a[field], b[field]
                if field in state_fields and chunk_a["start"] == 0 and not np.array_equal(av[0], bv[0]):
                    raise ValueError(f"Unmatched model initial state: {field}")
                column_ids = ids if field in ("soma", "M") else terminals[:, 0] if field == "terminal_info" else ports[:, 0]
                for name, group_rows in groups.items():
                    mask = np.flatnonzero(np.isin(column_ids, ids[group_rows]))
                    different = av[:, mask] != bv[:, mask]
                    checked[name][field] += different.size
                    changed = np.flatnonzero(np.any(different, axis=tuple(range(1, different.ndim))))
                    if len(changed) and first[name][field] is None:
                        row = int(changed[0])
                        tick = chunk_a["start"] + row - (field in state_fields)
                        first[name][field] = tick
                        if field == "inputs":
                            column, channel = np.argwhere(different[row])[0]
                            port_column = int(mask[column])
                            nid, sid = map(int, ports[port_column])
                            witnesses[name] = {"tick": tick, "neuron_id": nid,
                                "root": str(roots[np.flatnonzero(ids == nid)[0]]), "port": sid,
                                "channel": int(channel), "global": float(av[row, port_column, channel]),
                                "local": float(bv[row, port_column, channel])}
    return {"scope": "matched global-to-passive-cable replacement, not physiological acceptance",
            "condition": ma["condition"], "reference": str(reference), "variant": str(variant),
            "manifests_sha256": [sha256(p / "manifest.json") for p in (reference, variant)],
            "first_exact_field_divergence": first, "compared_values": checked,
            "first_receiving_input_witnesses": witnesses,
            "non_apl_all_recorded_fields_identical": all(t is None for name in ("KC", "PN") for t in first[name].values()),
            "caution": "Exact divergence includes numeric precision differences. APL mean release is not the same observation as global graded release. Downstream ports provide the comparable delivered signal."}


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
    spatial = m["assumptions"].get("spatial")
    p = build_paula(graph, Dynamics(**m["assumptions"]["parameters"]),
                    spatial=Path(spatial["analysis_path"]).parent if spatial else None)
    if list(p.root_to_id) != roots.tolist() or list(p.root_to_id.values()) != ids.tolist():
        raise ValueError("Replay cell identities differ")
    cells = list(p.network.network.neurons.values())
    posts = [s for cell in cells for s in cell.postsynaptic_points.values()]
    terminals = [t for cell in cells for t in cell.presynaptic_points.values()]
    checked = {key: 0 for key in ("soma", "M", "post_weight", "terminal_info")}
    cable_cells = [c for c in cells if hasattr(c, "cable")]
    if cable_cells:
        checked.update(cable_voltage=0, cable_terminal_release=0)
    tick_checked = {key: 0 for key in ("cable_current", "cable_arrived_current", "cable_checks")} if cable_cells else {}

    def check(data, row):
        states = {"soma": [[getattr(c, field) for field in SOMA_FIELDS] for c in cells],
                  "M": [c.M_vector for c in cells],
                  "post_weight": [s.u_i.info for s in posts],
                  "terminal_info": [t.u_o.info for t in terminals]}
        if cable_cells:
            states.update(cable_voltage=np.concatenate([c.cable.voltage for c in cable_cells]),
                          cable_terminal_release=np.concatenate([c.terminal_release for c in cable_cells]))
        for key, values in states.items():
            if not np.array_equal(values, data[key][row]):
                raise ValueError(f"Unobserved replay mismatch: {key}, after {p.network.current_tick} ticks")
            checked[key] += np.asarray(values).size
        if row and cable_cells:
            actual = {"cable_current": np.concatenate([c.cable.last_current for c in cable_cells]),
                      "cable_arrived_current": np.concatenate([c.arrived_port_current for c in cable_cells]),
                      "cable_checks": [[c.cable.last_mass_residual, c.native_mean_error] for c in cable_cells]}
            for key, value in actual.items():
                if not np.array_equal(value, data[key][row-1]):
                    raise ValueError(f"Unobserved replay mismatch: {key}, after {p.network.current_tick} ticks")
                tick_checked[key] += np.asarray(value).size

    for chunk in m["recording"]["chunks"]:
        with np.load(directory / chunk["file"], allow_pickle=False) as compressed:
            data = {key: compressed[key] for key in (checked.keys() | tick_checked.keys())}
        if chunk["start"] == 0:
            check(data, 0)
        for row, tick in enumerate(range(chunk["start"], chunk["stop"]), start=1):
            for i in np.flatnonzero(drive[tick]):
                p.stimulate(str(roots[i]), float(drive[tick, i]))
            p.network.run_tick()
            check(data, row)
    return {"scope": "exact recorded-state agreement without tick instrumentation; no biological acceptance",
            "ticks": p.network.current_tick, "equal_values": {**checked, **tick_checked},
            "manifest_sha256": report["manifest_sha256"]}


def main(directory: Path, output: Path, reference: Path | None = None, source: Path | None = None,
         compare_models: bool = False):
    if source is not None and reference is not None:
        raise ValueError("Choose comparison or uninstrumented verification, not both")
    if compare_models and reference is None:
        raise ValueError("Model comparison requires --reference")
    if source is not None:
        from .connectome import Subgraph
        report = verify_unobserved(Subgraph.load(source), directory)
    else:
        report = inspect_record(directory)[0] if reference is None else compare_apl_models(reference, directory) if compare_models else compare(reference, directory)
    with output.open("x") as out:
        json.dump(report, out, indent=2)
    def compact(value):
        if isinstance(value, dict):
            return {k: compact(v) for k, v in value.items() if "each_tick" not in k}
        return value
    print(json.dumps(compact(report), indent=2))


if __name__ == "__main__":
    typer.run(main)
