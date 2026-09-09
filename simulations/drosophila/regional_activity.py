"""Read full APL traces through recorded anatomical volumes, without neural edits.

These are model-voltage and local-release observations, not calcium predictions.
Keep every tick, source-tagged input, regional charge balance and missing region.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .amin_spatial import write_new
from .connectome import sha256
from .intervention_analysis import inspect_record

SOURCES = ("KC", "PN", "APL", "boundary", "experimental")


def contact_projection(nodes, ports, n_ports, masks):
    """Fraction of each port's contacts in each region; overlaps stay overlaps."""
    nodes, ports = np.asarray(nodes), np.asarray(ports)
    counts = np.bincount(ports, minlength=n_ports)
    totals = np.column_stack([
        np.bincount(ports, weights=m[nodes], minlength=n_ports) for m in masks.T])
    fractions = np.divide(totals, counts[:, None], out=np.zeros_like(totals), where=counts[:, None] > 0)
    return fractions, counts


def weighted_mean(values, weights):
    denominator = weights.sum(axis=0)
    return np.divide(values @ weights, denominator, out=np.full((len(values), weights.shape[1]), np.nan),
                     where=denominator > 0)


def regional_balance(voltage, capacity, masks, current, alpha):
    """Positive axial_inflow means net charge arriving across the region boundary."""
    charge = voltage @ (capacity[:, None] * masks)
    return (charge[1:] - (1-alpha)*charge[:-1]) / alpha - current


def analyze(record: Path, regions: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    m = json.loads((record / "manifest.json").read_text())
    params = m["assumptions"]["parameters"]
    if params.get("apl_representation") != "local_cable":
        raise ValueError("Regional voltage needs a local cable recording")
    b = json.loads((regions / "analysis.json").read_text())
    spatial = m["assumptions"]["spatial"]
    if (b["schema"] != "flywire-apl-neuropil-binding-v1" or b["root"] != spatial["root"]
            or b["anatomy_sha256"] != spatial["anatomy_sha256"]
            or sha256(regions / "node_regions.npz") != b["node_regions_sha256"]):
        raise ValueError("Unmatched regional anatomy")
    with np.load(regions / "node_regions.npz", allow_pickle=False) as f:
        ids, xyz, inside, ambiguous, names = (f[k] for k in ("node_ids", "xyz_nm", "inside", "ambiguous", "region_names"))
    if (inside.shape != ambiguous.shape or inside.shape != (len(ids), len(names))
            or inside.dtype != bool or ambiguous.dtype != bool
            or len(set(names.tolist())) != len(names) or np.any(inside & ambiguous)):
        raise ValueError("Invalid region memberships")
    geometry = Path(spatial["analysis_path"]).parent / "anatomy.npz"
    if sha256(geometry) != spatial["anatomy_sha256"]:
        raise ValueError("Changed anatomical geometry")
    with np.load(geometry, allow_pickle=False) as f:
        if not np.array_equal(ids, f["node_ids"]) or not np.array_equal(xyz, f["xyz_nm"]):
            raise ValueError("Region node identities or coordinates differ")
        contacts, pair_rows = f["contacts"], f["pair_source_rows"]
    if sha256(record / "columns.npz") != m["recording"]["columns_sha256"]:
        raise ValueError("Changed recording columns")
    with np.load(record / "columns.npz", allow_pickle=False) as f:
        columns = {k: f[k] for k in ("cell_ids", "root_ids", "cable_nodes", "cable_capacity",
                                   "edge_bindings", "incoming_boundary_ports", "cable_input_ports")}
    cell_ids, roots = columns["cell_ids"], columns["root_ids"]
    apl = int(cell_ids[m["protocol"]["apl_row"]])
    if (not np.array_equal(columns["cable_nodes"][:, 1], ids)
            or not np.all(columns["cable_nodes"][:, 0] == apl)):
        raise ValueError("Recorded voltage columns do not match regional nodes")
    # The independent full-record audit checks every equation, contact placement,
    # delay, source blockade and actual release before any reduction is accepted.
    audit = inspect_record(record)[0]
    unknown = ~(inside | ambiguous).any(axis=1)
    unresolved = ambiguous.any(axis=1)
    masks = np.column_stack([inside, unknown, unresolved, np.ones(len(ids), dtype=bool)])
    names = np.r_[names, ["outside_all", "ambiguous_any", "whole_APL"]]
    cap = columns["cable_capacity"]
    n_ports = len(columns["cable_input_ports"])
    group = {int(cell_ids[i]): label for label, key in (("KC", "kc_rows"), ("PN", "pn_rows"))
             for i in m["protocol"][key]}
    group[apl] = "APL"
    source_names = np.full(n_ports, "experimental", dtype="U12")
    pair_to_port = {}
    for row, pre, _, post, port in columns["edge_bindings"]:
        if post == apl:
            pair_to_port[int(row)] = int(port)
            source_names[port] = group[int(pre)]
    for row, _, post, port in columns["incoming_boundary_ports"]:
        if post == apl:
            pair_to_port[int(row)] = int(port)
            source_names[port] = "boundary"
    root = int(spatial["root"])
    incoming = (contacts[:, 2] == root) & (pair_rows >= 0)
    nodes = contacts[incoming, 6]
    ports = np.array([pair_to_port[int(row)] for row in pair_rows[incoming]], dtype=int)
    projection, counts = contact_projection(nodes, ports, n_ports, masks)
    uniform = np.flatnonzero(counts == 0)
    if len(uniform) != 1 or source_names[uniform[0]] != "experimental":
        raise ValueError("Unexpected contact-free input ports")
    projection[uniform[0]] = cap @ masks
    output_nodes = contacts[contacts[:, 1] == root, 5]
    kc_roots = roots[m["protocol"]["kc_rows"]].astype(np.int64)
    kc_output = (contacts[:, 1] == root) & (pair_rows >= 0) & np.isin(contacts[:, 2], kc_roots)
    kc_nodes = contacts[kc_output, 5]
    contact_weights = {}
    for name, rows in (("all_output", output_nodes), ("kc_output", kc_nodes)):
        contact_weights[name] = np.bincount(rows, minlength=len(ids))[:, None] * masks
    n_ticks, n_regions = m["protocol"]["ticks"], len(names)
    arrays = {"voltage_mean": np.zeros((n_ticks+1, n_regions)),
              "all_output_release_mean": np.zeros((n_ticks+1, n_regions)),
              "kc_output_release_mean": np.zeros((n_ticks+1, n_regions)),
              "all_output_saturated_fraction": np.zeros((n_ticks+1, n_regions)),
              "axial_inflow": np.zeros((n_ticks, n_regions))}
    for key in ("input_signed", "input_positive", "input_negative"):
        arrays[key] = np.zeros((n_ticks, n_regions, len(SOURCES)))
    maximum_current_error = 0.
    for chunk in m["recording"]["chunks"]:
        path = record / chunk["file"]
        if sha256(path) != chunk["sha256"]:
            raise ValueError("Recording changed during regional analysis")
        start, stop = chunk["start"], chunk["stop"]
        with np.load(path, allow_pickle=False) as f:
            v, arrived, current = f["cable_voltage"], f["cable_arrived_current"], f["cable_current"]
        destination = slice(start, stop+1)
        arrays["voltage_mean"][destination] = weighted_mean(v, cap[:, None] * masks)
        release = np.clip(v * params["apl_graded_gain"], 0, params["apl_release_max"])
        for label, weights in contact_weights.items():
            arrays[label+"_release_mean"][destination] = weighted_mean(release, weights)
        arrays["all_output_saturated_fraction"][destination] = weighted_mean(
            v * params["apl_graded_gain"] >= params["apl_release_max"], contact_weights["all_output"])
        for i, source in enumerate(SOURCES):
            selected = source_names == source
            for key, values in (("input_signed", arrived), ("input_positive", np.maximum(arrived, 0)),
                                ("input_negative", np.minimum(arrived, 0))):
                arrays[key][start:stop, :, i] = values[:, selected] @ projection[selected]
        expected = arrays["input_signed"][start:stop].sum(axis=2)
        measured = current @ masks
        maximum_current_error = max(maximum_current_error, float(np.abs(expected-measured).max()))
        np.testing.assert_allclose(expected, measured, atol=1e-9, rtol=1e-11)
        arrays["axial_inflow"][start:stop] = regional_balance(v, cap, masks, measured, 1/params["lambda_ticks"])
    epochs = []
    for epoch in m["protocol"]["epochs"]:
        start, stop = epoch["start"], epoch["stop"]
        per_region = {}
        for j, name in enumerate(names):
            entry = {}
            for key in ("voltage_mean", "all_output_release_mean", "kc_output_release_mean", "all_output_saturated_fraction"):
                values = arrays[key][start+1:stop+1, j]
                entry[key] = float(values.mean()) if np.isfinite(values).all() else None
            entry["input_signed_sum"] = dict(zip(SOURCES, arrays["input_signed"][start:stop, j].sum(axis=0).tolist()))
            entry["axial_inflow_sum"] = float(arrays["axial_inflow"][start:stop, j].sum())
            per_region[str(name)] = entry
        epochs.append({**epoch, "regions": per_region})
    output.mkdir(parents=True)
    with (output / "per_tick.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays, state_after_tick=np.arange(-1, n_ticks), tick=np.arange(n_ticks),
                            region_names=names, source_names=np.array(SOURCES),
                            node_count=masks.sum(axis=0), area_fraction=cap @ masks,
                            all_output_contact_count=contact_weights["all_output"].sum(axis=0),
                            kc_output_contact_count=contact_weights["kc_output"].sum(axis=0),
                            input_contact_count_by_source=np.array([
                                counts[source_names == s] @ projection[source_names == s] for s in SOURCES]).T)
    write_new(output / "full_record_audit.json", audit)
    result = {"schema": "flywire-apl-regional-activity-v1", "condition": m["condition"],
        "record": str(record.resolve()), "region_binding": str(regions.resolve()),
        "manifest_sha256": sha256(record / "manifest.json"), "region_analysis_sha256": sha256(regions / "analysis.json"),
        "per_tick_sha256": sha256(output / "per_tick.npz"), "source_code_sha256": sha256(Path(__file__)),
        "parameters": params, "epochs_not_acceptance": epochs,
        "max_regional_input_reconstruction_error": maximum_current_error,
        "limits": ["Atlas neuropils are not optical ROIs and not same-animal physiology",
            "Voltage and local release have no fitted calcium or biological-time mapping",
            "Source current is the actual native delayed input, divided by contact counts as in the recorded model",
            "Regions classify modeled skeleton attachment sites, not connector centroids",
            "Axial inflow is inferred from regional charge balance, already checked per node by the full-record audit",
            "Regional means are unnormalized; initial state has tick -1, absent readouts are NaN with zero denominator",
            "Overlapping mesh memberships remain overlapping; these readouts need not partition the cell",
            "Local release is before pair averaging, learned terminal gain and blockade; not the delivered synaptic event",
            "Whole-KC output blockade also removes other KC projections, not only KC-to-APL feedback",
            "External PN drive is an artificial course; this is not physiological acceptance"]}
    write_new(output / "analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("record", "regions", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = analyze(args.record, args.regions, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "epochs_not_acceptance"}, indent=2))


if __name__ == "__main__":
    main()
