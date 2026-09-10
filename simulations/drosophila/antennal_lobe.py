"""Reconstruct a declared DL5 upstream neighborhood without truncating new cells.

Promoting a boundary node from a previously cut graph is insufficient: its
connections to other boundary nodes were never present. Re-extract every
incident pair from the original source, retaining the prior preparation intact.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .connectome import Subgraph, extract_subgraph, iter_edges, read_catalog, verify_sources, sha256
from .pn_current import dl5_cut
from .prisco import dump_new


def upstream_selection(path, catalog, parent):
    root, _ = dl5_cut(parent)
    orns = {r for r in catalog.roots if catalog.annotations[r]["hemibrain_type"] == "ORN_DL5"}
    if not orns:
        raise ValueError("No annotated ORN_DL5 population")
    targets = np.asarray(sorted(orns | {root}), dtype=np.int64)
    allns = np.asarray([r for r in catalog.roots if catalog.annotations[r]["cell_class"] == "ALLN"], dtype=np.int64)
    selected_lns, witnesses = set(), []
    for edges in iter_edges(path, catalog):
        toward = np.isin(edges[:, 0], allns) & np.isin(edges[:, 1], targets)
        away = np.isin(edges[:, 1], allns) & np.isin(edges[:, 0], targets)
        selected_lns.update(map(str, edges[toward, 0]))
        selected_lns.update(map(str, edges[away, 1]))
        if (toward | away).any():
            witnesses.append(edges[toward | away])
    selection = set(parent.selected) | orns | selected_lns
    ordered = tuple(r for r in catalog.roots if r in selection)
    if len(ordered) != len(selection):
        raise ValueError("Parent selection is absent from source catalog")
    return ordered, orns, selected_lns, np.concatenate(witnesses) if witnesses else np.empty((0, 9), dtype=np.int64)


def verify_extension(parent, expanded):
    """Check all prior incident rows, not just internal pair totals."""
    parent.validate()
    expanded.validate()
    if not set(parent.selected) <= set(expanded.selected):
        raise ValueError("Expansion removed a previously simulated neuron")
    targets = np.asarray(parent.selected, dtype=np.int64)
    inherited = expanded.edges[np.isin(expanded.edges[:, 0], targets) | np.isin(expanded.edges[:, 1], targets)]
    old = parent.edges[np.argsort(parent.edges[:, 8])]
    inherited = inherited[np.argsort(inherited[:, 8])]
    np.testing.assert_array_equal(old, inherited)
    for r, node in parent.nodes.items():
        if expanded.nodes.get(r) != node:
            raise ValueError(f"Changed prior annotation/index: {r}")
    # Since every prior incident row and its original order survives, all
    # existing input/terminal port indices remain the same in build_paula.
    return {"prior_incident_pairs_exact": len(old), "prior_nodes_exact": len(parent.nodes),
            "prior_cell_port_order_preserved": True, "new_cells_need_full_source_reextraction": True}


def missing_incident_inventory(parent, expanded, added):
    records = []
    for root in sorted(added, key=lambda r: expanded.nodes[r]["global_index"]):
        raw = int(root)
        true_in = expanded.edges[:, 1] == raw
        true_out = expanded.edges[:, 0] == raw
        old_in = parent.edges[:, 1] == raw
        old_out = parent.edges[:, 0] == raw
        a = expanded.nodes[root]["annotation"]
        records.append({"root": root, "type": a["hemibrain_type"], "class": a["cell_class"],
            "side": a.get("side", ""), "top_nt": a.get("top_nt", ""), "known_nt": a.get("known_nt", ""),
            "incoming_pairs": int(true_in.sum()), "outgoing_pairs": int(true_out.sum()),
            "incoming_pairs_missing_from_parent_cut": int(true_in.sum()-old_in.sum()),
            "outgoing_pairs_missing_from_parent_cut": int(true_out.sum()-old_out.sum())})
    return records


def prepare(source, parent_path, output):
    if output.exists():
        raise FileExistsError(output)
    sources = verify_sources(source)
    parent = Subgraph.load(parent_path)
    for filename, checked in sources.items():
        if parent.provenance["sources"][filename]["sha256"] != checked["sha256"]:
            raise ValueError("Parent and expansion use different source releases")
    catalog = read_catalog(source)
    path = source/"Connectivity_783.parquet"
    selected, orns, lns, witnesses = upstream_selection(path, catalog, parent)
    definition = {"inherited": "all prior selected PN/KC/APL neurons",
        "orn": "every cell annotated ORN_DL5, both hemispheres, whether or not it projects to the target PN",
        "ln": "every ALLN with at least one counted pair in either direction with an ORN_DL5 or the identified DL5 PN",
        "minimum_pair_count": 1, "filter_by_sign_or_transmitter": False,
        "orn_roots": sorted(orns), "ln_roots": sorted(lns),
        "recruitment": "anatomical neighborhood, not a physiologically sufficient or closed antennal lobe"}
    expanded = extract_subgraph(path, catalog, selected, {
        "sources": sources, "parent_manifest_sha256": sha256(parent_path/"manifest.json"),
        "selection": definition, "materialization": 783,
        "boundary_policy": "all incident pairs retained; boundary neurons not automatically stimulated or simulated",
        "sign_evidence": "source model signs; no receptor or presynaptic-versus-postsynaptic mechanism inferred from counts"})
    fidelity = verify_extension(parent, expanded)
    added = set(expanded.selected)-set(parent.selected)
    inventory = missing_incident_inventory(parent, expanded, added)
    # Check address feasibility without clipping or merging any port.
    receiving, sending = Counter(map(int, expanded.edges[:, 3])), Counter(map(int, expanded.edges[:, 2]))
    overflow = []
    for r in expanded.selected:
        nid = expanded.nodes[r]["global_index"]
        if receiving[nid]+1 > 4096 or sending[nid] > 4096:
            overflow.append({"root": r, "inputs_including_electrode_port": receiving[nid]+1, "outputs": sending[nid]})
    output.mkdir(parents=True)
    expanded.save(output/"graph")
    with (output/"selection-witnesses.npz").open("xb") as f:
        np.savez_compressed(f, edges=witnesses)
    report = {"schema": 1, "selection": definition, "summary": expanded.summary(), "fidelity": fidelity,
        "added_cells": inventory, "added_classes": dict(Counter(expanded.nodes[r]["annotation"]["cell_class"] for r in added)),
        "address_overflow": overflow, "source_code_sha256": sha256(Path(__file__)),
        "parent_manifest_sha256": sha256(parent_path/"manifest.json"),
        "graph_manifest_sha256": sha256(output/"graph/manifest.json"),
        "witness_sha256": sha256(output/"selection-witnesses.npz"),
        "claim": "Exact anatomical expansion only; no new dynamics, receptor mechanisms, odor coding or behavioral acceptance"}
    dump_new(output/"analysis.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "parent", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    result = prepare(a.source, a.parent, a.output)
    print(json.dumps({k: result[k] for k in ("summary", "fidelity", "added_classes", "address_overflow")}, indent=2))


if __name__ == "__main__":
    main()
