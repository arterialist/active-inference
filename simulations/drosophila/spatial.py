"""Read-only CATMAID snapshots and spatial/count audits, never neural dynamics.

The public m783 import exposes connector locations and their nearest skeleton
attachments. These are not measured conductances, propagation times or separate
pre/post density coordinates. Keep this evidence separate from pair aggregates.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
from http.cookiejar import CookieJar
import json
from pathlib import Path
import re
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from .connectome import Subgraph, _root, sha256

BASE = "https://fafb-flywire.catmaid.org"
APL_LEFT = "720575940624547622"
MAX_RESPONSE = 96 * 1024**2
CONTACT_COLUMNS = ["connector_id", "pre_root", "post_root", "pre_treenode",
                   "post_treenode", "pre_local_index", "post_local_index"]


def _integer(value, *, positive=True):
    if type(value) is not int or value >= 2**63 or value < (1 if positive else 0):
        raise ValueError(f"Invalid exact integer: {value!r}")
    return value


def root_from_name(name):
    match = re.fullmatch(r"flywire: ([0-9]+)", name)
    if not match:
        raise ValueError(f"Unresolved FlyWire identity: {name!r}")
    return _root(match[1])


class PublicSnapshot:
    """Explicit downloads only; store every response and its request provenance."""

    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.cookies = CookieJar()
        self.opener = build_opener(HTTPCookieProcessor(self.cookies))
        self.records = {}
        # Ordinary anonymous session and CSRF exchange, no login or credentials.
        with self.opener.open(BASE + "/", timeout=55) as response:
            response.read(MAX_RESPONSE + 1)
        tokens = [c.value for c in self.cookies if c.name.startswith("csrftoken")]
        if len(tokens) != 1:
            raise ValueError("Expected one public session CSRF cookie")
        self.csrf = tokens[0]

    def get(self, filename, route, data=None, *, post=False):
        url = BASE + route
        data = data or {}
        encoded = urlencode(data).encode("ascii")
        headers = {"Referer": BASE + "/"}
        if post:
            headers.update({"X-CSRFToken": self.csrf,
                            "Content-Type": "application/x-www-form-urlencoded"})
        elif data:
            url += "?" + encoded.decode("ascii")
        request = Request(url, data=encoded if post else None, headers=headers)
        with self.opener.open(request, timeout=55) as response:
            payload = response.read(MAX_RESPONSE + 1)
            if len(payload) > MAX_RESPONSE:
                raise ValueError("Response exceeds the per-request storage limit")
            value = json.loads(payload)
        path = self.directory / filename
        with path.open("xb") as output:
            with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
                compressed.write(payload)
        self.records[filename] = {
            "url": url, "method": "POST" if post else "GET",
            "form": data if post else None, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "response_bytes": len(payload), "response_sha256": hashlib.sha256(payload).hexdigest(),
            "stored_bytes": path.stat().st_size, "sha256": sha256(path),
        }
        print(f"saved {filename}: {len(payload):,} response bytes", flush=True)
        return value


def acquire(directory: Path, root=APL_LEFT):
    root = _root(root)
    snapshot = PublicSnapshot(directory)
    manifest = {"schema": "flywire-catmaid-spatial-v1", "root": root,
                "project_id": 1, "materialization": 783, "status": "incomplete",
                "files": snapshot.records,
                "limits": ["Public server snapshot, not a versioned immutable API",
                           "Connector coordinates and nearest-tree attachments, not cable physiology",
                           "No assumption of identical filtering to Shiu pair counts"]}
    try:
        projects = snapshot.get("projects.json.gz", "/projects/")
        if not any(p["id"] == 1 and p["title"] == "FlyWire m783" for p in projects):
            raise ValueError("Public project no longer identifies itself as FlyWire m783")
        identity = snapshot.get("identity.json.gz", "/1/annotations/query-targets", {
            "name": f"flywire: {root}", "name_exact": "true",
            "with_annotations": "true", "range_length": 2}, post=True)
        if identity["totalRecords"] != 1 or len(identity["entities"]) != 1:
            raise ValueError("Expected one exact neuron identity")
        entity = identity["entities"][0]
        if root_from_name(entity["name"]) != root or len(entity["skeleton_ids"]) != 1:
            raise ValueError("Ambiguous root-to-skeleton identity")
        skeleton = _integer(entity["skeleton_ids"][0])
        manifest["skeleton_id"] = skeleton
        snapshot.get("relations.json.gz", "/1/connectors/types/")
        snapshot.get("compact.json.gz", f"/1/skeletons/{skeleton}/compact-detail", {
            "with_connectors": "true", "with_tags": "true", "with_annotations": "true"})
        connectors = snapshot.get("connectors.json.gz", "/1/connectors/", {
            "skeleton_ids[0]": skeleton, "with_partners": "true", "with_tags": "false"}, post=True)
        partners = sorted({_integer(link[2]) for links in connectors["partners"].values() for link in links})
        for offset in range(0, len(partners), 400):
            chunk = partners[offset:offset + 400]
            names = snapshot.get(f"names-{offset:06d}.json.gz", "/1/skeleton/neuronnames", {
                f"skids[{i}]": skid for i, skid in enumerate(chunk)}, post=True)
            if set(names) != {str(skid) for skid in chunk}:
                raise ValueError("Incomplete partner identity response")
            for name in names.values():
                root_from_name(name)
        manifest["status"] = "complete"
    finally:
        with (directory / "manifest.json").open("x") as out:
            json.dump(manifest, out, indent=2, sort_keys=True)


def load_snapshot(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest["schema"] != "flywire-catmaid-spatial-v1" or manifest["status"] != "complete"
            or manifest["materialization"] != 783 or manifest["project_id"] != 1):
        raise ValueError("Incomplete or unsupported spatial snapshot")
    data = {}
    for filename, record in manifest["files"].items():
        if Path(filename).name != filename:
            raise ValueError("Invalid snapshot filename")
        path = directory / filename
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Changed spatial source: {filename}")
        with gzip.open(path, "rb") as stream:
            payload = stream.read(MAX_RESPONSE + 1)
        if len(payload) != record["response_bytes"] or hashlib.sha256(payload).hexdigest() != record["response_sha256"]:
            raise ValueError(f"Changed response bytes: {filename}")
        data[filename] = json.loads(payload)
    names = {}
    for filename, value in data.items():
        if filename.startswith("names-"):
            if names.keys() & value.keys():
                raise ValueError("Duplicate partner identity batch")
            names.update(value)
    identity = data["identity.json.gz"]
    if identity["totalRecords"] != 1 or len(identity["entities"]) != 1:
        raise ValueError("Ambiguous snapshot identity")
    entity = identity["entities"][0]
    if (root_from_name(entity["name"]) != manifest["root"]
            or entity["skeleton_ids"] != [manifest["skeleton_id"]]
            or root_from_name(names[str(manifest["skeleton_id"])]) != manifest["root"]):
        raise ValueError("Snapshot identity disagreement")
    if not any(p["id"] == 1 and p["title"] == "FlyWire m783" for p in data["projects.json.gz"]):
        raise ValueError("Snapshot project disagreement")
    return manifest, data, names


def parse_spatial(compact, connectors, relation_types, names, skeleton):
    """Validate two independent endpoint representations before counting pairs.

    Polyads retain every pre/post link combination. Connectors lacking a linked
    partner survive with root/treenode 0, never a guessed neuron. Autapses occur
    once per pair, with both local attachment indices retained.
    """
    if len(compact) not in (3, 5):
        raise ValueError("Unsupported compact representation")
    nodes, local_links = compact[:2]
    if not nodes or any(len(row) != 8 for row in nodes):
        raise ValueError("Invalid tree rows")
    ids = np.array([_integer(row[0]) for row in nodes], dtype=np.int64)
    index = {int(node): i for i, node in enumerate(ids)}
    if len(index) != len(ids):
        raise ValueError("Duplicate treenode")
    parent = np.array([-1 if row[1] is None else index[_integer(row[1])] for row in nodes], dtype=np.int64)
    xyz = np.array([row[3:6] for row in nodes], dtype=np.float64)
    radius = np.array([row[6] for row in nodes], dtype=np.float64)
    if not np.isfinite(xyz).all() or not np.isfinite(radius).all():
        raise ValueError("Nonfinite morphology")
    children = np.flatnonzero(parent >= 0)
    if np.any(parent[children] == children):
        raise ValueError("Self-parent tree node")
    lengths = np.linalg.norm(xyz[children] - xyz[parent[children]], axis=1)
    cable = coo_matrix((np.r_[lengths, lengths],
                       (np.r_[children, parent[children]], np.r_[parent[children], children])),
                      shape=(len(ids), len(ids))).tocsr()
    components, _ = connected_components(cable, directed=False)
    if len(children) != len(ids) - components:
        raise ValueError("Cycle in morphology")
    relations = {x["relation"]: _integer(x["relation_id"]) for x in relation_types}
    pre_relation, post_relation = relations["presynaptic_to"], relations["postsynaptic_to"]
    if pre_relation == post_relation:
        raise ValueError("Ambiguous synaptic relations")
    roots = {_integer(int(k)): int(root_from_name(v)) for k, v in names.items()}
    if len(set(roots.values())) != len(roots):
        raise ValueError("Multiple skeletons mapped to one FlyWire root")
    root = roots[skeleton]
    compact_links = {}
    for row in local_links:
        if len(row) != 6 or row[2] not in (0, 1):
            raise ValueError("Unsupported local connector relation")
        node, connector = _integer(row[0]), _integer(row[1])
        if node not in index:
            raise ValueError("Connector attached outside morphology")
        key = (connector, node, pre_relation if row[2] == 0 else post_relation)
        if key in compact_links:
            raise ValueError("Duplicate local connector link")
        compact_links[key] = tuple(row[3:6])
    connector_ids = [_integer(row[0]) for row in connectors["connectors"]]
    if (len(set(connector_ids)) != len(connector_ids)
            or set(connector_ids) != {key[0] for key in compact_links}
            or set(connector_ids) != {int(k) for k in connectors["partners"]}):
        raise ValueError("Connector sets disagree")
    seen_local = set()
    seen_link_ids = set()
    contacts, locations = [], []
    for row in connectors["connectors"]:
        connector, location = row[0], tuple(row[1:4])
        if len(row) != 9 or not np.isfinite(location).all():
            raise ValueError("Invalid connector row")
        links = connectors["partners"][str(connector)]
        for link in links:
            # This public deployment appends creator and two timestamps.
            if len(link) not in (5, 8) or link[3] not in (pre_relation, post_relation):
                raise ValueError("Unsupported partner link")
            for value in link[:4]:
                _integer(value)
            if link[0] in seen_link_ids:
                raise ValueError("Duplicate partner link ID")
            seen_link_ids.add(link[0])
            if link[2] not in roots:
                raise ValueError("Missing partner identity")
            if link[2] == skeleton:
                key = (connector, link[1], link[3])
                if key not in compact_links or compact_links[key] != location or key in seen_local:
                    raise ValueError("Local attachment representations disagree")
                seen_local.add(key)
        pres = [link for link in links if link[3] == pre_relation]
        posts = [link for link in links if link[3] == post_relation]
        if len(pres) > 1:
            raise ValueError("Multiple presynaptic partners at a synaptic connector")
        for pre in pres or [None]:
            for post in posts or [None]:
                if not any(link is not None and link[2] == skeleton for link in (pre, post)):
                    continue
                endpoints = [roots[link[2]] if link else 0 for link in (pre, post)]
                treenodes = [link[1] if link else 0 for link in (pre, post)]
                local = [index[link[1]] if link and link[2] == skeleton else -1 for link in (pre, post)]
                contacts.append([connector, *endpoints, *treenodes, *local])
                locations.append(location)
    if seen_local != set(compact_links):
        raise ValueError("Missing local attachments")
    return {"node_ids": ids, "parents": parent, "xyz_nm": xyz, "radius_nm": radius,
            "contacts": np.array(contacts, dtype=np.int64).reshape(-1, 7),
            "connector_xyz_nm": np.array(locations, dtype=np.float64).reshape(-1, 3),
            "cable": cable, "root": root, "components": components,
            "tree_roots": int((parent < 0).sum()), "cable_length_nm": float(lengths.sum()),
            "zero_length_edges": int((lengths == 0).sum()),
            "raw_connectors": len(connector_ids), "local_links": len(local_links)}


def compare_pairs(contacts, root, graph):
    if str(root) not in graph.selected:
        raise ValueError("Spatial neuron is not selected in the graph")
    observed = Counter((int(r[1]), int(r[2])) for r in contacts if r[1] and r[2])
    expected = {(int(row[0]), int(row[1])): int(row[4]) for row in graph.edges if root in row[:2]}
    differences = [{"pre_root": str(pre), "post_root": str(post),
                    "pair_table": expected.get((pre, post), 0),
                    "spatial_snapshot": observed.get((pre, post), 0)}
                   for pre, post in sorted(expected.keys() | observed.keys())
                   if expected.get((pre, post), 0) != observed.get((pre, post), 0)]
    return {"pair_table_pairs": len(expected), "spatial_pairs": len(observed),
            "pair_table_synapses": sum(expected.values()), "spatial_paired_contacts": sum(observed.values()),
            "exact_pair_counts": sum(expected.get(key) == observed.get(key) for key in expected.keys() | observed.keys()),
            "unpaired_contacts": int(np.sum((contacts[:, 1:3] == 0).any(axis=1))),
            "all_counts_match": not differences, "differences": differences}


def bind_pair_rows(contacts, root, graph):
    """Map each located contact back to its exact original aggregate source row.

    Open connectors keep -1, not a made-up boundary port. Refuse an automatic
    join unless every directed count agrees, including weak and boundary pairs.
    """
    comparison = compare_pairs(contacts, root, graph)
    if not comparison["all_counts_match"]:
        raise ValueError("Cannot bind spatial contacts: pair counts disagree")
    rows = {(int(r[0]), int(r[1])): int(r[8]) for r in graph.edges if root in r[:2]}
    return np.array([rows[(int(r[1]), int(r[2]))] if r[1] and r[2] else -1
                     for r in contacts], dtype=np.int64)


def _quantiles(values):
    a = np.asarray(values)
    finite = a[np.isfinite(a)]
    return {"n": int(a.size), "nonfinite": int(a.size - finite.size),
            "min_q25_median_q75_max": np.quantile(finite, [0, .25, .5, .75, 1]).tolist() if finite.size else []}


def analyze(directory: Path, graph_directory: Path, output: Path):
    source_hash = sha256(Path(__file__))
    manifest, raw, names = load_snapshot(directory)
    parsed = parse_spatial(raw["compact.json.gz"], raw["connectors.json.gz"],
                           raw["relations.json.gz"], names, manifest["skeleton_id"])
    graph = Subgraph.load(graph_directory)
    pairs = compare_pairs(parsed["contacts"], parsed["root"], graph)
    contacts, xyz = parsed["contacts"], parsed["xyz_nm"]
    arrays = {key: parsed[key] for key in
              ("node_ids", "parents", "xyz_nm", "radius_nm", "contacts", "connector_xyz_nm")}
    if pairs["all_counts_match"]:
        arrays["pair_source_rows"] = bind_pair_rows(contacts, parsed["root"], graph)
    report = {"schema": "flywire-spatial-audit-v1", "root": manifest["root"],
              "source_manifest_sha256": sha256(directory / "manifest.json"),
              "pair_manifest_sha256": sha256(graph_directory / "manifest.json"),
              "nodes": len(xyz), "components": parsed["components"], "tree_roots": parsed["tree_roots"],
              "cable_length_um": parsed["cable_length_nm"] / 1000,
              "zero_length_edges": parsed["zero_length_edges"],
              "raw_connectors": parsed["raw_connectors"], "local_links": parsed["local_links"],
              "coordinate_bounds_nm": [xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()],
              "pair_comparison": pairs, "input_populations": {}, "nearest_cable_distances_um": {},
              "interpretation": "Anatomical audit only. No simulated current, calcium, propagation delay or physiological fit."}
    report["analysis_source_sha256"] = source_hash
    report["array_units"] = {"xyz_nm": "nm", "radius_nm": "nm", "connector_xyz_nm": "nm",
                             "distance_from_*_um": "um along imported tree; one value per node_id"}
    report["open_connector_convention"] = "Root/treenode 0 is unlinked; local index and pair source row -1 are absent. These are not neurons or silently added graph edges."
    children = np.flatnonzero(parsed["parents"] >= 0)
    report["tree_edge_length_um"] = _quantiles(np.linalg.norm(xyz[children] - xyz[parsed["parents"][children]], axis=1) / 1000)
    report["directions"] = {}
    for label, column in (("incoming", 2), ("outgoing", 1)):
        directed = contacts[contacts[:, column] == parsed["root"]]
        report["directions"][label] = {"contacts": len(directed),
            "unpaired": int(np.sum((directed[:, 1:3] == 0).any(axis=1)))}
    residuals = []
    for column in (5, 6):
        mask = contacts[:, column] >= 0
        residuals.extend(np.linalg.norm(xyz[contacts[mask, column]] - parsed["connector_xyz_nm"][mask], axis=1) / 1000)
    report["connector_to_attached_node_um"] = _quantiles(residuals)
    groups = {}
    for glomerulus in ("DM3", "VA1d", "DC3"):
        roots = [int(root) for root in graph.selected
                 if graph.nodes[root]["annotation"]["cell_class"] == "ALPN"
                 and graph.nodes[root]["annotation"]["hemibrain_type"].startswith(glomerulus + "_")]
        mask = np.isin(contacts[:, 1], roots) & (contacts[:, 2] == parsed["root"])
        local = contacts[mask, 6]
        groups[glomerulus] = np.unique(local)
        report["input_populations"][glomerulus] = {
            "roots": [str(root) for root in roots], "contacts_onto_apl": int(mask.sum()),
            "distinct_attachment_nodes": len(groups[glomerulus]),
            "coordinate_mean_nm": xyz[local].mean(axis=0).tolist() if len(local) else None,
            "grouping": "All selected ALPN types with this glomerulus prefix, not an equal or purely excitatory odor drive",
            "per_root": {str(root): {
                **{key: graph.nodes[str(root)]["annotation"][key] for key in
                   ("hemibrain_type", "side", "top_nt", "known_nt")},
                "contacts_onto_apl": int(np.sum(mask & (contacts[:, 1] == root)))} for root in roots}}
    for source, indices in groups.items():
        if not len(indices):
            continue
        # Distance to the nearest source site, not an electrical space constant.
        distance = dijkstra(parsed["cable"], indices=indices, min_only=True, directed=False)
        arrays[f"distance_from_{source}_um"] = distance / 1000
        for target, targets in groups.items():
            if source != target and len(targets):
                report["nearest_cable_distances_um"][f"{source} -> {target}"] = _quantiles(distance[targets] / 1000)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "anatomy.npz").open("xb") as out:
        np.savez_compressed(out, **arrays)
    report["anatomy_sha256"] = sha256(output / "anatomy.npz")
    report["contact_columns"] = CONTACT_COLUMNS
    if sha256(Path(__file__)) != source_hash:
        raise RuntimeError("Spatial analyzer source changed during execution")
    with (output / "analysis.json").open("x") as out:
        json.dump(report, out, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in report.items() if k != "pair_comparison"}, indent=2))
    print(json.dumps({k: v for k, v in pairs.items() if k != "differences"}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    acquire_parser = commands.add_parser("acquire", help="Explicit bounded read-only public download")
    acquire_parser.add_argument("directory", type=Path)
    acquire_parser.add_argument("--root", default=APL_LEFT)
    audit = commands.add_parser("analyze", help="Offline spatial and pair-count audit")
    audit.add_argument("directory", type=Path)
    audit.add_argument("graph", type=Path)
    audit.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "acquire":
        acquire(args.directory, args.root)
    else:
        analyze(args.directory, args.graph, args.output)


if __name__ == "__main__":
    main()
