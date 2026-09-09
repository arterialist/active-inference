"""Pinned FlyWire 783 neuron-pair counts, identities and explicit cut boundaries.

This reads the 783 table distributed by Shiu and colleagues, not their Brian2
model. Counts are anatomical observations; ``Excitatory`` is their model sign.
The table has no synapse coordinates, electrical junctions or measured delays.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

SHIU_COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
ANNOTATION_COMMIT = "ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0"
SOURCES = {
    "Connectivity_783.parquet": {
        "url": f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{SHIU_COMMIT}/Connectivity_783.parquet",
        "sha256": "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347",
    },
    "Completeness_783.csv": {
        "url": f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{SHIU_COMMIT}/Completeness_783.csv",
        "sha256": "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311",
    },
    "annotations.tsv": {
        "url": f"https://raw.githubusercontent.com/flyconnectome/flywire_annotations/{ANNOTATION_COMMIT}/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
        "sha256": "30be6c73975a70c56d930e27911f36455d3886e15abf383b78edd2a5d679e0b6",
    },
}
RAW_COLUMNS = (
    "Presynaptic_ID", "Postsynaptic_ID", "Presynaptic_Index",
    "Postsynaptic_Index", "Connectivity", "Excitatory",
    "Excitatory x Connectivity", "__index_level_0__",
)
# Exact int64 arrays on disk. Convert root IDs to strings if exporting to JSON/JS.
EDGE_COLUMNS = (
    "pre_root", "post_root", "pre_index", "post_index", "synapse_count",
    "source_model_sign", "source_signed_count", "source_index", "source_row",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_sources(directory: Path) -> dict:
    """Verify exact release bytes before doing biological selection."""
    records = {}
    for filename, source in SOURCES.items():
        path = directory / filename
        digest = sha256(path)
        if digest != source["sha256"]:
            raise ValueError(f"Source mismatch: {path}; expected {source['sha256']}, got {digest}")
        records[filename] = {**source, "bytes": path.stat().st_size}
    return records


def _root(value: str) -> str:
    # No float intermediate: FlyWire IDs exceed JavaScript's exact integer range.
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValueError(f"Root ID must be an exact decimal string: {value!r}")
    if str(int(value)) != value or not 0 < int(value) < 2**63:
        raise ValueError(f"Invalid root ID: {value!r}")
    return value


@dataclass
class Catalog:
    roots: tuple[str, ...]  # Original global model index order, never cut-local IDs.
    annotations: dict[str, dict[str, str]]

    def __post_init__(self):
        if len(set(self.roots)) != len(self.roots):
            raise ValueError("Duplicate neuron identity in global index")
        for root in self.roots:
            _root(root)
        missing = set(self.roots) - self.annotations.keys()
        if missing:
            raise ValueError(f"Missing annotation identities: {sorted(missing)[:5]}")
        for root, annotation in self.annotations.items():
            if _root(root) != annotation.get("root_id"):
                raise ValueError(f"Annotation identity mismatch: {root}")


def read_catalog(directory: Path) -> Catalog:
    with (directory / "Completeness_783.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["", "Completed"]:
            raise ValueError("Unexpected Completeness_783.csv schema")
        roots = []
        for row in reader:
            if row["Completed"] not in {"True", "False"}:
                raise ValueError("Invalid completeness flag")
            roots.append(_root(row[""]))
    with (directory / "annotations.tsv").open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"root_id", "side", "cell_class", "hemibrain_type", "top_nt", "known_nt"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Missing annotation fields")
        annotations = {}
        for row in reader:
            root = _root(row["root_id"])
            if root in annotations:
                raise ValueError(f"Duplicate annotation: {root}")
            annotations[root] = row
    return Catalog(tuple(roots), annotations)


def iter_edges(path: Path, catalog: Catalog, batch_size: int = 131072) -> Iterator[np.ndarray]:
    """Check every row's indices/count/sign, stream without loading the full graph.

    Duplicate directed pairs are checked after selection, across all retained
    batches. No claim is made about duplicates among discarded rows.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = pq.ParquetFile(path)
    for name in RAW_COLUMNS:
        if name not in source.schema_arrow.names or not pa.types.is_int64(source.schema_arrow.field(name).type):
            raise ValueError(f"Missing/non-int64 connectivity column: {name}")
    indexed_roots = np.asarray(catalog.roots, dtype=np.int64)
    offset = 0
    for batch in source.iter_batches(batch_size=batch_size, columns=list(RAW_COLUMNS)):
        if any(col.null_count for col in batch.columns):
            raise ValueError(f"Null connectivity value near row {offset}")
        cols = [col.to_numpy(zero_copy_only=False) for col in batch.columns]
        edges = np.column_stack([*cols, np.arange(offset, offset + len(batch), dtype=np.int64)])
        for root_col, index_col in ((0, 2), (1, 3)):
            indices = edges[:, index_col]
            if np.any(indices < 0) or np.any(indices >= len(indexed_roots)):
                raise ValueError(f"Connectivity index outside global catalog near row {offset}")
            if not np.array_equal(edges[:, root_col], indexed_roots[indices]):
                raise ValueError(f"Root/index disagreement near row {offset}")
        if np.any(edges[:, 4] <= 0):
            raise ValueError(f"Non-positive synapse count near row {offset}")
        if not np.isin(edges[:, 5], [-1, 1]).all():
            raise ValueError(f"Unknown source model sign near row {offset}")
        if not np.array_equal(edges[:, 6], edges[:, 4] * edges[:, 5]):
            raise ValueError(f"Signed-count disagreement near row {offset}")
        yield edges
        offset += len(batch)


def kc_apl_roots(catalog: Catalog, side: str) -> tuple[str, ...]:
    if side not in {"left", "right"}:
        raise ValueError("Select one explicit hemisphere: left or right")
    apl = [r for r in catalog.roots if catalog.annotations[r]["side"] == side
           and catalog.annotations[r]["hemibrain_type"] == "APL"]
    if len(apl) != 1:
        raise ValueError(f"Expected one {side} APL identity, found {len(apl)}")
    core = tuple(r for r in catalog.roots if catalog.annotations[r]["side"] == side
                 and (catalog.annotations[r]["cell_class"] == "Kenyon_Cell" or r in apl))
    if len(core) < 2:
        raise ValueError("No Kenyon cells selected")
    return core


def alpn_providers(path: Path, catalog: Catalog, core: tuple[str, ...]) -> set[str]:
    """Include only ALPNs with measured projections into the core, on either side."""
    targets = np.asarray(core, dtype=np.int64)
    alpn = np.asarray([r for r in catalog.roots if catalog.annotations[r]["cell_class"] == "ALPN"], dtype=np.int64)
    providers = set()
    for edges in iter_edges(path, catalog):
        keep = np.isin(edges[:, 1], targets) & np.isin(edges[:, 0], alpn)
        providers.update(str(r) for r in edges[keep, 0])
    return providers


@dataclass
class Subgraph:
    selected: tuple[str, ...]
    nodes: dict[str, dict]  # Selected AND boundary endpoints, including global index.
    edges: np.ndarray  # Every edge incident to the selected set, once.
    provenance: dict

    def validate(self) -> None:
        if not self.selected or len(set(self.selected)) != len(self.selected):
            raise ValueError("Empty/duplicate selected identity")
        indices = set()
        for root, node in self.nodes.items():
            _root(root)
            if node["annotation"]["root_id"] != root:
                raise ValueError("Node annotation identity mismatch")
            idx = node["global_index"]
            if type(idx) is not int or not 0 <= idx < 2**36 or idx in indices:
                raise ValueError("Invalid/duplicate global index")
            indices.add(idx)
        if not set(self.selected) <= self.nodes.keys():
            raise ValueError("Selected neuron missing from node table")
        e = self.edges
        if e.dtype != np.dtype("int64") or e.ndim != 2 or e.shape[1] != len(EDGE_COLUMNS):
            raise ValueError("Incorrect edge array schema")
        if len(np.unique(e[:, :2], axis=0)) != len(e):
            raise ValueError("Duplicate directed pair; refusing to silently aggregate")
        if len(np.unique(e[:, 8])) != len(e) or np.any(e[:, 8] < 0):
            raise ValueError("Duplicate/invalid source row")
        if np.any(e[:, 4] <= 0) or not np.isin(e[:, 5], [-1, 1]).all():
            raise ValueError("Invalid count or source model sign")
        if not np.array_equal(e[:, 6], e[:, 4] * e[:, 5]):
            raise ValueError("Signed count mismatch")
        for root_col, index_col in ((0, 2), (1, 3)):
            for root, idx in np.unique(e[:, [root_col, index_col]], axis=0):
                if str(root) not in self.nodes or self.nodes[str(root)]["global_index"] != int(idx):
                    raise ValueError("Edge endpoint/global index mismatch")
        inside_pre, inside_post = self.membership()
        if not (inside_pre | inside_post).all():
            raise ValueError("Nonincident edge in subgraph")

    def membership(self) -> tuple[np.ndarray, np.ndarray]:
        selected = np.asarray(self.selected, dtype=np.int64)
        return np.isin(self.edges[:, 0], selected), np.isin(self.edges[:, 1], selected)

    @property
    def internal(self) -> np.ndarray:
        pre, post = self.membership()
        return self.edges[pre & post]

    def summary(self) -> dict:
        pre, post = self.membership()
        result = {"selected_neurons": len(self.selected), "boundary_neurons": len(self.nodes) - len(self.selected)}
        for name, mask in (("internal", pre & post), ("incoming_boundary", ~pre & post), ("outgoing_boundary", pre & ~post)):
            e = self.edges[mask]
            result[name] = {"directed_pairs": len(e), "synapses": int(e[:, 4].sum())}
        result["classes"] = dict(Counter(self.nodes[r]["annotation"]["cell_class"] for r in self.selected))
        motifs = Counter()
        for edge in self.edges[pre & post]:
            names = []
            for root in edge[:2]:
                a = self.nodes[str(root)]["annotation"]
                names.append("APL" if a["hemibrain_type"] == "APL" else a["cell_class"])
            motifs[" -> ".join(names)] += int(edge[4])
        result["internal_motif_synapses"] = dict(sorted(motifs.items()))
        return result

    def save(self, directory: Path) -> None:
        self.validate()
        directory.mkdir(parents=True, exist_ok=False)
        with (directory / "nodes.json").open("x") as out:
            json.dump({"selected": self.selected, "nodes": self.nodes}, out, sort_keys=True)
        with (directory / "edges.npz").open("xb") as out:
            np.savez_compressed(out, edges=self.edges)
        manifest = {"schema": "flywire-pair-subgraph-v1", "edge_columns": EDGE_COLUMNS,
                    "provenance": self.provenance, "summary": self.summary(),
                    "files": {f: sha256(directory / f) for f in ("nodes.json", "edges.npz")}}
        with (directory / "manifest.json").open("x") as out:
            json.dump(manifest, out, indent=2, sort_keys=True)

    @classmethod
    def load(cls, directory: Path) -> Subgraph:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["schema"] != "flywire-pair-subgraph-v1" or manifest["edge_columns"] != list(EDGE_COLUMNS):
            raise ValueError("Unsupported subgraph schema")
        for filename in ("nodes.json", "edges.npz"):
            if sha256(directory / filename) != manifest["files"][filename]:
                raise ValueError(f"Changed artifact: {filename}")
        data = json.loads((directory / "nodes.json").read_text())
        with np.load(directory / "edges.npz", allow_pickle=False) as arrays:
            edges = arrays["edges"]
        graph = cls(tuple(data["selected"]), data["nodes"], edges, manifest["provenance"])
        graph.validate()
        if graph.summary() != manifest["summary"]:
            raise ValueError("Subgraph summary disagrees with retained anatomy")
        return graph


def extract_subgraph(path: Path, catalog: Catalog, selected: tuple[str, ...], provenance: dict) -> Subgraph:
    if not set(selected) <= set(catalog.roots) or len(set(selected)) != len(selected):
        raise ValueError("Unknown or duplicate selection")
    selected_array = np.asarray(selected, dtype=np.int64)
    chunks = []
    total_rows = total_synapses = 0
    for edges in iter_edges(path, catalog):
        total_rows += len(edges)
        total_synapses += int(edges[:, 4].sum())
        mask = np.isin(edges[:, 0], selected_array) | np.isin(edges[:, 1], selected_array)
        if mask.any():
            chunks.append(edges[mask])
    retained = np.concatenate(chunks) if chunks else np.empty((0, len(EDGE_COLUMNS)), dtype=np.int64)
    endpoints = {str(r) for r in np.unique(retained[:, :2])} | set(selected)
    nodes = {r: {"global_index": i, "annotation": catalog.annotations[r]}
             for i, r in enumerate(catalog.roots) if r in endpoints}
    graph = Subgraph(selected, nodes, retained, {**provenance,
                     "source_rows_checked": total_rows, "source_synapses": total_synapses,
                     "duplicate_check_scope": "all retained directed pairs, including cut boundary"})
    graph.validate()
    return graph


def prepare_mushroom_body(directory: Path, side: str = "left") -> Subgraph:
    sources = verify_sources(directory)
    catalog = read_catalog(directory)
    path = directory / "Connectivity_783.parquet"
    core = kc_apl_roots(catalog, side)
    providers = alpn_providers(path, catalog, core)
    selected_set = set(core) | providers
    selected = tuple(r for r in catalog.roots if r in selected_set)
    return extract_subgraph(path, catalog, selected, {
        "materialization": 783, "annotation_version": "2.1.0", "sources": sources,
        "selection": {"side": side, "core": "cell_class Kenyon_Cell or hemibrain_type APL",
                      "providers": "all ALPNs with >=1 counted synapse onto a core neuron, either side",
                      "core_neurons": len(core), "provider_neurons": len(providers)},
        "boundary_policy": "retained as anatomy; not automatically clamped, simulated, or imputed",
        "missing": ["individual synapse coordinates", "electrical junctions", "measured physiological strengths", "conduction times"],
        "sign_evidence": "Shiu model Excitatory column, not measured receptor polarity",
    })
