"""Include measured compartment-matched DAN providers to selected KC cells."""
import argparse
import json
from pathlib import Path

from ..connectome import Subgraph, read_catalog, extract_subgraph, verify_sources, sha256


def prepare(source, parent):
    source, parent_path = Path(source), Path(parent)
    parent = Subgraph.load(parent_path)
    sources = verify_sources(source); catalog = read_catalog(source)
    added = set(); routes = []
    for e in parent.edges:
        pre, post = str(e[0]), str(e[1])
        if post not in parent.selected:
            continue
        a, b = catalog.annotations[pre], catalog.annotations[post]
        match = (a["hemibrain_type"] == "PAM11" and b["hemibrain_type"].startswith("KCab-")) or (
            a["hemibrain_type"] in {"PAM07", "PAM08"} and b["hemibrain_type"].startswith("KCg-"))
        if match:
            added.add(pre)
            routes.append(dict(source=pre, target=post, source_row=int(e[8]), count=int(e[4])))
    selected = set(parent.selected)|added
    provenance = dict(parent.provenance, sources=sources,
        parent_graph_sha256=sha256(parent_path/"manifest.json"),
        receptor_provider_addition=sorted(added-set(parent.selected)), receptor_routes=routes,
        receptor_selection="Existing 235-cell cut plus actual PAM11-to-selected-KCab and PAM07/PAM08-to-selected-KCg providers. No unobserved connection or compartment location is invented.",
        receptor_limit="Cell-pair anatomy does not identify subcellular receptor or terminal compartments; coupling of matched cell-type groups remains an explicit hypothesis.")
    return extract_subgraph(source/"Connectivity_783.parquet", catalog,
        tuple(r for r in catalog.roots if r in selected), provenance)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path); p.add_argument("parent", type=Path); p.add_argument("output", type=Path)
    a=p.parse_args(); g=prepare(a.source,a.parent); g.save(a.output); print(json.dumps(g.summary(), indent=2))
