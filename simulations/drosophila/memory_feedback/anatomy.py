"""Select the alpha1 prerequisite without modifying the sensory preparation."""
import argparse
from collections import Counter
from pathlib import Path
import json

import numpy as np

from ..connectome import read_catalog, iter_edges, extract_subgraph, verify_sources


def prepare(source: Path, kc_count: int = 64):
    if kc_count < 4 or kc_count % 2:
        raise ValueError("Need an even number of at least four controlled KCs")
    sources = verify_sources(source)
    catalog = read_catalog(source)
    types = {"MBON07", "PAM11", "SMP353", "SMP108", "APL"}
    core = {r for r, a in catalog.annotations.items()
            if a["side"] == "left" and a["hemibrain_type"] in types}
    mbons = {r for r in core if catalog.annotations[r]["hemibrain_type"] == "MBON07"}
    counts = Counter()
    for edges in iter_edges(source / "Connectivity_783.parquet", catalog):
        for row in edges[np.isin(edges[:, 1], np.array(list(mbons), dtype=np.int64))]:
            root = str(row[0])
            a = catalog.annotations[root]
            if a["side"] == "left" and a["hemibrain_type"].startswith("KCab-"):
                counts[root] += int(row[4])
    # Anatomical rank only, before any execution. Alternate ranks balance the
    # two disjoint controlled codes approximately; neither is a public odor.
    ranked = sorted(counts, key=lambda r: (-counts[r], int(r)))[:kc_count]
    if len(ranked) != kc_count:
        raise ValueError("Insufficient alpha/beta KCs")
    selected = tuple(r for r in catalog.roots if r in core or r in ranked)
    graph = extract_subgraph(source / "Connectivity_783.parquet", catalog, selected, {
        "materialization": 783, "annotation_version": "2.1.0", "sources": sources,
        "selection": "left MBON07, PAM11, SMP353, SMP108 and APL; strongest KCab afferents by summed MBON07 counts",
        "controlled_codes": {"A": ranked[::2], "C": ranked[1::2]},
        "core_types": sorted(types), "kc_count": kc_count,
        "reference": "All directed pairs among included cells, all incident boundary pairs and identities retained",
        "boundary": "Other KCs, MBON14, lateral-horn drive and SMP108 student DAN targets remain identified but unexecuted",
        "specimen_limit": "FlyWire 783 identities and counts only; hemibrain type labels are correspondences, not hemibrain cell identities",
        "sign_evidence": "Imported Shiu model signs, not measured receptor polarities",
        "selection_limit": "SMP354 is not an exact type annotation in this pinned catalog; no identity was invented or substituted",
    })
    return graph


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    graph = prepare(a.source)
    graph.save(a.output)
    print(json.dumps(graph.summary(), indent=2))
