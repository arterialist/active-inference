"""Reassign controlled cues using anatomical loads, without changing anatomy.

The original B alone activates eight gamma KCs with 13 direct contacts to
SMP108. Its alpha/beta afferents were also selected after the stronger A/C
pool. Allocate the same 192 KCs into three disjoint equal-size codes, separately
within alpha/beta and gamma types, using only measured contact loads to each
retained MBON07, MBON04, SMP108 and APL. No simulation or behavioral outcome
enters allocation. This remains an engineered cue panel, not natural odors.
"""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from ..connectome import Subgraph, sha256


def reassign(graph):
    original = graph.provenance["controlled_codes"]
    roots = [r for rr in original.values() for r in rr]
    if len(roots) != len(set(roots)):
        raise ValueError("Controlled codes must be disjoint")
    targets = sorted((r for r in graph.selected if graph.nodes[r]["annotation"]["hemibrain_type"]
        in {"MBON07", "MBON04", "SMP108", "APL"}), key=int)
    features = {r: np.zeros(len(targets), dtype=float) for r in roots}
    columns = {r: i for i, r in enumerate(targets)}
    for edge in graph.internal:
        pre, post = str(edge[0]), str(edge[1])
        if pre in features and post in columns:
            features[pre][columns[post]] += int(edge[4])
    codes = {c: [] for c in "ABC"}; allocation = {}
    for prefix in ("KCab-", "KCg-"):
        pool = [r for r in roots if graph.nodes[r]["annotation"]["hemibrain_type"].startswith(prefix)]
        if not pool or len(pool) % 3:
            raise ValueError("Need three equal KC pools per type")
        total = sum((features[r] for r in pool), np.zeros(len(targets)))
        scale = np.where(total > 0, total, 1.)
        normalized = {r: np.r_[features[r]/scale, 1./len(pool)] for r in pool}
        order = sorted(pool, key=lambda r: (-float(normalized[r].max()), -float(normalized[r].sum()), int(r)))
        loads = {c: np.zeros(len(targets)+1) for c in codes}
        bins = {c: [] for c in codes}
        for root in order:
            f = normalized[root]
            eligible = [c for c in codes if len(bins[c]) < len(pool)//3]
            cue = min(eligible, key=lambda c: (float(np.sum((loads[c]+f)**2-loads[c]**2)), len(bins[c]), c))
            bins[cue].append(root); loads[cue] += f
        allocation[prefix] = {c: dict(roots=rr, contacts_by_target={t: int(sum(features[r][i] for r in rr))
            for i, t in enumerate(targets)}) for c, rr in bins.items()}
        for cue in codes:
            codes[cue].extend(bins[cue])
    if sorted(sum(codes.values(), [])) != sorted(roots):
        raise ValueError("Reassignment changed controlled membership")
    provenance = deepcopy(graph.provenance)
    provenance["controlled_codes"] = codes
    provenance["student_codes"] = {c: allocation["KCg-"][c]["roots"] for c in codes}
    provenance["code_reassignment"] = dict(original_codes=original, allocation=allocation,
        targets={r: graph.nodes[r]["annotation"]["hemibrain_type"] for r in targets},
        algorithm="Deterministic greedy allocation by smallest increase in squared normalized anatomical loads; equal cardinality per KC type, no outcome fitting",
        source_sha256=sha256(Path(__file__)),
        limit="Approximate balance for declared targets only. Labels require reciprocal controls before generalization. Every measured pair and KC identity remains.")
    return Subgraph(graph.selected, graph.nodes, graph.edges, provenance)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parent", type=Path); p.add_argument("output", type=Path)
    a=p.parse_args(); g=Subgraph.load(a.parent); changed=reassign(g)
    changed.provenance["code_reassignment"]["parent_manifest_sha256"] = sha256(a.parent/"manifest.json")
    changed.save(a.output)
    for name in ("nodes.json", "edges.npz"):
        if sha256(a.parent/name) != sha256(a.output/name):
            raise AssertionError("Anatomical artifact changed during cue reassignment")
