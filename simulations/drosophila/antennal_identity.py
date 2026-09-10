"""Keep imported model signs, predictions and curated transmitter evidence apart."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .connectome import Subgraph
from .prisco import digest, dump_new


def identity_audit(graph):
    records = []
    for root in graph.selected:
        a = graph.nodes[root]["annotation"]
        if a["cell_class"] != "ALLN":
            continue
        outgoing = graph.edges[graph.edges[:, 0] == int(root)]
        signs = np.unique(outgoing[:, 5]).tolist()
        known = a.get("known_nt", "")
        tokens = {s.strip().lower() for s in known.split(",") if s.strip()}
        prediction = a.get("top_nt", "")
        confidence = a.get("top_nt_conf", "")
        confidence = float(confidence) if confidence else None
        if confidence is not None and not 0 <= confidence <= 1:
            raise ValueError("Invalid transmitter confidence")
        records.append({"root": root, "global_index": graph.nodes[root]["global_index"],
            "type": a["hemibrain_type"], "side": a.get("side", ""),
            "source_model_signs": signs, "outgoing_pairs": len(outgoing),
            "outgoing_contacts": int(outgoing[:, 4].sum()),
            "top_nt": prediction, "top_nt_conf": confidence,
            "known_nt": known, "known_nt_source": a.get("known_nt_source", ""),
            "prediction_outside_curated_transmitter_set": bool(tokens and prediction not in tokens),
            "curated_gaba_positive_model_candidate": "gaba" in tokens and signs == [1]})
    groups = Counter((r["known_nt"], r["known_nt_source"], tuple(r["source_model_signs"])) for r in records)
    return {"schema": 1, "cells": records,
        "known_source_sign_groups": [{"known_nt": k[0], "source": k[1], "signs": list(k[2]), "cells": n}
                                     for k, n in sorted(groups.items())],
        "gaba_positive_candidates": [r["root"] for r in records if r["curated_gaba_positive_model_candidate"]],
        "prediction_curated_conflicts": sum(r["prediction_outside_curated_transmitter_set"] for r in records),
        "limits": ["known_nt is a curated annotation, not an assay of each FlyWire cell or receptor.",
                   "A transmitter annotation does not determine every target's response or peptide action.",
                   "Positive GABA candidates motivate a signed-current sensitivity control, not an automatic correction.",
                   "No prediction threshold, sign replacement or anatomical edit is applied by this audit."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("graph", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    report = identity_audit(Subgraph.load(a.graph))
    report["graph_manifest_sha256"] = digest(a.graph / "manifest.json")
    report["analysis_source_sha256"] = digest(Path(__file__))
    dump_new(a.output, report)


if __name__ == "__main__":
    main()
