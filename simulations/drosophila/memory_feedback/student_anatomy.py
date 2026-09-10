"""Add the identified gamma4 student compartment to the alpha1 reference.

Use MBON04 cells with measured input to the selected SMP108, and their actual
PAM07/PAM08 providers. Include both sides where the measured route crosses.
Preserve the parent A/C alpha-beta codes; add a third controlled code B. All
included pair directions and boundary identities remain from FlyWire 783.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from ..connectome import Subgraph, read_catalog, iter_edges, verify_sources, extract_subgraph


def prepare(source, parent):
    source, parent = Path(source), Subgraph.load(Path(parent))
    sources = verify_sources(source); catalog = read_catalog(source)
    smp = next(r for r in parent.selected if parent.nodes[r]["annotation"]["hemibrain_type"] == "SMP108")
    student = {str(e[0]) for e in parent.edges if str(e[1]) == smp and
               parent.nodes[str(e[0])]["annotation"]["hemibrain_type"] == "MBON04"}
    if not student:
        raise ValueError("No identified gamma4 output to SMP108")
    teachers = {r for r in parent.selected if parent.nodes[r]["annotation"]["hemibrain_type"] == "MBON07"}
    gamma, alpha, dans = Counter(), Counter(), set()
    targets = np.array(list(student|teachers),dtype=np.int64)
    for edges in iter_edges(source/"Connectivity_783.parquet",catalog):
        for e in edges[np.isin(edges[:,1],targets)]:
            pre, post = str(e[0]), str(e[1]); a = catalog.annotations[pre]
            if post in student:
                if a["hemibrain_type"] in ("PAM07","PAM08"):
                    dans.add(pre)
                if a["hemibrain_type"].startswith("KCg-"):
                    gamma[pre] += int(e[4])
            if post in teachers and a["side"] == "left" and a["hemibrain_type"].startswith("KCab-"):
                alpha[pre] += int(e[4])
    ranked_gamma = sorted(gamma,key=lambda r:(-gamma[r],int(r)))[:96]
    new_alpha = sorted((r for r in alpha if r not in parent.selected),key=lambda r:(-alpha[r],int(r)))[:32]
    if len(ranked_gamma)!=96 or len(new_alpha)!=32:
        raise ValueError("Insufficient controlled afferents")
    sides = {catalog.annotations[r]["side"] for r in ranked_gamma}
    apl = {r for r,a in catalog.annotations.items() if a["hemibrain_type"]=="APL" and a["side"] in sides}
    selected_set = set(parent.selected)|student|dans|set(ranked_gamma)|set(new_alpha)|apl
    codes = {k:list(v) for k,v in parent.provenance["controlled_codes"].items()}
    codes["B"] = new_alpha
    student_codes = {cue:ranked_gamma[i::3] for i,cue in enumerate(("A","B","C"))}
    for cue, roots in student_codes.items():
        codes[cue] += roots
    return extract_subgraph(source/"Connectivity_783.parquet",catalog,
        tuple(r for r in catalog.roots if r in selected_set), dict(
            materialization=783,annotation_version="2.1.0",sources=sources,
            parent_selection=parent.provenance,controlled_codes=codes,student_codes=student_codes,
            student_MBONS=sorted(student),student_DANS=sorted(dans),teaching_interneuron=smp,
            selection="Parent alpha1 cut plus MBON04 afferents to SMP108, actual PAM07/PAM08 providers, 96 strongest gamma KCs, their APLs and 32 new alpha-beta KCs",
            boundary="Every pair incident to the selection retained; all directed pairs within selection execute; other boundary cells remain absent",
            limit="Controlled codes and rank-based KC sample, not natural odor processing. Imported uniform transmitter signs remain assumptions. No mixing with hemibrain counts."))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path);p.add_argument("parent",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();g=prepare(a.source,a.parent);g.save(a.output);print(json.dumps(g.summary(),indent=2))
