"""Estimate measured dopamine-provider coverage from the failed eight-pair course."""
import argparse
import json
from pathlib import Path

import numpy as np

from ..connectome import Subgraph, sha256


def run(base, output):
    base, output = Path(base), Path(output)
    graph = Subgraph.load(base/"memory-balanced-codes-20260910")
    parent=base/"memory-interface-paired-20260910"
    m=json.loads((parent/"manifest.json").read_text())
    paths={n:base/f"memory-interface-eight-{n}-20260910" for n in ("intact","cut")}
    reports={n:json.loads((p/"summary.json").read_text()) for n,p in paths.items()}
    for n,p in paths.items(): assert sha256(p/"soma.npy") == reports[n]["artifacts"]["soma.npy"]
    with np.load(parent/"identities.npz") as z: rows=dict(zip(z["roots"].tolist(),range(len(z["roots"]))))
    phase=next(p for p in reports["intact"]["phases"] if p["name"]=="pair0_A")
    a,c=[np.load(paths[n]/"soma.npy",mmap_mode="r")[phase["begin"]:phase["end"]] for n in ("intact","cut")]
    predictions=[]
    for root in sorted({e["target"] for e in reports["intact"]["projection"]}):
        aa,cc=a[:,rows[root]],c[:,rows[root]]
        if np.any(cc[:,1]): raise ValueError("Background provider spikes invalidate this first-crossing comparison")
        spikes=int(aa[:,1].sum())
        if spikes:
            predictions.append(dict(root=root,observed_A_spikes=spikes,multiplier=1.,tick=None))
            continue
        delta=aa[:,0]-cc[:,0]
        values=np.divide(aa[:,2]-cc[:,0],delta,out=np.full(len(delta),np.inf),where=delta>1e-9)
        t=int(np.argmin(values)); assert np.isfinite(values[t])
        predictions.append(dict(root=root,observed_A_spikes=0,multiplier=float(values[t]),tick=t,
                                threshold=float(aa[t,2]),background=float(cc[t,0]),projection_delta=float(delta[t])))
    # Use all three cue codes and anatomical KC-to-MBON counts, not a selected
    # B behavior or a fitted spike-count objective, to specify the next bound.
    outputs=[e for e in graph.internal if str(e[1]) in m["roles"]["MBON04"]]
    candidates=[]
    for multiplier in (1,2,4,8):
        providers={p["root"] for p in predictions if p["multiplier"]<=multiplier}
        kcs={str(e[1]) for e in graph.internal if str(e[0]) in providers}
        coverage={cue:sum(int(e[4]) for e in outputs if str(e[0]) in code and str(e[0]) in kcs)/
                  sum(int(e[4]) for e in outputs if str(e[0]) in code) for cue,code in m["codes"].items()}
        candidates.append(dict(multiplier=multiplier,providers=len(providers),contact_weighted_coverage=coverage))
    proposed=next(x for x in candidates if min(x["contact_weighted_coverage"].values())>=.5)
    result=dict(source_sha256=sha256(Path(__file__)),input_soma_sha256={n:reports[n]["artifacts"]["soma.npy"] for n in paths},
        parent_manifest_sha256=sha256(parent/"manifest.json"),graph_sha256=m["graph_sha256"],phase=phase,
        targets=predictions,candidates=candidates,proposed=proposed,
        total_source_efficacy=64*proposed["multiplier"],
        limit="Held-input first-crossing estimates for currently silent cells, not coupled-network predictions or measured fly efficacies. The explicit coverage criterion is at least half of each cue's anatomical KC-to-MBON04 contact mass within providers' measured KC footprint. A uniform source-efficacy hypothesis must be tested from birth with paired and unpaired nutrients before any memory-transfer claim.")
    output.mkdir(parents=True,exist_ok=True)
    (output/"recruitment-coverage.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("base",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();run(a.base,a.output)
