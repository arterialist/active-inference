"""Audit repeated exposure and imposed expression bounds without new training."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..connectome import sha256
from .plastic_reach import reached_mask


def read(path): return json.loads(Path(path).read_text())


def run(base, output):
    base, output = Path(base), Path(output)
    parent = base/"memory-interface-paired-20260910"
    m = read(parent/"manifest.json")
    with np.load(parent/"identities.npz") as z:
        selected = z["selected"]; roots = z["roots"].tolist(); ids = z["cells"].tolist()
    mapping = dict(zip(roots, ids, strict=True)); rows = dict(zip(roots, range(len(ids)), strict=True))
    mask = np.isin(selected[:, 1], [mapping[r] for r in m["codes"]["B"]]) & np.isin(
        selected[:, 3], [mapping[r] for r in m["roles"]["MBON04"]])
    paths = {n:base/f"memory-interface-eight-{n}-20260910" for n in ("intact", "cut")}
    result = dict(source_sha256=sha256(Path(__file__)), graph_sha256=m["graph_sha256"], courses={}, bounds={})
    release = {}
    for name, p in paths.items():
        r = read(p/"summary.json")
        assert r["cut"] == (name == "cut") and not r["displaced"]
        assert r["pairings"] == 8 and r["phases"][-1]["end"] == 11880
        assert r["nutrient_j"] == 0 and r["branch_rng_preserved"]
        assert r["minimum_eta_post"] > 0 and r["minimum_eta_retro"] > 0
        assert r["input_checkpoint_sha256"] == sha256(parent/"retention.paula")
        for f, digest in r["artifacts"].items(): assert sha256(p/f) == digest
        q = np.load(p/"release.npy", mmap_mode="r"); release[name] = q[-1].copy()
        r["mean_B_release"] = float(q[-1, mask].mean())
        r["B_release_after_each_pair"] = [float(q[ph["end"], mask].mean()) for ph in r["phases"] if ph["name"].endswith("recovery")]
        r["record"] = p.name; r["summary_sha256"] = sha256(p/"summary.json")
        result["courses"][name] = r
    with np.load(paths["intact"]/"probe-B.npz") as a, np.load(paths["cut"]/"probe-B.npz") as b:
        result["B_body_exact"] = bool(np.array_equal(a["body"], b["body"]))
        result["B_output_events_exact"] = bool(np.array_equal(a["soma"][:, rows[m["roles"]["SMP108"][0]], 1], b["soma"][:, rows[m["roles"]["SMP108"][0]], 1]))
        spikes = a["soma"][:, :, 1].sum(axis=0)
    reached = reached_mask(mask, release["intact"], release["cut"])
    result["coverage"] = {}
    w = np.load(paths["cut"]/"weights.npy", mmap_mode="r")[-1]
    for root in m["roles"]["MBON04"]:
        target = mask & (selected[:, 3] == mapping[root]); hit = target & reached
        result["coverage"][root] = dict(B_terminals=int(target.sum()), reached_terminals=int(hit.sum()),
            B_probe_spikes=int(spikes[rows[root]]), reached_source_rows=selected[hit, 0].tolist(),
            reached_fraction_of_receiving_weight=float(w[hit].sum()/w[target].sum()),
            retained_weighted_release_reduction=float(1-np.dot(w[target],release["intact"][target])/np.dot(w[target],release["cut"][target])))
    for name in ("sham", "reached", "all"):
        p=base/f"memory-interface-eight-bound-{name}-20260910"; r=read(p/"summary.json")
        assert sha256(p/"trace.npz") == r["trace_sha256"]
        assert r["expression_bound"]["all_other_runtime_state_exact"]
        assert r["expression_bound"]["reached_source_rows"] == selected[reached,0].tolist()
        r["record"]=p.name; result["bounds"][name]=r
    with np.load(base/"memory-interface-eight-bound-sham-20260910/trace.npz") as a, np.load(paths["intact"]/"probe-B.npz") as b:
        result["bound_sham_replay_exact"] = all(np.array_equal(a[k], b[k]) for k in ("soma", "body"))
        assert result["bound_sham_replay_exact"]
    p=base/"memory-interface-eight-projection-20260910"; r=read(p/"summary.json")
    assert sha256(p/"trace.npz") == r["trace_sha256"]
    r["record"]=p.name; result["fourfold_projection"]=r
    result["conclusion"] = (
        "Eight pairings increase feedback-dependent B-terminal depression but leave retained B physical trajectories exactly unchanged against feedback block. "
        "A feeding remains 1.000 J. Only five of 32 B-to-MBON04 terminals show a depressive feedback contrast. "
        "Imposed complete depression of those five changes SMP108 output from nine to ten spikes, versus 17 when all 32 are zeroed. "
        "These expression bounds are not acquired B behavior. The fixed fourfold log-depression projection is also diagnostic, not a full-course outcome.")
    output.mkdir(parents=True,exist_ok=True)
    (output/"repeated-interface.json").write_text(json.dumps(result,indent=2)+"\n")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig, ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for n in ("intact", "cut"):
        ax[0].plot(range(1,9),result["courses"][n]["B_release_after_each_pair"],"o-",label=n)
    ax[0].set(xlabel="Completed pairings",ylabel="Mean B terminal release",title="More depression, unchanged B action")
    ax[0].legend(frameon=False)
    labels=["Eight pairs", "Fourfold\nprojection", "Reached five\nset to zero", "All 32\nset to zero"]
    values=[result["bounds"]["sham"]["spikes"]["SMP108"],r["spikes"]["SMP108"],result["bounds"]["reached"]["spikes"]["SMP108"],result["bounds"]["all"]["spikes"]["SMP108"]]
    bars=ax[1].bar(labels,values,color=["#176B87","#B35932","#B35932","#B35932"])
    ax[1].bar_label(bars);ax[1].set(ylim=(0,19),ylabel="Retained B probe SMP108 spikes",title="Orange bars are imposed diagnostics")
    fig.savefig(output/"repeated-interface.png",dpi=170);plt.close(fig)
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("base",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();run(a.base,a.output)
