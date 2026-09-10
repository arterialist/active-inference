"""Audit the student interfaces, retained A benefit and two-pairing controls."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..connectome import sha256


def read(p): return json.loads(Path(p).read_text())


def run(base, output):
    base, output = Path(base), Path(output)
    parent = base/"memory-interface-paired-20260910"
    m = read(parent/"manifest.json")
    with np.load(parent/"identities.npz") as z:
        selected = z["selected"]
        mapping = dict(zip(z["roots"].tolist(), z["cells"].tolist(), strict=True))
    masks = {role+"_"+cue: np.isin(selected[:, 1], [mapping[r] for r in m["codes"][cue]]) &
        np.isin(selected[:, 3], [mapping[r] for r in m["roles"][role]])
        for role in ("MBON07", "MBON04") for cue in ("A", "B", "C")}
    result = dict(source_sha256=sha256(Path(__file__)), graph_sha256=m["graph_sha256"],
        interface=m["assumptions"]["student_interface"], diagnostics={}, courses={}, expression={}, continuation={})
    for mode in ("sensory", "both"):
        for state in ("sham", "zero"):
            p = base/f"memory-interface-probe-{mode}-{state}-20260910"
            r = read(p/"summary.json")
            assert sha256(p/"trace.npz") == r["trace_sha256"]
            result["diagnostics"][mode+"_"+state] = r
    result["diagnostic_neural_equality"] = {}
    for state in ("sham", "zero"):
        with np.load(base/f"memory-interface-probe-sensory-{state}-20260910/trace.npz") as a, np.load(base/f"memory-interface-probe-both-{state}-20260910/trace.npz") as b:
            exact = all(np.array_equal(a[k], b[k]) for k in ("soma", "release", "weights"))
            assert exact
            result["diagnostic_neural_equality"][state] = exact
    for name in ("paired", "unpaired"):
        p = base/f"memory-interface-{name}-20260910"; phases=read(p/"summary.json")
        assert phases[-1]["end"] == 10120
        retention = next(ph["end"] for ph in phases if ph["name"] == "retention")
        q=np.load(p/"release.npy",mmap_mode="r")
        result["courses"][name] = dict(record=p.name, retained_A=next(ph for ph in phases if ph["name"] == "retained_A"),
            retained_release={k: float(q[retention, mask].mean()) for k, mask in masks.items()},
            artifacts={f: sha256(p/f) for f in ("manifest.json", "summary.json", "retention.paula", "soma.npy", "body.npy", "release.npy", "weights.npy")})
    for name in ("A-sham", "A-erased", "A-transferred", "C-paired", "C-unpaired"):
        p=base/f"memory-interface-expression-{name}-20260910"; r=read(p/"summary.json")
        assert sha256(p/"trace.npz") == r["trace_sha256"]
        with np.load(base/r["state"]/"retention-body.npz") as z: initial=float(z["energy"][:2].sum())
        with np.load(p/"trace.npz") as z:
            r["stored_energy_gain_J"] = float(z["body"][-1, 7:9].sum()-initial)
        result["expression"][name]=r
    paths={name:base/f"memory-interface-second-{name}-20260910" for name in ("intact", "cut", "unpaired", "displaced")}
    control=base/"memory-interface-teacher-removed-20260910"
    paths["teacher_removed"]=control/"continuation"
    for name,p in paths.items():
        r=read(p/"summary.json")
        assert r["nutrient_j"] == 0 and r["phases"][-1]["end"] == 3720 and r["branch_rng_preserved"]
        for f,digest in r["artifacts"].items(): assert sha256(p/f)==digest
        q=np.load(p/"release.npy",mmap_mode="r")
        r["record"]=str(p.relative_to(base)); r["summary_sha256"]=sha256(p/"summary.json")
        r["release"]={k:dict(before=float(q[0,mask].mean()),retained=float(q[-1,mask].mean())) for k,mask in masks.items()}
        r["course_spikes"]={role:sum(ph["spikes"][role] for ph in r["phases"]) for role in m["roles"]}
        result["continuation"][name]=r
    receipt=read(control/"intervention.json")
    assert receipt["all_other_runtime_state_exact"]
    assert receipt["continuation_summary_sha256"] == sha256(control/"continuation/summary.json")
    assert receipt["substituted_checkpoint_sha256"] == sha256(control/"receiver/retention.paula")
    result["teacher_intervention"] = receipt
    result["comparisons"] = {}
    for name in ("cut", "unpaired", "displaced", "teacher_removed"):
        with np.load(paths["intact"]/"probe-B.npz") as a,np.load(paths[name]/"probe-B.npz") as b:
            result["comparisons"][name]=dict(B_body_exact=bool(np.array_equal(a["body"],b["body"])),
                extra_mean_B_depression=result["continuation"][name]["release"]["MBON04_B"]["retained"]-result["continuation"]["intact"]["release"]["MBON04_B"]["retained"])
    result["storage"] = dict(method="Transparent APFS compression of completed .npy records; paths and logical SHA-256 hashes unchanged; checkpoints and active records excluded",
        audit_sha256=sha256(base/"transparent-compression.jsonl"),
        compressed_files=len((base/"transparent-compression.jsonl").read_text().splitlines()),
        audited_allocated_bytes_saved=sum(r["allocated_before"]-r["allocated_after"] for line in (base/"transparent-compression.jsonl").read_text().splitlines() for r in [json.loads(line)]))
    result["conclusion"] = (
        "The explicit student cue-input and motor-output interfaces provide a physical response range to imposed B-terminal depression. "
        "From-birth paired acquisition retains an A-specific feeding and stored-energy benefit, transferred and removed by A-terminal substitution. "
        "Two nutrient-free B-before-A pairings add teacher-memory-dependent B depression but no detectable B movement benefit. "
        "The subsequent eight-pairing comparison is audited separately in repeated-interface.json. "
        "Small body differences between other controls can reflect inherited physical state, so any future positive B effect needs within-state memory substitution.")
    output.mkdir(parents=True,exist_ok=True)
    (output/"student-interface.json").write_text(json.dumps(result,indent=2)+"\n")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    for mode,state,label in (("sensory","zero","Cue input only, imposed depression"),("both","sham","Both interfaces, unchanged memory"),("both","zero","Both interfaces, imposed depression")):
        with np.load(base/f"memory-interface-probe-{mode}-{state}-20260910/trace.npz") as z: ax[0].plot(np.arange(200)*.004,z["body"][:,0],label=label)
    ax[0].set(title="The assay can express student depression",xlabel="Dry probe time, seconds",ylabel="Hinge angle, radians")
    ax[0].legend(frameon=False,fontsize=7)
    names=("A-sham","A-erased","A-transferred","C-paired","C-unpaired")
    bars=ax[1].bar(range(5),[result["expression"][n]["well_j"] for n in names],color=["#176B87","#B35932","#176B87","#999999","#BBBBBB"])
    ax[1].bar_label(bars,fmt="%.3f")
    ax[1].set_xticks(range(5),["A learned","A memory\nremoved","A memory\ntransferred","C paired","C unpaired"],fontsize=7)
    ax[1].set(title="Stored A memory improves feeding",ylabel="Ingested energy, J",ylim=(0,1.2))
    ax[2].axis("off")
    ax[2].text(0,1,"Two-pairing transfer remains unexpressed\n\nA memory adds B-terminal depression.\nA-terminal removal eliminates that addition.\n\nB output remains 9 SMP108 spikes.\nIntact and feedback-cut B body traces match.\n\nEight-pairing results: separate audit.\nNo claim of acquired B action.",va="top",linespacing=1.8)
    fig.savefig(output/"student-interface.png",dpi=170);plt.close(fig)
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("records",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();run(a.records,a.output)
