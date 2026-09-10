"""Interpret the single additional hop without extending or tuning it."""
import argparse
import json
from pathlib import Path

import numpy as np

from ..connectome import Subgraph,sha256


def read(path): return json.loads(Path(path).read_text())


def screen(base,manifest,roots,selected,mapping):
    rows=dict(zip(roots,range(len(roots))))
    a,c=[np.load(base/f"memory-coverage-eight-diagnostics-20260910/{name}/trace.npz")["soma"] for name in ("sham","removed")]
    providers=manifest["roles"]["PAM07"]+manifest["roles"]["PAM08"]
    changed=[r for r in providers if not np.array_equal(a[:,rows[r],1],c[:,rows[r],1])]
    graph=Subgraph.load(base/"memory-balanced-codes-20260910")
    kcs={str(e[1]) for e in graph.internal if str(e[0]) in changed and str(e[1]) in manifest["codes"]["C"]}
    mask=np.isin(selected[:,1],[mapping[r] for r in manifest["codes"]["C"]])&np.isin(selected[:,3],[mapping[r] for r in manifest["roles"]["MBON04"]])
    q=np.load(base/"memory-coverage-eight-intact-20260910/release.npy",mmap_mode="r")[-1]
    return dict(B_teacher_spikes=[int(x[:,rows[manifest["roles"]["SMP108"][0]],1].sum()) for x in (a,c)],
        changed_DAN_roots=changed,C_KCs_reached_by_changed_DAN_events=sorted(kcs),C_terminal_mean_at_start=float(q[mask].mean()),
        limit="Read-only screen of existing isolated B probes. Nonzero neural access permits a new comparison but does not predict coupled C-before-B learning or expression.")


def run(base,output):
    base,output=Path(base),Path(output)
    original=base/"memory-coverage-eight-intact-20260910"
    removed_donor=base/"memory-coverage-eight-blocked-20260910"
    genetic=read(base/"memory-coverage-paired-20260910/manifest.json")
    with np.load(original/"identities.npz") as z:
        selected=z["selected"];roots=z["roots"].tolist();mapping=dict(zip(roots,z["cells"].tolist()))
    masks={role+"_"+cue:np.isin(selected[:,1],[mapping[r] for r in genetic["codes"][cue]]) &
           np.isin(selected[:,3],[mapping[r] for r in genetic["roles"][role]]) for role in ("MBON07","MBON04") for cue in ("A","B","C")}
    result=dict(source_sha256=sha256(Path(__file__)),screen=screen(base,genetic,roots,selected,mapping),courses={},substitutions={})
    for name in ("intact","removed"):
        root=base/f"memory-reuse-{name}-20260910";p=root/"course";parent=root/"receiver"
        m=read(parent/"manifest.json");receipt=read(parent/"intervention.json");r=read(p/"summary.json")
        assert m["assumptions"]==genetic["assumptions"] and m["codes"]==genetic["codes"] and m["graph_sha256"]==genetic["graph_sha256"]
        assert receipt["all_other_runtime_state_exact"] and receipt["removed"]==(name=="removed")
        assert receipt["original_state_sha256"]==sha256(original/"retention.paula")
        assert receipt["donor_state_sha256"]==sha256(removed_donor/"retention.paula")
        assert sha256(parent/"retention-body.npz")==sha256(original/"retention-body.npz")
        assert r["input_checkpoint_sha256"]==sha256(parent/"retention.paula")
        assert r["nutrient_j"]==0 and r["minimum_eta_post"]>0 and r["minimum_eta_retro"]>0 and r["branch_rng_preserved"]
        assert r["phases"][-1]["end"]==11880 and r["phases"][-1]["begin"]==10880
        assert {ph["cue"] for ph in r["phases"]}=={"","B","C"}
        assert sum(ph["end"]-ph["begin"] for ph in r["phases"] if ph["cue"]=="C")==1120
        assert sum(ph["end"]-ph["begin"] for ph in r["phases"] if ph["cue"]=="B")==1600
        for f,digest in r["artifacts"].items(): assert sha256(p/f)==digest
        q=np.load(p/"release.npy",mmap_mode="r");w=np.load(p/"weights.npy",mmap_mode="r")
        original_q=np.load(original/"release.npy",mmap_mode="r")[-1]
        mask=masks["MBON04_B"]
        assert np.array_equal(q[0,~mask],original_q[~mask])
        expected=np.load((removed_donor if name=="removed" else original)/"release.npy",mmap_mode="r")[-1]
        assert np.array_equal(q[0,mask],expected[mask])
        assert np.array_equal(w[0],np.load(original/"weights.npy",mmap_mode="r")[-1])
        assert np.any(w[-1]!=w[0])
        r["release"]={k:dict(before=float(q[0,v].mean()),retained=float(q[-1,v].mean()),max_absolute_change=float(np.max(np.abs(q[-1,v]-q[0,v])))) for k,v in masks.items()}
        with np.load(p/"retention-body.npz") as z: initial=float(z["energy"][:2].sum())
        for label,probe in r["probes"].items():
            assert sha256(p/label/"trace.npz")==probe["trace_sha256"]
            assert probe["input_checkpoint_sha256"]==sha256(p/"retention.paula")
            with np.load(p/label/"trace.npz") as z: probe["stored_energy_gain_J"]=float(z["body"][-1,7:9].sum()-initial)
        r["record"]=str(p.relative_to(base));r["intervention"]=receipt
        result["courses"][name]=r
    for label,receiver,donor in (("removed","intact","removed"),("transferred","removed","intact")):
        state=base/f"memory-reuse-{receiver}-20260910/course";source=base/f"memory-reuse-{donor}-20260910/course"
        for mode in ("dry","food"):
            p=base/f"memory-reuse-C-{label}-{mode}-20260910";r=read(p/"summary.json")
            assert sha256(p/"trace.npz")==r["trace_sha256"]
            assert sha256(state/"retention.paula")==r["input_checkpoint_sha256"]
            assert sha256(source/"retention.paula")==r["donor_checkpoint_sha256"]
            with np.load(p/"trace.npz") as a,np.load(state/f"C-{mode}/trace.npz") as b:
                r["same_state_body_exact"]=bool(np.array_equal(a["body"],b["body"]))
                r["same_state_somatic_output_exact"]=bool(np.array_equal(a["soma"][:,:,1],b["soma"][:,:,1]))
                cmask=masks["MBON07_C"]|masks["MBON04_C"]
                assert np.array_equal(a["release"][0,~cmask],b["release"][0,~cmask])
                assert np.array_equal(a["weights"][0],b["weights"][0])
            r["record"]=p.name;result["substitutions"][label+"_"+mode]=r
    with np.load(original/"retention-body.npz") as z: initial=float(z["energy"][:2].sum())
    baseline_paths={"A":original/"probe-A.npz",
        "B":base/"memory-coverage-eight-feeding-sham-20260910/trace.npz",
        "C":base/"memory-coverage-eight-specificity-20260910/sham/trace.npz"}
    result["before_hop_food_probes"]={}
    for cue,path in baseline_paths.items():
        with np.load(path) as z:
            result["before_hop_food_probes"][cue]=dict(well_j=float(z["body"][:,5].sum()),
                stored_energy_gain_J=float(z["body"][-1,7:9].sum()-initial),trace_sha256=sha256(path))
    result["screen_source_hashes"]={name:sha256(base/f"memory-coverage-eight-diagnostics-20260910/{name}/trace.npz") for name in ("sham","removed")}
    result["additional_C_depression_from_acquired_B"]=result["courses"]["removed"]["release"]["MBON04_C"]["retained"]-result["courses"]["intact"]["release"]["MBON04_C"]["retained"]
    result["functional_C_memory_effect_detected"]=any(not r["same_state_body_exact"] for r in result["substitutions"].values())
    result["conclusion"]=("Acquired B adds a small C-terminal depression increment, but both branches acquire the same C output and food response. C-terminal substitutions leave each receiver's physical response unchanged. The additional acquired teaching signal is not functionally expressed at this hop. A feeding is retained. B action and intake increase in the intact branch, while its stored-energy gain declines from the pre-hop probe. Stop after this fixed additional hop; the accepted A-to-B result is unchanged."
        if not result["functional_C_memory_effect_detected"] else "A same-state C-terminal effect is present; inspect the recorded action and energy differences before interpreting functional reuse.")
    result["limits"]="This comparison isolates the contribution of acquired B memory, not C-only learning versus baseline B cue feedback. C's pre-existing terminal state was preserved. No gain, timing, exposure or further-hop sweep was run."
    output.mkdir(parents=True,exist_ok=True)
    (output/"memory-reuse.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("base",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();run(a.base,a.output)
