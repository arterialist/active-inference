"""Audit the from-birth coverage efficacy hypothesis and direct acquisition."""
import argparse
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..connectome import Subgraph, sha256
from ...active_inference.core.runtime_checkpoint import load_checkpoint, _serializer
from neuron.neuron import setup_neuron_logger


def read(path): return json.loads(Path(path).read_text())


def audit_student_probe(base,path,probe,mask):
    state=base/probe["state"];donor=base/probe["donor_state"]
    assert sha256(state/"retention.paula")==probe["input_checkpoint_sha256"]
    assert sha256(donor/"retention.paula")==probe["donor_checkpoint_sha256"]
    assert sha256(path/"trace.npz")==probe["trace_sha256"]
    q=np.load(state/"release.npy",mmap_mode="r")[-1]
    w=np.load(state/"weights.npy",mmap_mode="r")[-1]
    with np.load(path/"trace.npz") as z:
        assert np.array_equal(z["release"][0,~mask],q[~mask])
        assert np.array_equal(z["weights"][0],w)
        probe["initial_unselected_releases_exact"]=True
        probe["initial_receiving_weights_exact"]=True


def figure(base,output,result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    names=[n for n in ("intact","cut","teacher_removed") if n in result["continuation"]]
    labels={"intact":"Intact","cut":"Feedback\nblocked","teacher_removed":"A memory\nremoved"}
    values=[100*(1-result["continuation"][n]["B_release"]["retained"]) for n in names]
    bars=axes[0].bar([labels[n] for n in names],values,color=["#176B87","#777777","#B35932"][:len(names)])
    axes[0].bar_label(bars,labels=[f"{v:.2f}" if abs(v)>=.005 else f"{v:.4f}" for v in values],padding=3)
    axes[0].set(title="B depression after two pairings",ylabel="Mean release reduction from 1, %",ylim=(-.2,max([.1]+values)*1.25))
    for n,label,style in (("sham","Acquired, two pairings","-"),("removed","B memory replaced",":"),("projection","Imposed fourfold projection","--")):
        p=base/"memory-coverage-second-diagnostics-20260910"/n/"trace.npz"
        if p.exists():
            with np.load(p) as z: axes[1].plot(np.arange(200)*.004,z["body"][:,0],style,label=label)
    axes[1].set(title="Two-pairing memory has no action effect",xlabel="Dry B probe, seconds",ylabel="Hinge angle, radians")
    axes[1].legend(frameon=False,fontsize=7)
    if all("eight_"+n in result["continuation"] for n in ("intact","cut")):
        for n,label in (("intact","Intact"),("cut","Feedback blocked")):
            p=base/result["continuation"]["eight_"+n]["record"]/"probe-B.npz"
            with np.load(p) as z: axes[2].plot(np.arange(200)*.004,z["body"][:,0],label=label)
        axes[2].set(title="Actual eight-pairing comparison",xlabel="Dry B probe, seconds",ylabel="Hinge angle, radians")
        axes[2].legend(frameon=False,fontsize=7)
    else:
        axes[2].axis("off")
        axes[2].text(0,1,"Eight-pairing courses are pending.\n\nThe projection is an imposed diagnostic.\nIt is not evidence of learned B action.",va="top",linespacing=1.8)
    fig.savefig(output/"coverage-acquisition.png",dpi=170);plt.close(fig)


def audit_birth(reference, candidate):
    reference, candidate = Path(reference), Path(candidate)
    original, changed = [load_checkpoint(p/"birth.paula",trusted=True) for p in (reference,candidate)]
    a,b = [read(p/"manifest.json") for p in (reference,candidate)]
    changes=b["assumptions"]["coverage_efficacy"]["changed_pairs"]
    for r in changes:
        ref=original.network.network.neurons[r["target_id"]].postsynaptic_points[r["synapse_id"]]
        point=changed.network.network.neurons[r["target_id"]].postsynaptic_points[r["synapse_id"]]
        assert point.u_i.info == r["after"] and ref.u_i.info == r["before"]
        point.u_i.info=ref.u_i.info
    _,pickler=_serializer()
    def blob(branch):
        buf=io.BytesIO();pickler(buf,protocol=5).dump(dict(network=branch.network,python_rng=branch.python_rng,numpy_rng=branch.numpy_rng))
        return buf.getvalue()
    exact=blob(original)==blob(changed)
    assert exact
    ba=dict(b["assumptions"]);ba.pop("coverage_efficacy")
    assert ba==a["assumptions"]
    for key in ("graph_sha256","codes","roles","factor","motor_gain","well_angle","energy_parameters","seed"):
        assert a[key]==b[key]
    assert sha256(reference/"birth-body.npz")==sha256(candidate/"birth-body.npz")
    graph=Subgraph.load(Path(b["graph"]))
    net=changed.network.network
    assert Counter((a,c) for a,t,c,s in net.connections)==Counter((int(e[2]),int(e[3])) for e in graph.internal)
    assert all(net.neurons[c].synapse_sources[s]==(a,t) for a,t,c,s in net.connections)
    incoming=sum(len(n.postsynaptic_points) for n in net.neurons.values())
    outgoing=sum(len(n.presynaptic_points) for n in net.neurons.values())
    stats=graph.summary()
    assert incoming==len(graph.internal)+stats["incoming_boundary"]["directed_pairs"]+len(net.neurons)
    assert outgoing==len(graph.internal)+stats["outgoing_boundary"]["directed_pairs"]
    return dict(reference=reference.name,candidate=candidate.name,changed_pairs=len(changes),
        all_other_runtime_state_exact=exact,body_exact=True,
        anatomy=dict(internal_connections_exact=True,source_registration_exact=True,neurons=len(net.neurons),
                     directed_internal_pairs=len(net.connections),input_ports_including_boundary_and_experimental=incoming,
                     output_terminals_including_boundary=outgoing,graph=stats),
        reference_checkpoint_sha256=sha256(reference/"birth.paula"),candidate_checkpoint_sha256=sha256(candidate/"birth.paula"))


def run(base,output):
    setup_neuron_logger("CRITICAL")
    base,output=Path(base),Path(output)
    result=dict(source_sha256=sha256(Path(__file__)),birth={},courses={},expression={},continuation={})
    for name in ("paired","unpaired"):
        p=base/f"memory-coverage-{name}-20260910"
        result["birth"][name]=audit_birth(base/f"memory-interface-{name}-20260910",p)
        phases=read(p/"summary.json")
        completed=bool(phases and phases[-1]["name"]=="final_recovery")
        result["courses"][name]=dict(record=p.name,completed=completed)
        if completed:
            result["courses"][name].update(retained_A=next(ph for ph in phases if ph["name"]=="retained_A"),
                artifacts={f:sha256(p/f) for f in ("manifest.json","summary.json","retention.paula","retention-body.npz","soma.npy","release.npy","weights.npy","body.npy")})
    for name in ("sham","erased","transferred"):
        p=base/f"memory-coverage-expression-A-{name}-20260910"
        if not (p/"summary.json").exists(): continue
        r=read(p/"summary.json");assert sha256(p/"trace.npz")==r["trace_sha256"]
        with np.load(base/r["state"]/"retention-body.npz") as z: initial=float(z["energy"][:2].sum())
        with np.load(p/"trace.npz") as z: r["stored_energy_gain_J"]=float(z["body"][-1,7:9].sum()-initial)
        r["record"]=p.name;result["expression"][name]=r
    parent=base/"memory-coverage-paired-20260910";m=read(parent/"manifest.json")
    with np.load(parent/"identities.npz") as z:
        selected=z["selected"];mapping=dict(zip(z["roots"].tolist(),z["cells"].tolist()))
        rows=dict(zip(z["roots"].tolist(),range(len(z["roots"]))))
    mask=np.isin(selected[:,1],[mapping[r] for r in m["codes"]["B"]])&np.isin(selected[:,3],[mapping[r] for r in m["roles"]["MBON04"]])
    branches=[]
    teacher=base/"memory-coverage-teacher-removed-20260910"
    for exposure,pairings in (("second",2),("eight",8)):
        for mode in ("intact","cut","unpaired","displaced"):
            label="blocked" if mode=="cut" else mode
            name=mode if exposure=="second" else "eight_"+mode
            p=base/f"memory-coverage-{exposure}-{label}-20260910"
            expected_parent=base/f"memory-coverage-{'unpaired' if mode=='unpaired' else 'paired'}-20260910"
            branches.append((name,p,expected_parent,pairings,mode))
    branches.extend((("teacher_removed",teacher/"continuation",teacher/"receiver",2,"teacher_removed"),
        ("eight_teacher_removed",base/"memory-coverage-eight-teacher-removed-20260910",teacher/"receiver",8,"teacher_removed")))
    alpha=base/"memory-coverage-alpha-removed-20260910"
    branches.extend((("alpha_removed",alpha/"continuation",alpha/"receiver",2,"alpha_removed"),
        ("eight_alpha_removed",base/"memory-coverage-eight-alpha-removed-20260910",alpha/"receiver",8,"alpha_removed")))
    for name,p,expected_parent,pairings,mode in branches:
        if not (p/"summary.json").exists(): continue
        r=read(p/"summary.json")
        if "artifacts" not in r: continue
        assert r["cut"] == (mode=="cut") and r["displaced"] == (mode=="displaced")
        assert r["input_checkpoint_sha256"]==sha256(expected_parent/"retention.paula")
        assert r.get("pairings",2)==pairings
        assert r["nutrient_j"]==0 and r["phases"][-1]["end"]==1360*pairings+1000 and r["branch_rng_preserved"]
        assert r["minimum_eta_post"]>0 and r["minimum_eta_retro"]>0
        assert r["removed_events"]==(r["feedback_events"] if mode=="cut" else 0)
        for f,digest in r["artifacts"].items(): assert sha256(p/f)==digest
        q=np.load(p/"release.npy",mmap_mode="r")
        r["B_release"]=dict(before=float(q[0,mask].mean()),retained=float(q[-1,mask].mean()))
        ss=np.load(p/"soma.npy",mmap_mode="r")
        phase=next(ph for ph in r["phases"] if ph["name"]=="pair0_A")
        r["first_A_provider_spikes"]={root:int(ss[phase["begin"]:phase["end"],rows[root],1].sum())
            for role in ("PAM07","PAM08") for root in m["roles"][role]}
        r["record"]=str(p.relative_to(base));result["continuation"][name]=r
    for prefix,exposure in (("","second"),("eight_","eight")):
        if not all(prefix+n in result["continuation"] for n in ("intact","cut")): continue
        paths=[base/result["continuation"][prefix+n]["record"] for n in ("intact","cut")]
        q,c=[np.load(p/"release.npy",mmap_mode="r")[-1] for p in paths]
        result[prefix+"reached_B_terminals"]=int(np.sum(mask & (c-q>1e-6)))
        with np.load(paths[0]/"probe-B.npz") as a,np.load(paths[1]/"probe-B.npz") as b:
            result[prefix+"B_body_exact_against_feedback_block"]=bool(np.array_equal(a["body"],b["body"]))
    result["diagnostics"]={}
    for exposure in ("second","eight"):
        p=base/f"memory-coverage-{exposure}-diagnostics-20260910"
        if not (p/"summary.json").exists(): continue
        r=read(p/"summary.json")
        for name,probe in r["probes"].items(): audit_student_probe(base,p/name,probe,mask)
        r["record"]=p.name;result["diagnostics"][exposure]=r
    result["B_feeding"]={}
    for name in ("sham","removed","transferred","blocked-sham"):
        p=base/f"memory-coverage-eight-feeding-{name}-20260910"
        if not (p/"summary.json").exists(): continue
        r=read(p/"summary.json")
        assert r["cue"]=="B" and r["well"] and r["targets"]==["MBON04"]
        audit_student_probe(base,p,r,mask)
        with np.load(base/r["state"]/"retention-body.npz") as z: initial=float(z["energy"][:2].sum())
        with np.load(p/"trace.npz") as z: r["stored_energy_gain_J"]=float(z["body"][-1,7:9].sum()-initial)
        r["record"]=p.name;result["B_feeding"][name]=r
    for key,record in (("cue_specificity","memory-coverage-eight-specificity-20260910"),
                       ("teacher_chain","memory-coverage-eight-teacher-chain-20260910")):
        p=base/record
        if not (p/"summary.json").exists(): continue
        r=read(p/"summary.json")
        for name,probe in r["probes"].items():
            audit_student_probe(base,p/name,probe,mask)
            if probe["well"]:
                with np.load(base/probe["state"]/"retention-body.npz") as z: initial=float(z["energy"][:2].sum())
                with np.load(p/name/"trace.npz") as z: probe["stored_energy_gain_J"]=float(z["body"][-1,7:9].sum()-initial)
        r["record"]=record;result[key]=r
    if (teacher/"intervention.json").exists():
        r=read(teacher/"intervention.json")
        assert r["all_other_runtime_state_exact"]
        assert r["substituted_checkpoint_sha256"]==sha256(teacher/"receiver/retention.paula")
        assert r["continuation_summary_sha256"]==sha256(teacher/"continuation/summary.json")
        result["teacher_intervention"]=r
    if (alpha/"intervention.json").exists():
        r=read(alpha/"intervention.json")
        assert r["targets"]==["MBON07"] and r["all_other_runtime_state_exact"]
        assert r["substituted_checkpoint_sha256"]==sha256(alpha/"receiver/retention.paula")
        assert r["continuation_summary_sha256"]==sha256(alpha/"continuation/summary.json")
        result["alpha_intervention"]=r
    result["launch_correction"]="memory-coverage-second-cut-20260910 was accidentally launched without --cut and is excluded from control inference. Its summary preserves cut=False. The actual blocked comparison is memory-coverage-second-blocked-20260910."
    result["limit"]="Only completed records are audited. Any B action difference against feedback block still requires teacher-memory and within-state B-terminal controls before a second-order learning claim. Incomplete courses carry no outcome claim."
    output.mkdir(parents=True,exist_ok=True)
    (output/"coverage-acquisition.json").write_text(json.dumps(result,indent=2)+"\n")
    figure(base,output,result)
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("base",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();run(a.base,a.output)
