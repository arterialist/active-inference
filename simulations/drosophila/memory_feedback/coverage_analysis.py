"""Audit the from-birth coverage efficacy hypothesis and direct acquisition."""
import argparse
import io
import json
from pathlib import Path

import numpy as np

from ..connectome import sha256
from ...active_inference.core.runtime_checkpoint import load_checkpoint, _serializer
from neuron.neuron import setup_neuron_logger


def read(path): return json.loads(Path(path).read_text())


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
    return dict(reference=reference.name,candidate=candidate.name,changed_pairs=len(changes),
        all_other_runtime_state_exact=exact,body_exact=True,
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
    mask=np.isin(selected[:,1],[mapping[r] for r in m["codes"]["B"]])&np.isin(selected[:,3],[mapping[r] for r in m["roles"]["MBON04"]])
    for name in ("intact","cut","unpaired","displaced"):
        # The initial directory labelled 'cut' was launched without --cut.
        # Its metadata correctly says cut=False; preserve it as a replay only.
        label="blocked" if name=="cut" else name
        p=base/f"memory-coverage-second-{label}-20260910"
        if not (p/"summary.json").exists(): continue
        r=read(p/"summary.json")
        if "artifacts" not in r: continue
        assert r["cut"] == (name=="cut") and r["displaced"] == (name=="displaced")
        expected_parent=base/f"memory-coverage-{'unpaired' if name=='unpaired' else 'paired'}-20260910"
        assert r["input_checkpoint_sha256"]==sha256(expected_parent/"retention.paula")
        assert r["nutrient_j"]==0 and r["phases"][-1]["end"]==3720 and r["branch_rng_preserved"]
        for f,digest in r["artifacts"].items(): assert sha256(p/f)==digest
        q=np.load(p/"release.npy",mmap_mode="r")
        r["B_release"]=dict(before=float(q[0,mask].mean()),retained=float(q[-1,mask].mean()))
        r["record"]=p.name;result["continuation"][name]=r
    if all(n in result["continuation"] for n in ("intact","cut")):
        with np.load(base/"memory-coverage-second-intact-20260910/probe-B.npz") as a,np.load(base/"memory-coverage-second-blocked-20260910/probe-B.npz") as b:
            result["B_body_exact_against_feedback_block"]=bool(np.array_equal(a["body"],b["body"]))
    result["launch_correction"]="memory-coverage-second-cut-20260910 was accidentally launched without --cut and is excluded from control inference. Its summary preserves cut=False. The actual blocked comparison is memory-coverage-second-blocked-20260910."
    result["limit"]="Only completed records are audited. Any B action difference against feedback block still requires teacher-memory and within-state B-terminal controls before a second-order learning claim. Incomplete courses carry no outcome claim."
    output.mkdir(parents=True,exist_ok=True)
    (output/"coverage-acquisition.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("base",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();run(a.base,a.output)
