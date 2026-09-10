"""Audit the from-birth coverage efficacy hypothesis and direct acquisition."""
import argparse
import io
import json
from pathlib import Path

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
    result=dict(source_sha256=sha256(Path(__file__)),birth={},courses={})
    for name in ("paired","unpaired"):
        p=base/f"memory-coverage-{name}-20260910"
        result["birth"][name]=audit_birth(base/f"memory-interface-{name}-20260910",p)
        phases=read(p/"summary.json")
        completed=bool(phases and phases[-1]["name"]=="final_recovery")
        result["courses"][name]=dict(record=p.name,completed=completed)
        if completed:
            result["courses"][name].update(retained_A=next(ph for ph in phases if ph["name"]=="retained_A"),
                artifacts={f:sha256(p/f) for f in ("manifest.json","summary.json","retention.paula","retention-body.npz","soma.npy","release.npy","weights.npy","body.npy")})
    result["limit"]="Birth-state specificity and direct acquisition only. Second-order behavior and teacher-memory dependence require subsequent nutrient-free continuations and causal expression controls. Incomplete courses carry no outcome claim."
    output.mkdir(parents=True,exist_ok=True)
    (output/"coverage-acquisition.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("base",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();run(a.base,a.output)
