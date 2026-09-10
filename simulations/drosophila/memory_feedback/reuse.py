"""One additional C-before-B hop from the accepted retained A/B preparation.

Keep anatomy, equations and all birth parameters fixed. Compare acquired B
terminal memory with only that acquired component replaced by the matched
feedback-blocked values. Do not reset C or extend the chain beyond this hop.
"""
import argparse
import io
import json
from pathlib import Path
import shutil

import numpy as np

from . import feeding, interface_controls as controls, student_interface
from .second_order import ProjectionGate
from ..connectome import sha256
from ...active_inference.core.runtime_checkpoint import load_checkpoint,save_checkpoint,_serializer
from neuron.neuron import setup_neuron_logger


def read(path): return json.loads(Path(path).read_text())


def prepare(base,output,removed):
    parent=base/"memory-coverage-paired-20260910"
    state=base/"memory-coverage-eight-intact-20260910"
    donor=base/"memory-coverage-eight-blocked-20260910"
    m,ids,roots,selected=controls.pair(parent,parent)
    a=load_checkpoint(state/"retention.paula",trusted=True)
    b=load_checkpoint(donor/"retention.paula",trusted=True)
    _,pickler=_serializer()
    def blob():
        f=io.BytesIO();pickler(f,protocol=5).dump(dict(network=a.network,python_rng=a.python_rng,numpy_rng=a.numpy_rng));return f.getvalue()
    before=blob();mask=controls.mask_for(m,ids,roots,selected,"B",("MBON04",))
    changes,originals=controls.swap(a,b if removed else a,selected,mask)
    output.mkdir(parents=True)
    save_checkpoint(a,output/"retention.paula",sources=[__file__])
    for point,value in originals: point.u_o.info=value
    assert blob()==before
    receipt=dict(removed=removed,intervention_cue="B",targets=["MBON04"],changes=changes,
        original_state_sha256=sha256(state/"retention.paula"),donor_state_sha256=sha256(donor/"retention.paula"),
        all_other_runtime_state_exact=True,source_sha256=sha256(Path(__file__)))
    m["reuse_start"]=receipt
    (output/"manifest.json").write_text(json.dumps(m,indent=2)+"\n")
    for f in ("identities.npz",): shutil.copyfile(parent/f,output/f)
    shutil.copyfile(state/"retention-body.npz",output/"retention-body.npz")
    (output/"intervention.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return output


def protocol():
    return [(f"pair{i}_{name}",cue,ticks) for i in range(8)
            for name,cue,ticks in (("C","C",140),("gap","",20),("B","B",200),("recovery","",1000))]+[("retention","",1000)]


def run(base,output,*,removed=False):
    setup_neuron_logger("CRITICAL")
    base,output=Path(base),Path(output)
    if output.exists(): raise FileExistsError(output)
    if shutil.disk_usage(base).free<4*1024**3+400*1024**2: raise RuntimeError("Shared-volume reserve")
    parent=prepare(base,output/"receiver",removed)
    course=output/"course";course.mkdir()
    m,ids,roots,selected=controls.pair(parent,parent)
    mapping=dict(zip(roots.tolist(),ids.tolist()));rows=dict(zip(roots.tolist(),range(len(roots))))
    branch=load_checkpoint(parent/"retention.paula",trusted=True);net=branch.network
    organism=feeding.FeedingBody();organism.restore(parent/"retention-body.npz")
    cells=[net.network.neurons[int(i)] for i in ids]
    phases=protocol();total=sum(t for _,_,t in phases)
    def array(name,shape): return np.lib.format.open_memmap(course/(name+".npy"),mode="w+",dtype=np.float64,shape=shape)
    soma=array("soma",(total,len(ids),len(feeding.SOMA_FIELDS)));body=array("body",(total,len(feeding.BODY_FIELDS)))
    mods=array("modulation",(total,len(ids),2));q=array("release",(total+1,len(selected)));w=array("weights",q.shape)
    def coefficients():
        return ([net.network.neurons[int(e[1])].presynaptic_points[int(e[2])].u_o.info for e in selected],
                [net.network.neurons[int(e[3])].postsynaptic_points[int(e[4])].u_i.info for e in selected])
    q[0],w[0]=coefficients()
    gate=ProjectionGate(net,{mapping[r] for r in m["roles"]["SMP108"]},
                        {mapping[r] for role in ("PAM07","PAM08") for r in m["roles"][role]},False)
    reports=[];t=0
    for name,cue,ticks in phases:
        assert cue in ("B","C","")
        begin=t
        for _ in range(ticks):
            gate.before_step()
            body[t]=student_interface.step(net,mapping,m["roles"],m["codes"],cue,organism,factor=m["factor"],advance=branch.step)
            soma[t]=[[getattr(c,f) for f in feeding.SOMA_FIELDS] for c in cells];mods[t]=[c.M_vector for c in cells]
            q[t+1],w[t+1]=coefficients()
            if not all(np.isfinite(a).all() for a in (body[t],soma[t],mods[t],q[t+1],w[t+1])): raise FloatingPointError("Nonfinite hop")
            t+=1
        r=dict(name=name,cue=cue,begin=begin,end=t,feedback_events=gate.observed,
               spikes={role:int(soma[begin:t,[rows[root] for root in rr],1].sum()) for role,rr in m["roles"].items()})
        reports.append(r);print(json.dumps(r),flush=True)
    assert not np.any(body[:,4:6])
    save_checkpoint(branch,course/"retention.paula",sources=[__file__]);organism.save(course/"retention-body.npz")
    for a in (soma,body,mods,q,w): a.flush()
    shutil.copyfile(parent/"identities.npz",course/"identities.npz")
    result=dict(removed_B_memory=removed,phases=reports,nutrient_j=0.,input_checkpoint_sha256=sha256(parent/"retention.paula"),
        source_sha256=sha256(Path(__file__)),branch_rng_preserved=True,minimum_eta_post=min(c.params.eta_post for c in cells),
        minimum_eta_retro=min(c.params.eta_retro for c in cells),probes={})
    # Write ancestry before independent expression probes. A is never supplied
    # during acquisition; these separate probes test retention/discrimination.
    (course/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    for cue in ("A","B","C"):
        for well in (False,True):
            label=cue+("-food" if well else "-dry")
            result["probes"][label]=controls.probe(parent,parent,course/label,state=course,donor_state=course,
                                                  cue=cue,well=well,student_only=False)
    result["artifacts"]={f.name:sha256(f) for f in course.iterdir() if f.is_file() and f.name!="summary.json"}
    result["limit"]="One fixed additional hop, eight C-before-B pairings without A or nutrients. C's existing state is preserved. Independent post-course A/B/C probes do not affect acquisition or each other. No gain/exposure sweep or subsequent hop is authorized by this experiment."
    (course/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("base",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--removed",action="store_true");a=p.parse_args();run(a.base,a.output,removed=a.removed)
