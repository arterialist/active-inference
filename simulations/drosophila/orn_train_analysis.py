"""Equation audits, exact target replay and population recruitment per epoch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .connectome import Subgraph
from .orn_train import pulse_course, CONDITIONS
from .pn_current import dl5_cut
from .pn_current_steps import prepare
from .prisco import digest, dump_new


def audit_sources(before, current, command, soma, intrinsic, *, first_tick, last_fire, previous_S):
    """Independent source membrane/reset check, including missing pulse responses."""
    np.testing.assert_array_equal(before, np.vstack([previous_S,soma[:-1,:,0]]))
    predicted = before+(1/intrinsic[:,:,3])*(-before+(current+command))
    clipped=np.clip(predicted,-1000.,1000.)
    if not np.isin(soma[:,:,1],[0.,1.]).all():
        raise ValueError("Source spike flags must be binary")
    spikes=np.zeros(predicted.shape)
    for t in range(len(before)):
        tick=first_tick+t
        if np.any((soma[t,:,1]>0)&(tick-last_fire<3)):
            raise AssertionError("Source spike violates refractory period")
        threshold=np.where(tick-last_fire<=3,intrinsic[t,:,2],intrinsic[t,:,1])
        threshold=np.where(np.abs(clipped[t])<.005,intrinsic[t,:,1],threshold)
        fired=(clipped[t]>=threshold)&(tick-last_fire>=3)
        spikes[t]=fired
        last_fire=np.where(soma[t,:,1]>0,tick,last_fire)
    # Ordinary PAULA can promote/demote NumPy scalar types after a reset or
    # weight update. The recording stores values, not those runtime dtype tags.
    # Bound one native float32 update, and disclose threshold ambiguity rather
    # than pretend that a double-precision reconstruction is bit-exact.
    tolerance=8*np.finfo(np.float32).eps*np.maximum(1.,np.abs(clipped))
    mismatch=(soma[:,:,1]>0)!=(spikes>0)
    margin=np.minimum(np.abs(clipped-intrinsic[:,:,1]),np.abs(clipped-intrinsic[:,:,2]))
    if np.any(mismatch&(margin>tolerance)):
        raise AssertionError("Source spike contradicts current and thresholds")
    expected=np.where(soma[:,:,1]>0,0.,clipped)
    if np.any(np.abs(soma[:,:,0]-expected)>tolerance):
        raise AssertionError("Source membrane equation exceeded float32 error bound")
    return last_fire,soma[-1,:,0].copy(),int(np.sum(clipped!=predicted)),float(np.max(np.abs(soma[:,:,0]-expected))),int(np.sum(mismatch))


def audit_resources(available,used,native,effective,source_spikes,offsets,depletion,previous):
    for tick in range(len(available)):
        previous += (1-previous)*-np.expm1(-1/893.)
        expected=np.zeros_like(previous)
        for i,fired in enumerate(source_spikes[tick]>0):
            if fired:
                sec=slice(offsets[i],offsets[i+1])
                expected[sec]=previous[sec]
                previous[sec]*=1-depletion
        np.testing.assert_allclose(used[tick],expected,rtol=0,atol=1e-13)
        np.testing.assert_allclose(available[tick],previous,rtol=0,atol=1e-13)
        # Native terminal coefficients may be NumPy float32 after a return
        # update; multiplying by the Python resource scalar can round twice.
        np.testing.assert_allclose(effective[tick],native[tick]*used[tick],rtol=2*np.finfo(np.float32).eps,atol=0)
    return previous


def analyze_condition(graph,intrinsic,tail,directory):
    manifest=json.loads((directory/"analysis.json").read_text())
    if "electrical_junction" in manifest["assumptions"]:
        raise ValueError("Electrical recordings require the extension-aware electrical_analysis audit")
    if not manifest["complete_course"] or manifest["ticks"]!=4200:
        raise ValueError("A partial smoke run cannot pass a complete-course analysis")
    if manifest["assumptions"]["release_depression"]["recovery_ticks"]!=893.:
        raise ValueError("This audit implements the declared 893-tick recovery control")
    for p,h in manifest["source_hashes"].items():
        if digest(Path(p))!=h:raise ValueError(f"Changed source: {p}")
    if digest(directory/"structure.npz")!=manifest["structure_sha256"]:
        raise ValueError("Changed structure")
    with np.load(directory/"structure.npz") as f:
        roots=f["roots"].tolist();orns=f["orn_roots"].tolist();lns=f["ln_roots"].tolist();offsets=f["release_offsets"]
    index={r:i for i,r in enumerate(roots)}
    source_rows=[index[r] for r in orns]
    pnrow=index[manifest["root"]]
    command,epochs=pulse_course(len(orns))
    if epochs!=manifest["epochs"]:raise ValueError("Changed epoch definition")
    root,cut=dl5_cut(graph)
    prep,target=prepare(cut,intrinsic,tail)
    np.testing.assert_array_equal([p.u_i.info for p in target.postsynaptic_points.values()],manifest["initial_post_weight"])
    spike_counts=np.zeros((len(epochs),len(roots)),dtype=np.int64)
    pulse_misses=np.zeros((len(epochs),len(orns)),dtype=np.int64)
    extra_spikes=np.zeros_like(pulse_misses)
    pn_current_sum=np.zeros((len(epochs),target.params.num_inputs))
    pn_state=np.zeros((4200,7));orn_spikes=np.zeros((4200,len(orns)),dtype=bool)
    resource_min=np.ones(len(epochs));apl_peak=np.zeros(len(epochs));native_min=np.zeros(len(epochs))
    release_counts=np.zeros((len(epochs),3),dtype=np.int64)
    last_fire=np.full(len(orns),-np.inf);previous_S=np.zeros(len(orns));resources=np.ones(int(offsets[-1]))
    clipping=cursor=ambiguous=0;max_source_residual=0.;first_input=np.full(target.params.num_inputs,-1,dtype=int)
    final_post=final_terminal=None
    for record in manifest["chunks"]:
        start,stop=record["start"],record["stop"]
        if start!=cursor or stop<=start or stop>4200:raise ValueError("Missing, duplicated or extra ticks")
        path=directory/record["file"]
        if digest(path)!=record["sha256"]:raise ValueError("Changed trace")
        with np.load(path) as a:
            data={k:a[k] for k in a.files}
        if any(len(x)!=stop-start or not np.isfinite(x).all() for x in data.values()):
            raise ValueError("Invalid record shape or value")
        source_soma=data["soma"][:,source_rows]
        last_fire,previous_S,clips,residual,uncertain=audit_sources(data["orn_before_S"],data["orn_native_current"],command[start:stop],
            source_soma,data["orn_intrinsic"],first_tick=start,last_fire=last_fire,previous_S=previous_S)
        clipping+=clips
        ambiguous+=uncertain;max_source_residual=max(max_source_residual,residual)
        resources=audit_resources(data["available"],data["release_used"],data["release_native"],data["release_effective"],
            source_soma[:,:,1],offsets,manifest["assumptions"]["release_depression"]["depletion_fraction"],resources)
        for tick in range(start,stop):
            i=tick-start
            target.input_buffer[:]=data["pn_inputs"][i]
            target.tick({},tick)
            np.testing.assert_array_equal([target.S,target.O,target.F_avg],data["soma"][i,pnrow])
            np.testing.assert_array_equal([target.t_ref,target.r,target.b,target.total_current],data["pn_intrinsic"][i])
            np.testing.assert_array_equal(target.last_port_current,data["pn_current"][i])
            np.testing.assert_array_equal([p.u_i.info for p in target.postsynaptic_points.values()],data["pn_post_weight"][i])
        for port in range(target.params.num_inputs):
            if first_input[port]<0:
                seen=np.flatnonzero(data["pn_inputs"][:,port,0]>0)
                if len(seen):first_input[port]=start+seen[0]
        for j,e in enumerate(epochs):
            lo,hi=max(start,e["start"]),min(stop,e["stop"])
            if lo>=hi:continue
            sec=slice(lo-start,hi-start)
            spikes=source_soma[sec,:,1]>0;pulses=command[lo:hi]>0
            spike_counts[j]+=(data["soma"][sec,:,1]>0).sum(0)
            pulse_misses[j]+=(pulses&~spikes).sum(0)
            extra_spikes[j]+=(~pulses&spikes).sum(0)
            pn_current_sum[j]+=data["pn_current"][sec].sum(0)
            resource_min[j]=min(resource_min[j],float(data["available"][sec].min()))
            apl_peak[j]=max(apl_peak[j],float(data["apl"][sec,2].max()))
            native_min[j]=min(native_min[j],float(data["orn_native_current"][sec].min()))
            release_counts[j]+=data["ln_events"][sec].sum(axis=(0,1))
        pn_state[start:stop]=np.c_[data["soma"][:,pnrow],data["pn_intrinsic"]]
        orn_spikes[start:stop]=source_soma[:,:,1]>0
        final_post=data["pn_post_weight"][-1];final_terminal=data["pn_terminal_weight"][-1]
        cursor=stop
    if cursor!=4200:raise ValueError("Incomplete record")
    if first_input[-1]>=0:raise ValueError("PN was artificially driven at its experimental port")
    catalog=[e for e in sorted(cut.edges,key=lambda e:int(e[8])) if str(e[1])==root]
    reports=[]
    for j,e in enumerate(epochs):
        populations={}
        for cls in ("olfactory","ALLN","ALPN","Kenyon_Cell"):
            ids=[i for i,r in enumerate(roots) if graph.nodes[r]["annotation"]["cell_class"]==cls]
            populations[cls]={"spikes":int(spike_counts[j,ids].sum()),"cells_fired":int(np.count_nonzero(spike_counts[j,ids]))}
        by_source={}
        for port,edge in enumerate(catalog):
            info=graph.nodes[str(edge[0])]["annotation"]
            label="APL" if info["hemibrain_type"]=="APL" else info["cell_class"]
            key=f"{label}/source-sign{int(edge[5]):+d}"
            by_source[key]=by_source.get(key,0.)+float(pn_current_sum[j,port])
        reports.append({**e,"populations":populations,"target_spikes":int(spike_counts[j,pnrow]),
            "requested_pulses":int(np.count_nonzero(command[e["start"]:e["stop"]])),
            "pulses_without_same_tick_spike":int(pulse_misses[j].sum()),"nonpulse_spikes":int(extra_spikes[j].sum()),
            "minimum_available_fraction":float(resource_min[j]),"maximum_apl_local_release":float(apl_peak[j]),
            "minimum_orn_native_current":float(native_min[j]),"pn_current_sums_by_source_class":by_source,
            "ln_event_counts":release_counts[j].tolist()})
    ports=[{"port":p,"source_row":int(e[8]),"source_root":str(e[0]),
            "type":graph.nodes[str(e[0])]["annotation"]["hemibrain_type"],
            "first_receptor_tick":int(first_input[p]),"current_sums_per_epoch":pn_current_sum[:,p].tolist()}
           for p,e in enumerate(catalog) if first_input[p]>=0]
    result={"condition":manifest["condition"],"epochs":reports,"active_target_inputs":ports,
        "source_membrane_ticks_audited":4200*len(orns),"source_membrane_clipping_events":clipping,
        "source_max_membrane_equation_residual":max_source_residual,"source_precision_ambiguous_spikes":ambiguous,
        "target_ticks_replayed_exactly":4200,"release_resource_values_audited":4200*int(offsets[-1]),
        "target_post_weights_changed":int(np.count_nonzero(final_post!=manifest["initial_post_weight"])),
        "target_terminal_weights_changed":int(np.count_nonzero(final_terminal!=manifest["initial_terminal_weight"])),
        "record_manifest_sha256":digest(directory/"analysis.json"),
        "audit_limits":"Source voltage and release equations are checked within explicit float32 rounding bounds against recorded currents and native amplitudes, not full ORN synaptic-history replay. Runtime scalar dtypes and target terminal return events are not replayed."}
    return result,pn_state,orn_spikes


def analyze(graph_path,intrinsic_path,tail_path,directory,output):
    if output.exists():raise FileExistsError(output)
    graph=Subgraph.load(graph_path)
    intrinsic=json.loads(intrinsic_path.read_text());tail=json.loads(tail_path.read_text())["fits"]["2"]["all_cells"]
    results,states,spikes={},{},{}
    for name in CONDITIONS:
        results[name],states[name],spikes[name]=analyze_condition(graph,intrinsic,tail,directory/name)
    with np.load(directory/CONDITIONS[0]/"structure.npz") as reference:
        for name in CONDITIONS[1:]:
            with np.load(directory/name/"structure.npz") as other:
                for key in reference.files:np.testing.assert_array_equal(reference[key],other[key])
    comparisons=[]
    for left,right in (("no_depression","depressing"),("depressing","depressing_ln_block")):
        def first(mask):
            x=np.flatnonzero(mask);return int(x[0]) if len(x) else None
        comparisons.append({"left":left,"right":right,
            "first_target_S_difference":first(states[left][:,0]!=states[right][:,0]),
            "first_target_spike_difference":first(states[left][:,1]!=states[right][:,1]),
            "first_orn_spike_difference":first(np.any(spikes[left]!=spikes[right],axis=1)),
            "orn_spike_disagreement_entries":int(np.count_nonzero(spikes[left]!=spikes[right]))})
    report={"schema":1,"conditions":results,"comparisons":comparisons,"all_bindings_exact":True,
        "analysis_source_sha256":digest(Path(__file__)),"claim":"Audited neural train/recruitment comparison, not odor or behavior acceptance"}
    dump_new(output,report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("graph","intrinsic","tail","directory","output"):p.add_argument(name,type=Path)
    a=p.parse_args();analyze(a.graph,a.intrinsic,a.tail,a.directory,a.output)


if __name__=="__main__":main()
