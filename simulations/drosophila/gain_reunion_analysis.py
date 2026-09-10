"""Locate preservation or loss of a conditional sensory function after reunion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .connectome import Subgraph
from .electrical_analysis import prefix
from .gain_reunion import reunion_cut,pathway_bindings,regulator_afferent_bindings
from .ln_gain import PN,LN
from .ln_gain_analysis import read_record
from .ln_input_replay import cut_cells
from .pn_current_steps import prepare
from .prisco import digest,dump_new
from .antennal_identity import identity_audit


def polarity_roots(graph,meta):
    return set(identity_audit(graph)["gaba_positive_candidates"]) if meta.get("polarity_control") is not None else set()


def audit_polarity(graph,meta,structure):
    """Reconstruct the declared intervention from anatomy, not saved weights."""
    from types import SimpleNamespace
    spec=meta.get("polarity_control")
    if spec is None:return {"enabled":False,"pairs":0}
    candidates=polarity_roots(graph,meta)
    if set(spec["source_roots"])!=candidates or len(spec["source_roots"])!=len(candidates):
        raise ValueError("Different polarity candidate identities")
    ids={graph.nodes[r]["global_index"] for r in candidates}
    bindings=structure["edge_bindings"]
    feedback={int(e[0]) for e in pathway_bindings(SimpleNamespace(edge_bindings=bindings),graph,"positive_LN_or_PN_to_LN")}
    edges={int(e[8]):e for e in graph.edges}
    expected=[]
    for row,pre,terminal,post,port in bindings:
        if int(pre) not in ids:continue
        edge=edges[int(row)]
        if int(pre)!=graph.nodes[str(edge[0])]["global_index"] or int(post)!=graph.nodes[str(edge[1])]["global_index"]:
            raise ValueError("Binding disagrees with anatomical identity")
        weight=edge[6]*meta["assumptions"]["parameters"]["weight_per_count"]
        if int(row) in feedback:weight*=meta["feedback_initialization"]["scale"]
        expected.append([int(row),int(pre),int(terminal),int(post),int(port),weight,-weight])
    np.testing.assert_array_equal(np.array(spec["changed_receiving_coefficients"]).reshape(-1,7),np.array(expected).reshape(-1,7))
    return {"enabled":True,"source_roots":sorted(candidates),"pairs":len(expected),
            "claim":"Fast-current polarity sensitivity only; no target-specific receptor physiology established."}


def apply_pn_polarity(pn,graph,meta):
    """Restore receptor signs in a PN-only replay whose sources are absent."""
    candidates=polarity_roots(graph,meta)
    rows=sorted(graph.edges[graph.edges[:,1]==int(PN)],key=lambda e:int(e[8]))
    for port,e in enumerate(rows):
        if str(e[0]) in candidates:
            point=pn.postsynaptic_points[port]
            if point.u_i.info<=0:raise ValueError("Replay polarity control applied twice or to a negative receptor")
            point.u_i.info=-point.u_i.info


def first_difference(a,b):
    if a.shape!=b.shape:raise ValueError("Cannot compare different trace shapes")
    rows=np.flatnonzero(np.any(a!=b,axis=tuple(range(1,a.ndim)))) if a.ndim>1 else np.flatnonzero(a!=b)
    return int(rows[0]) if len(rows) else None


def course_bounds(meta):
    """Reject partial or ambiguous courses instead of silently slicing 2200 ticks."""
    if len(meta["epochs"])!=1:raise ValueError("Need one complete stimulus/recovery course")
    e=meta["epochs"][0]
    start,stop,end=e["start"],e["stop"],e["recovery_stop"]
    if any(type(t) is not int for t in (start,stop,end,meta["ticks"])) or not 0<start<stop<end or end!=meta["ticks"]:
        raise ValueError("Invalid complete-course bounds")
    return start,stop,end


def port_groups(graph,negative_roots=()):
    rows=sorted(graph.edges[graph.edges[:,1]==int(PN)],key=lambda e:int(e[8]))
    names=[]
    for e in rows:
        a=graph.nodes[str(e[0])]["annotation"]
        cls=a["cell_class"]
        if a["hemibrain_type"]=="ORN_DL5":name="ORN_DL5"
        elif a["hemibrain_type"]=="APL":name="APL"
        elif cls in ("ALLN","ALPN"):name=cls+("_positive" if e[5]>0 and str(e[0]) not in negative_roots else "_negative")
        elif cls=="Kenyon_Cell":name="KC"
        else:name="other"
        names.append(name)
    names.append("experimental")
    return {name:np.array([n==name for n in names]) for name in sorted(set(names))}


def conditional_replay(graph,intrinsic,tail,inputs,keep,meta=None):
    _,pn=prepare(cut_cells(graph,(PN,)),intrinsic,tail)
    apply_pn_polarity(pn,graph,meta or {})
    trace=np.zeros((len(inputs),7))
    for tick,values in enumerate(inputs):
        pn.input_buffer[:]=values
        pn.input_buffer[~keep]=0
        pn.tick({},tick)
        trace[tick]=[pn.S,pn.O,pn.F_avg,pn.t_ref,pn.r,pn.b,pn.total_current]
    return trace


def epochs(data,graph,roots,intervals=((0,200),(200,400),(400,1200),(1200,2200))):
    results=[];pn_col=roots.index(PN);ln_col=roots.index(LN)
    for lo,hi in intervals:
        if hi<=lo:continue
        if lo<0 or hi>len(data["soma"]):raise ValueError("Requested interval outside recording")
        groups={}
        for cls in ("olfactory","ALLN","ALPN","Kenyon_Cell"):
            cols=[i for i,r in enumerate(roots) if graph.nodes[r]["annotation"]["cell_class"]==cls]
            counts=data["soma"][lo:hi,cols,1].sum(0)
            groups[cls]={"cells_in_preparation":len(cols),"spikes":int(counts.sum()),"cells_fired":int(np.count_nonzero(counts)),
                "max_cell_spikes":int(counts.max(initial=0))}
        results.append({"start":lo,"stop":hi,"populations":groups,
            "PN_spikes":int(data["soma"][lo:hi,pn_col,1].sum()),
            "LN_spikes":int(data["soma"][lo:hi,ln_col,1].sum()),
            "mean_ORN_gate_fraction":float(data["inhibition"][lo:hi,:,1].mean()),
            "max_APL_release":float(data["apl"][lo:hi,2].max()) if "apl" in data else None})
    return results


def audit_command_shift(early,late):
    """Equal injected dose is not assumed to mean equal neural intervention."""
    if early.shape!=late.shape or early.ndim!=2 or early.shape[1]<2:
        raise ValueError("Different command shapes")
    if not np.isfinite(early).all() or not np.isfinite(late).all():
        raise ValueError("Nonfinite command")
    np.testing.assert_array_equal(early[:,:-1],late[:,:-1])
    a=np.flatnonzero(early[:,-1]);b=np.flatnonzero(late[:,-1])
    if not len(a) or len(a)!=len(b):raise ValueError("Different or empty pulse trains")
    shifts=b-a
    if shifts[0]<=0 or np.any(shifts!=shifts[0]):raise ValueError("Not a pure delayed pulse train")
    np.testing.assert_array_equal(early[a,-1],late[b,-1])
    return {"pulses":len(a),"shift_ticks":int(shifts[0]),
            "injected_current_sum":float(early[:,-1].sum())}


def regulator_pulses(data,structure,cooldown):
    """Annotate delivered current against the actual preceding spike history."""
    soma=data["soma"][:,structure["roots"].tolist().index(LN)]
    src=structure["source_roots"].tolist().index(LN)
    spikes=np.flatnonzero(soma[:,1]);rows=[]
    for t in np.flatnonzero(data["command"][:,src]):
        previous=spikes[spikes<t]
        elapsed=int(t-previous[-1]) if len(previous) else None
        rows.append({"tick":int(t),"ticks_since_previous_spike":elapsed,
            "cooldown_prevents_spike":elapsed is not None and elapsed<cooldown,
            "fired":bool(soma[t,1]),"post_tick_S":float(soma[t,0]),
            "native_current":float(data["source_current"][t,src,0]),
            "total_current":float(data["source_current"][t,src,1])})
    return rows


def audit_recorded_sources(meta,data,structure):
    """Current delivery and somatic decisions, not a full synaptic replay."""
    from .orn_train_analysis import audit_sources
    if meta["assumptions"]["parameters"]["cooldown_ticks"]!=3:
        raise ValueError("Source equation audit requires this course's three-tick cooldown")
    roots=structure["roots"].tolist()
    cols=[roots.index(r) for r in structure["source_roots"]]
    source_soma=data["soma"][:,cols]
    before=np.vstack([np.zeros((1,len(cols))),source_soma[:-1,:,0]])
    np.testing.assert_allclose(data["source_current"][:,:,1],data["source_current"][:,:,0]+data["command"],rtol=1e-7,atol=1e-7)
    _,_,clips,residual,ambiguous=audit_sources(before,data["source_current"][:,:,1],np.zeros_like(data["command"]),
        source_soma,data["source_intrinsic"],first_tick=0,last_fire=np.full(len(cols),-np.inf),previous_S=np.zeros(len(cols)))
    return {"cell_ticks":len(data["soma"])*len(cols),"clips":clips,
            "maximum_residual":residual,"threshold_ambiguous_ticks":ambiguous}


def schedule_followup(graph_path,early,late,sustained,unassisted):
    """Matched schedule intervention, retaining source and output distinctions.

    The first pair shifts a fixed dose. The other two courses establish what
    changed after early assistance ended and whether delayed assistance had any
    outgoing effect. No activity threshold is used to label a run successful.
    """
    paths=dict(early=early,late=late,sustained=sustained,unassisted=unassisted)
    records={name:prefix(path,2200,current_source=False) for name,path in paths.items()}
    base_meta,base_data,base_structure=records["early"]
    graph=reunion_cut(Subgraph.load(graph_path),base_meta["scope"])
    roots=base_structure["roots"].tolist()
    if set(roots)!=set(graph.selected):raise ValueError("Different graph selection")
    result={"command_shift":audit_command_shift(base_data["command"],records["late"][1]["command"]),
            "conditions":{},"comparisons":{},"manifest_sha256":{name:digest(path/"analysis.json") for name,path in paths.items()}}
    for name,(meta,data,structure) in records.items():
        if meta["ticks"]!=2200 or meta.get("sensory_precondition") is not None:
            raise ValueError("This legacy timing comparison requires an unpreconditioned 2200-tick course")
        if meta.get("polarity_control")!=base_meta.get("polarity_control"):
            raise ValueError("Concurrent polarity change")
        for key in ("scope","gain","direct","seed","lesion","epochs","assumptions","anatomy","pathway_intervention"):
            if meta[key]!=base_meta[key]:raise ValueError(f"Concurrent {key} change in {name}")
        if meta["feedback_initialization"]["scale"]!=base_meta["feedback_initialization"]["scale"]:
            raise ValueError("Concurrent feedback strength change")
        if meta.get("regulator_afferent_initialization",{}).get("scale",1.)!=base_meta.get("regulator_afferent_initialization",{}).get("scale",1.):
            raise ValueError("Concurrent regulator afferent strength change")
        for path,h in base_meta["source_hashes"].items():
            if Path(path).name!="gain_reunion.py" and meta["source_hashes"].get(path)!=h:
                raise ValueError(f"Concurrent model/calibration change: {path}")
        for key,value in base_structure.items():np.testing.assert_array_equal(value,structure[key])
        for path,spec in ((paths[name],meta),(early,base_meta)):
            if digest(path/"feedback-initial.npz")!=spec["feedback_initialization"]["initial_sha256"]:
                raise ValueError("Changed initial weights")
        with np.load(early/"feedback-initial.npz") as a,np.load(paths[name]/"feedback-initial.npz") as b:
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(data["command"][:,:-1],base_data["command"][:,:-1])
        result["conditions"][name]={"intervals":epochs(data,graph,roots,
            ((200,300),(300,400),(400,600),(600,700),(700,1200),(1200,2200),(2000,2200))),
            "regulator_pulses":regulator_pulses(data,structure,3),
            "source_equation_audit":audit_recorded_sources(meta,data,structure)}
    for a,b in (("early","late"),("early","sustained"),("late","unassisted")):
        x,y=records[a][1],records[b][1]
        changed=first_difference(x["command"],y["command"])
        if changed is None:raise ValueError("No scheduled intervention")
        for key in x:np.testing.assert_array_equal(x[key][:changed],y[key][:changed])
        result["comparisons"][a+"_vs_"+b]={"first_command_difference":changed,
            "first_state_difference":first_difference(x["soma"],y["soma"]),
            "first_spike_difference":first_difference(x["soma"][:,:,1],y["soma"][:,:,1]),
            "first_regulator_spike_difference":first_difference(x["soma"][:,roots.index(LN),1],y["soma"][:,roots.index(LN),1]),
            "first_gate_difference":first_difference(x["inhibition"],y["inhibition"]),
            "first_target_input_difference":first_difference(x["pn_inputs"],y["pn_inputs"]),
            "first_regulator_release_event_count_difference":first_difference(x["ln_events"],y["ln_events"])}
    result["limits"]=["Single sensory channel, timing seed and 2,200-tick course; not physiological acceptance.",
        "Source equation audit uses recorded current and thresholds, not a replay of every source receptor or return event.",
        "No shifted-window result establishes an autonomous neural recruitment mechanism.",
        "Source hash comparison allows the recording driver to gain windowing; shared dynamics and initial bindings/weights are checked."]
    return result


def audit_pathway_events(events,start):
    if events.ndim!=2 or events.shape[1]!=3 or not 0<=start<len(events):
        raise ValueError("Invalid intervention event recording")
    if not np.isfinite(events).all() or np.any(events<0) or np.any(events!=np.floor(events)):
        raise ValueError("Invalid intervention event counts")
    expected=events[:,0].copy();expected[start:]=0
    np.testing.assert_array_equal(events[:,1],expected)
    withheld=int(events[start:,0].sum())
    if not withheld:raise ValueError("Intervention did not withhold any event")
    return {"withheld_forward_events":withheld,"return_events_preserved":int(events[start:,2].sum())}


def compare_pathway(reference,directory,graph,meta,data,structure):
    """An intact shared past plus a closed-loop, time-specific intervention."""
    from types import SimpleNamespace
    old_meta,old,old_structure=prefix(reference,2200,current_source=False)
    if old_meta["ticks"]!=2200 or meta["ticks"]!=2200:
        raise ValueError("This pathway comparison requires complete 2200-tick courses")
    if old_meta.get("sensory_precondition")!=meta.get("sensory_precondition"):
        raise ValueError("Concurrent sensory precondition change")
    if old_meta.get("polarity_control")!=meta.get("polarity_control"):
        raise ValueError("Concurrent polarity change")
    if old_meta.get("feedback_initialization",{}).get("scale",1.)!=meta.get("feedback_initialization",{}).get("scale",1.):
        raise ValueError("Concurrent feedback-strength change")
    if old_meta.get("regulator_afferent_initialization",{}).get("scale",1.)!=meta.get("regulator_afferent_initialization",{}).get("scale",1.):
        raise ValueError("Concurrent regulator-afferent strength change")
    if old_meta.get("pathway_intervention",{}).get("pathway","none")!="none":
        raise ValueError("Reference is already intervened")
    for key in ("scope","gain","direct","lateral","seed","lesion","epochs"):
        if old_meta[key]!=meta[key]:raise ValueError(f"Unmatched boundary: {key}")
    if old_meta.get("lateral_window")!=meta.get("lateral_window"):
        raise ValueError("Concurrent lateral command-window change")
    # The recording driver gained the intervention; shared cellular dynamics,
    # assembly, graph and calibration sources must not change across courses.
    for path,h in old_meta["source_hashes"].items():
        if Path(path).name!="gain_reunion.py" and meta["source_hashes"].get(path)!=h:
            raise ValueError(f"Concurrent source change: {path}")
    for key,value in old_structure.items():np.testing.assert_array_equal(value,structure[key])
    spec=meta["pathway_intervention"];start=spec["start"]
    if spec["pathway"]=="none":raise ValueError("No intervention")
    if digest(directory/"intervention.npz")!=spec["bindings_sha256"]:raise ValueError("Changed intervention bindings")
    with np.load(directory/"intervention.npz") as saved:
        selected=saved["bindings"]
    np.testing.assert_array_equal(selected,pathway_bindings(SimpleNamespace(edge_bindings=structure["edge_bindings"]),graph,spec["pathway"]))
    if len(selected)!=spec["pairs"] or len(set(selected[:,1]))!=spec["sources"]:
        raise ValueError("Different declared intervention size")
    parity=0
    first_withheld=np.flatnonzero(data["pathway_events"][:,0]!=data["pathway_events"][:,1])
    shared_stop=int(first_withheld[0]) if len(first_withheld) else start
    for key,values in old.items():
        if key=="pathway_events":continue
        np.testing.assert_array_equal(values[:shared_stop],data[key][:shared_stop]);parity+=values[:shared_stop].size
    np.testing.assert_array_equal(old["command"],data["command"])
    event_audit=audit_pathway_events(data["pathway_events"],start)
    roots=structure["roots"].tolist()
    result={"reference_manifest_sha256":digest(reference/"analysis.json"),"intervention":spec,
        "pre_intervention_exact_values":parity,"events":event_audit,
        "shared_recorded_past_stop":shared_stop,
        "first_state_difference":first_difference(old["soma"],data["soma"]),
        "first_spike_difference":first_difference(old["soma"][:,:,1],data["soma"][:,:,1]),
        "cells_with_changed_spikes":int(np.any(old["soma"][:,:,1]!=data["soma"][:,:,1],axis=0).sum()),
        "epochs":{}}
    for name,record in (("intact",old),("intervened",data)):
        rows=[]
        for lo,hi in ((200,start),(start,1200),(1200,2200)):
            if hi<=lo:continue
            populations={}
            for cls in ("olfactory","ALLN","ALPN","Kenyon_Cell"):
                cols=[i for i,r in enumerate(roots) if graph.nodes[r]["annotation"]["cell_class"]==cls]
                counts=record["soma"][lo:hi,cols,1].sum(0)
                populations[cls]={"spikes":int(counts.sum()),"cells_fired":int(np.count_nonzero(counts))}
            rows.append({"start":lo,"stop":hi,"populations":populations,
                "PN_spikes":int(record["soma"][lo:hi,roots.index(PN),1].sum()),
                "regulating_LN_spikes":int(record["soma"][lo:hi,roots.index(LN),1].sum()),
                "mean_gate_fraction":float(record["inhibition"][lo:hi,:,1].mean())})
        result["epochs"][name]=rows
    return result


def audit_feedback_gain(directory,graph,meta,structure,*,afferent=False):
    from types import SimpleNamespace
    spec=meta["regulator_afferent_initialization" if afferent else "feedback_initialization"]
    stem="regulator-afferent" if afferent else "feedback"
    for filename,key in ((stem+"-initial.npz","initial_sha256"),(stem+"-final.npz","final_sha256")):
        if digest(directory/filename)!=spec[key]:raise ValueError("Changed feedback weight record")
    with np.load(directory/(stem+"-initial.npz")) as saved:
        bindings=saved["bindings"];before=saved["reference_weights"];initial=saved["initial_weights"]
    with np.load(directory/(stem+"-final.npz")) as saved:final=saved["weights"]
    prep=SimpleNamespace(edge_bindings=structure["edge_bindings"])
    expected=regulator_afferent_bindings(prep,graph) if afferent else pathway_bindings(prep,graph,"positive_LN_or_PN_to_LN")
    np.testing.assert_array_equal(bindings,expected)
    rows={int(e[8]):e for e in graph.edges}
    np.testing.assert_array_equal(before,[rows[int(e[0])][6]*meta["assumptions"]["parameters"]["weight_per_count"] for e in bindings])
    if not np.isfinite(spec["scale"]) or spec["scale"]<=0 or (not afferent and spec["scale"]>1) or len(bindings)!=spec["pairs"]:
        raise ValueError("Invalid receiving gain declaration")
    candidates=polarity_roots(graph,meta)
    signs=np.array([-1 if str(rows[int(e[0])][0]) in candidates else 1 for e in bindings])
    np.testing.assert_array_equal(initial,before*spec["scale"]*signs)
    if final.shape!=initial.shape or not np.isfinite(final).all():raise ValueError("Invalid final feedback weights")
    changed=int(np.count_nonzero(final!=initial))
    if changed!=spec["changed_by_learning"]:raise ValueError("Incorrect learning count")
    return {"scale":spec["scale"],"pairs":len(bindings),"nonzero_initial_weights":int(np.count_nonzero(initial)),
        "changed_by_learning":changed,"maximum_absolute_learning_change":float(np.max(np.abs(final-initial),initial=0.)),
        "limit":"Initial and final selected receiving weights only; this does not replay every intervening learning update or verify untouched weights."}


def compare_polarity(graph_path,reference,directory):
    """Matched closed-loop sign sensitivity, including the earliest divergence."""
    a,x,s=prefix(reference,2200,current_source=False)
    b,y,t=prefix(directory,2200,current_source=False)
    if a["ticks"]!=2200 or b["ticks"]!=2200:
        raise ValueError("This polarity comparison requires complete 2200-tick courses")
    if a.get("sensory_precondition")!=b.get("sensory_precondition"):
        raise ValueError("Concurrent sensory precondition change")
    if a.get("polarity_control") is not None or b.get("polarity_control") is None:
        raise ValueError("Need an original-sign reference and a polarity intervention")
    for key in ("scope","gain","direct","lateral","seed","lesion","epochs","lateral_window","assumptions","anatomy"):
        if a.get(key)!=b.get(key):raise ValueError(f"Concurrent polarity-test change: {key}")
    for key in ("pathway","start","pairs","sources","bindings_sha256"):
        if a["pathway_intervention"][key]!=b["pathway_intervention"][key]:
            raise ValueError("Concurrent pathway intervention change")
    for key in ("feedback_initialization","regulator_afferent_initialization"):
        if a[key]["scale"]!=b[key]["scale"]:raise ValueError("Concurrent receiving gain change")
    for path,h in a["source_hashes"].items():
        if Path(path).name!="gain_reunion.py" and b["source_hashes"].get(path)!=h:
            raise ValueError(f"Concurrent model/calibration change: {path}")
    graph=reunion_cut(Subgraph.load(graph_path),a["scope"])
    audit=audit_polarity(graph,b,t)
    if not audit["pairs"]:raise ValueError("Empty polarity intervention")
    for key in s:
        if key!="pn_initial_weights":np.testing.assert_array_equal(s[key],t[key])
    expected=s["pn_initial_weights"].copy()
    pn_id=graph.nodes[PN]["global_index"]
    for _,_,_,post,port,_,_ in b["polarity_control"]["changed_receiving_coefficients"]:
        if post==pn_id:expected[port]*=-1
    np.testing.assert_array_equal(expected,t["pn_initial_weights"])
    for path,meta,structure in ((reference,a,s),(directory,b,t)):
        audit_feedback_gain(path,graph,meta,structure)
        audit_feedback_gain(path,graph,meta,structure,afferent=True)
    np.testing.assert_array_equal(x["command"],y["command"])
    state=first_difference(x["soma"],y["soma"])
    spike=first_difference(x["soma"][:,:,1],y["soma"][:,:,1])
    roots=s["roots"].tolist()
    def changed_cells(tick,spikes_only):
        if tick is None:return []
        left,right=x["soma"][tick],y["soma"][tick]
        changed=(left[:,1]!=right[:,1]) if spikes_only else np.any(left!=right,axis=1)
        return [{"root":roots[i],"type":graph.nodes[roots[i]]["annotation"]["hemibrain_type"],
                 "original":left[i].tolist(),"control":right[i].tolist()} for i in np.flatnonzero(changed)]
    courses={}
    for name,data in (("original",x),("control",y)):
        last={}
        for cls in ("olfactory","ALLN","ALPN"):
            cols=[i for i,r in enumerate(roots) if graph.nodes[r]["annotation"]["cell_class"]==cls]
            ticks=np.flatnonzero(np.any(data["soma"][:,cols,1],axis=1))
            last[cls]=int(ticks[-1]) if len(ticks) else None
        courses[name]={"epochs":epochs(data,graph,roots,((200,1200),(1200,2200),(2000,2200))),"last_spike":last}
    return {"polarity_audit":audit,"courses":courses,"first_state_difference":state,"first_spike_difference":spike,
        "first_state_cells":changed_cells(state,False),"first_spike_cells":changed_cells(spike,True),
        "first_target_PN_spike_difference":first_difference(x["soma"][:,roots.index(PN),1],y["soma"][:,roots.index(PN),1]),
        "first_regulator_spike_difference":first_difference(x["soma"][:,roots.index(LN),1],y["soma"][:,roots.index(LN),1]),
        "first_gate_difference":first_difference(x["inhibition"],y["inhibition"]),
        "source_hashes":{str(p.resolve()):digest(p) for p in (Path(__file__),reference/"analysis.json",directory/"analysis.json")},
        "limits":["All candidate signs change together; no individual candidate is established necessary or sufficient.",
                  "A finite recovery window cannot prove permanent stability or a dynamical attractor."]}


def compare_sensory_history(graph_path,low,preconditioned,low_reference,high_reference):
    """Different sensory pasts, identical future commands and fixed circuitry."""
    from .gain_reunion import reunion_commands,window_lateral_command
    paths={"constant_low":low,"high_then_low":preconditioned}
    records={}
    for name,path in paths.items():
        bounds=course_bounds(json.loads((path/"analysis.json").read_text()))
        records[name]=prefix(path,bounds[-1],current_source=False)
    a,x,s=records["constant_low"];b,y,t=records["high_then_low"]
    start,stop,end=course_bounds(a)
    pre=b.get("sensory_precondition")
    if a.get("sensory_precondition") is not None or pre is None or pre["rate"]<=a["direct"]:
        raise ValueError("Need constant low and higher-input preconditioned courses")
    switch=pre["stop"]
    if not start<switch<stop or pre["start"]!=start:raise ValueError("Invalid history boundary")
    fixed=("scope","gain","lateral","seed","lesion","lateral_window","assumptions","anatomy","polarity_control")
    def same_circuit(left,right,left_structure,right_structure,*,old_driver=False):
        for key in fixed:
            if left.get(key)!=right.get(key):raise ValueError(f"Concurrent circuit/boundary change: {key}")
        for key in ("feedback_initialization","regulator_afferent_initialization"):
            if left[key]["scale"]!=right[key]["scale"]:raise ValueError("Concurrent receiving gain change")
        if left["pathway_intervention"]["pathway"]!="none" or right["pathway_intervention"]["pathway"]!="none":
            raise ValueError("History comparison does not permit a concurrent pathway block")
        for path,h in left["source_hashes"].items():
            if old_driver and Path(path).name=="gain_reunion.py":continue
            if right["source_hashes"].get(path)!=h:raise ValueError(f"Concurrent source change: {path}")
        for key,values in left_structure.items():np.testing.assert_array_equal(values,right_structure[key])
    same_circuit(a,b,s,t)
    if a["direct"]!=b["direct"] or course_bounds(b)!=(start,stop,end):raise ValueError("Different future protocol")
    graph=reunion_cut(Subgraph.load(graph_path),a["scope"]);roots=s["roots"].tolist()
    if set(roots)!=set(graph.selected):raise ValueError("Different anatomical selection")
    for name,(meta,data,structure) in records.items():
        spec=meta.get("sensory_precondition")
        expected,_=reunion_commands(len(structure["source_roots"])-1,meta["direct"],meta["lateral"],meta["seed"],
            stimulus_duration=stop-start,sensory_precondition=None if spec is None else (spec["rate"],spec["stop"]))
        expected=window_lateral_command(expected,meta.get("lateral_window"))
        np.testing.assert_array_equal(data["command"],expected)
        audit_polarity(graph,meta,structure)
        audit_feedback_gain(paths[name],graph,meta,structure)
        audit_feedback_gain(paths[name],graph,meta,structure,afferent=True)
    np.testing.assert_array_equal(x["command"][switch:],y["command"][switch:])
    first_command=first_difference(x["command"],y["command"])
    if first_command is None or first_command>=switch:raise ValueError("No distinct input history")
    for key in x:np.testing.assert_array_equal(x[key][:first_command],y[key][:first_command])
    prefix_checks={}
    for name,path,expected_rate,until in (("constant_low",low_reference,a["direct"],None),
                                          ("high_then_low",high_reference,pre["rate"],switch)):
        rm=json.loads((path/"analysis.json").read_text())
        rs,re,rt=course_bounds(rm)
        until=re if until is None else until
        if not rs<until<=re or rm.get("sensory_precondition") is not None or rm["direct"]!=expected_rate:
            raise ValueError("Invalid stationary reference prefix")
        rm,rd,rstruct=prefix(path,until,current_source=False)
        meta,data,structure=records[name]
        same_circuit(rm,meta,rstruct,structure,old_driver=True)
        values=0
        for key in rd:
            np.testing.assert_array_equal(rd[key],data[key][:until]);values+=rd[key].size
        prefix_checks[name]={"exclusive_stop":until,"exact_recorded_values":values,"manifest_sha256":digest(path/"analysis.json")}
    results={}
    for name,(meta,data,structure) in records.items():
        last={}
        for cls in ("olfactory","ALLN","ALPN"):
            cols=[i for i,r in enumerate(roots) if graph.nodes[r]["annotation"]["cell_class"]==cls]
            ts=np.flatnonzero(np.any(data["soma"][:,cols,1],axis=1))
            last[cls]=int(ts[-1]) if len(ts) else None
        results[name]={"intervals":epochs(data,graph,roots,((start,switch),(switch,stop),(stop,end),(end-200,end))),
            "hundred_tick_intervals":epochs(data,graph,roots,tuple((lo,min(lo+100,end)) for lo in range(start,end,100))),
            "regulator_spike_ticks":np.flatnonzero(data["soma"][:,roots.index(LN),1]).tolist(),
            "last_spike":last,"source_equation_audit":audit_recorded_sources(meta,data,structure)}
    orn_cols=[roots.index(r) for r in s["source_roots"] if r!=LN]
    sensory_spike_difference=first_difference(x["soma"][switch:,orn_cols,1],y["soma"][switch:,orn_cols,1])
    return {"switch_tick":switch,"stimulus_stop":stop,"recording_stop":end,
        "first_future_sensory_spike_difference":None if sensory_spike_difference is None else switch+sensory_spike_difference,
        "identical_future_command_values":int(x["command"][switch:].size),"stationary_prefix_checks":prefix_checks,
        "conditions":results,"manifest_sha256":{name:digest(path/"analysis.json") for name,path in paths.items()},
        "analysis_sha256":digest(Path(__file__)),
        "limits":["Sensory history changes the complete adaptive state; this does not separate fast state from learned weights.",
                  "One timing seed and finite input/recovery windows do not establish an attractor or robustness distribution."]}


def analyze(graph_path,intrinsic_path,tail_path,isolated,directory,output,*,reference=None):
    if output.exists():raise FileExistsError(output)
    # Driver/observation code may evolve after a completed run. Historical
    # records retain their hashes; exact current PN replay below checks the
    # receiving dynamics instead of pretending an old driver is current.
    start,stop,ticks=course_bounds(json.loads((directory/"analysis.json").read_text()))
    meta,data,structure=prefix(directory,ticks,current_source=False)
    intervals=((0,start),(start,min(start+200,stop)),(min(start+200,stop),stop),(stop,ticks))
    graph=reunion_cut(Subgraph.load(graph_path),meta["scope"])
    if set(structure["roots"])!=set(graph.selected):raise ValueError("Different selected neurons")
    polarity_audit=audit_polarity(graph,meta,structure)
    roots=structure["roots"].tolist();pn_col=roots.index(PN)
    intrinsic=json.loads(intrinsic_path.read_text());tail=json.loads(tail_path.read_text())["fits"]["2"]["all_cells"]
    iso_meta=json.loads((isolated/"analysis.json").read_text())
    if iso_meta.get("polarity_control")!=meta.get("polarity_control"):
        # An isolated course may be reused only when neither sign hypothesis
        # changes any source in that cut. Never transfer this exception to reunion.
        isolated_graph=reunion_cut(Subgraph.load(graph_path),"isolated")
        if polarity_roots(isolated_graph,meta) or polarity_roots(isolated_graph,iso_meta):
            raise ValueError("Unmatched isolated polarity")
    if iso_meta.get("regulator_afferent_initialization",{}).get("scale",1.)!=meta.get("regulator_afferent_initialization",{}).get("scale",1.):
        raise ValueError("Unmatched isolated regulator afferent strength")
    if iso_meta.get("scope")=="isolated":
        iso_meta,iso,iso_structure=prefix(isolated,ticks,current_source=False)
        for key in ("seed","direct","lateral","lesion","gain","epochs","lateral_window","sensory_precondition","ticks"):
            if iso_meta.get(key)!=meta.get(key):raise ValueError(f"Unmatched isolated boundary: {key}")
        if iso_meta["feedback_initialization"]["scale"]!=meta["feedback_initialization"]["scale"]:
            raise ValueError("Unmatched isolated feedback strength")
        iso["roots"]=iso_structure["roots"]
    else:
        if ticks!=2200 or meta.get("sensory_precondition") is not None:
            raise ValueError("Changed stimulation requires a matching isolated course")
        if meta.get("lateral_window") is not None:raise ValueError("Windowed stimulation requires a matching isolated course")
        matches=[r for r in iso_meta["records"] if r["seed"]==meta["seed"] and r["direct_command_hz"]==meta["direct"]
            and r["lateral_command_hz"]==meta["lateral"] and r["lesion"]==meta["lesion"]
            and (r.get("inhibition_gain") or 0)==meta["gain"]]
        if len(matches)!=1:raise ValueError("Need one matched isolated course")
        iso=read_record(isolated,matches[0])
    iso_roots=iso["roots"].tolist()
    np.testing.assert_array_equal(data["command"],iso["command"][:ticks])
    np.testing.assert_array_equal(structure["source_roots"],iso["roots"][:-2].tolist()+[LN])
    # The current receptor and terminal IDs remain stable across cuts.
    groups=port_groups(graph,polarity_roots(graph,meta))
    _,pn=prepare(cut_cells(graph,(PN,)),intrinsic,tail)
    apply_pn_polarity(pn,graph,meta)
    np.testing.assert_array_equal([p.u_i.info for p in pn.postsynaptic_points.values()],structure["pn_initial_weights"])
    for t in range(ticks):
        pn.input_buffer[:]=data["pn_inputs"][t];pn.tick({},t)
        np.testing.assert_array_equal([pn.S,pn.O,pn.F_avg],data["soma"][t,pn_col])
        np.testing.assert_array_equal([pn.t_ref,pn.r,pn.b,pn.total_current],data["pn_intrinsic"][t])
        np.testing.assert_array_equal(pn.last_port_current,data["pn_current"][t])
        np.testing.assert_array_equal([p.u_i.info for p in pn.postsynaptic_points.values()],data["pn_weights"][t])
    gate=data["inhibition"];before=np.vstack([np.zeros((1,gate.shape[1])),gate[:-1,:,0]])
    np.testing.assert_allclose(gate[:,:,0],before*np.exp(-1/100.)+gate[:,:,2],atol=1e-14,rtol=2e-15)
    np.testing.assert_array_equal(gate[:,:,1],1/(1+meta["gain"]*gate[:,:,0]))
    np.testing.assert_array_equal(gate[:,:,4],gate[:,:,3]*gate[:,:,1])
    # One ordinary cleft tick and the native float32 receiving buffer. The
    # sensory cell with no target pair is excluded explicitly, not indexed -1.
    valid=structure["pn_orn_ports"]>=0
    np.testing.assert_array_equal(data["pn_inputs"][1:,structure["pn_orn_ports"][valid],0],
                                  gate[:-1,valid,4].astype(np.float32))
    masks={"all":np.ones(data["pn_inputs"].shape[1],dtype=bool),"ORN_only":groups["ORN_DL5"],
        "without_ORN":~groups["ORN_DL5"],
        "without_positive_ALLN":~groups.get("ALLN_positive",np.zeros(data["pn_inputs"].shape[1],dtype=bool))}
    replay={};replay_summary={}
    for name,mask in masks.items():
        trace=conditional_replay(graph,intrinsic,tail,data["pn_inputs"],mask,meta)
        if name=="all":np.testing.assert_array_equal(trace,np.c_[data["soma"][:,pn_col],data["pn_intrinsic"]])
        replay[name]=trace
        replay_summary[name]={"stimulus_spikes":int(trace[start:stop,1].sum()),
                              "recovery_spikes":int(trace[stop:ticks,1].sum())}
    charge=[]
    for lo,hi in intervals[1:]:
        if hi<=lo:continue
        charge.append({"start":lo,"stop":hi,"signed_current_sum":{name:float(data["pn_current"][lo:hi,mask].sum()) for name,mask in groups.items()}})
    i=iso_roots.index(PN)
    comparisons={"first_PN_state_difference":first_difference(data["soma"][:,pn_col],iso["soma"][:ticks,i,:3]),
        "first_PN_spike_difference":first_difference(data["soma"][:,pn_col,1],iso["soma"][:ticks,i,1]),
        "isolated_PN_stimulus_spikes":int(iso["soma"][start:stop,i,1].sum()),
        "isolated_PN_recovery_spikes":int(iso["soma"][stop:ticks,i,1].sum()),
        "reunited_PN_stimulus_spikes":int(data["soma"][start:stop,pn_col,1].sum()),
        "reunited_PN_recovery_spikes":int(data["soma"][stop:ticks,pn_col,1].sum())}
    for root,label in ((LN,"regulating_LN"),):
        comparisons[label+"_first_spike_difference"]=first_difference(data["soma"][:,roots.index(root),1],iso["soma"][:ticks,iso_roots.index(root),1])
    output.mkdir(parents=True)
    with (output/"conditional-PN.npz").open("xb") as f:np.savez_compressed(f,**replay)
    result={"schema":1,"scope":meta["scope"],"direct":meta["direct"],"gain":meta["gain"],
        "ticks":ticks,"stimulus_start":start,"stimulus_stop":stop,"sensory_precondition":meta.get("sensory_precondition"),
        "polarity_audit":polarity_audit,
        "source_group_polarity":"Initial effective receiving sign, including declared curated-GABA control; not transmitter ground truth.",
        "lateral_window":meta.get("lateral_window"),"lateral_command_pulses":int(np.count_nonzero(data["command"][:,-1])),
        "lateral_injected_current_sum":float(data["command"][:,-1].sum()),
        "epochs":epochs(data,graph,roots,intervals),"vs_isolated":comparisons,"PN_current_by_source":charge,
        "conditional_receiving_tests":replay_summary,"exact_PN_replay_ticks":ticks,
        "source_equation_audit":audit_recorded_sources(meta,data,structure),
        "gate_and_target_routing_ticks":ticks,"source_hashes":{str(p.resolve()):digest(p) for p in
            (Path(__file__),directory/"analysis.json",isolated/"analysis.json",graph_path/"manifest.json",intrinsic_path,tail_path)},
        "limits":["Conditional receiving tests retain intact source histories and are not closed-loop lesion predictions.",
            "Only a single sensory channel is tested; KC differences do not establish odor-identity discrimination.",
            "One timing realization and one full recovery window are not an acceptance distribution.",
            "Gate/routing and PN replay checks do not verify all unrecorded intracellular states or fit physiology."]}
    if reference is not None:
        result["closed_loop_pathway_test"]=compare_pathway(reference,directory,graph,meta,data,structure)
    if meta.get("pathway_intervention",{}).get("pathway","none")!="none":
        result["limits"].append("Isolated comparison is the intact functional reference; the pathway intervention's causal comparison uses a matched connected reference.")
    if "feedback_initialization" in meta:
        result["feedback_gain_audit"]=audit_feedback_gain(directory,graph,meta,structure)
    if "regulator_afferent_initialization" in meta:
        result["regulator_afferent_gain_audit"]=audit_feedback_gain(directory,graph,meta,structure,afferent=True)
    dump_new(output/"analysis.json",result)
    print(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("graph","intrinsic","tail","isolated","directory","output"):p.add_argument(name,type=Path)
    p.add_argument("--reference",type=Path)
    a=p.parse_args();analyze(a.graph,a.intrinsic,a.tail,a.isolated,a.directory,a.output,reference=a.reference)


if __name__=="__main__":main()
