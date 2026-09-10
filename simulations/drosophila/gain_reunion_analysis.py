"""Locate preservation or loss of a conditional sensory function after reunion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .connectome import Subgraph
from .electrical_analysis import prefix
from .gain_reunion import reunion_cut,pathway_bindings
from .ln_gain import PN,LN
from .ln_gain_analysis import read_record
from .ln_input_replay import cut_cells
from .pn_current_steps import prepare
from .prisco import digest,dump_new


def first_difference(a,b):
    if a.shape!=b.shape:raise ValueError("Cannot compare different trace shapes")
    rows=np.flatnonzero(np.any(a!=b,axis=tuple(range(1,a.ndim)))) if a.ndim>1 else np.flatnonzero(a!=b)
    return int(rows[0]) if len(rows) else None


def port_groups(graph):
    rows=sorted(graph.edges[graph.edges[:,1]==int(PN)],key=lambda e:int(e[8]))
    names=[]
    for e in rows:
        a=graph.nodes[str(e[0])]["annotation"]
        cls=a["cell_class"]
        if a["hemibrain_type"]=="ORN_DL5":name="ORN_DL5"
        elif a["hemibrain_type"]=="APL":name="APL"
        elif cls in ("ALLN","ALPN"):name=cls+("_positive" if e[5]>0 else "_negative")
        elif cls=="Kenyon_Cell":name="KC"
        else:name="other"
        names.append(name)
    names.append("experimental")
    return {name:np.array([n==name for n in names]) for name in sorted(set(names))}


def conditional_replay(graph,intrinsic,tail,inputs,keep):
    _,pn=prepare(cut_cells(graph,(PN,)),intrinsic,tail)
    trace=np.zeros((len(inputs),7))
    for tick,values in enumerate(inputs):
        pn.input_buffer[:]=values
        pn.input_buffer[~keep]=0
        pn.tick({},tick)
        trace[tick]=[pn.S,pn.O,pn.F_avg,pn.t_ref,pn.r,pn.b,pn.total_current]
    return trace


def epochs(data,graph,roots):
    results=[];pn_col=roots.index(PN);ln_col=roots.index(LN)
    for lo,hi in ((0,200),(200,400),(400,1200),(1200,2200)):
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
    if old_meta.get("pathway_intervention",{}).get("pathway","none")!="none":
        raise ValueError("Reference is already intervened")
    for key in ("scope","gain","direct","lateral","seed","lesion","epochs"):
        if old_meta[key]!=meta[key]:raise ValueError(f"Unmatched boundary: {key}")
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
    for key,values in old.items():
        if key=="pathway_events":continue
        np.testing.assert_array_equal(values[:start],data[key][:start]);parity+=values[:start].size
    np.testing.assert_array_equal(old["command"],data["command"])
    event_audit=audit_pathway_events(data["pathway_events"],start)
    roots=structure["roots"].tolist()
    result={"reference_manifest_sha256":digest(reference/"analysis.json"),"intervention":spec,
        "pre_intervention_exact_values":parity,"events":event_audit,
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


def analyze(graph_path,intrinsic_path,tail_path,isolated,directory,output,*,reference=None):
    if output.exists():raise FileExistsError(output)
    # Driver/observation code may evolve after a completed run. Historical
    # records retain their hashes; exact current PN replay below checks the
    # receiving dynamics instead of pretending an old driver is current.
    meta,data,structure=prefix(directory,2200,current_source=False)
    graph=reunion_cut(Subgraph.load(graph_path),meta["scope"])
    if set(structure["roots"])!=set(graph.selected):raise ValueError("Different selected neurons")
    roots=structure["roots"].tolist();pn_col=roots.index(PN)
    intrinsic=json.loads(intrinsic_path.read_text());tail=json.loads(tail_path.read_text())["fits"]["2"]["all_cells"]
    iso_meta=json.loads((isolated/"analysis.json").read_text())
    matches=[r for r in iso_meta["records"] if r["seed"]==meta["seed"] and r["direct_command_hz"]==meta["direct"]
        and r["lateral_command_hz"]==meta["lateral"] and r["lesion"]==meta["lesion"]
        and (r.get("inhibition_gain") or 0)==meta["gain"]]
    if len(matches)!=1:raise ValueError("Need one matched isolated course")
    iso=read_record(isolated,matches[0]);iso_roots=iso["roots"].tolist()
    np.testing.assert_array_equal(data["command"],iso["command"][:2200])
    np.testing.assert_array_equal(structure["source_roots"],iso["roots"][:-2].tolist()+[LN])
    # The current receptor and terminal IDs remain stable across cuts.
    groups=port_groups(graph)
    _,pn=prepare(cut_cells(graph,(PN,)),intrinsic,tail)
    np.testing.assert_array_equal([p.u_i.info for p in pn.postsynaptic_points.values()],structure["pn_initial_weights"])
    for t in range(2200):
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
        trace=conditional_replay(graph,intrinsic,tail,data["pn_inputs"],mask)
        if name=="all":np.testing.assert_array_equal(trace,np.c_[data["soma"][:,pn_col],data["pn_intrinsic"]])
        replay[name]=trace
        replay_summary[name]={"stimulus_spikes":int(trace[200:1200,1].sum()),
                              "recovery_spikes":int(trace[1200:2200,1].sum())}
    charge=[]
    for lo,hi in ((200,400),(400,1200),(1200,2200)):
        charge.append({"start":lo,"stop":hi,"signed_current_sum":{name:float(data["pn_current"][lo:hi,mask].sum()) for name,mask in groups.items()}})
    i=iso_roots.index(PN)
    comparisons={"first_PN_state_difference":first_difference(data["soma"][:,pn_col],iso["soma"][:2200,i,:3]),
        "first_PN_spike_difference":first_difference(data["soma"][:,pn_col,1],iso["soma"][:2200,i,1]),
        "isolated_PN_stimulus_spikes":int(iso["soma"][200:1200,i,1].sum()),
        "isolated_PN_recovery_spikes":int(iso["soma"][1200:2200,i,1].sum()),
        "reunited_PN_stimulus_spikes":int(data["soma"][200:1200,pn_col,1].sum()),
        "reunited_PN_recovery_spikes":int(data["soma"][1200:2200,pn_col,1].sum())}
    for root,label in ((LN,"regulating_LN"),):
        comparisons[label+"_first_spike_difference"]=first_difference(data["soma"][:,roots.index(root),1],iso["soma"][:2200,iso_roots.index(root),1])
    output.mkdir(parents=True)
    with (output/"conditional-PN.npz").open("xb") as f:np.savez_compressed(f,**replay)
    result={"schema":1,"scope":meta["scope"],"direct":meta["direct"],"gain":meta["gain"],
        "epochs":epochs(data,graph,roots),"vs_isolated":comparisons,"PN_current_by_source":charge,
        "conditional_receiving_tests":replay_summary,"exact_PN_replay_ticks":2200,
        "gate_and_target_routing_ticks":2200,"source_hashes":{str(p.resolve()):digest(p) for p in
            (Path(__file__),directory/"analysis.json",isolated/"analysis.json",graph_path/"manifest.json",intrinsic_path,tail_path)},
        "limits":["Conditional receiving tests retain intact source histories and are not closed-loop lesion predictions.",
            "Only a single sensory channel at two intensities is tested; KC differences do not establish odor-identity discrimination.",
            "One timing realization and one full recovery window are not an acceptance distribution.",
            "Gate/routing and PN replay checks do not verify all unrecorded intracellular states or fit physiology."]}
    if reference is not None:
        result["closed_loop_pathway_test"]=compare_pathway(reference,directory,graph,meta,data,structure)
    dump_new(output/"analysis.json",result)
    print(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("graph","intrinsic","tail","isolated","directory","output"):p.add_argument(name,type=Path)
    p.add_argument("--reference",type=Path)
    a=p.parse_args();analyze(a.graph,a.intrinsic,a.tail,a.isolated,a.directory,a.output,reference=a.reference)


if __name__=="__main__":main()
