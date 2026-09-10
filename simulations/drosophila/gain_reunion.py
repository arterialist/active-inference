"""Reunite a fixed presynaptic gain mechanism with its actual neural partners.

Controlled electrodes are identical to the isolated assay. Neither the input
commands nor the cellular gate are retuned after restoring reciprocal paths.
The consumer cut is a selective intermediate; full includes all 3,005 cells.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import inspect
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np

from .connectome import Subgraph
from .ln_gain import PN, LN, commands, blocked_terminals
from .ln_input_replay import cut_cells
from .paula import Neuron
from .presynaptic_prepare import prepare_inhibited, PresynapticInhibitionNeuron
from .prisco import digest, dump_new
from neuron.neuron import RetrogradeSignalEvent

PATHWAYS = ("none", "positive_LN_to_LN", "positive_LN_to_target_PN",
            "positive_PN_to_LN", "positive_LN_or_PN_to_LN",
            "regulator_to_LN", "regulator_to_PN", "regulator_to_LN_and_PN")


def reunion_cut(graph, scope):
    if scope == "full": return graph
    if scope == "isolated":
        from .ln_gain import select
        return select(graph)
    if scope == "antennal":
        roots=tuple(r for r in graph.selected if graph.nodes[r]["annotation"]["cell_class"] in
            ("olfactory","ALLN","ALPN") and graph.nodes[r]["annotation"]["hemibrain_type"]!="APL")
        return cut_cells(graph,roots)
    if scope != "consumers": raise ValueError("Unknown reunion scope")
    roots = tuple(r for r in graph.selected if r == LN or
        graph.nodes[r]["annotation"]["cell_class"] in ("ALPN", "Kenyon_Cell") or
        graph.nodes[r]["annotation"]["hemibrain_type"] in ("ORN_DL5", "APL"))
    return cut_cells(graph,roots)


def target_bindings(prep,source_roots,target_id):
    terminals=[];ports=[]
    for root in source_roots:
        rows=prep.edge_bindings[(prep.edge_bindings[:,1]==prep.root_to_id[root])&(prep.edge_bindings[:,3]==target_id)]
        if len(rows)>1:raise ValueError("Duplicate aggregated source-to-target pair")
        # A selected sensory neuron may reach other PNs without a direct pair
        # to this particular target. -1 denotes absence, never a fabricated port.
        terminals.append(int(rows[0,2]) if len(rows) else -1)
        ports.append(int(rows[0,4]) if len(rows) else -1)
    return terminals,ports


def pathway_bindings(prep, graph, pathway):
    """Select internal measured pairs by annotation and initial model sign.

    This is an experimental transmission block, not a sign correction or a
    claim about receptor physiology. Anatomical bindings remain unchanged.
    """
    if pathway not in PATHWAYS:
        raise ValueError("Unknown recurrent pathway intervention")
    rows = set()
    for e in graph.edges:
        source, target = graph.nodes[str(e[0])]["annotation"], graph.nodes[str(e[1])]["annotation"]
        if pathway.startswith("regulator_to_"):
            target_ln=target["cell_class"]=="ALLN" and target["hemibrain_type"]!="APL"
            target_pn=target["cell_class"]=="ALPN"
            if str(e[0])==LN and ((target_ln and pathway in ("regulator_to_LN","regulator_to_LN_and_PN"))
                    or (target_pn and pathway in ("regulator_to_PN","regulator_to_LN_and_PN"))):
                rows.add(int(e[8]))
            continue
        if pathway == "none" or e[5] <= 0 or source["hemibrain_type"] == "APL":
            continue
        source_ln=source["cell_class"]=="ALLN"
        source_pn=source["cell_class"]=="ALPN"
        target_ln=target["cell_class"]=="ALLN" and target["hemibrain_type"]!="APL"
        if ((pathway == "positive_LN_to_LN" and source_ln and target_ln)
                or (pathway == "positive_PN_to_LN" and source_pn and target_ln)
                or (pathway == "positive_LN_or_PN_to_LN" and (source_ln or source_pn) and target_ln)
                or (pathway == "positive_LN_to_target_PN" and source_ln and str(e[1]) == PN)):
            rows.add(int(e[8]))
    return np.array([e for e in prep.edge_bindings if int(e[0]) in rows], dtype=np.int64).reshape(-1, 5)


def filter_forward(events, blocked):
    if any(not isinstance(e,RetrogradeSignalEvent) and not (isinstance(e,tuple) and len(e)==3) for e in events):
        raise TypeError("Unknown event type")
    return [e for e in events if not isinstance(e,tuple) or e[1] not in blocked]


def initialize_feedback_gain(prep,graph,scale):
    """An initial receptor-weight hypothesis, not an online rate controller."""
    if not np.isfinite(scale) or not 0<scale<=1:
        raise ValueError("Feedback gain must be finite and in (0, 1]")
    bindings=pathway_bindings(prep,graph,"positive_LN_or_PN_to_LN")
    before=[];after=[]
    for _,_,_,target,port in bindings:
        point=prep.network.network.neurons[int(target)].postsynaptic_points[int(port)]
        if point.u_i.info<=0:raise ValueError("Selected feedback receptor is not positive")
        before.append(point.u_i.info)
        if scale!=1.:point.u_i.info*=scale
        after.append(point.u_i.info)
    return bindings,np.array(before),np.array(after)


def regulator_afferent_bindings(prep,graph):
    rows={int(e[8]) for e in graph.edges if str(e[1])==LN and e[5]>0
          and graph.nodes[str(e[0])]["annotation"]["hemibrain_type"]=="ORN_DL5"}
    return np.array([e for e in prep.edge_bindings if int(e[0]) in rows],dtype=np.int64).reshape(-1,5)


def initialize_regulator_afferent_gain(prep,graph,scale):
    """Sensitivity of measured sensory inputs to the regulator, before tick zero.

    No added connection, prescribed regulator firing, or online gain adjustment.
    The default is an exact no-op. This is not a physiological strength fit.
    """
    if not np.isfinite(scale) or scale<=0:raise ValueError("Afferent gain must be positive and finite")
    bindings=regulator_afferent_bindings(prep,graph)
    if not len(bindings):raise ValueError("No measured sensory afferents to regulator")
    before=[];after=[]
    for _,_,_,target,port in bindings:
        point=prep.network.network.neurons[int(target)].postsynaptic_points[int(port)]
        if point.u_i.info<=0:raise ValueError("Selected sensory receptor is not positive")
        before.append(point.u_i.info)
        if scale!=1.:point.u_i.info*=scale
        after.append(point.u_i.info)
    return bindings,np.array(before),np.array(after)


def window_lateral_command(command, window):
    if window is None:return command
    if len(window)!=2 or any(type(t) is not int for t in window) or not 0<=window[0]<window[1]<=len(command):
        raise ValueError("Invalid lateral electrode window")
    result=command.copy()
    result[:window[0],-1]=0.;result[window[1]:,-1]=0.
    return result


def reunion_commands(n,direct,lateral,seed,*,stimulus_duration=1000,sensory_precondition=None):
    """Prescribed sensory history; no state-dependent stimulus or phase reset."""
    command,epochs=commands(n,direct,lateral,seed,duration=stimulus_duration,trials=1)
    if sensory_precondition is None:return command,epochs
    if len(sensory_precondition)!=2:raise ValueError("Precondition requires a rate and stop tick")
    rate,stop=sensory_precondition
    if not np.isfinite([rate,stop]).all() or stop!=int(stop) or not 200<stop<epochs[0]["stop"]:
        raise ValueError("Invalid sensory precondition boundary")
    high,_=commands(n,rate,lateral,seed,duration=stimulus_duration,trials=1)
    command[200:int(stop),:-1]=high[200:int(stop),:-1]
    return command,epochs


def run(graph_path,intrinsic_path,tail_path,spatial,output,*,scope="full",gain=1.,
        direct=50.,lateral=80.,seed=11,lesion="intact",chunk=250,
        pathway="none",block_start=600,feedback_scale=1.,lateral_window=None,regulator_afferent_scale=1.,curated_gaba_control=False,
        stimulus_duration=1000,sensory_precondition=None):
    if output.exists():raise FileExistsError(output)
    if type(chunk) is not int or chunk<1:raise ValueError("Invalid chunk size")
    if shutil.disk_usage(output.parent).free < 512*1024**2:
        raise OSError("Less than 512 MiB free; do not start another recording")
    started=time.perf_counter()
    graph=reunion_cut(Subgraph.load(graph_path),scope)
    intrinsic=json.loads(intrinsic_path.read_text())
    tail=json.loads(tail_path.read_text())["fits"]["2"]["all_cells"]
    for p,h in intrinsic["source_hashes"].items():
        if digest(Path(p))!=h:raise ValueError(f"Changed calibration source: {p}")
    has_apl=any(graph.nodes[r]["annotation"]["hemibrain_type"]=="APL" for r in graph.selected)
    prep,pn=prepare_inhibited(graph,intrinsic,tail,LN,gain,100.,spatial=spatial if has_apl else None)
    feedback_bindings,feedback_before,feedback_initial=initialize_feedback_gain(prep,graph,feedback_scale)
    afferent_bindings,afferent_before,afferent_initial=initialize_regulator_afferent_gain(prep,graph,regulator_afferent_scale)
    polarity=None
    if curated_gaba_control:
        from .orn_onset import negative_gaba_control
        polarity=negative_gaba_control(prep,graph)
        feedback_initial=np.array([prep.network.network.neurons[int(target)].postsynaptic_points[int(port)].u_i.info
                                   for _,_,_,target,port in feedback_bindings])
    roots=list(prep.root_to_id)
    cells=[prep.network.network.neurons[prep.root_to_id[r]] for r in roots]
    orns=[r for r in roots if graph.nodes[r]["annotation"]["hemibrain_type"]=="ORN_DL5"]
    source_roots=orns+[LN]
    source_cells=[prep.network.network.neurons[prep.root_to_id[r]] for r in source_roots]
    source_cols={c.id:i for i,c in enumerate(source_cells)}
    apl=next((c for r,c in zip(roots,cells) if graph.nodes[r]["annotation"]["hemibrain_type"]=="APL"),None)
    ln_id=prep.root_to_id[LN]; blocked=blocked_terminals(prep,graph,lesion)
    pn_terminals,pn_ports=target_bindings(prep,orns,pn.id)
    command,epochs=reunion_commands(len(orns),direct,lateral,seed,stimulus_duration=stimulus_duration,
                                   sensory_precondition=sensory_precondition)
    command=window_lateral_command(command,lateral_window)
    ticks=len(command)
    if type(block_start) is not int or not 0 <= block_start < ticks:
        raise ValueError("Intervention onset outside recorded ticks")
    path_bindings=pathway_bindings(prep,graph,pathway)
    if pathway != "none" and not len(path_bindings):raise ValueError("Empty selected pathway")
    path_terminals={}
    for _,source,terminal,_,_ in path_bindings:
        path_terminals.setdefault(int(source),set()).add(int(terminal))
    files=[Path(__file__),Path(__file__).with_name("presynaptic_prepare.py"),Path(__file__).with_name("ln_gain.py"),
        Path(__file__).with_name("paula.py"),Path(__file__).with_name("pn_current_steps.py"),
        Path(__file__).with_name("spatial_paula.py"),Path(inspect.getfile(Neuron)),
        Path(inspect.getfile(PresynapticInhibitionNeuron)),graph_path/"manifest.json",intrinsic_path,tail_path,spatial/"analysis.json"]
    if curated_gaba_control:
        files.extend(Path(__file__).with_name(name) for name in ("orn_onset.py","antennal_identity.py"))
    hashes={str(p.resolve()):digest(p) for p in files}
    output.mkdir(parents=True)
    with (output/"structure.npz").open("xb") as f:
        np.savez_compressed(f,roots=np.array(roots),source_roots=np.array(source_roots),
            edge_bindings=prep.edge_bindings,incoming_boundary_ports=prep.incoming_boundary_ports,
            outgoing_boundary_terminals=prep.outgoing_boundary_terminals,
            pn_orn_terminals=np.array(pn_terminals),pn_orn_ports=np.array(pn_ports),
            pn_initial_weights=np.array([p.u_i.info for p in pn.postsynaptic_points.values()]))
    with (output/"intervention.npz").open("xb") as f:
        np.savez_compressed(f,bindings=path_bindings)
    with (output/"feedback-initial.npz").open("xb") as f:
        np.savez_compressed(f,bindings=feedback_bindings,reference_weights=feedback_before,initial_weights=feedback_initial)
    with (output/"regulator-afferent-initial.npz").open("xb") as f:
        np.savez_compressed(f,bindings=afferent_bindings,reference_weights=afferent_before,initial_weights=afferent_initial)
    arrays={};start=0
    native_hillock,native_tick=Neuron._hillock_current,Neuron.tick
    target_hillock=type(pn)._hillock_current

    def electrode(cell,tick,dt):
        current=native_hillock(cell,tick,dt); col=source_cols.get(cell.id)
        if col is None:return current
        total=current if command[tick,col]==0 else current+command[tick,col]
        arrays["source_current"][tick-start,col]=[current,total]
        return total

    def measure(cell,tick,dt):
        if cell is not pn:return target_hillock(cell,tick,dt)
        arrays["pn_inputs"][tick-start]=cell.input_buffer
        current=target_hillock(cell,tick,dt)
        arrays["pn_current"][tick-start]=cell.last_port_current
        return current

    def release(cell,inputs,tick,dt=1.):
        events=native_tick(cell,inputs,tick,dt)
        admitted=events
        if cell.id in path_terminals:
            selected=path_terminals[cell.id]
            attempted=sum(isinstance(e,tuple) and e[1] in selected for e in events)
            if tick>=block_start:admitted=filter_forward(events,selected)
            arrays["pathway_events"][tick-start]+=[attempted,
                sum(isinstance(e,tuple) and e[1] in selected for e in admitted),
                sum(isinstance(e,RetrogradeSignalEvent) for e in events)]
        if cell.id==ln_id:
            admitted=filter_forward(admitted,blocked)
            forward=sum(isinstance(e,tuple) for e in events)
            arrays["ln_events"][tick-start]=[forward,sum(isinstance(e,tuple) for e in admitted),len(events)-forward]
        return admitted

    chunks=[]
    with ExitStack() as stack:
        stack.enter_context(patch.object(Neuron,"_hillock_current",electrode))
        stack.enter_context(patch.object(type(pn),"_hillock_current",measure))
        stack.enter_context(patch.object(Neuron,"tick",release))
        for start in range(0,ticks,chunk):
            if shutil.disk_usage(output).free < 256*1024**2:
                raise OSError("Recording stopped before disk exhaustion; this course is incomplete")
            stop=min(start+chunk,ticks);n=stop-start
            arrays={"soma":np.zeros((n,len(cells),3)),"command":command[start:stop],
                "source_current":np.zeros((n,len(source_cells),2)),
                "source_intrinsic":np.zeros((n,len(source_cells),4)),
                "pn_inputs":np.zeros((n,pn.params.num_inputs,4),dtype=np.float32),
                "pn_current":np.zeros((n,pn.params.num_inputs)),
                "pn_weights":np.zeros((n,pn.params.num_inputs)),"pn_intrinsic":np.zeros((n,4)),
                "inhibition":np.zeros((n,len(orns),5)),"ln_events":np.zeros((n,3),dtype=np.int64),
                "pathway_events":np.zeros((n,3),dtype=np.int64)}
            if apl is not None:arrays["apl"]=np.zeros((n,3))
            for tick in range(start,stop):
                i=tick-start
                prep.network.run_tick()
                arrays["soma"][i]=[[c.S,c.O,c.F_avg] for c in cells]
                arrays["pn_intrinsic"][i]=[pn.t_ref,pn.r,pn.b,pn.total_current]
                arrays["pn_weights"][i]=[p.u_i.info for p in pn.postsynaptic_points.values()]
                arrays["source_intrinsic"][i]=[[c.t_ref,c.r,c.b,c.params.lambda_param] for c in source_cells]
                arrays["inhibition"][i]=[[c.inhibition_state,c.inhibition_fraction,c.inhibition_arriving_drive,
                    c.inhibition_last_native.get(t,0.),c.inhibition_last_effective.get(t,0.)]
                    for c,t in zip(source_cells[:-1],pn_terminals)]
                if apl is not None:
                    arrays["apl"][i]=[apl.cable.voltage.min(),apl.cable.voltage.max(),apl.terminal_release.max(initial=0)]
            if any(not np.isfinite(a).all() for a in arrays.values()):raise ValueError("Nonfinite recording")
            name=f"ticks-{start:06d}-{stop:06d}.npz"
            with (output/name).open("xb") as f:np.savez_compressed(f,**arrays)
            chunks.append({"file":name,"start":start,"stop":stop,"sha256":digest(output/name)})
            print(f"{scope} gain={gain:g} direct={direct:g}: {stop}/{ticks}, {time.perf_counter()-started:.1f}s",flush=True)
    if any(c.params.eta_post<=0 or c.params.eta_retro<=0 or c._ablation for c in cells):raise ValueError("Adaptation disabled")
    if any(digest(Path(p))!=h for p,h in hashes.items()):raise ValueError("Source changed during recording")
    feedback_final=np.array([prep.network.network.neurons[int(target)].postsynaptic_points[int(port)].u_i.info
                             for _,_,_,target,port in feedback_bindings])
    with (output/"feedback-final.npz").open("xb") as f:np.savez_compressed(f,weights=feedback_final)
    afferent_final=np.array([prep.network.network.neurons[int(target)].postsynaptic_points[int(port)].u_i.info
                            for _,_,_,target,port in afferent_bindings])
    with (output/"regulator-afferent-final.npz").open("xb") as f:np.savez_compressed(f,weights=afferent_final)
    result={"schema":1,"scope":scope,"ticks":ticks,"chunks":chunks,"source_hashes":hashes,
        "structure_sha256":digest(output/"structure.npz"),"anatomy":graph.summary(),"assumptions":prep.assumptions,
        "gain":gain,"direct":direct,"lateral":lateral,"seed":seed,"lesion":lesion,"epochs":epochs,"apl_present":has_apl,
        "polarity_control":polarity,
        "sensory_precondition":None if sensory_precondition is None else {"rate":float(sensory_precondition[0]),
            "start":200,"stop":int(sensory_precondition[1]),"after":"unchanged baseline-rate train on original source phases"},
        "lateral_window":list(lateral_window) if lateral_window is not None else None,
        "lateral_command_pulses":int(np.count_nonzero(command[:,-1])),
        "lateral_injected_current_sum":float(command[:,-1].sum()),
        "regulator_afferent_initialization":{"scale":regulator_afferent_scale,"pairs":len(afferent_bindings),
            "initial_sha256":digest(output/"regulator-afferent-initial.npz"),
            "final_sha256":digest(output/"regulator-afferent-final.npz"),
            "changed_by_learning":int(np.count_nonzero(afferent_final!=afferent_initial)),
            "action":"scale only initial internal ORN_DL5-to-identified-regulator receiving info weights; native ongoing learning and all connections retained",
            "status":"uncalibrated sensory recruitment strength hypothesis, not measured physiology"},
        "feedback_initialization":{"scale":feedback_scale,"pairs":len(feedback_bindings),
            "initial_sha256":digest(output/"feedback-initial.npz"),"final_sha256":digest(output/"feedback-final.npz"),
            "changed_by_learning":int(np.count_nonzero(feedback_final!=feedback_initial)),
            "action":"scale initial internal positive-model LN/PN-to-LN receiving info weights once before tick 0; boundary weights, all events, connections and native adaptation retained",
            "status":"uncalibrated pathway-strength sensitivity hypothesis, not measured physiology"},
        "pathway_intervention":{"pathway":pathway,"start":block_start,
            "pairs":len(path_bindings),"sources":len(path_terminals),
            "bindings_sha256":digest(output/"intervention.npz"),
            "action":"withhold only selected forward terminal events from onset; preserve return events, ports, native states and learning"},
        "recording":{"soma":["S","O","F_avg"],"source_current":["native","total"],
            "source_intrinsic":["t_ref","r","b","lambda"],"pn_intrinsic":["t_ref","r","b","total_current"],
            "inhibition":["state","fraction","arriving_drive","native_PN_terminal_release","effective_PN_terminal_release"],
            "apl":["minimum_voltage","maximum_voltage","maximum_local_release"] if has_apl else None,
            "ln_events":["attempted_forward","admitted_forward","returned"],
            "pathway_events":["attempted_selected_forward","admitted_selected_forward","preserved_return_events_from_selected_sources"]},
        "runtime_seconds":time.perf_counter()-started,
        "claim":"Fixed local gain mechanism under restored reciprocal neural coupling; not embodied or physiological acceptance",
        "limits":["One timing seed and one complete train/recovery cycle per course, not an acceptance distribution.",
            "Controlled LN current remains an imposed boundary, not a public sensory-input circuit.",
            "Consumers-only excludes other LNs explicitly; full restores all 208 selected LNs.",
            "Antennal-only retains all selected ORNs/LNs/PNs but excludes KCs and APL; missing APL state is not recorded as zero.",
            "Full soma and target receiving state are recorded; other cells' full ports and the full APL tree are not.",
            "Gate gain and 100-tick decay are unchanged isolated-assay hypotheses, not fitted physiological parameters."]}
    dump_new(output/"analysis.json",result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("graph","intrinsic","tail","spatial","output"):p.add_argument(name,type=Path)
    p.add_argument("--scope",choices=("full","consumers","antennal","isolated"),default="full")
    p.add_argument("--gain",type=float,default=1.)
    p.add_argument("--direct",type=float,default=50.)
    p.add_argument("--lateral",type=float,default=80.)
    p.add_argument("--seed",type=int,default=11)
    p.add_argument("--lesion",choices=("intact","LN_to_ORN_block","LN_to_PN_block","LN_release_block"),default="intact")
    p.add_argument("--pathway",choices=PATHWAYS,default="none")
    p.add_argument("--block-start",type=int,default=600)
    p.add_argument("--feedback-scale",type=float,default=1.)
    p.add_argument("--lateral-window",nargs=2,type=int,metavar=("START","STOP"))
    p.add_argument("--regulator-afferent-scale",type=float,default=1.)
    p.add_argument("--curated-gaba-control",action="store_true",help="Experimental negative receiving signs for curated GABA / positive-model ALLNs; preserve wiring and magnitudes")
    p.add_argument("--stimulus-duration",type=int,default=1000)
    p.add_argument("--sensory-precondition",nargs=2,type=float,metavar=("RATE","STOP"))
    a=p.parse_args()
    run(a.graph,a.intrinsic,a.tail,a.spatial,a.output,scope=a.scope,gain=a.gain,direct=a.direct,
        lateral=a.lateral,seed=a.seed,lesion=a.lesion,pathway=a.pathway,block_start=a.block_start,
        feedback_scale=a.feedback_scale,lateral_window=a.lateral_window,regulator_afferent_scale=a.regulator_afferent_scale,
        curated_gaba_control=a.curated_gaba_control,stimulus_duration=a.stimulus_duration,
        sensory_precondition=a.sensory_precondition)


if __name__=="__main__":main()
