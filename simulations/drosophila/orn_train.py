"""Native ORN spikes drive the reunited DL5/ALLN/PN/KC/APL preparation.

This is a current-pulse experiment, not an odor transduction model. Staggered
electrical pulses may be vetoed or supplemented by the actual recurrent
network. Inhibitory ALLN-to-ORN pairs still use ordinary somatic inputs here;
this deliberately does not claim to implement presynaptic GABA inhibition.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import inspect
import json
from pathlib import Path
import random
import time
from unittest.mock import patch

import numpy as np

from .connectome import Subgraph
from .paula import ORNReleaseDepression, Neuron
from .pn_current_steps import prepare
from .prisco import digest, dump_new
from neuron.extensions.experimental.release_depression import DepressingReleaseNeuron
from neuron.extensions.experimental.input_current import InputCurrentNeuron
from neuron.extensions.experimental.passive_cable import LocalCableGradedNeuron
from neuron.neuron import RetrogradeSignalEvent
from neuron.extensions.experimental.electrical import ElectricalCoupling

CONDITIONS = ("no_depression", "depressing", "depressing_ln_block")


def pulse_course(n):
    if type(n) is not int or n < 1:
        raise ValueError("Need a nonempty declared source population")
    command = np.zeros((4200,n))
    epochs = [{"start":0,"stop":200,"phase":"baseline"},
        {"start":200,"stop":1200,"phase":"10Hz"},
        {"start":1200,"stop":2200,"phase":"recovery"},
        {"start":2200,"stop":3200,"phase":"50Hz"},
        {"start":3200,"stop":4200,"phase":"recovery"}]
    for start,interval in ((200,100),(2200,20)):
        for col in range(n):
            phase = col*interval//n
            command[np.arange(start+phase,start+1000,interval),col]=40.
    return command,epochs


def run(graph_path, intrinsic_path, tail_path, spatial, output, condition, *, chunk=500, stop_tick=None,
        junction=None):
    if condition not in CONDITIONS or type(chunk) is not int or chunk < 1:
        raise ValueError("Invalid condition/chunk")
    if output.exists():
        raise FileExistsError(output)
    started=time.perf_counter()
    graph=Subgraph.load(graph_path)
    intrinsic=json.loads(intrinsic_path.read_text())
    tail=json.loads(tail_path.read_text())["fits"]["2"]["all_cells"]
    for p,h in intrinsic["source_hashes"].items():
        if digest(Path(p))!=h:
            raise ValueError(f"Calibration source changed: {p}")
    orns=tuple(r for r in graph.selected if graph.nodes[r]["annotation"]["hemibrain_type"]=="ORN_DL5")
    lns=tuple(r for r in graph.selected if graph.nodes[r]["annotation"]["cell_class"]=="ALLN")
    if not orns or not lns:
        raise ValueError("Need reconstructed ORNs and ALLNs, not promoted boundary stubs")
    spec=ORNReleaseDepression(orns,0. if condition=="no_depression" else .22,893.)
    random.seed(0)
    electrical_roots=() if junction is None else tuple(junction[:2])
    prep,target=prepare(graph,intrinsic,tail,spatial=spatial,release_depression=spec,
                        electrical_roots=electrical_roots)
    stepper=prep.network
    electrical_cells=[]
    if junction is not None:
        if len(junction)!=3:
            raise ValueError("Need exactly two roots and one declared conductance")
        electrical_cells=[prep.network.network.neurons[prep.root_to_id[r]] for r in electrical_roots]
        stepper=ElectricalCoupling(prep.network,[(electrical_cells[0].id,electrical_cells[1].id,junction[2])])
        prep.assumptions["electrical_junction"]={"roots":list(electrical_roots),"g":junction[2],
            "status":"hypothetical somatic electrical contact; not inferred from chemical counts or established for these identities"}
    cells=[prep.network.network.neurons[i] for i in prep.root_to_id.values()]
    roots=list(prep.root_to_id)
    source_cells=[prep.network.network.neurons[prep.root_to_id[r]] for r in orns]
    source_cols={c.id:i for i,c in enumerate(source_cells)}
    ln_cols={prep.root_to_id[r]:i for i,r in enumerate(lns)}
    release_offsets=np.r_[0,np.cumsum([len(c.release_terminals) for c in source_cells])]
    command,epochs=pulse_course(len(orns))
    ticks=len(command) if stop_tick is None else stop_tick
    if type(ticks) is not int or not 1<=ticks<=len(command):
        raise ValueError("Invalid stop tick")
    file_sources=[Path(__file__),graph_path/"manifest.json",intrinsic_path,tail_path,
        spatial/"analysis.json",Path(__file__).with_name("paula.py"),
        Path(__file__).with_name("pn_current_steps.py"),Path(__file__).with_name("spatial_paula.py"),
        Path(inspect.getfile(Neuron)),Path(inspect.getfile(DepressingReleaseNeuron)),
        Path(inspect.getfile(InputCurrentNeuron)),Path(inspect.getfile(LocalCableGradedNeuron))]
    if junction is not None:
        file_sources.append(Path(inspect.getfile(ElectricalCoupling)))
    hashes={str(p.resolve()):digest(p) for p in file_sources}
    output.mkdir(parents=True)
    with (output/"structure.npz").open("xb") as f:
        np.savez_compressed(f,roots=np.array(roots),orn_roots=np.array(orns),ln_roots=np.array(lns),
            release_offsets=release_offsets,edge_bindings=prep.edge_bindings,
            incoming_boundary_ports=prep.incoming_boundary_ports,outgoing_boundary_terminals=prep.outgoing_boundary_terminals)
    initial_post=np.array([p.u_i.info for p in target.postsynaptic_points.values()])
    initial_terminal=np.array([p.u_o.info for p in target.presynaptic_points.values()])
    arrays,start={},0
    native_hillock=Neuron._hillock_current
    native_tick=Neuron.tick
    target_hillock=type(target)._hillock_current
    source_counts=np.zeros(len(orns),dtype=int)

    def electrode(cell,tick,dt):
        current=native_hillock(cell,tick,dt)
        col=source_cols.get(cell.id)
        if col is None:
            return current
        arrays["orn_native_current"][tick-start,col]=current
        arrays["orn_before_S"][tick-start,col]=float(cell.S)
        source_counts[col]+=1
        return current if command[tick,col]==0 else current+command[tick,col]

    def observe_target(cell,tick,dt):
        if cell is not target:
            return target_hillock(cell,tick,dt)
        arrays["pn_inputs"][tick-start]=cell.input_buffer
        current=target_hillock(cell,tick,dt)
        arrays["pn_current"][tick-start]=cell.last_port_current
        return current

    def block_ln_release(cell,inputs,tick,dt=1.):
        events=native_tick(cell,inputs,tick,dt)
        col=ln_cols.get(cell.id)
        if col is None:
            return events
        if any(not isinstance(e,RetrogradeSignalEvent) and not (isinstance(e,tuple) and len(e)==3) for e in events):
            raise TypeError("Unknown event type")
        forward=sum(isinstance(e,tuple) for e in events)
        block=condition=="depressing_ln_block"
        arrays["ln_events"][tick-start,col]=[forward,0 if block else forward,len(events)-forward]
        return [e for e in events if isinstance(e,RetrogradeSignalEvent)] if block else events

    chunks=[]
    apl=next(c for r,c in zip(roots,cells) if graph.nodes[r]["annotation"]["hemibrain_type"]=="APL")
    with ExitStack() as stack:
        stack.enter_context(patch.object(Neuron,"_hillock_current",electrode))
        stack.enter_context(patch.object(type(target),"_hillock_current",observe_target))
        stack.enter_context(patch.object(Neuron,"tick",block_ln_release))
        for start in range(0,ticks,chunk):
            stop=min(start+chunk,ticks);n=stop-start
            arrays={"soma":np.zeros((n,len(cells),3)),
                "orn_native_current":np.zeros((n,len(orns))),"orn_before_S":np.zeros((n,len(orns))),
                "orn_intrinsic":np.zeros((n,len(orns),4)),
                "available":np.zeros((n,int(release_offsets[-1]))),
                "release_used":np.zeros((n,int(release_offsets[-1]))),
                "release_native":np.zeros((n,int(release_offsets[-1]))),
                "release_effective":np.zeros((n,int(release_offsets[-1]))),
                "pn_inputs":np.zeros((n,target.params.num_inputs,4),dtype=np.float32),
                "pn_current":np.zeros((n,target.params.num_inputs)),
                "pn_intrinsic":np.zeros((n,4)),
                "pn_post_weight":np.zeros((n,len(initial_post))),
                "pn_terminal_weight":np.zeros((n,len(initial_terminal))),
                "ln_events":np.zeros((n,len(lns),3),dtype=np.int32),"apl":np.zeros((n,3))}
            if junction is not None:
                arrays["electrical"]=np.zeros((n,2,9))
            for tick in range(start,stop):
                row=tick-start
                before=[float(c.S) for c in electrical_cells]
                stepper.run_tick()
                arrays["soma"][row]=[[float(c.S),float(c.O),float(c.F_avg)] for c in cells]
                arrays["pn_intrinsic"][row]=[target.t_ref,target.r,target.b,target.total_current]
                arrays["pn_post_weight"][row]=[p.u_i.info for p in target.postsynaptic_points.values()]
                arrays["pn_terminal_weight"][row]=[p.u_o.info for p in target.presynaptic_points.values()]
                for i,c in enumerate(source_cells):
                    sec=slice(release_offsets[i],release_offsets[i+1])
                    arrays["orn_intrinsic"][row,i]=[c.t_ref,c.r,c.b,c.params.lambda_param]
                    for name,attribute in (("available","release_available"),("release_used","release_used_fraction"),
                        ("release_native","release_native_amplitude"),("release_effective","release_effective_amplitude")):
                        arrays[name][row,sec]=getattr(c,attribute)
                arrays["apl"][row]=[float(apl.cable.voltage.min()),float(apl.cable.voltage.max()),float(apl.terminal_release.max(initial=0))]
                for i,c in enumerate(electrical_cells):
                    arrays["electrical"][row,i]=[before[i],c.S,c.electrical_native_current,
                        c.electrical_current,c.O,c.F_avg,c.t_ref,c.r,c.b]
            if any(not np.isfinite(a).all() for a in arrays.values()):
                raise ValueError("Nonfinite recording")
            filename=f"ticks-{start:06d}-{stop:06d}.npz"
            with (output/filename).open("xb") as f:
                np.savez_compressed(f,**arrays)
            chunks.append({"file":filename,"start":start,"stop":stop,"sha256":digest(output/filename)})
            print(f"{condition}: {stop}/{ticks} ticks, {time.perf_counter()-started:.1f}s",flush=True)
    np.testing.assert_array_equal(source_counts,ticks)
    if any(c.params.eta_post<=0 or c.params.eta_retro<=0 or c._ablation for c in cells):
        raise ValueError("Adaptation was disabled")
    if hashes!={str(p.resolve()):digest(p) for p in file_sources}:
        raise ValueError("Source changed during experiment")
    result={"schema":1,"condition":condition,"ticks":ticks,"complete_course":ticks==len(command),
        "epochs":epochs,"root":intrinsic["root"],"assumptions":prep.assumptions,"anatomy":graph.summary(),
        "chunks":chunks,"structure_sha256":digest(output/"structure.npz"),"source_hashes":hashes,
        "initial_post_weight":initial_post.tolist(),"initial_terminal_weight":initial_terminal.tolist(),
        "runtime_seconds":time.perf_counter()-started,
        "protocol":{"clock":"nominal 1 ms/tick, not full-circuit physiological calibration",
            "current_pulse_model_units":40.,"pulse_length_ticks":1,"per_source_phase":"source-order rank times period divided by population size, floored",
            "source_spikes_clamped":False,"depression":"effective single-pool control: f=.78 and recovery 893 ms from Nagel et al. Figure 1, transferred across glomeruli",
            "ln_block":"all forward release from all 208 selected ALLNs, regardless of sign; state and return events retained" if condition=="depressing_ln_block" else None},
        "recording":{"soma":["S","O","F_avg"],"orn_intrinsic":["t_ref","r","b","lambda"],"pn_intrinsic":["t_ref","r","b","total_current"],
            "ln_events":["attempted_forward","admitted_to_router","returned"],"apl":["min_compartment_voltage","max_compartment_voltage","max_local_release"],
            "omitted":["full event queues","full APL voltage tree","all source receptor buffers and weights","other cells' per-port histories"],"checkpoint":False},
        "limits":["No odor transduction, ORN intrinsic calibration or empirical spike-train matching.",
            "Uniform simple-depression dynamics cannot reproduce the paper's distinct fast/slow mechanisms.",
            "ALLN-to-ORN inhibition is still somatic in this adapter, not physiological presynaptic release regulation.",
            "Other neuron classes and synaptic gains remain uncalibrated; added anatomy is not physiological validation."],
        "claim":"Reunited neural train diagnostic and explicit simple-depression control; not odor or behavioral reproduction"}
    if junction is not None:
        result["recording"]["electrical"]={"roots":list(electrical_roots),
            "fields":["S_before","S_after","chemical_current","electrical_current","O","F_avg","t_ref","r","b"],
            "pn_intrinsic_total_current":"retains chemical-only current; electrical current is separate"}
        result["limits"].append("The added passive electrical contact is hypothetical; no electrical anatomy, conductance, or LN intrinsic calibration was established.")
    dump_new(output/"analysis.json",result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("graph","intrinsic","tail","spatial","output"):
        p.add_argument(name,type=Path)
    p.add_argument("condition",choices=CONDITIONS)
    p.add_argument("--stop-tick",type=int)
    p.add_argument("--junction",nargs=3,metavar=("ROOT_A","ROOT_B","G"))
    a=p.parse_args()
    junction=None if a.junction is None else (a.junction[0],a.junction[1],float(a.junction[2]))
    run(a.graph,a.intrinsic,a.tail,a.spatial,a.output,a.condition,stop_tick=a.stop_tick,junction=junction)


if __name__=="__main__":main()
