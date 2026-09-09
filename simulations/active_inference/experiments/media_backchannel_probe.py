"""Trace and selectively interrupt native retrograde feedback during recall.

Four short full-graph branches cross learned/reset selected input weights with
intact/cut feedback from those same synapses. No adaptation rates are zeroed.
The interruption is an experimental lesion, not a proposed brain controller.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import gzip
import inspect
import json
from pathlib import Path
import time
import numpy as np

from .association_balance_audit import checked
from .association_route_probe import digest
from .composition_probe import encode, fingerprint
from .eligibility_association_probe import dynamic_snapshot
from .eligibility_media_probe import record
from .graded_recall_factors import intervene
from .multimodal_pairing_probe import fresh, WeightObserver
from .population_state_branch import TickDriver
from neuron.neuron import Neuron, RetrogradeSignalEvent
from neuron.extensions.experimental.graded_eligibility import GradedEligibilityNeuron


class TerminalObserver(WeightObserver):
    def __init__(self, neurons, synapses):
        super().__init__(neurons,synapses)
        self.terminal_rows=[]

    def __call__(self):
        super().__call__()
        self.terminal_rows.append([[p.u_o.info,*p.u_o.mod] for p in self.terminals])


@contextmanager
def selected_feedback(net, ports, cut):
    selected={(n,sid) for n,sid,_ in ports};events=[];removed=[]
    original_tick=GradedEligibilityNeuron.tick;owned='tick' in GradedEligibilityNeuron.__dict__
    original_retro=Neuron.process_retrograde_signal
    def tick(n, external, current_tick, dt=1.):
        output=original_tick(n,external,current_tick,dt)
        if not cut or net.network.neurons.get(n.id) is not n:return output
        kept=[]
        for e in output:
            if isinstance(e,RetrogradeSignalEvent) and (e.source_neuron_id,e.source_synapse_id) in selected:
                removed.append([current_tick,e.source_neuron_id,e.source_synapse_id,e.target_neuron_id,e.target_terminal_id,*e.error_vector])
            else:kept.append(e)
        return kept
    def retro(n,e):
        match=net.network.neurons.get(n.id) is n and (e.source_neuron_id,e.source_synapse_id) in selected
        before=float(n.presynaptic_points[e.target_terminal_id].u_o.info) if match else None
        result=original_retro(n,e)
        if match:
            after=float(n.presynaptic_points[e.target_terminal_id].u_o.info)
            events.append([net.current_tick,e.source_neuron_id,e.source_synapse_id,n.id,e.target_terminal_id,before,after,*e.error_vector])
        return result
    GradedEligibilityNeuron.tick=tick;Neuron.process_retrograde_signal=retro
    try:yield events,removed
    finally:
        if owned:GradedEligibilityNeuron.tick=original_tick
        else:delattr(GradedEligibilityNeuron,'tick')
        Neuron.process_retrograde_signal=original_retro


def run(source,output,ticks=64):
    source,output=Path(source).resolve(),Path(output).resolve()
    if not 1<=ticks<=300:raise ValueError('Need a recorded probe prefix')
    m=json.loads((source/'manifest.json').read_text());s=json.loads((source/'summary.json').read_text())
    if any(digest(p)!=h for p,h in m['source_hashes'].items()):raise ValueError('Source runtime changed')
    features=[]
    for clip in (0,1):
        path=next(Path(p) for p in m['physical_sources'] if Path(p).name==f'sensory-{clip}.npz')
        if digest(path)!=m['physical_sources'][str(path)]:raise ValueError('Physical source changed')
        with np.load(path) as z:features.append({k:z[k] for k in z.files})
    hashes=fingerprint()
    for obj in (run,record,intervene,fresh,TickDriver,dynamic_snapshot):
        path=Path(inspect.getfile(obj)).resolve();hashes[str(path)]=digest(path)
    output.mkdir(parents=True,exist_ok=False);began=time.perf_counter()
    net,core,members,points=fresh(source/'config.json',m['seed'],GradedEligibilityNeuron)
    birth=json.loads(dynamic_snapshot(net));ports=m['selected_ports'];health=WeightObserver(members,points)
    for item in s['training']:
        data=record(net,core,members,points,features,m['groups'],item['trial'],ports,health)
        with np.load(checked(source,item)) as z:
            if any(not np.array_equal(data[k],z[k]) for k in z.files):raise ValueError('Acquisition replay differs')
    parent=dynamic_snapshot(net)
    with gzip.open(source/'trained-state.json.gz','rt') as f:
        if json.loads(parent)!=json.load(f):raise ValueError('Trained state differs')
    print('exact acquisition complete',round(time.perf_counter()-began,2),flush=True)
    q0=np.array([birth['neurons'][str(n)]['synapses'][str(sid)][0] for n,sid,_ in ports])
    q1=np.array([net.network.neurons[n].postsynaptic_points[sid].u_i.info for n,sid,_ in ports])
    branches=[]
    for reset,q in ((False,q1),(True,q0)):
        for cut in (False,True):
            branch=deepcopy(net);intervene(branch,ports,m['groups']['vision'],q,'spiking')
            members=list(branch.network.neurons.values());points=[p for n in members for p in n.postsynaptic_points.values()]
            observer=TerminalObserver(members,points);t=branch.current_tick
            trial=dict(start=t,stop=t+ticks,visual_clip=0,audio_clip=None)
            with selected_feedback(branch,ports,cut) as (retro,removed):
                data=record(branch,TickDriver(branch),members,points,features,m['groups'],trial,ports,observer)
            if not cut:
                ref=next(p for p in s['probes'] if p['state']==('reset_selected' if reset else 'trained')
                         and p['expression']=='spiking' and p['sense']=='visual' and p['clip']==0)
                with np.load(checked(source,ref)) as z:
                    for k in ('cells','weights','pre','post','arrivals','eta','weight_health'):
                        if not np.array_equal(data[k],z[k][:ticks]):raise ValueError('Observer changed probe prefix')
            if cut and retro:raise ValueError('Cut selected feedback was delivered')
            if cut and not removed:raise ValueError('Lesion did not intercept events')
            if not cut and not retro:raise ValueError('Selected feedback is not driven')
            data.update(terminals=np.asarray(observer.terminal_rows),retro_events=np.asarray(retro).reshape(-1,11),
                        removed_retro=np.asarray(removed).reshape(-1,9))
            name=f'{"reset" if reset else "learned"}-{ "cut" if cut else "intact"}.npz'
            np.savez_compressed(output/name,**data)
            branches.append(dict(reset=reset,cut=cut,file=name,sha256=digest(output/name),trial=trial))
            if dynamic_snapshot(net)!=parent:raise ValueError('Parent changed')
            print(name,'delivered',len(retro),'removed',len(removed),flush=True)
    if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('Runtime changed')
    manifest=dict(source=str(source),groups=m['groups'],selected_ports=ports,source_hashes=hashes,
        terminal_order=[(n.id,tid) for n in members for tid in n.presynaptic_points],
        terminal_fields=['info','mod0','mod1'],retro_fields=['tick','from_neuron','from_synapse','to_neuron','to_terminal','info_before','info_after','error_info','error_plast','error_mod0','error_mod1'],
        limits='One history, one cue prefix. Selected return events are removed only after native local computation. '
               'Other feedback and all positive learning rates remain active. No claim that eliminating feedback improves memory.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    result=dict(branches=branches,exact_acquisition_replay=True,intact_prefixes_exact=True,parent_unchanged=True,
                ticks=m['trials'][-1]['stop']+4*ticks,seconds=time.perf_counter()-began)
    (output/'summary.json').write_text(encode(result)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ticks',type=int,default=64)
    a=p.parse_args();run(a.source,a.output,a.ticks)
