"""Observe release feedback and test a 193-cell inhibitory gate under live plasticity.

This isolated preparation keeps the measured 192-target feedback fanout. The
input schedule is an external diagnostic stimulus, not a neural controller.
The optional local magnitude rule is a model hypothesis, not an established
biological retrograde learning law. Every terminal and target is recorded.
"""
import argparse
from pathlib import Path
import shutil
import time

import numpy as np

from . import context_organization as base
from neuron.extensions.experimental.magnitude_retrograde import MagnitudeRetrogradeNeuron


class ReleaseObserver:
    """Read-only tick wrapper. Record actual returns in their delivery order."""
    def __init__(self, net, context):
        self.net=net;self.context=context;self.original=net.run_tick
        self.terminal_ids=[(nid,tid) for nid,n in net.network.neurons.items()
                           for tid in n.presynaptic_points]
        self.terminals=[net.network.neurons[nid].presynaptic_points[tid]
                        for nid,tid in self.terminal_ids]
        n=net.network.neurons[context]
        if len(n.presynaptic_points)!=1 or n._rate_boost!=0:
            raise ValueError('Context observer needs one terminal and native return rate')
        self.tid=next(iter(n.presynaptic_points));self.neuron=n
        self.releases=[];self.context_trace=[];self.events=[];self.offsets=[0];self.before_dtypes=[]
        self.initial=np.array([p.u_o.info for p in self.terminals])

    def tick(self):
        net=self.net;slot=net.current_tick % net.wheel_size
        returns=[s.event for s in net.retrograde_wheel[slot]
                 if s.event.target_neuron_id==self.context and s.event.target_terminal_id==self.tid]
        p=self.neuron.presynaptic_points[self.tid];before=p.u_o.info;expected=before
        self.before_dtypes.append(32 if isinstance(before,np.float32) else
                                  64 if isinstance(before,np.float64) else 0)
        for e in returns:
            if e.error_vector.dtype!=np.float32:
                raise ValueError('Recorder currently requires native float32 return events')
            # Preserve native scalar promotion and sequential clipping.
            expected+=self.neuron.params.eta_retro*e.error_vector[0]
            expected=np.clip(expected,-100.,100.)
            self.events.append([net.current_tick,e.source_neuron_id,e.source_synapse_id,
                                *e.error_vector])
        arriving=sum(float(s.event[2]) for s in net.presynaptic_wheel[slot]
                     if isinstance(s.event,tuple) and s.event[:2]==(self.context,self.tid))
        result=self.original()
        if p.u_o.info!=expected:raise ValueError('Context terminal recurrence differs')
        self.context_trace.append([before,p.u_o.info,len(returns),arriving])
        self.offsets.append(len(self.events))
        self.releases.append([p.u_o.info for p in self.terminals])
        return result

    def __enter__(self):
        self.net.run_tick=self.tick
        return self

    def __exit__(self,*args):
        self.net.run_tick=self.original

    def arrays(self):
        return dict(terminal_ids=np.array(self.terminal_ids),terminal_initial=self.initial,
            terminal_info=np.asarray(self.releases),context_terminal=np.asarray(self.context_trace),
            context_before_dtype=np.asarray(self.before_dtypes,dtype=np.int8),
            retrograde_events=np.asarray(self.events,dtype=float).reshape(-1,7),
            retrograde_offsets=np.asarray(self.offsets,dtype=np.int64))


def configure(enabled=False,width=192):
    if type(width) is not int or not 1<=width<=192:raise ValueError('Need 1..192 targets')
    cfg=dict(metadata={},global_params=dict(num_inputs=2,num_neuromodulators=2),
        simulation_params=dict(max_history=1),neurons=[],synaptic_points=[],connections=[],external_inputs=[])
    for nid in range(1,width+2):
        n=base.k.neuron(nid,lam=2,c=3,eta_post=1e-7,eta_retro=1e-7,delta_decay=.99,
            meta=dict(graded_gain=1.,bounded_plasticity=True,retrograde_magnitude_error=enabled))
        n['params']['num_inputs']=2;cfg['neurons'].append(n)
        cfg['synaptic_points'].extend([base.k.term(nid),base.k.syn(nid,0,1.,adapt=[0.,0.]),
            base.k.syn(nid,1,0. if nid==1 else -4.,adapt=[0.,0.])])
        cfg['external_inputs'].append(base.k.ext(nid,0))
        if nid>1:cfg['connections'].append(base.k.conn(1,nid,1))
    return cfg


def stimulus(tick,width):
    # Sustained prefix spans the old failure. Then withdrawal and varying drive
    # distinguish working conditional inhibition from a permanently silent bank.
    gate=1. if tick<12000 else (0. if tick<13000 else .75+.25*np.sin((tick-13000)/37.))
    target=.25+.125*np.sin(tick/23.+np.arange(width)*2*np.pi/width)
    return np.r_[gate,target]


def run(output,enabled=False,ticks=16000,width=192):
    output=Path(output).resolve()
    if not 1<=ticks<=16000:raise ValueError('Need 1..16000 ticks')
    if output.exists():raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free<3*1024**3:raise OSError('Need 3 GiB reserve')
    output.mkdir();cfg=configure(enabled,width)
    (output/'config.json').write_text(base.encode(cfg)+'\n')
    net,_,neurons,_=base.fresh(output/'config.json',11,MagnitudeRetrogradeNeuron)
    hashes=base.fingerprint();hashes[str(Path(__file__).resolve())]=base.digest(__file__)
    (output/'manifest.json').write_text(base.encode(dict(enabled=enabled,ticks=ticks,width=width,
        source_hashes=hashes,cell_fields=base.FIELDS,seed=11,
        context_columns=['before','after','return_count','arriving_release'],
        return_columns=['tick','source_neuron','source_port','info','plast','mod0','mod1'],
        limits=__doc__))+'\n')
    began=time.perf_counter();chunks=[]
    for start in range(0,ticks,512):
        cells=[];drive=[];weights=[]
        with ReleaseObserver(net,1) as observer:
            for t in range(start,min(start+512,ticks)):
                values=stimulus(t,width)
                for n,v in zip(neurons,values):net.set_external_input(n.id,0,float(v))
                net.run_tick();cells.append(base.cellular(neurons));drive.append(values)
                weights.append([[p.u_i.info for p in n.postsynaptic_points.values()] for n in neurons])
        data=dict(cells=np.asarray(cells),drive=np.asarray(drive),weights=np.asarray(weights),
            neuron_ids=np.array([n.id for n in neurons]),**observer.arrays())
        if any(not np.isfinite(a).all() for a in data.values()):raise ValueError('Nonfinite record')
        name=f'ticks-{start:05d}.npz';np.savez_compressed(output/name,**data)
        chunks.append(dict(file=name,sha256=base.digest(output/name),start=start,ticks=len(cells)))
        if shutil.disk_usage(output).free<3*1024**3:raise OSError('Storage reserve reached')
        print(base.encode(dict(ticks=net.current_tick,enabled=enabled,seconds=time.perf_counter()-began)),flush=True)
    if any(base.digest(p)!=h for p,h in hashes.items()):raise ValueError('Source changed during run')
    result=dict(chunks=chunks,executed_ticks=ticks,seconds=time.perf_counter()-began)
    (output/'summary.json').write_text(base.encode(result)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    p.add_argument('--enabled',action='store_true');p.add_argument('--ticks',type=int,default=16000)
    a=p.parse_args();run(a.output,a.enabled,a.ticks)
