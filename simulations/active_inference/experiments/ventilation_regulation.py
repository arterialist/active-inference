"""Compose acquired sensorimotor learning with neural organ feedback.

Continue the actual return/current checkpoint at tick 6352. New organs and
seven new cells are declared additions, not retroactive acquisition history.
All old cells, weights, event queues and 64-sample physical history survive.
Organ afferents have a separate 64-sample delay initially filled with the
initial organ measurements. No host deficit comparator or motor selector.

Compare feedback, oxygen-output zero-q, energy-output zero-q and tonic neural
recruitment. All retain the same graph and positive adaptation. This first
course keeps drag .8; it tests composition, not necessity over changing worlds.
"""
import argparse
from collections import deque
from copy import deepcopy
import inspect
import json
from pathlib import Path
import shutil

import numpy as np

from . import context_organization as base
from .active_sweep_memory import restore
from .active_sweep_credit import record_credit
from .crossed_av_continuation import isolated_rng
from ..components.arbitration.ventilation_feedback import append_ventilation_feedback, install_ventilation_feedback, MODES
from ..components.body.ventilation import VentilatedHinge, VentilationOrgans
from ..components.body.energy_budget import EnergyBudget
from ..components.arbitration.energy_feedback import append_energy_feedback
from ..components.motor.sensory_correction import install_on_runtime
from ..core.external_input_state import synchronize_quiescent_external_inputs
from neuron.extensions.experimental.cascade_eligibility import CascadeEligibilityNeuron
from neuron.neuron import RetrogradeSignalEvent


def organ_afferents(body):
    e=body.organs.energy.energy_store
    return np.array([e,1.-e,body.organs.oxygen_ml/body.organs.params.oxygen_capacity_ml])


class OrganDelay:
    def __init__(self,initial):
        a=np.asarray(initial,dtype=float)
        if a.shape!=(64,3) or not np.isfinite(a).all() or np.any((a<0)|(a>1)):
            raise ValueError('Expected sixty-four organ samples')
        self.queue=deque(row.copy() for row in a)
    def state(self): return np.asarray(self.queue)
    def step(self,value):
        self.queue.append(np.asarray(value).copy());return self.queue.popleft()


class CoupledHinge(VentilatedHinge):
    def __init__(self,net,groups):
        super().__init__(drag=.8);self.net=net;self.groups=groups
        self.rows={k:[] for k in ('organs','exchange','muscles')}
    def step(self,command):
        m=np.array([self.net.network.neurons[n].O for n in self.groups['muscle']])
        if command != m[0]-m[1]: raise ValueError('Command is not actual neural muscle difference')
        result=self.step_muscles(m)
        self.rows['organs'].append(self.organs.state())
        self.rows['exchange'].append(self.last_exchange.copy());self.rows['muscles'].append(m)
        return result


class ResourceInputs:
    def __init__(self,net,body,meta,delay):
        self.net,self.body,self.meta,self.delay=net,body,meta,delay
        self.rows={k:[] for k in ('organ_before','organ_raw','organ_drive')}
    def __getattr__(self,name): return getattr(self.net,name)
    def run_tick(self):
        before=self.body.organs.state();raw=organ_afferents(self.body);drive=self.delay.step(raw)
        for nid,v in zip(self.meta['sensory_ids'],drive): self.net.set_external_input(nid,0,float(v))
        for key,v in zip(self.rows,(before,raw,drive)): self.rows[key].append(v.copy())
        return self.net.run_tick()


def record(net,body,physical_delay,organ_delay,features,groups,meta,ticks):
    ids=meta['neurons']+groups['muscle'];chosen=set(ids)
    width=max(net.network.neurons[n].params.num_inputs for n in ids)
    rows={k:[] for k in ('reg_inputs','reg_q_before','reg_q_after','reg_scheduled')}
    queues=[[nid,t,s,float(v)] for nid in ids for t,_,v,s in net.network.neurons[nid].propagation_queue]
    initial=base.cellular(list(net.network.neurons.values()))
    initial_buffers=np.zeros((len(ids),width,4),dtype=np.float32)
    cpg_ids=list(groups['cpg']);cpg_inputs=[]
    cpg_initial=np.zeros((len(cpg_ids),2,4),dtype=np.float32)
    if any(net.network.neurons[n].params.num_inputs!=2 for n in cpg_ids):
        raise ValueError('Expected two observed ports per rhythm cell')
    positions={nid:i for i,nid in enumerate(ids)}
    for signal in net.presynaptic_wheel[net.current_tick%net.wheel_size]:
        if isinstance(signal.event,tuple):
            src,term,value=signal.event
            for target,sid in net.network.connection_cache.get((src,term),()):
                if target in chosen: initial_buffers[positions[target],sid,0]+=value
                if target in cpg_ids: cpg_initial[cpg_ids.index(target),sid,0]+=value
    wrapper=ResourceInputs(net,body,meta,organ_delay)
    old_tick=CascadeEligibilityNeuron.tick;observed={};returns=[];offsets=[0]
    start_organ=body.organs.state();start_delay=organ_delay.state()
    def tick(n,external,t,dt=1.):
        if n.id in cpg_ids:
            if n.id==cpg_ids[0]: cpg_inputs.append([])
            cpg_inputs[-1].append(n.input_buffer.copy())
        if n.id not in chosen: return old_tick(n,external,t,dt)
        inputs=np.zeros((width,4),dtype=np.float32);inputs[:n.params.num_inputs]=n.input_buffer
        before=np.zeros(width);before[:n.params.num_inputs]=[p.u_i.info for p in n.postsynaptic_points.values()]
        events=old_tick(n,external,t,dt)
        after=np.zeros(width);after[:n.params.num_inputs]=[p.u_i.info for p in n.postsynaptic_points.values()]
        scheduled=np.zeros(width,dtype=np.float32)
        for sid,p in n.postsynaptic_points.items():
            if inputs[sid,0]>0: scheduled[sid]=p.potential
        observed[n.id]=(inputs,before,after,scheduled)
        for e in events:
            if isinstance(e,RetrogradeSignalEvent):
                returns.append([e.source_neuron_id,e.source_synapse_id,e.target_neuron_id,e.target_terminal_id,*e.error_vector])
        # Record order explicitly; newly added cells occur after old muscles.
        if n.id==meta['neurons'][-1]:
            for i,key in enumerate(rows): rows[key].append([observed[nid][i] for nid in ids])
            offsets.append(len(returns))
        return events
    CascadeEligibilityNeuron.tick=tick
    try: data=record_credit(wrapper,body,physical_delay,features,groups,ticks)
    finally: CascadeEligibilityNeuron.tick=old_tick
    for d in (rows,wrapper.rows,body.rows): data.update({k:np.asarray(v) for k,v in d.items()})
    data.update(reg_ids=np.array(ids),reg_cells_initial=initial,reg_queues_initial=np.asarray(queues).reshape(-1,4),
        reg_arriving_initial=initial_buffers,organ_initial=start_organ,organ_delay_initial=start_delay,
        organ_delay_final=organ_delay.state(),reg_returns=np.asarray(returns).reshape(-1,8),
        cpg_ids=np.array(cpg_ids),cpg_inputs=np.asarray(cpg_inputs),cpg_arriving_initial=cpg_initial,
        reg_return_offsets=np.asarray(offsets),intervention=np.array([1.,1.,1.]))
    return data


def assemble(parent,mode,output):
    m=json.loads((parent/'manifest.json').read_text());row=next(r for r in m['rows'] if r['kind']=='current')
    if row['end']!=6352: raise ValueError('Expected actual current state after return')
    for key,h in (('checkpoint','checkpoint_sha256'),('physical','physical_sha256')):
        if base.digest(parent/row[key])!=row[h]: raise ValueError('Acquired checkpoint changed')
    original=json.loads(Path(m['config']).read_text());g=m['groups']
    cfg,meta=append_ventilation_feedback(original,g,mode=mode)
    path=output/'config.json';path.write_text(base.encode(cfg)+'\n')
    net,oldbody,delay=restore(parent/row['checkpoint'],parent/row['physical'])
    with isolated_rng(): fresh=base.fresh(path,m['seed'],CascadeEligibilityNeuron)[0]
    old_cells=dict(net.network.neurons)
    install_ventilation_feedback(net,fresh,original,cfg)
    if not all(net.network.neurons[n] is cell for n,cell in old_cells.items()):
        raise ValueError('Old cell was replaced')
    body=CoupledHinge(net,g);body.restore(oldbody.state(),next_gate=oldbody.next_gate,crossings=oldbody.crossings)
    organ_delay=OrganDelay(np.tile(organ_afferents(body),(64,1)))
    return net,body,delay,organ_delay,cfg,meta,m


def run(parent,output,mode='feedback',ticks=1024):
    from .ventilation_regulation_audit import audit
    parent,output=Path(parent).resolve(),Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    if ticks not in (128,1024,2048) or mode not in MODES: raise ValueError('Undeclared course')
    if shutil.disk_usage(output.parent).free<3*1024**3: raise OSError('Need 3 GiB reserve')
    output.mkdir();net,body,delay,organ_delay,cfg,meta,m=assemble(parent,mode,output)
    sources={str(parent/'manifest.json'):base.digest(parent/'manifest.json')}
    for p,h in m['sources'].items():
        if Path(p).exists() and base.digest(p)!=h: raise ValueError('Changed source: '+p)
    # Retain source identities without requiring deleted, unrelated prior branches.
    sources.update({p:h for p,h in m['sources'].items() if Path(p).exists()})
    for obj in (run,audit,append_ventilation_feedback,install_ventilation_feedback,VentilatedHinge,
                EnergyBudget,append_energy_feedback,install_on_runtime,record_credit,
                synchronize_quiescent_external_inputs):
        p=str(Path(inspect.getfile(obj)).resolve());sources[p]=base.digest(p)
    row=next(r for r in m['rows'] if r['kind']=='current')
    for key in ('checkpoint','physical'):
        p=str(parent/row[key]);sources[p]=base.digest(p)
    protocol=dict(parent=str(parent),mode=mode,seed=m['seed'],ticks=ticks,start=net.current_tick,
                  groups=m['groups'],meta=meta,sources=sources,limits=__doc__,cell_fields=base.FIELDS,
                  organ_fields=VentilationOrgans.fields,exchange_fields=VentilationOrgans.exchange_fields)
    (output/'protocol.json').write_text(base.encode(protocol)+'\n')
    with np.load(m['media']) as z: features={k:z[k] for k in ('visual','auditory')}
    base.save_checkpoint(net,output/'initial.paula',sources=list(sources))
    np.savez_compressed(output/'initial-body.npz',state=body.state(),delay=delay.state(),
        organ=body.organs.state(),organ_delay=organ_delay.state(),gate=[body.crossings,body.next_gate])
    data=record(net,body,delay,organ_delay,features,m['groups'],meta,ticks)
    path=output/'ticks.npz';np.savez_compressed(path,**data)
    base.save_checkpoint(net,output/'final.paula',sources=list(sources))
    np.savez_compressed(output/'final-body.npz',state=body.state(),delay=delay.state(),
        organ=body.organs.state(),organ_delay=organ_delay.state(),gate=[body.crossings,body.next_gate])
    # Save before auditing so a failed audit leaves the actual evidence available.
    residual=audit(data,cfg,m['groups'],features)
    if any(base.digest(p)!=h for p,h in sources.items()): raise ValueError('Source changed during run')
    failures={name:np.flatnonzero(data['organs'][:,col]>1e-12).tolist()
              for name,col in [('oxygen',4),('energy',10)]}
    result=dict(protocol,sha256=base.digest(path),audited_ticks=ticks,max_neural_residual=residual,
                deficits=failures,final_organs=data['organs'][-1],neurons=len(net.network.neurons))
    (output/'manifest.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(seed=m['seed'],mode=mode,ticks=ticks,residual=residual,
        first_deficit={k:(v[0] if v else None) for k,v in failures.items()},final_organs=data['organs'][-1])),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--mode',choices=MODES,default='feedback');p.add_argument('--ticks',type=int,default=1024)
    a=p.parse_args();run(a.parent,a.output,a.mode,a.ticks)
