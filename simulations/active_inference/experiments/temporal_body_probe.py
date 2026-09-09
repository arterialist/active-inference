"""Matched temporal-history populations in the retained embodied brain.

Both conditions append 128 graded history cells and 20 prediction/readout
cells. Short and multiscale variants differ only in history-cell lambda.
No input labels, desired position or host prediction enter the brain.
"""
import argparse
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from .association_route_probe import digest
from .composition_probe import encode,fingerprint
from .multimodal_pairing_probe import fresh
from .proprioceptive_loop_probe import record_loop
from .proprioceptive_learning_audit import verify_physics
from .predictive_bridge_probe import audit_record
from ..components.body.radian_research_rower import RadianResearchRower
from ..components.learning.temporal_basis import append_temporal_basis
from ..components.learning.predictive_bridge import append_predictive_bridge
from ..core.runtime_checkpoint import save_checkpoint
from neuron.extensions.experimental.predictive_receptor import PredictiveReceptorNeuron


def record_history(net,body,motor,bridge,basis,ticks,*,gain=.08):
    cells=[net.network.neurons[n] for n in basis];ids=set(basis)
    maxdelay=max(n.distances[0] for n in cells)
    due=np.zeros((maxdelay+1,len(cells)))
    for j,n in enumerate(cells):
        for t,_,v,s in n.propagation_queue:
            if not 0<=t-net.current_tick<=maxdelay:raise ValueError('Unexpected queued history input')
            due[t-net.current_tick,j]+=v*n.params.delta_decay**n.distances[s]
    start_weights=np.array([n.postsynaptic_points[0].u_i.info for n in cells])
    rows={key:[] for key in ('history_arrivals','history_scheduled','history_weights','history_input_vectors')}
    observed={};original=PredictiveReceptorNeuron.tick
    def observed_tick(n,external_inputs,current_tick,dt=1.):
        if n.id not in ids:return original(n,external_inputs,current_tick,dt)
        vector=n.input_buffer[0].copy()
        arriving=float(n.input_buffer[0,0])
        events=original(n,external_inputs,current_tick,dt)
        observed[n.id]=(arriving,float(n.postsynaptic_points[0].potential) if arriving>0 else 0.,
                         n.postsynaptic_points[0].u_i.info,vector)
        # Neurons execute in allocation order. Capture one complete population row.
        if n.id==basis[-1]:
            for column,key in enumerate(rows):rows[key].append([observed[i][column] for i in basis])
        return events
    PredictiveReceptorNeuron.tick=observed_tick
    try:data=record_loop(net,body,motor,bridge,ticks,gain=gain)
    finally:PredictiveReceptorNeuron.tick=original
    data.update({k:np.asarray(v) for k,v in rows.items()})
    data['history_neuron_ids']=np.array(basis)
    data['start_history_weights']=start_weights;data['start_history_due']=due
    return data


def audit_history(data,cfg,basis):
    if not np.array_equal(data['history_neuron_ids'],basis):raise ValueError('History column mismatch')
    index={int(n):i for i,n in enumerate(data['neuron_ids'])}
    nodes={n['id']:n for n in cfg['neurons']}
    locations=[index[n] for n in basis]
    source=[index[nodes[n]['metadata']['history_source']] for n in basis]
    delay=np.array([nodes[n]['metadata']['history_delay'] for n in basis])
    lam=np.array([nodes[n]['params']['lambda_param'] for n in basis],dtype=np.float32)
    attenuation=np.array([nodes[n]['params']['delta_decay']**d for n,d in zip(basis,delay)],dtype=np.float32)
    eta=np.array([nodes[n]['params']['eta_post'] for n in basis])
    weight_decay=np.array([nodes[n]['metadata'].get('plasticity_magnitude_decay',.02) for n in basis])
    prev=data['start_cells'];terminal=data['start_terminals'];q=data['start_history_weights']
    residual=0.
    def same(a,b,label):
        nonlocal residual
        d=float(np.max(np.abs(a-b)));residual=max(residual,d)
        if not np.allclose(a,b,rtol=0,atol=2e-12):raise ValueError(f'{label}: {d}')
    for t,cells in enumerate(data['cells']):
        arriving=(prev[:,1]*terminal)[source].astype(np.float32)
        same(data['history_arrivals'][t],arriving,'history release delivery')
        vectors=data['history_input_vectors'][t].astype(np.float32)
        same(vectors[:,0],arriving,'history information vector')
        # This preparation has no plastic or modulatory source release.
        # Refuse a broader interpretation rather than assume those channels away.
        same(vectors[:,1:],np.zeros_like(vectors[:,1:]),'history noninformation inputs')
        same(data['history_scheduled'][t],arriving*q.astype(np.float32),'history synaptic current')
        expected_weights=q.copy()
        for j in np.flatnonzero(arriving>0):
            v=vectors[j].copy();v[0]-=np.float32(q[j])
            error=float(np.linalg.norm(v))
            expected_weights[j]=q[j]*math.exp(-float(eta[j])*(error+float(weight_decay[j])))
        same(data['history_weights'][t],expected_weights,'history bounded local plasticity')
        current=np.zeros(len(basis),dtype=np.float32)
        for j,d in enumerate(delay):
            current[j]=data['start_history_due'][t,j] if t<d else np.float32(data['history_scheduled'][t-d,j])*attenuation[j]
        s=prev[locations,0].astype(np.float32)
        expected=s+(1/lam)*(-s+current)
        same(cells[locations,0],expected,'history membrane integration')
        same(cells[locations,1],np.maximum(0.,expected),'history graded release')
        q=data['history_weights'][t];prev=cells;terminal=data['terminals'][t]
    return residual


def run(source,output,*,mode='multiscale',ticks=1024,seed=11):
    source,output=Path(source).resolve(),Path(output).resolve()
    if not 128<=ticks<=2048:raise ValueError('Expected a bounded course')
    if shutil.disk_usage(output.parent).free<2*1024**3:raise OSError('Keep 2 GiB free')
    m=json.loads((source/'manifest.json').read_text())
    original=json.loads((source/'config.json').read_text())
    if len(original['neurons'])!=1793:raise ValueError('Expected retained embodied graph')
    for p,h in m['source_hashes'].items():
        if digest(p)!=h:raise ValueError('Parent runtime changed: '+p)
    motor=m['motor'];oldbridge=deepcopy(original['metadata']['predictive_bridge'])
    cfg,basis=append_temporal_basis(original,motor['cpg']+motor['muscles'],mode=mode)
    cfg,bridge,edges,selected=append_predictive_bridge(cfg,basis,motor['joint_position'],
        seed=seed,fanin=len(basis),consumers=8)
    cfg['metadata']['temporal_predictive_bridge']=cfg['metadata'].pop('predictive_bridge')
    cfg['metadata']['predictive_bridge']=oldbridge
    output.mkdir(exist_ok=False);path=output/'config.json';path.write_text(encode(cfg)+'\n')
    net,_,members,_=fresh(path,seed,PredictiveReceptorNeuron)
    # Preserve acquired audiovisual input weights from the parent birth state.
    oldpoints={}
    for p in original['synaptic_points']:
        if p['type']=='postsynaptic':oldpoints.setdefault(p['neuron_id'],[]).append(p['synapse_id'])
    with np.load(source/'closed-loop.npz') as z:weights=z['start_incoming_info']
    keys=[(n['id'],s) for n in original['neurons'] for s in oldpoints[n['id']]]
    if len(weights)!=len(keys):raise ValueError('Original weight identity mismatch')
    for (n,s),w in zip(keys,weights):net.network.neurons[n].postsynaptic_points[s].u_i.info=float(w)
    hashes=fingerprint()
    for obj in (run,append_temporal_basis,append_predictive_bridge,record_loop,verify_physics,RadianResearchRower):
        p=Path(inspect.getfile(obj)).resolve();hashes[str(p)]=digest(p)
    manifest=dict(source=str(source),source_config_sha256=digest(source/'config.json'),
        source_record_sha256=digest(source/'closed-loop.npz'),mode=mode,seed=seed,
        fields=m['fields'],motor=motor,old_bridge=m['bridge'],bridge=bridge,basis=basis,
        selected=selected,edges=edges,source_hashes=hashes,ticks=ticks,gain=.08,
        limits='One graph, motor-driven course; not autonomous action selection or accepted hierarchy. '
        'Existing graph rebuilt with retained incoming weights; initial and final full runtimes saved. '
        'Short/multiscale match cell count, delays, edges, basal rates, birth conductance and capacity. '
        'Comparison to original predictor also changes fan-in and maximum total conductance.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    save_checkpoint(net,output/'initial.neural-checkpoint',sources=(__file__,))
    started=time.perf_counter();body=RadianResearchRower()
    data=record_history(net,body,motor,bridge,basis,ticks)
    raw=output/'closed-loop.npz';np.savez_compressed(raw,**data)
    history_residual=audit_history(data,cfg,basis)
    prediction_residual=audit_record(data,cfg,bridge)
    physics_residual=verify_physics(data,.08)
    save_checkpoint(net,output/'final.neural-checkpoint',sources=(__file__,))
    if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('Runtime changed')
    result=dict(neurons=len(members),ticks=ticks,mode=mode,seconds=time.perf_counter()-started,
        history_residual=history_residual,prediction_residual=prediction_residual,
        physics_residual=physics_residual,raw_bytes=raw.stat().st_size,raw_sha256=digest(raw),
        weight_change_max=float(np.max(np.abs(data['weights']-data['start_weights']))))
    (output/'summary.json').write_text(encode(result)+'\n');print(encode(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--mode',choices=('short','multiscale'),default='multiscale')
    p.add_argument('--ticks',type=int,default=1024);p.add_argument('--seed',type=int,default=11)
    run(**vars(p.parse_args()))
