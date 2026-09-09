"""Neural temporal contrast consumer of recorded individual PAULA outputs.

Thirty-two graded PAULA integrators feed a graded mean-inhibition cell and
32 ordinary spiking contrast cells. Input is the recorded individual source
spike history, not a decoded class, spike count or confidence. This isolated
boundary replay does not reproduce feedback into the original source terminals.
"""
import argparse
from copy import deepcopy
import gzip
import inspect
import json
from pathlib import Path
import time
import numpy as np

from .composition_probe import encode, fingerprint, snapshot, k
from .association_balance_audit import checked
from .association_route_probe import digest
from .multimodal_pairing_probe import fresh
from neuron.extensions.graded import GradedNeuron

CONDITIONS = ('evidence_slow', 'evidence_fast', 'saturated_slow')


def consumer_config(condition):
    if condition not in CONDITIONS: raise ValueError(condition)
    neurons=[];points=[];edges=[];external=[]
    for nid in range(1,66):
        integrator=nid<=32;pool=nid==33;graded=integrator or pool
        n=k.neuron(nid,r=.5 if graded else .01,b=.5 if graded else .01,
            c=3,lam=1. if not integrator or condition=='evidence_fast' else 16.,
            delta_decay=1.,eta_post=1e-6,eta_retro=1e-7,
            meta=dict(role='integrator' if integrator else 'mean_inhibition' if pool else 'contrast',
                      graded_gain=1. if graded else 0.,graded_S0=0.,graded_max=0.))
        n['params']['num_inputs']=6 if integrator else 32 if pool else 2
        neurons.append(n);points.append(k.term(nid,mod=[0.,0.]))
        if integrator:
            for sid in range(6):
                points.append(k.syn(nid,sid,1/6,0,adapt=[0.,0.]));external.append(k.ext(nid,sid))
        elif pool:
            for sid,src in enumerate(range(1,33)):
                points.append(k.syn(nid,sid,1/32,0,adapt=[0.,0.]));edges.append(k.conn(src,nid,sid))
        else:
            points.extend((k.syn(nid,0,8.,1,adapt=[0.,0.]),k.syn(nid,1,-8.,0,adapt=[0.,0.])))
            edges.extend((k.conn(nid-33,nid,0),k.conn(33,nid,1)))
    return dict(metadata=dict(preparation='neural-evidence-consumer-v0',condition=condition),
        global_params=dict(num_inputs=1,num_neuromodulators=2),simulation_params=dict(max_history=1),
        neurons=neurons,synaptic_points=points,connections=edges,external_inputs=external)


def record(net, tape):
    if tape.ndim!=2 or tape.shape[1]!=192 or not np.isin(tape,[0,1]).all():raise ValueError('Need individual source spike channels')
    members=list(net.network.neurons.values());rows=[];captured=[];order=[]
    original=GradedNeuron.tick
    def observed(n,ext,tick,dt=1.):
        if net.network.neurons[n.id] is not n:return original(n,ext,tick,dt)
        ports=list(n.postsynaptic_points.values())
        captured.append((n.input_buffer[:,0].copy(),[p.u_i.info for p in ports]))
        order.append(n.id)
        return original(n,ext,tick,dt)
    GradedNeuron.tick=observed
    weights=[];terminals=[]
    try:
        for t in range(len(tape)):
            # One cleft tick at the recorded module boundary. Every source
            # record used here ends in silence, so no input packet crosses a
            # trial boundary. This is an isolated stimulus replay, not a body.
            if t:
                for index in np.flatnonzero(tape[t-1]):net.set_external_input(int(index)//6+1,int(index)%6,1.)
            net.run_tick()
            rows.append([[n.S,n.O,n.F_avg,*n.M_vector,n.r,n.b,n.t_ref] for n in members])
            weights.append([p.u_i.info for n in members for p in n.postsynaptic_points.values()])
            terminals.append([next(iter(n.presynaptic_points.values())).u_o.info for n in members])
    finally:GradedNeuron.tick=original
    if tape[-1].any():raise ValueError('Pending boundary input requires an explicit continuation driver')
    if order!=list(range(1,66))*len(tape):raise AssertionError('Unexpected neuron order')
    incoming=[];before=[]
    for start in range(0,len(captured),65):
        incoming.append(np.concatenate([x[0] for x in captured[start:start+65]]))
        before.append(np.concatenate([x[1] for x in captured[start:start+65]]))
    data=dict(states=np.asarray(rows),incoming=np.asarray(incoming),before=np.asarray(before),
              after=np.asarray(weights),terminal_info=np.asarray(terminals),source=tape.astype(np.uint8))
    if any(not np.isfinite(v).all() for v in data.values()):raise ValueError('Nonfinite recording')
    return data


def source_tape(raw, condition):
    cells=raw['states']
    if cells.shape[1:]!=(368,8):raise ValueError('Requires recorded 368-cell preparation')
    if condition=='saturated_slow':
        # Coordinate-preserving fan-out to six equal ports. Normalized input
        # conductance is one, as for the six actual evidence cells. This is a
        # unit-amplitude spike-channel control, not an outgoing-release replay.
        return np.repeat(cells[:,64:96,1]>0,6,axis=1)
    return cells[:,176:,1]>0


def run(source, output):
    source,output=Path(source).resolve(),Path(output).resolve()
    m=json.loads((source/'manifest.json').read_text());s=json.loads((source/'summary.json').read_text())
    if not s['baseline_exact'] or m['architecture']!='diverse':raise ValueError('Need completed diverse evidence preparation')
    for p,h in m['source_hashes'].items():
        if digest(p)!=h:raise ValueError('Source runtime changed')
    output.mkdir(parents=True,exist_ok=False)
    hashes=fingerprint();hashes[str(Path(__file__).resolve())]=digest(__file__)
    nets={};training=[];probes=[];starts={};started=time.perf_counter()
    for condition in CONDITIONS:
        cfg=consumer_config(condition);path=output/f'{condition}.json';path.write_text(encode(cfg)+'\n')
        nets[condition]=fresh(path,m['seed'],GradedNeuron)[0]
    manifest=dict(source=str(source),seed=m['seed'],mapping=m['mapping'],masks=m['masks'],
        conditions=CONDITIONS,source_hashes=hashes,
        limits='Isolated unit-amplitude replay of individual recorded neural outputs. Existing graded extension is phenomenological; its native timing rule depresses inputs because it has no spikes. Positive weak adaptation remains active. No learned upper association, source-terminal feedback, complete-brain or body integration is tested.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    for i,item in enumerate(s['training']):
        with np.load(checked(source,item)) as z:raw=dict(states=z['states'])
        for condition,net in nets.items():
            data=record(net,source_tape(raw,condition));name=f'{condition}-train-{i:03d}.npz'
            np.savez_compressed(output/name,**data);training.append(dict(condition=condition,index=i,file=name,sha256=digest(output/name)))
        cp=i+1
        if cp not in (32,128):continue
        for condition,net in nets.items():
            parent=encode(snapshot(net));name=f'{condition}-{cp}-start.json.gz'
            with gzip.open(output/name,'wt') as f:f.write(parent+'\n')
            starts[f'{condition}/{cp}']=dict(file=name,sha256=digest(output/name))
            for item in s['probes']:
                if item['checkpoint']!=cp:continue
                with np.load(checked(source,item)) as z:raw=dict(states=z['states'])
                branch=deepcopy(net);data=record(branch,source_tape(raw,condition))
                name=f'{condition}-{cp}-{item["state"]}-{item["case"].replace("/","--")}.npz'
                np.savez_compressed(output/name,**data)
                probes.append(dict(condition=condition,checkpoint=cp,case=item['case'],state=item['state'],file=name,sha256=digest(output/name),source_file=item))
            if encode(snapshot(net))!=parent:raise AssertionError('Probe changed consumer acquisition')
        print(encode(dict(checkpoint=cp,seconds=time.perf_counter()-started)),flush=True)
    if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('Runtime changed')
    result=dict(training=training,probes=probes,starts=starts,seconds=time.perf_counter()-started,
                ticks=3*s['ticks'])
    (output/'summary.json').write_text(encode(result)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
