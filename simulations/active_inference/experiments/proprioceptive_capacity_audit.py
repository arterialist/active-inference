"""Conditional output envelope for the recorded motor-context interface.

This is an offline diagnostic, not a decoder or fitting step in the brain.
It gives every contextual synapse its maximum permitted weight, bounds the
initial membrane by PAULA's global cap, and propagates a conservative envelope.
It cannot prove impossibility under changed upstream dynamics or new circuitry.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .association_route_probe import digest
from .composition_probe import encode
from .proprioceptive_learning_audit import intervals


def envelope(arrivals,lam,decay,cap):
    if np.any(arrivals<0) or np.any(lam<1):raise ValueError('Envelope needs nonnegative inputs and nonnegative retention')
    upper=np.full(arrivals.shape[1],1000.)
    rows=[upper.copy()]
    for t in range(1,len(arrivals)):
        # All contributions are nonnegative. Give each multiply/add eight
        # machine epsilons of upward slack per input, then guard integration.
        current=decay*cap*arrivals[t-1].astype(float).sum(axis=1)
        margin=8*np.finfo(np.float32).eps*arrivals.shape[2]*np.maximum(1.,current)
        current+=margin
        rounding=8*np.finfo(np.float32).eps*np.maximum.reduce([np.ones_like(upper),upper,current])
        upper=(1-1/lam)*upper+current/lam+rounding
        rows.append(upper.copy())
    return np.array(rows)


def run(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    m=json.loads((source/'manifest.json').read_text())
    cfg=json.loads((Path(m['parent'])/'config.json').read_text())
    with np.load(source/'closed-loop.npz') as z:data={k:z[k] for k in z.files}
    nodes={n['id']:n for n in cfg['neurons']};ps=m['bridge']['prediction']
    points={(p['neuron_id'],p['synapse_id']):p for p in cfg['synaptic_points'] if p['type']=='postsynaptic'}
    for n in ps:
        node=nodes[n];ports=node['metadata']['prediction_ports']
        for sid in range(node['params']['num_inputs']):
            p=points[n,sid]
            if p['distance_to_hillock']!=1 or p['u_i']['plast']!=0:
                raise ValueError('Unsupported propagation or plastic throughput')
            if sid not in ports and p['u_i']['info']!=0:
                raise ValueError('Unbounded extra current path')
        if node['metadata']['graded_gain']!=1 or node['metadata'].get('graded_S0',0)!=0:
            raise ValueError('Unsupported release transformation')
    lam=np.array([nodes[n]['params']['lambda_param'] for n in ps])
    decay=np.array([nodes[n]['params']['delta_decay'] for n in ps])
    cap=np.array([nodes[n]['metadata']['prediction_cap'] for n in ps])
    if np.any(data['weights']>cap[None,:,None]) or np.any(data['weights']<0):
        raise ValueError('Actual weights violate assumed bounds')
    upper=envelope(data['arrivals'],lam,decay,cap)
    ix={int(n):i for i,n in enumerate(data['neuron_ids'])}
    actual=data['cells'][:,[ix[n] for n in ps],1]
    if np.any(actual>upper):raise ValueError('Envelope fails to contain recorded predictor')
    target=data['cells'][:,[ix[n] for n in m['motor']['joint_position']],1]
    exceeds=target>upper+.01
    exceeds[:64]=False
    output.mkdir(exist_ok=False)
    np.savez_compressed(output/'envelope-per-tick.npz',ticks=data['ticks'],upper=upper,
        actual=actual,target=target,exceeds=exceeds)
    report=dict(source=str(source),raw_sha256=digest(source/'closed-loop.npz'),
        exceeds_ticks_per_channel=exceeds.sum(axis=0),
        exceeds_intervals=[intervals(exceeds[:,i]) for i in range(len(ps))],
        criterion='Target receptor output exceeds the conservative maximum contextual prediction by .01, after index63.',
        scope='Conditional on recorded upstream releases, configured caps, one-tick dendrites and this linear graded predictor. '
        'Not a theorem about all PAULA networks or a demand for zero instantaneous sensory error. '
        'This distinguishes insufficient temporal context from insufficient weight training.')
    (output/'summary.json').write_text(encode(report)+'\n');print(encode(report),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    run(**vars(p.parse_args()))
