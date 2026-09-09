"""Reconstruct reference-cell synaptic and somatic dynamics from recorded sources.

Checks weights from local tick 1 and membrane/output from tick 2. Earlier
in-flight history is not inferred: those entries have explicit unchecked masks.
Native float32 potentials and heap accumulation order are reproduced. This is
an observer, never a reference generator supplied to the neural network.
"""
import math

import numpy as np

from . import context_organization as base


def audit_pools(data,config):
    nodes={n['id']:n for n in config['neurons']}
    points={(p['neuron_id'],p['synapse_id']):p for p in config['synaptic_points'] if p['type']=='postsynaptic'}
    ids=list(data['neuron_ids']);terms=list(map(tuple,data['terminal_ids']));length=len(data['cells'])
    residual=np.zeros((length,len(data['pool_ids']),3));checked=np.zeros_like(residual,dtype=bool)
    for pool,nid in enumerate(data['pool_ids']):
        node=nodes[int(nid)];p=node['params'];md=node['metadata']
        if (p['lambda_param']!=2 or p['delta_decay']!=.99 or p['eta_post']!=1e-7
                or md.get('graded_gain')!=1 or md.get('graded_S0',0)!=0 or md.get('graded_max',0)!=0
                or not md.get('bounded_plasticity') or md.get('plasticity_rate_boost',0)!=0):
            raise ValueError('Pool equation differs from declared audit')
        if np.any(data['cells'][:,ids.index(nid),3:5]):raise ValueError('Unexpected pool modulation')
        sources=data['pool_sources'][pool]
        for sid,(source,tid) in enumerate(sources):
            point=points[int(nid),sid]
            if point['distance_to_hillock']!=1 or point['u_i']['plast']!=0:
                raise ValueError('Unexpected pool dendrite')
            edges=[c for c in config['connections'] if c['target_neuron']==nid and c['target_synapse']==sid]
            if len(edges)!=1 or (edges[0]['source_neuron'],edges[0]['source_terminal'])!=(source,tid):
                raise ValueError('Pool source identity differs')
        arrivals=np.zeros((length,len(sources)),dtype=np.float32)
        for sid,(source,tid) in enumerate(sources):
            release=data['cells'][:-1,ids.index(source),base.FIELDS.index('O')]*data['terminal_info'][:-1,terms.index((source,tid))]
            arrivals[1:,sid]=release.astype(np.float32)
        for t in range(1,length):
            before=data['pool_weights'][t-1,pool];expected=before.copy()
            for sid,a in enumerate(arrivals[t]):
                if a>0:
                    # Native local error norm uses float32 scalar arithmetic.
                    error=float(np.linalg.norm(np.array([a-float(before[sid]),np.float32(0),np.float32(0),np.float32(0)])))
                    expected[sid]=float(before[sid])*math.exp(-1e-7*(error+.02))
            residual[t,pool,0]=abs(expected-data['pool_weights'][t,pool]).max()
            checked[t,pool,0]=True
        for t in range(2,length):
            weights=data['pool_weights'][t-2,pool]
            potentials=[(a*(float(w)+0.),sid) for sid,(a,w) in enumerate(zip(arrivals[t-1],weights)) if a>0]
            current=0.
            for potential,_ in sorted(potentials):current+=potential*(.99**1)
            state=np.float32(data['cells'][t-1,ids.index(nid),base.FIELDS.index('S')])
            state+=(1./2.)*(-state+current)
            state=min(1000.,max(-1000.,state));out=max(0.,float(state))
            residual[t,pool,1]=abs(float(state)-data['cells'][t,ids.index(nid),base.FIELDS.index('S')])
            residual[t,pool,2]=abs(out-data['cells'][t,ids.index(nid),base.FIELDS.index('O')])
            checked[t,pool,1:]=True
    if not np.isfinite(residual).all() or np.any(residual[checked]>2e-12):
        where=np.unravel_index(np.argmax(residual),residual.shape)
        raise ValueError(f'Pool intracellular reconstruction differs at {where}: {residual[where]}')
    return dict(residual=residual,checked=checked,pool_ids=data['pool_ids'].copy())
