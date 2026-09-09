"""Conditional capacity certificates for the recorded pre-feedback interface.

Offline diagnosis only. No optimizer weights are saved or installed in a brain.
For fixed recorded presynaptic histories and constant selected q, reconstruct
the actual delayed predictor membrane response. Bound the best possible minimum
signed output for q_positive-q_negative in [-1,1]. Counterfactual q can change
upstream return paths, so this is NOT a capacity theorem for the coupled network.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from . import context_organization as base
from .opponent_context_analysis import read_record
from .temporal_verification import verify_learning


def margin_certificate(c):
    """A feasible lower bound and convex-combination upper bound, no trained policy."""
    c=np.asarray(c,dtype=float)
    if c.ndim!=2 or min(c.shape)==0 or not np.isfinite(c).all():
        raise ValueError('Need finite nonempty event-by-input matrix')
    objective=np.zeros(c.shape[1]+1);objective[-1]=-1
    result=linprog(objective,A_ub=np.column_stack((-c,np.ones(len(c)))),
        b_ub=np.zeros(len(c)),bounds=[(-1,1)]*c.shape[1]+[(None,None)],method='highs')
    if not result.success:raise ValueError(f'Conditional capacity solve failed: {result.message}')
    w=np.clip(result.x[:-1],-1,1)
    lower=float((c@w).min())
    lam=np.maximum(0.,-result.ineqlin.marginals)
    if not lam.sum()>0:raise ValueError('Missing dual certificate')
    lam/=lam.sum()
    # For any legal w, min_i C_i w <= sum_i lam_i C_i w
    # <= ||sum_i lam_i C_i||_1. This upper bound needs no fitted readout.
    upper=float(abs(lam@c).sum())
    if lower>upper+1e-9:raise ValueError('Inconsistent numerical certificate')
    return dict(feasible_minimum=lower,upper_bound=upper,numerical_slack=1e-9,
                certificate_events=np.flatnonzero(lam>0).tolist(),
                certificate_mass=lam[lam>0].tolist())


def filtered_context(arrivals):
    out=np.zeros_like(arrivals,dtype=float);state=np.zeros(arrivals.shape[1])
    for t in range(1,len(arrivals)):
        state=.75*state+.25*.99*arrivals[t-1]
        out[t]=state
    return out


def native_predictor(arrivals,q):
    """Reproduce float32 local potentials, heap order and native somatic arithmetic."""
    state=0.;out=[0.]
    for t in range(1,len(arrivals)):
        events=[]
        for sid,a in enumerate(arrivals[t-1]):
            if a>0:events.append((np.float32(a)*(float(q[sid])+0.),sid))
        current=0.
        for potential,_ in sorted(events):current+=potential*(.99**1)
        state+=(1./4.)*(-state+current)
        out.append(float(state))
    return np.array(out)


def rounding_guard(arrays):
    """Conservative two-channel float32 guard on this bounded positive interface.

    Each q lies in [0,1]. Scalar casts, product, attenuation and sum have
    <=n+4 roundings, including weak Python-scalar conversion to float32;
    gamma(n) bounds their accumulated relative error. With I<=Imax and S<=2Imax,
    three somatic operations contribute <=2.75*gamma(3)*Imax per step.
    Summing the .75-leaky recurrence gives gamma(n+4)+11*gamma(3) per channel.
    Add 1e-9 for negligible subnormal/float64 bookkeeping contributions.
    """
    unit=2.**-24
    n=arrays[0].shape[1]
    gamma=lambda k:k*unit/(1-k*unit)
    imax=max(float((.99*a.sum(axis=1)).max()) for a in arrays)
    if n+4>=1/unit or not 0<=imax<500 or any(np.any(a<0) for a in arrays):
        raise ValueError('Rounding guard outside its nonnegative, unclamped domain')
    return 2*imax*(gamma(n+4)+11*gamma(3))+1e-9


def inspect(roots,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    cases=[];traces={};checked=0
    for root in map(lambda p:Path(p).resolve(),roots):
        m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text())
        for p,h in {**m['source_hashes'],**m['physical_sources']}.items():
            if base.digest(p)!=h:raise ValueError('Source changed')
        cfg=json.loads((root/'config.json').read_text())
        nodes={n['id']:n for n in cfg['neurons']}
        points={(p['neuron_id'],p['synapse_id']):p for p in cfg['synaptic_points'] if p['type']=='postsynaptic'}
        for nid,sid,_ in m['selected']:
            n=nodes[nid];point=points[nid,sid]
            if (n['params']['lambda_param']!=4 or n['params']['delta_decay']!=.99
                    or n['metadata']['graded_gain']!=1 or n['metadata'].get('graded_S0',0)!=0
                    or n['metadata']['graded_max']!=0 or point['distance_to_hillock']!=1
                    or point['u_i']['plast']!=0):raise ValueError('Conditional equation no longer matches')
            if n['metadata']['prediction_ports']!=list(range(len(n['metadata']['prediction_ports']))):
                raise ValueError('Native reconstruction requires recorded contiguous port order')
        for weights in ('birth','learned'):
            signed=[];incoming=[];ids0=None;ideal_residual=0.
            for video,audio in ((0,0),(0,1),(1,0),(1,1)):
                row=next(r for r in s['probes'] if (r['kind'],r['weights'],r['presentation'],r['video'],r['audio'])
                         ==('resting',weights,'both',video,audio))
                z=read_record(root,row,m,learning_auditor=verify_learning)
                if np.any(z['weights'][:64]!=z['weights_initial']) or np.any(z['drive'][:64,194:198]):
                    raise ValueError('Selected weights or pre-feedback condition changed')
                neuron_ids=list(z['neuron_ids']);features=[]
                for j,nid in enumerate(m['groups']['prediction']):
                    x=filtered_context(z['arrivals'][:64,j]);features.append(x)
                    predicted=x@z['weights_initial'][j]
                    actual=z['cells'][:64,neuron_ids.index(nid),base.FIELDS.index('O')]
                    ideal_residual=max(ideal_residual,float(abs(predicted-actual).max()))
                    residual=float(abs(native_predictor(z['arrivals'][:64,j],z['weights_initial'][j])-actual).max())
                    if residual>2e-12:raise ValueError(f'Predictor reconstruction failed: {residual}')
                first,second=z['context_source_ids']
                if ids0 is None:ids0=first.copy()
                if not np.array_equal(ids0,first):raise ValueError('Input identity changed')
                reorder=[list(second).index(i) for i in first]
                if not np.array_equal(features[0],features[1][:,reorder]):
                    raise ValueError('Opponent predictors do not receive identical source histories')
                sign=1 if (video^audio^int(m['reverse']))==0 else -1
                signed.append(sign*features[0]);incoming.append(z['arrivals'][:64,0]);checked+=64
            prefix=f's{m["seed"]}_r{int(m["reverse"])}_{weights}'
            # Full per-tick/channel matrix retains the certificate's exact domain.
            traces[prefix]=np.stack(signed)
            for start,stop in ((16,32),(32,64),(16,64)):
                c=np.concatenate([x[start:stop] for x in signed])
                cases.append(dict(key=prefix,seed=m['seed'],reverse=m['reverse'],weights=weights,
                    start=start,stop=stop,source_ids=ids0.tolist(),
                    two_channel_float32_guard=rounding_guard(incoming),
                    observed_ideal_reconstruction_residual=ideal_residual,**margin_certificate(c)))
    if not cases:raise ValueError('No experimental evidence')
    output.mkdir();np.savez_compressed(output/'conditional-inputs.npz',**traces)
    summary=dict(cases=cases,checked_predictor_ticks=2*checked,
        evidence=[dict(root=str(Path(r).resolve()),summary_sha256=base.digest(Path(r)/'summary.json')) for r in roots],
        limits='Fixed factual incoming histories only, including their native adaptation. '
               'Counterfactual selected weights can alter those histories via retrograde paths. '
               'Zero birth neural state, no current bodily evidence, q constant over tested prefix. '
               'No global fit installed. A small margin is not a requirement of perfect compensation.')
    (output/'summary.json').write_text(base.encode(summary)+'\n')
    print(base.encode(dict(cases=len(cases),checked_predictor_ticks=2*checked)),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('roots',type=Path,nargs='+')
    a=p.parse_args();inspect(a.roots,a.output)
