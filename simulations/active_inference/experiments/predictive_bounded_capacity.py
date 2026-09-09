"""Offline bounded-readout diagnostic on actual neural arrival amplitudes.

Uses the first two complete audiovisual presentations, one of each cue. No
optimized weights enter any brain. The static mean-map family is conditional
on these recorded histories, not a bound on the adapting dynamical system.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from .association_route_probe import digest
from .composition_probe import encode


def bounded_projection(points, target, cap=1.):
    a=np.asarray(points,dtype=float).T;b=np.asarray(target,dtype=float)
    if (a.ndim!=2 or a.shape[0]!=2 or a.shape[1]==0 or b.shape!=(2,) or
            not np.isfinite(a).all() or not np.isfinite(b).all() or
            np.any(a<0) or np.any(b<0) or not np.isfinite(cap) or cap<=0):
        raise ValueError('Invalid bounded two-cue problem')
    fit=lsq_linear(a,b,bounds=(0.,cap),method='bvls',tol=1e-12,max_iter=1000)
    q=fit.x
    if not fit.success or not np.isfinite(q).all() or np.any(q < -1e-12) or np.any(q > cap+1e-12):
        raise ArithmeticError('No feasible converged capacity witness')
    # BVLS can return a boundary value a few ulps outside its box. Make the
    # witness actually feasible, then recompute BOTH error and tangent bound.
    # Never reuse the optimizer's objective as the certificate after clipping.
    q=np.clip(q,0.,cap)
    predicted=a@q;residual=predicted-b;upper=float(residual@residual)
    gradient=2*a.T@residual
    # Convex tangent plane: f(z)>=f(q)+g.(z-q). Minimize that plane
    # over the box to bound the global minimum independently of solver status.
    lower=max(0.,upper+float(np.sum(gradient*np.where(gradient>=0,-q,cap-q))))
    if upper-lower>1e-9:
        raise ArithmeticError(f'Capacity witness gap {upper-lower} exceeds tolerance')
    return dict(weights=q,closest=predicted,error_lower=lower,error_upper=upper,gap=upper-lower)


def run(raw,contrast,output):
    records={};witnesses={};provenance=[]
    for label,root in (('raw',Path(raw).resolve()),('contrast',Path(contrast).resolve())):
        m=json.loads((root/'manifest.json').read_text());cfg=json.loads((root/'config.json').read_text())
        progress=json.loads((root/'progress.json').read_text())
        rows=[r for r in progress['entries'] if r['phase']=='experience'][:2]
        if len(rows)!=2 or {r['trial']['visual_clip'] for r in rows}!={0,1}:
            raise ValueError('Need complete first presentation of both cues')
        nodes={n['id']:n for n in cfg['neurons']};index={n['id']:i for i,n in enumerate(cfg['neurons'])}
        targets=cfg['metadata']['predictive_bridge']['target_ids']
        x=[];y=[]
        for row in sorted(rows,key=lambda r:r['trial']['visual_clip']):
            path=root/row['file']
            if digest(path)!=row['sha256']:raise ValueError('Neural record changed')
            with np.load(path) as z:
                # Actual predictor-port arrivals, with the predictor's DC dendritic attenuation.
                x.append(z['arrivals'].mean(axis=0)*np.array([nodes[n]['params']['delta_decay'] for n in m['bridge']['prediction']])[:,None])
                ti=[index[n] for n in targets]
                y.append((z['cells'][:,ti,1]*z['terminals'][:,ti]).mean(axis=0))
            provenance.append(dict(file=str(path),sha256=row['sha256'],trial=row['trial']))
        x=np.array(x);y=np.array(y);channels=[];q=[]
        for i,n in enumerate(m['bridge']['prediction']):
            result=bounded_projection(x[:,i,:].T,y[:,i],nodes[n]['metadata']['prediction_cap'])
            q.append(result.pop('weights'));channels.append(result)
        records[label]=dict(channels=channels,
            positive_lower_bounds=sum(r['error_lower']>1e-12 for r in channels),
            total_error_lower=sum(r['error_lower'] for r in channels),
            total_error_upper=sum(r['error_upper'] for r in channels),
            maximum_gap=max(r['gap'] for r in channels))
        witnesses[label+'_weights']=np.array(q);witnesses[label+'_context_means']=x;witnesses[label+'_target_means']=y
    output=Path(output).resolve();output.mkdir(exist_ok=False)
    np.savez_compressed(output/'bounded-witnesses.npz',**witnesses)
    result=dict(records=records,sources=provenance,source_hash=digest(__file__),
        witness_sha256=digest(output/'bounded-witnesses.npz'),
        scope='Conditional static map between observed neural means at the first two audiovisual '
        'presentations, including actual arrival scaling and per-synapse caps. Not exact finite-time '
        'membrane reconstruction, a lower bound for changing weights, or accepted recall. '
        'Raw and contrast histories may differ through their active return pathways. '
        'Optimized witnesses remain offline diagnostic data and are never inserted into a brain.')
    (output/'summary.json').write_text(encode(result)+'\n')
    print(encode({label:{k:v for k,v in r.items() if k!='channels'} for label,r in records.items()}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('raw','contrast','output'):p.add_argument('--'+key,type=Path,required=True)
    run(**vars(p.parse_args()))
