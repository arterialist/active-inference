"""A necessary representational check for a positive, fixed-weight readout.

For two cue-mean vectors x0,x1 and q_i >= 0, the output-mean pair
(q dot x0, q dot x1) must lie in the positive cone of (x0_i,x1_i).
We project each requested auditory mean pair onto that two-dimensional cone.
No weights are fitted into or supplied to the brain. This optimistic family
even permits unbounded q, so a residual cannot be repaired by a tighter cap.

Scope is explicit: static nonnegative linear decoding of declared mean physical
features, or steady mean responses of linear unity-DC filters with held weights.
This is NOT a bound on the complete adapting nonlinear PAULA brain, finite-time
transients, or a richer recurrent/contextual code. Its purpose is to expose a
missing representation before tuning learning rates or duplicating predictors.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .association_route_probe import digest
from .composition_probe import encode


def cone_projection(points, target):
    points=np.asarray(points,dtype=float); target=np.asarray(target,dtype=float)
    if (points.ndim!=2 or points.shape[1]!=2 or target.shape!=(2,) or
            not np.isfinite(points).all() or not np.isfinite(target).all() or
            np.any(points<0) or np.any(target<0)):
        raise ValueError('Need nonnegative finite two-cue features and target')
    keep=np.flatnonzero(np.any(points>0,axis=1))
    if not len(keep) or not np.any(target):
        return np.zeros(2), np.zeros(len(points))
    angles=np.arctan2(points[keep,1],points[keep,0]); a=int(keep[np.argmin(angles)]); b=int(keep[np.argmax(angles)])
    theta=np.arctan2(target[1],target[0])
    witness=np.zeros(len(points))
    if a!=b and angles.min()<theta<angles.max():
        coefficients=np.linalg.solve(points[[a,b]].T,target)
        if np.any(coefficients < -1e-12): raise ArithmeticError('Invalid positive-cone witness')
        witness[[a,b]]=np.maximum(coefficients,0)
        return points.T@witness,witness
    candidates=[]
    for i in (a,b):
        scale=max(0.,float(target@points[i]/(points[i]@points[i])))
        candidates.append((float(np.sum((scale*points[i]-target)**2)),i,scale))
    _,i,scale=min(candidates);witness[i]=scale
    return points.T@witness,witness


def run(source, output):
    source,output=Path(source).resolve(),Path(output).resolve()
    m=json.loads((source/'manifest.json').read_text());cfg=json.loads((source/'config.json').read_text())
    features=[]
    for path,h in sorted(m['physical_sources'].items()):
        if digest(path)!=h: raise ValueError('Physical input changed')
        with np.load(path) as z: features.append({k:z[k] for k in ('visual','auditory')})
    visual=np.stack([f['visual'].mean(axis=0) for f in features])
    auditory=np.stack([f['auditory'].mean(axis=0) for f in features])
    vi={nid:i for i,nid in enumerate(m['groups']['vision'])}
    predictors=m['bridge']['prediction']
    columns=[[vi[src] for n,_,src in m['selected'] if n==target] for target in predictors]
    if len(predictors)!=auditory.shape[1]:raise ValueError('Target channels differ')
    predictions={};rows=[];witnesses={}
    for mapping in ('paired','swapped'):
        target=auditory if mapping=='paired' else auditory[::-1]
        fitted=[];weights=[]
        for channel,selected in enumerate(columns):
            points=visual[:,selected].T
            pred,witness=cone_projection(points,target[:,channel]);fitted.append(pred);weights.append(witness)
            residual=float(np.sum((pred-target[:,channel])**2))
            rows.append(dict(mapping=mapping,channel=channel,target=target[:,channel],closest_mean=pred,
                squared_error_lower_bound=residual,nonzero_target=bool(np.any(target[:,channel])),
                feasible_at_1e12=residual<=1e-12))
        predictions[mapping]=np.array(fitted).T;witnesses[mapping]=np.array(weights)
    output.mkdir(exist_ok=False)
    np.savez_compressed(output/'cone-witnesses.npz',visual=visual,auditory=auditory,
        columns=np.array(columns),**{k+'_prediction':v for k,v in predictions.items()},
        **{k+'_weights':v for k,v in witnesses.items()})
    report=dict(source=str(source),source_manifest_sha256=digest(source/'manifest.json'),
        source_config_sha256=digest(source/'config.json'),physical_sources=m['physical_sources'],
        inference='Capacity of static positive readout on physical means, not an impossibility theorem for the adapting brain',
        channels=rows,summary={mapping:dict(infeasible_channels=sum(not r['feasible_at_1e12'] for r in rows if r['mapping']==mapping),
            active_target_channels=sum(r['nonzero_target'] for r in rows if r['mapping']==mapping),
            total_mean_pair_squared_error_lower_bound=sum(r['squared_error_lower_bound'] for r in rows if r['mapping']==mapping))
            for mapping in ('paired','swapped')})
    (output/'summary.json').write_text(encode(report)+'\n')
    print(encode(report['summary']),flush=True);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(**vars(p.parse_args()))
