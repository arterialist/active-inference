"""Combine already equation-audited crossed assignments across both orders.

Full channel trajectories are retained. Physical spectral projection is an
offline observer, never accuracy, significance or a learned neural decoder.
Counterbalancing removes sequence marginals, not arbitrary nonlinear history.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .association_route_probe import digest
from .composition_probe import encode
from .predictive_bridge_audit import windows
from .media_order_control import protocol


def combine(orders):
    if len(orders)!=2: raise ValueError('Need both presentation orders')
    axis=np.asarray(orders[0]['physical_profile_axis'])
    if axis.ndim!=1 or not np.isfinite(axis).all() or axis@axis<=0:
        raise ValueError('Invalid physical reference')
    if not np.array_equal(axis,orders[1]['physical_profile_axis']):
        raise ValueError('Different physical reference axes')
    result={'physical_profile_axis':axis}
    for layer in ('prediction','consumer'):
        a,b=[np.asarray(o[layer+'_assignment_by_cue_interaction']) for o in orders]
        if a.shape!=b.shape or a.ndim!=2 or not np.isfinite([a,b]).all():
            raise ValueError('Mismatched or nonfinite channel trajectories')
        result[layer+'_balanced']=(a+b)/2
        result[layer+'_order_difference']=b-a
    p=np.stack([o['prediction_assignment_by_cue_interaction']@axis/(axis@axis) for o in orders])
    result.update(projection_by_order=p,projection_balanced=p.mean(axis=0))
    return result


def run(order0,order1,output):
    roots=[Path(p).resolve() for p in (order0,order1)]
    arrays=[]; sources=[]; reference=None
    for order,root in enumerate(roots):
        report=json.loads((root/'summary.json').read_text())
        if not report['valid'] or report['max_equation_residual']!=0:
            raise ValueError('Source equation audit did not pass exactly')
        path=root/'effects-per-tick.npz'
        if digest(path)!=report['effects_sha256']: raise ValueError('Effect record changed')
        history_roots=sorted({Path(row['file']).parent for row in report['sources']})
        if len(history_roots)!=2: raise ValueError('Need two assignment histories per order')
        mappings=[]
        for history in history_roots:
            m=json.loads((history/'manifest.json').read_text());mappings.append(m['mapping'])
            if m['order']!=order or m['trials']!=protocol(300,m['repeats'],m['mapping'],order):
                raise ValueError('Unexpected order or physical course')
            signature=(digest(history/'config.json'),m['physical_sources'],m['bridge_seed'],m['repeats'])
            if reference is None: reference=signature
            elif signature!=reference: raise ValueError('Graph, media, seed or acquisition differs')
        if sorted(mappings)!=['paired','swapped']: raise ValueError('Missing assignment control')
        for row in report['sources']:
            if digest(row['file'])!=row['sha256']: raise ValueError('Raw source changed')
        with np.load(path) as z:arrays.append({k:z[k] for k in z.files})
        sources.append(dict(path=str(root),summary_sha256=digest(root/'summary.json'),effects_sha256=digest(path)))
    effects=combine(arrays)
    output=Path(output).resolve();output.mkdir(exist_ok=False)
    np.savez_compressed(output/'effects-per-tick.npz',**effects)
    p=effects['projection_by_order'];balanced=effects['projection_balanced']
    result=dict(sources=sources,graph_seed=reference[2],orders=[0,1],
        both_positive_ticks=np.flatnonzero(np.all(p>0,axis=0)).tolist(),
        sign_disagreement_ticks=np.flatnonzero(p[0]*p[1]<0).tolist(),
        zero_both_ticks=np.flatnonzero(np.all(p==0,axis=0)).tolist(),
        per_order=[windows(row) for row in p],balanced=windows(balanced),
        effects_sha256=digest(output/'effects-per-tick.npz'),
        limits='One base graph and two clips. Ticks are correlated, not independent replications. '
        'Order balance is not arbitrary-history control, semantic recall, useful neural action '
        'or proof that a recurrent generative regime is learned.')
    (output/'summary.json').write_text(encode(result)+'\n')
    print(encode({k:v for k,v in result.items() if k not in ('per_order','balanced','sources')}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('order0','order1','output'):p.add_argument('--'+key,type=Path,required=True)
    run(**vars(p.parse_args()))
