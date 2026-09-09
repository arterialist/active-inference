"""Audit weight-only recall probes, including a within-cell weight shuffle.

Preserve complete channel-by-tick effects. A positive physical-profile
projection remains an observer-dependent mechanistic effect, not a semantic
label, sound reconstruction, or independent graph replication.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .association_route_probe import digest
from .composition_probe import encode
from .predictive_bridge_probe import audit_record
from .predictive_bridge_audit import windows
from .predictive_weight_transplant import arrange_weights


def interactions(records,axis):
    axis=np.asarray(axis,dtype=float);norm=float(axis@axis)
    if axis.ndim!=1 or not np.isfinite(axis).all() or norm<=0:
        raise ValueError('Undefined physical profile contrast')
    result={'physical_profile_axis':axis}
    for layer in ('prediction','consumer'):
        for control in ('birth','shuffled'):
            effects={}
            for mapping in ('paired','swapped'):
                for cue in (0,1):
                    effect=records[mapping,'learned',cue,layer]-records[mapping,control,cue,layer]
                    result[f'{mapping}_cue{cue}_{layer}_minus_{control}']=effect
                    effects[mapping,cue]=effect
                    if layer=='prediction':
                        sign=(1 if mapping=='paired' else -1)*(1 if cue==0 else -1)
                        result[f'{mapping}_cue{cue}_minus_{control}_signed_projection']=sign*(effect@axis)/norm
            effect=(effects['paired',0]-effects['swapped',0])-(effects['paired',1]-effects['swapped',1])
            result[f'{layer}_minus_{control}_assignment_by_cue']=effect
            if layer=='prediction':result[f'minus_{control}_projection']=effect@axis/norm
    return result


def run(paired,swapped,output):
    records={};sources=[];reference=None;maximum_residual=0.;physical=None
    for mapping,root in (('paired',Path(paired).resolve()),('swapped',Path(swapped).resolve())):
        m=json.loads((root/'manifest.json').read_text());source=Path(m['source'])
        parent=json.loads((source/'manifest.json').read_text());cfg=json.loads((source/'config.json').read_text())
        if m['mapping']!=mapping or m['source_manifest_sha256']!=digest(source/'manifest.json'):
            raise ValueError('Source assignment changed')
        signature=(digest(source/'config.json'),parent['physical_sources'],m['order'],m['shuffle_seed'])
        if reference is None:reference=signature;physical=parent['physical_sources']
        elif signature!=reference:raise ValueError('Unmatched birth graph, media, order or shuffle')
        for p,h in {**parent['source_hashes'],**m['source_hashes'],**physical}.items():
            if digest(p)!=h:raise ValueError('Runtime, checkpoint or media changed')
        done=json.loads((root/'completion.json').read_text())
        if len(done['entries'])!=6:raise ValueError('Incomplete transplant course')
        index={n['id']:i for i,n in enumerate(cfg['neurons'])}
        initial_weights={}
        for row in done['entries']:
            path=root/row['file']
            if digest(path)!=row['sha256']:raise ValueError('Raw transplant record changed')
            with np.load(path) as z:data={k:z[k] for k in z.files}
            maximum_residual=max(maximum_residual,audit_record(data,cfg,m['bridge']))
            sources.append(dict(file=str(path),sha256=row['sha256']))
            condition,cue=row['condition'],row['clip']
            if condition=='birth_control':continue
            initial_weights[condition,cue]=data['start_weights']
            baseline=source/f'probe-initial-{cue}.npz'
            if digest(baseline)!=row['initial_reference_sha256']:raise ValueError('Birth reference changed')
            with np.load(baseline) as z:
                for key,value in data.items():
                    if key.startswith('start_') and key not in ('start_weights','start_incoming_info'):
                        if not np.array_equal(value,z[key]):raise ValueError(f'Unexpected inherited activity: {key}')
                for layer,role in (('prediction','prediction'),('consumer','prediction_consumer')):
                    indices=[index[n] for n in m['bridge'][role]]
                    records[mapping,condition,cue,layer]=data['cells'][:,indices,1]
                    records[mapping,'birth',cue,layer]=z['cells'][:,indices,1]
        q=initial_weights['learned',0]
        if not np.array_equal(q,initial_weights['learned',1]):raise ValueError('Cue-specific weight injection')
        expected=arrange_weights(q,shuffled=True,seed=m['shuffle_seed'])
        for cue in (0,1):
            if not np.array_equal(expected,initial_weights['shuffled',cue]):raise ValueError('Unexpected weight shuffle')
    spectra=[]
    for p in sorted(physical):
        with np.load(p) as z:spectra.append(z['auditory'].mean(axis=0))
    effects=interactions(records,spectra[0]-spectra[1])
    output=Path(output).resolve();output.mkdir(exist_ok=False)
    np.savez_compressed(output/'effects-per-tick.npz',**effects)
    result=dict(sources=sources,max_equation_residual=maximum_residual,order=reference[2],shuffle_seed=reference[3],
        per_cue={k:windows(v) for k,v in effects.items() if k.endswith('signed_projection')},
        assignment_by_cue={name:windows(effects[f'minus_{name}_projection']) for name in ('birth','shuffled')},
        effects_sha256=digest(output/'effects-per-tick.npz'),
        scope='Only acquired contextual weights transfer. Original activity and all other '
        'parameters remain at birth. Positive adaptation continues. One graph seed, one '
        'permutation seed and two familiar clips; no category generalization or useful action claim.')
    (output/'summary.json').write_text(encode(result)+'\n')
    print(encode({k:v for k,v in result.items() if k not in ('sources','per_cue')}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('paired','swapped','output'):p.add_argument('--'+key,type=Path,required=True)
    run(**vars(p.parse_args()))
