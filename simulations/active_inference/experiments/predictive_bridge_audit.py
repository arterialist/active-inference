"""Re-audit completed predictive-bridge traces and expose weight-causal effects.

Physical sound profiles provide a declared offline axis, not a neural decoder
or category label. Retain the entire channel-by-tick effect alongside that
projection. One order, one graph seed and two recordings remain exploratory.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .association_route_probe import digest
from .composition_probe import encode
from .predictive_bridge_probe import audit_record


def windows(values):
    return [dict(start=a,stop=b,mean=float(values[a:b].mean()),
                 minimum=float(values[a:b].min()),maximum=float(values[a:b].max()),
                 positive=int(np.count_nonzero(values[a:b]>0)),negative=int(np.count_nonzero(values[a:b]<0)))
            for a,b in ((0,32),(32,64),(64,96),(96,128),(128,160),(160,192),(192,224),(224,256),(256,288),(288,300))]


def run(paired,swapped,output):
    roots=[Path(p).resolve() for p in (paired,swapped)]; output=Path(output).resolve()
    manifests=[json.loads((p/'manifest.json').read_text()) for p in roots]
    configs=[json.loads((p/'config.json').read_text()) for p in roots]
    if configs[0]!=configs[1]: raise ValueError('Birth graph differs')
    if [m['mapping'] for m in manifests]!=['paired','swapped']: raise ValueError('Need crossed physical assignments')
    if manifests[0]['physical_sources']!=manifests[1]['physical_sources']: raise ValueError('Physical stimuli differ')
    if any(digest(p)!=h for p,h in {**manifests[0]['source_hashes'],**manifests[0]['physical_sources']}.items()):
        raise ValueError('Runtime or physical source changed')
    reference=[]
    for path in sorted(manifests[0]['physical_sources']):
        with np.load(path) as z: reference.append(z['auditory'].mean(axis=0))
    axis=reference[0]-reference[1]; norm=float(axis@axis)
    if norm==0: raise ValueError('No physical channel contrast')
    output.mkdir(exist_ok=False)
    traces={}; entries=[]; residual=0.
    for root,m,cfg in zip(roots,manifests,configs):
        done=json.loads((root/'completion.json').read_text())
        probes={}; members=[n['id'] for n in cfg['neurons']]
        pi=[members.index(n) for n in m['bridge']['prediction']]
        ci=[members.index(n) for n in m['bridge']['prediction_consumer']]
        for row in done['entries']:
            path=root/row['file']
            if digest(path)!=row['sha256']: raise ValueError('Raw record changed')
            with np.load(path) as z:
                data={k:z[k] for k in z.files}
            residual=max(residual,audit_record(data,cfg,m['bridge']))
            entries.append(dict(file=str(path),sha256=row['sha256']))
            if row['phase']=='probe':
                probes[row['state'],row['clip']]=dict(prediction=data['cells'][:,pi,1],
                    consumer=data['cells'][:,ci,1],weights=data['weights'],
                    initial_weights=data['start_weights'],eta=data['eta'],
                    start={key:value for key,value in data.items() if key.startswith('start_')})
        for cue in (0,1):
            intact=probes['trained',cue]['start']; reset=probes['reset_selected',cue]['start']
            for key in intact:
                if key not in ('start_weights','start_incoming_info') and not np.array_equal(intact[key],reset[key]):
                    raise ValueError(f'Undeclared counterfactual state change: {key}')
            expected=intact['start_incoming_info'].copy()
            all_points=[p for n in cfg['neurons'] for p in cfg['synaptic_points']
                        if p['type']=='postsynaptic' and p['neuron_id']==n['id']]
            chosen={(n,s) for n,s,_ in m['selected']}
            for i,p in enumerate(all_points):
                if (p['neuron_id'],p['synapse_id']) in chosen: expected[i]=p['u_i']['info']
            if not np.array_equal(reset['start_incoming_info'],expected):
                raise ValueError('Reset changed something besides the selected information weights')
            for layer in ('prediction','consumer'):
                trained=probes['trained',cue][layer]; reset=probes['reset_selected',cue][layer]
                traces[f'{m["mapping"]}_cue{cue}_{layer}_weight_effect']=trained-reset
                traces[f'{m["mapping"]}_cue{cue}_{layer}_trained']=trained
                traces[f'{m["mapping"]}_cue{cue}_{layer}_reset']=reset
            expected=(1 if cue==0 else -1)*(1 if m['mapping']=='paired' else -1)
            effect=traces[f'{m["mapping"]}_cue{cue}_prediction_weight_effect']
            traces[f'{m["mapping"]}_cue{cue}_signed_physical_profile_projection']=expected*(effect@axis)/norm
        traces[m['mapping']+'_trained_blank_prediction']=probes['trained',None]['prediction']
    # Same physical cue, different acquired assignment. Subtracting cue effects
    # removes a cue-independent history offset, but not all order interactions.
    for layer in ('prediction','consumer'):
        effects={cue:traces[f'paired_cue{cue}_{layer}_weight_effect']-
                     traces[f'swapped_cue{cue}_{layer}_weight_effect'] for cue in (0,1)}
        traces[layer+'_assignment_by_cue_interaction']=effects[0]-effects[1]
    traces['physical_profile_axis']=axis
    projection=traces['prediction_assignment_by_cue_interaction']@axis/norm
    traces['assignment_by_cue_profile_projection']=projection
    np.savez_compressed(output/'effects-per-tick.npz',**traces)
    report=dict(valid=True,max_equation_residual=residual,records=len(entries),sources=entries,
        per_cue={key:windows(value) for key,value in traces.items() if key.endswith('signed_physical_profile_projection')},
        assignment_by_cue=windows(projection),
        limits='The axis is the difference of mean physical spectral features, not a fitted observer. '
               'A positive projection is not sound-waveform replay or semantic recall. '
               'This is one presentation order and one graph seed; recency and arbitrary history effects remain. '
               'The added consumer receives predictions, but useful action and hierarchical regulation are untested. '
               'Raw channels and each completed tick remain the primary evidence.',
        effects_sha256=digest(output/'effects-per-tick.npz'))
    (output/'summary.json').write_text(encode(report)+'\n')
    print(encode({k:v for k,v in report.items() if k!='sources'}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('paired','swapped','output'):p.add_argument('--'+key,type=Path,required=True)
    run(**vars(p.parse_args()))
