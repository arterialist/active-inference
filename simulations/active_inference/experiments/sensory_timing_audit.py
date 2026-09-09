"""Reconstruct changed sensory input and compare feature-specific recall effects."""
import argparse
import json
from pathlib import Path

import numpy as np

from .association_route_probe import digest
from .composition_probe import encode
from .predictive_bridge_probe import audit_record
from .predictive_bridge_audit import windows


def expected_external(features,ids,vision,clip,mode,length):
    result=np.zeros((length,len(ids)));index={n:i for i,n in enumerate(ids)}
    for t in range(length):
        values=features[clip]['visual'][t%int(features[clip]['ticks'])]
        for n,v in zip(vision,values):
            if mode=='continuous':result[t,index[n]]=.5*v
            elif mode in ('native','phase_shift_1'):
                if (t+(mode=='phase_shift_1'))%4==n%4:result[t,index[n]]=2*v
            else:raise ValueError('Unknown sensory schedule')
    return result


def feature_effect(records):
    return ((records['paired','learned',0]-records['paired','shuffled',0])-
            (records['swapped','learned',0]-records['swapped','shuffled',0])-
            (records['paired','learned',1]-records['paired','shuffled',1])+
            (records['swapped','learned',1]-records['swapped','shuffled',1]))


def run(paired,swapped,output):
    modes=('native','phase_shift_1','continuous');records={};sources=[];reference=None;maximum=0.;doses={}
    for mapping,root in (('paired',Path(paired).resolve()),('swapped',Path(swapped).resolve())):
        m=json.loads((root/'manifest.json').read_text());parent=Path(m['parent'])
        pm=json.loads((parent/'manifest.json').read_text());cfg=json.loads((parent/'config.json').read_text())
        signature=(digest(parent/'config.json'),m['physical_sources'],m['order'])
        if m['mapping']!=mapping:raise ValueError('Unexpected acquisition mapping')
        if reference is None:reference=signature
        elif signature!=reference:raise ValueError('Unmatched graph, media or acquisition order')
        for p,h in {**m['source_hashes'],**m['physical_sources']}.items():
            if digest(p)!=h:raise ValueError('Source changed')
        features=[]
        for p in sorted(m['physical_sources']):
            with np.load(p) as z:features.append({k:z[k] for k in z.files})
        done=json.loads((root/'completion.json').read_text())
        expected_keys={(mode,c,cue) for mode in modes for c in ('learned','shuffled') for cue in (0,1)}
        if len(done['entries'])!=12 or {(e['mode'],e['condition'],e['clip']) for e in done['entries']}!=expected_keys:
            raise ValueError('Incomplete or duplicated timing conditions')
        index={n['id']:i for i,n in enumerate(cfg['neurons'])}
        for row in done['entries']:
            path=root/row['file']
            if digest(path)!=row['sha256']:raise ValueError('Raw record changed')
            with np.load(path) as z:data={k:z[k] for k in z.files}
            mode,condition,cue=row['mode'],row['condition'],row['clip']
            length=32 if mode=='native' else 300
            if len(data['cells'])!=length:raise ValueError('Unexpected recording length')
            expected=expected_external(features,m['external_neuron_ids'],pm['groups']['vision'],cue,mode,length)
            if not np.array_equal(data['external_information'],expected):raise ValueError('Sensory schedule or dose differs')
            maximum=max(maximum,audit_record(data,cfg,m['bridge']))
            sources.append(dict(file=str(path),sha256=row['sha256']))
            golden=Path(m['source'])/f'{condition}-cue{cue}.npz'
            if digest(golden)!=m['source_hashes'][str(golden)]:raise ValueError('Reference transplant changed')
            with np.load(golden) as z:
                for key,value in data.items():
                    if key.startswith('start_'):
                        if not np.array_equal(value,z[key]):raise ValueError('Initial state differs across timing')
                    elif mode=='native' and key not in ('end_incoming_info','external_information'):
                        if not np.array_equal(value,z[key][:32]):raise ValueError('Native reference replay differs')
                # Native long trajectories are reused, not synthesized from prefixes.
                cells=z['cells'] if mode=='native' else data['cells']
                for layer,role in (('prediction','prediction'),('consumer','prediction_consumer')):
                    records[mode,layer,mapping,condition,cue]=cells[:,[index[n] for n in m['bridge'][role]],1]
            if mapping=='paired' and condition=='learned':
                drive=expected_external(features,m['external_neuron_ids'],pm['groups']['vision'],cue,mode,300)
                doses[mode,cue]=drive.sum(axis=0)
    axis=features[0]['auditory'].mean(axis=0)-features[1]['auditory'].mean(axis=0);norm=float(axis@axis)
    if norm<=0:raise ValueError('Physical profile axis is undefined')
    arrays={'physical_profile_axis':axis};report={};dose_report={}
    for mode in modes:
        report[mode]={}
        for layer in ('prediction','consumer'):
            r={(mapping,c,cue):records[mode,layer,mapping,c,cue] for mapping in ('paired','swapped') for c in ('learned','shuffled') for cue in (0,1)}
            effect=feature_effect(r);arrays[f'{mode}_{layer}_feature_interaction']=effect
            if layer=='prediction':
                p=effect@axis/norm;arrays[f'{mode}_projection']=p;report[mode]['windows']=windows(p)
                for mapping in ('paired','swapped'):
                    for cue in (0,1):
                        difference=r[mapping,'learned',cue]-r[mapping,'shuffled',cue]
                        arrays[f'{mode}_{mapping}_cue{cue}_feature_effect']=difference
        dose_report[mode]=[]
        for cue in (0,1):
            a,b=doses[mode,cue],doses['native',cue];positive=b>0
            arrays[f'{mode}_cue{cue}_external_dose']=a
            dose_report[mode].append(dict(cue=cue,max_absolute_delta=float(abs(a-b).max()),
                positive_native_channels=int(positive.sum()),
                relative_range=[float((a[positive]/b[positive]).min()),float((a[positive]/b[positive]).max())]))
    output=Path(output).resolve();output.mkdir(exist_ok=False)
    np.savez_compressed(output/'effects-per-tick.npz',**arrays)
    result=dict(sources=sources,order=reference[2],max_equation_residual=maximum,results=report,dose=dose_report,
        effects_sha256=digest(output/'effects-per-tick.npz'),
        scope='Weight-placement contribution under changed sensory timing, with active plasticity. '
        'Two familiar clips and one graph seed. Native 300-tick controls are reused raw records. '
        'Physical-profile projection is not accuracy; no category, neural action, recurrent regime or consciousness claim.')
    (output/'summary.json').write_text(encode(result)+'\n')
    print(encode({k:v for k,v in result.items() if k not in ('sources','results')}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('paired','swapped','output'):p.add_argument('--'+key,type=Path,required=True)
    run(**vars(p.parse_args()))
