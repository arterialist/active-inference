"""Audit actual sensory histories and full neural/physical crossed-pair traces."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import context_organization as base
from . import crossed_av_world as world
from .opponent_context_analysis import read_record
from .temporal_verification import verify_learning


def verify_stimuli(z,features,row,reverse):
    for tick in range(len(z['body'])):
        # Reconstruct the declared physical protocol independently of the
        # producer's paired-feature wrapper and source-index load adapter.
        expected=np.zeros(194);expected[192]=1.;load=0.
        if tick<300:
            sample=(tick+row.get('offset',0))%300
            if row.get('presentation','both')!='audio':
                expected[:96]=features[row['video']]['visual'][sample]
            if row.get('presentation','both')!='vision':
                expected[96:192]=features[row['audio']]['auditory'][sample]
            load=base.FORCE*(1 if row['video']==row['audio'] else -1)*(-1 if reverse else 1)
        if not np.array_equal(z['drive'][tick,:194],expected) or z['body'][tick,4]!=load:
            raise ValueError('Recorded sensory evidence or physical assignment differs')


def verify_missing_sense(a,b):
    """Conditional indistinguishability, not a claim about arbitrary histories."""
    for field in ('drive','cells','weights','arrivals','errors'):
        if not np.array_equal(a[field][:64],b[field][:64]):
            raise ValueError(f'Missing-sense histories differ: {field}')
    if not np.array_equal(a['body'][:64,4],-b['body'][:64,4]):
        raise ValueError('Missing-sense comparison does not require opposite loads')


def verify_continuity(previous,current):
    for prior,start in (('physical_states','body_initial'),('weights','weights_initial')):
        if not np.array_equal(previous[prior][-1],current[start]):
            raise ValueError(f'Acquisition continuity failed: {start}')
    if not np.array_equal(previous['delay_final'],current['delay_initial']):
        raise ValueError('Acquisition continuity failed: delay')
    if not np.array_equal(previous['errors'][-1,:,2],current['error_initial']):
        raise ValueError('Acquisition continuity failed: error')
    x=previous['context_initial'].copy();decay=math.exp(-1/64)
    for arrivals in previous['arrivals']:x=decay*x+(1-decay)*arrivals
    if not np.array_equal(x,current['context_initial']):
        raise ValueError('Acquisition continuity failed: local credit trace')


def analyze(roots,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    traces={};cases=[];sources=[];conditions={};checked_ticks=0;interactions=[]
    for root in map(lambda p:Path(p).resolve(),roots):
        m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text())
        identity=(m['seed'],m['reverse'])
        if identity in conditions:raise ValueError('Duplicate condition')
        for p,h in {**m['source_hashes'],**m['physical_sources']}.items():
            if base.digest(p)!=h:raise ValueError(f'Source changed: {p}')
        if base.digest(root/'config.json')!=m['config_sha256']:raise ValueError('Configuration changed')
        conditions[identity]=m['config_sha256']
        if not s['birth_replay_exact']:raise ValueError('Missing exact birth replay')
        expected=[(block,v,a) for block,pairs in enumerate(m['schedule']) for v,a in pairs]
        if len(s['training'])!=16 or [(r['block'],r['video'],r['audio']) for r in s['training']]!=expected:
            raise ValueError('Unbalanced or incomplete acquisition')
        features=[]
        for clip in (0,1):
            paths=[p for p in m['physical_sources'] if Path(p).name==f'sensory-{clip}.npz']
            if len(paths)!=1:raise ValueError('Ambiguous media')
            with np.load(paths[0]) as z:features.append({k:z[k] for k in ('visual','auditory')})
        prefix=f's{m["seed"]}_r{int(m["reverse"])}'
        previous=None
        for row in s['training']:
            z=read_record(root,row,m,learning_auditor=verify_learning)
            if len(z['body'])!=364:raise ValueError('Wrong acquisition duration')
            if previous is not None:verify_continuity(previous,z)
            verify_stimuli(z,features,row,m['reverse']);checked_ticks+=364
            traces[f'{prefix}_{Path(row["file"]).stem}']=z['body']
            previous=z
        final_weights=z['weights'][-1].copy()
        cache={};expected=set()
        for kind in ('acquired','resting','shifted'):
            for weights in (('learned',) if kind=='shifted' else ('birth','learned')):
                for mode in (world.PRESENTATIONS if kind=='resting' else ('both',)):
                    for v in (0,1):
                        for a in (0,1):expected.add((kind,weights,mode,v,a))
        for row in s['probes']:
            identity=tuple(row[k] for k in ('kind','weights','presentation','video','audio'))
            if identity in cache:raise ValueError('Duplicate probe')
            if row['offset']!=(64 if row['kind']=='shifted' else 0):raise ValueError('Unexpected onset')
            z=read_record(root,row,m,learning_auditor=verify_learning)
            if len(z['body'])!=96:raise ValueError('Wrong probe duration')
            verify_stimuli(z,features,row,m['reverse']);checked_ticks+=96;cache[identity]=z
            if row['weights']=='learned':
                if not np.array_equal(z['weights_initial'],final_weights):raise ValueError('Learned weights differ')
            elif np.any(z['weights_initial']):raise ValueError('Reset prediction weights not zero')
            if row['kind']!='acquired':
                if not np.array_equal(z['body_initial'],base.Arm().state()):raise ValueError('Body is not at rest')
                if any(np.any(z[f]) for f in ('delay_initial','context_initial','error_initial')):
                    raise ValueError('Residual history in birth-state probe')
                if np.any(z['drive'][:64,194:198]):raise ValueError('Somatic evidence arrived early')
            ids=list(z['neuron_ids'])
            p=z['cells'][:,[ids.index(n) for n in m['groups']['prediction']],base.FIELDS.index('O')]
            signed=p[:,0]-p[:,1];direction=np.sign(z['body'][:,4]);prediction=signed*direction
            torque=z['body'][:,4]+base.FORCE*z['body'][:,3]
            dq=abs(z['weights']-z['weights_initial']).max(axis=(1,2))
            key=f'{prefix}_{Path(row["file"]).stem}'
            traces[key]=np.column_stack((z['body'],torque,p,prediction,z['body'][:,1]*direction,
                                        z['errors'][:,:,1],z['eta'],dq))
            cases.append(dict(key=key,seed=m['seed'],reverse=m['reverse'],**{k:row[k] for k in
                ('kind','weights','presentation','video','audio','offset')},
                wrong_prediction_ticks=np.flatnonzero(prediction<0).tolist(),
                windows=[dict(start=a,stop=b,min_prediction=float(prediction[a:b].min()),
                    max_prediction=float(prediction[a:b].max()),mean_abs_torque=float(abs(torque[a:b]).mean()),
                    angle_end=float(z['body'][b-1,1]),max_q_change=float(dq[a:b].max()),
                    min_eta=float(z['eta'][a:b].min())) for a,b in ((0,32),(32,64),(64,96))]))
        if set(cache)!=expected:raise ValueError('Incomplete probe family')
        for kind in ('acquired','resting'):
            for mode in (('both',) if kind=='acquired' else world.PRESENTATIONS):
                for v in (0,1):
                    for a in (0,1):
                        left,right=(cache[kind,w,mode,v,a] for w in ('birth','learned'))
                        for field in ('body_initial','delay_initial','context_initial','error_initial'):
                            if not np.array_equal(left[field],right[field]):raise ValueError('Unmatched weight control')
        for weights in ('birth','learned'):
            for index in (0,1):
                verify_missing_sense(cache['resting',weights,'vision',index,0],cache['resting',weights,'vision',index,1])
                verify_missing_sense(cache['resting',weights,'audio',0,index],cache['resting',weights,'audio',1,index])
        # A nonzero factorial interaction diagnoses nonadditivity, not learning
        # or a useful linear readout. Retain every incoming channel and tick.
        for weights in ('birth','learned'):
            q={pair:cache['resting',weights,'both',*pair]['arrivals'][:64,0]
               for pair in ((0,0),(0,1),(1,0),(1,1))}
            interaction=q[0,0]+q[1,1]-q[0,1]-q[1,0]
            key=f'{prefix}_{weights}_mixed_interaction'
            traces[key]=interaction
            interactions.append(dict(key=key,seed=m['seed'],reverse=m['reverse'],weights=weights,
                max_abs=float(abs(interaction).max()),source_ids=cache['resting',weights,'both',0,0]['context_source_ids'][0].tolist()))
        sources.append(dict(root=str(root),manifest_sha256=base.digest(root/'manifest.json'),
                            summary_sha256=base.digest(root/'summary.json')))
    if not conditions:raise ValueError('No experimental conditions')
    for (seed,reverse),config_hash in conditions.items():
        if conditions.get((seed,not reverse))!=config_hash:raise ValueError('Missing or nonidentical paired brain')
    output.mkdir();np.savez_compressed(output/'per-tick.npz',**traces)
    result=dict(cases=cases,sources=sources,interactions=interactions,checked_ticks=checked_ticks,
        columns=['time_s','angle_rad','velocity_rad_s','command','load_Nm','net_torque_Nm',
                 'prediction_0','prediction_1','prediction_along_load','angle_along_load',
                 'teaching_0','teaching_1','eta_0','eta_1','max_selected_q_change'],
        limits='All 96 probe ticks retained; first 64 have no fresh somatic input only in birth/resting probes. '
               'Mixed factorial interaction is descriptive, not a learned or sufficient decoding proof.')
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(cases=len(cases),checked_ticks=checked_ticks)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('roots',type=Path,nargs='+')
    a=p.parse_args();analyze(a.roots,a.output)
