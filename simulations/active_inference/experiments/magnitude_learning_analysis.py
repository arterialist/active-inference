"""Independent completed-prefix audit of the embodied magnitude-feedback trial.

No terminal sign or all-pair endpoint alone constitutes memory acceptance. Keep
every prediction, physical trajectory and population member in the evidence.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import context_organization as base
from .crossed_av_analysis import verify_continuity,verify_stimuli
from .crossed_av_continuation_analysis import trajectory,factorial_modes,PAIR_ORDER
from .opponent_context_analysis import read_record
from .temporal_verification import verify_learning
from .magnitude_feedback_analysis import verify_returns


def analyze(roots,output,blocks):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if type(blocks) is not int or not 1<=blocks<=16:raise ValueError('Need completed prefix 1..16')
    cases=[];sources=[];traces={};modes={};seen=set();checked=0;gate_cases=[]
    for root in map(lambda p:Path(p).resolve(),roots):
        m=json.loads((root/'manifest.json').read_text())
        progress=root/f'completed-block-{blocks}.json';s=json.loads(progress.read_text())
        if m['seed'] in seen or s['blocks']!=blocks:raise ValueError('Duplicate seed or wrong block')
        seen.add(m['seed'])
        for p,h in {**m['source_hashes'],**m['physical_sources']}.items():
            if base.digest(p)!=h:raise ValueError(f'Source changed: {p}')
        if not m['preflight_exact'] or base.digest(root/'preflight.npz')!=m['preflight_sha256']:
            raise ValueError('Missing verified native replay')
        if base.digest(root/'config.json')!=m['config_sha256']:raise ValueError('Configuration changed')
        expected=[(b,v,a) for b in range(blocks) for v,a in m['schedule'][b]]
        if [(r['block'],r['video'],r['audio']) for r in s['training']]!=expected:
            raise ValueError('Unbalanced or missing training')
        if [c['blocks'] for c in s['checkpoints']]!=list(range(1,blocks+1)):
            raise ValueError('Missing checkpoint')
        for c in s['checkpoints']:
            if base.digest(root/c['neural'])!=c['neural_sha256'] or base.digest(root/c['physical'])!=c['physical_sha256']:
                raise ValueError('Checkpoint changed')
        features=[]
        for clip in (0,1):
            paths=[p for p in m['physical_sources'] if Path(p).name==f'sensory-{clip}.npz']
            if len(paths)!=1:raise ValueError('Ambiguous media')
            with np.load(paths[0]) as z:features.append({k:z[k] for k in ('visual','auditory')})
        previous=None;by_block={};gate=[]
        for row in s['training']:
            z=read_record(root,row,m,learning_auditor=verify_learning);verify_returns(z)
            verify_stimuli(z,features,row,False)
            if len(z['body'])!=364:raise ValueError('Wrong training duration')
            if previous is not None:
                verify_continuity(previous,z)
                if not np.array_equal(previous['terminal_info'][-1],z['terminal_initial']):
                    raise ValueError('Terminal state reset between episodes')
            ids=list(z['neuron_ids']);o=z['cells'][:,:,base.FIELDS.index('O')]
            spare=o[:,[ids.index(n) for n in m['groups']['mixed_1']]]
            # Keep all members in the derived record as well, not only bank totals.
            key=f's{m["seed"]}_{Path(row["file"]).stem}'
            traces[key]=trajectory(z,m['groups']);traces[key+'_spare_O']=spare
            gate.append(np.column_stack((np.arange(len(gate)*364,(len(gate)+1)*364),
                z['context_terminal'],spare.min(1),spare.max(1))))
            by_block[row['block']+1]=z['weights'][-1].copy();previous=z;checked+=364
        gate=np.concatenate(gate);traces[f's{m["seed"]}_gate']=gate
        gate_cases.append(dict(seed=m['seed'],negative_terminal_ticks=np.flatnonzero(gate[:,2]<0).tolist(),
            active_spare_ticks=np.flatnonzero(gate[:,6]>0).tolist(),
            final_context_release=float(gate[-1,2])))
        expected={(b,'resting','learned',v,a) for b in range(1,blocks+1) for v,a in PAIR_ORDER}
        expected|={(b,'acquired',w,v,a) for b in (4,8,12,16) if b<=blocks
                    for w in ('learned','reset') for v,a in PAIR_ORDER}
        observed=set();responses={};controls={}
        for row in s['probes']:
            identity=tuple(row[k] for k in ('blocks','kind','weights','video','audio'))
            if identity in observed:raise ValueError('Duplicate probe')
            observed.add(identity)
            z=read_record(root,row,m,learning_auditor=verify_learning);verify_returns(z)
            verify_stimuli(z,features,row,False)
            if len(z['body'])!=96:raise ValueError('Wrong probe duration')
            if row['weights']=='learned':
                if not np.array_equal(z['weights_initial'],by_block[row['blocks']]):
                    raise ValueError('Incorrect transferred weights')
            elif np.any(z['weights_initial']):raise ValueError('Reset weights not zero')
            if row['kind']=='resting':
                if not np.array_equal(z['body_initial'],base.Arm().state()):raise ValueError('Not resting')
                if any(np.any(z[k]) for k in ('context_initial','error_initial','delay_initial')):
                    raise ValueError('Residual state in weight-only probe')
            else:
                key=(row['blocks'],row['video'],row['audio'])
                initial={k:z[k].copy() for k in ('body_initial','delay_initial','context_initial',
                                               'error_initial','terminal_initial')}
                if key in controls and any(not np.array_equal(v,controls[key][k]) for k,v in initial.items()):
                    raise ValueError('Unmatched acquired reset control')
                controls[key]=initial
            key=f's{m["seed"]}_{Path(row["file"]).stem}'
            tr=trajectory(z,m['groups']);traces[key]=tr;checked+=96
            group=(row['blocks'],row['kind'],row['weights'])
            responses.setdefault(group,{})[(row['video'],row['audio'])]=tr[:,6]-tr[:,7]
            cases.append(dict(seed=m['seed'],**{k:row[k] for k in ('blocks','kind','weights','video','audio')},
                trace=key,wrong_prediction_ticks=np.flatnonzero(tr[:,8]<0).tolist(),
                pre_feedback_min=float(tr[16:64,8].min()),angle_at63=float(tr[63,1]),
                limits='Pre-feedback interpretation applies only to resting probes.'))
        if observed!=expected:raise ValueError('Incomplete probe family')
        for group,pairs in responses.items():
            modes[f's{m["seed"]}_'+ '_'.join(map(str,group))]=factorial_modes(np.column_stack([pairs[p] for p in PAIR_ORDER]))
        sources.append(dict(root=str(root),manifest_sha256=base.digest(root/'manifest.json'),
                            progress_sha256=base.digest(progress)))
    output.mkdir();np.savez_compressed(output/'per-tick.npz',**traces)
    np.savez_compressed(output/'factorial.npz',**modes)
    result=dict(blocks=blocks,cases=cases,gates=gate_cases,sources=sources,checked_ticks=checked,
        gate_columns=['tick','terminal_before','terminal_after','return_count','arriving_release','spare_min','spare_max'],
        trajectory_columns=['time','angle','velocity','command','load','net_torque','prediction0','prediction1',
                            'prediction_along_load','angle_along_load','teach0','teach1','eta0','eta1','max_dq'],
        factorial_columns=['common','visual','audio','joint'],
        limits='Completed-prefix audit, not full-run or agent acceptance. Functional comparison with '
               'the unchanged native courses and all unfavorable intervals remains necessary.')
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(blocks=blocks,cases=len(cases),checked_ticks=checked,gates=gate_cases)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--blocks',type=int,required=True)
    a=p.parse_args();analyze(a.roots,a.output,a.blocks)
