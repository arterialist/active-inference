"""Audit continuous acquisition and block-by-block embodied memory expression."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import context_organization as base
from .crossed_av_analysis import verify_stimuli,verify_continuity
from .opponent_context_analysis import read_record
from .temporal_verification import verify_learning


PAIR_ORDER = ((0, 0), (0, 1), (1, 0), (1, 1))
FACTORIAL = np.array([[1, 1, 1, 1], [1, 1, -1, -1],
                     [1, -1, 1, -1], [1, -1, -1, 1]], dtype=float) / 4


def factorial_modes(pair_responses):
    """Descriptive contrasts, not a decoder supplied to the neural system.

    Columns of the input are 00,01,10,11. Output columns are common, visual,
    auditory and joint interaction. A growing interaction is not sufficient
    for correct actions: main effects and common bias may still dominate it.
    """
    x = np.asarray(pair_responses, dtype=float)
    if x.ndim != 2 or x.shape[1] != 4 or not np.isfinite(x).all():
        raise ValueError('Need finite tick-by-four-pair responses')
    return x @ FACTORIAL.T


def trajectory(z,groups):
    ids=list(z['neuron_ids'])
    p=z['cells'][:,[ids.index(n) for n in groups['prediction']],base.FIELDS.index('O')]
    direction=np.sign(z['body'][:,4]);prediction=(p[:,0]-p[:,1])*direction
    torque=z['body'][:,4]+base.FORCE*z['body'][:,3]
    dq=abs(z['weights']-z['weights_initial']).max(axis=(1,2))
    return np.column_stack((z['body'],torque,p,prediction,z['body'][:,1]*direction,
                            z['errors'][:,:,1],z['eta'],dq))


def analyze(roots,output,blocks=16):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if type(blocks) is not int or not 5<=blocks<=16:raise ValueError('Need completed block count 5..16')
    traces={};cases=[];sources=[];checked=0;seen=set();checkpoints=[];factorial={}
    for root in map(lambda p:Path(p).resolve(),roots):
        m=json.loads((root/'manifest.json').read_text())
        completed=root/f'completed-block-{blocks}.json'
        progress=json.loads(completed.read_text())
        if progress['blocks']!=blocks:raise ValueError('Wrong completed block record')
        identity=(m['seed'],m['reverse'])
        if identity in seen:raise ValueError('Duplicate condition')
        seen.add(identity)
        for p,h in {**m['source_hashes'],**m['physical_sources']}.items():
            if base.digest(p)!=h:raise ValueError(f'Source changed: {p}')
        parent=Path(m['parent']);pm=json.loads((parent/'manifest.json').read_text())
        ps=json.loads((parent/'summary.json').read_text())
        for name,h in m['parent_evidence'].items():
            if base.digest(parent/name)!=h:raise ValueError('Parent evidence changed')
        if not json.loads((root/'preflight.json').read_text())['parent_expression_replay_exact']:
            raise ValueError('Parent expression did not replay')
        for c in progress['checkpoints']:
            if base.digest(root/c['neural'])!=c['neural_sha256'] or base.digest(root/c['physical'])!=c['physical_sha256']:
                raise ValueError('Block checkpoint changed')
            checkpoints.append(dict(seed=m['seed'],**c))
        if [c['blocks'] for c in progress['checkpoints']]!=list(range(5,blocks+1)):
            raise ValueError('Missing block checkpoint')
        expected=[(block,v,a) for block in range(4,blocks) for v,a in m['schedule'][block]]
        if [(r['block'],r['video'],r['audio']) for r in progress['training']]!=expected:
            raise ValueError('Acquisition order differs')
        features=[]
        for clip in (0,1):
            paths=[p for p in m['physical_sources'] if Path(p).name==f'sensory-{clip}.npz']
            if len(paths)!=1:raise ValueError('Ambiguous media')
            with np.load(paths[0]) as z:features.append({k:z[k] for k in ('visual','auditory')})
        previous=read_record(parent,ps['training'][-1],pm,learning_auditor=verify_learning)
        by_block={4:previous['weights'][-1].copy()};prefix=f's{m["seed"]}_r{int(m["reverse"])}'
        for row in progress['training']:
            z=read_record(root,row,m,learning_auditor=verify_learning)
            if len(z['body'])!=364:raise ValueError('Wrong episode length')
            verify_continuity(previous,z);verify_stimuli(z,features,row,m['reverse']);checked+=364
            traces[f'{prefix}_{Path(row["file"]).stem}']=trajectory(z,m['groups'])
            # The final entry of each ordered block replaces earlier episode endpoints.
            by_block[row['block']+1]=z['weights'][-1].copy();previous=z
        expected=set()
        for b in range(5,blocks+1):
            for v in (0,1):
                for a in (0,1):
                    expected.add((b,'resting','learned',v,a))
                    if b in (8,12,16):
                        expected.update((b,'acquired',w,v,a) for w in ('learned','reset'))
        records={};pairs={};responses={}
        # Retain the initial four-exposure reference in the SAME column format.
        references=[r for r in ps['probes'] if r['kind']=='resting' and r['presentation']=='both']
        all_rows=[(parent,dict(r,blocks=4),pm) for r in references]
        all_rows.extend((root,r,m) for r in progress['probes'])
        for source,row,metadata in all_rows:
            z=read_record(source,row,metadata,learning_auditor=verify_learning)
            verify_stimuli(z,features,row,m['reverse'])
            if len(z['body'])!=96:raise ValueError('Wrong probe duration')
            if source==root:
                identity=tuple(row[k] for k in ('blocks','kind','weights','video','audio'))
                if identity in records:raise ValueError('Duplicate probe')
                records[identity]=row;checked+=96
            target=by_block[row['blocks']] if row['weights']=='learned' else np.zeros_like(z['weights_initial'])
            if not np.array_equal(z['weights_initial'],target):raise ValueError('Wrong selected memory')
            if row['kind']=='resting':
                if not np.array_equal(z['body_initial'],base.Arm().state()):raise ValueError('Body not at rest')
                if any(np.any(z[k]) for k in ('delay_initial','context_initial','error_initial')):
                    raise ValueError('Residual state in stored-content probe')
                if np.any(z['drive'][:64,194:198]):raise ValueError('Early bodily evidence')
            else:
                pair=(row['blocks'],row['video'],row['audio'])
                initial={k:z[k].copy() for k in ('body_initial','delay_initial','context_initial','error_initial')}
                if pair in pairs and any(not np.array_equal(initial[k],pairs[pair][k]) for k in initial):
                    raise ValueError('Full-state weight controls differ')
                pairs[pair]=initial
            x=trajectory(z,m['groups']);key=f'{prefix}_b{row["blocks"]}_{row["kind"]}_{row["weights"]}_v{row["video"]}_a{row["audio"]}'
            traces[key]=x
            contrast_key=f'{prefix}_b{row["blocks"]}_{row["kind"]}_{row["weights"]}'
            responses.setdefault(contrast_key,{})[row['video'],row['audio']]=x[:,6]-x[:,7]
            cases.append(dict(key=key,seed=m['seed'],reverse=m['reverse'],**{k:row[k] for k in
                ('blocks','kind','weights','video','audio')},wrong_ticks=np.flatnonzero(x[:,8]<0).tolist(),
                windows=[dict(start=a,stop=b,min_prediction=float(x[a:b,8].min()),
                    max_prediction=float(x[a:b,8].max()),angle_end=float(x[b-1,1]),
                    mean_abs_torque=float(abs(x[a:b,5]).mean()),max_q_change=float(x[a:b,14].max()),
                    min_eta=float(x[a:b,12:14].min())) for a,b in ((0,32),(32,64),(64,96))]))
        if set(records)!=expected:raise ValueError('Missing probe conditions')
        for key,values in responses.items():
            if set(values)!=set(PAIR_ORDER):raise ValueError('Incomplete factorial comparison')
            factorial[key]=factorial_modes(np.column_stack([values[p] for p in PAIR_ORDER]))
        sources.append(dict(root=str(root),manifest_sha256=base.digest(root/'manifest.json'),
                            completed_record=completed.name,completed_sha256=base.digest(completed)))
    if not seen:raise ValueError('No evidence')
    output.mkdir();np.savez_compressed(output/'per-tick.npz',**traces)
    np.savez_compressed(output/'factorial.npz',**factorial)
    result=dict(cases=cases,sources=sources,checkpoints=checkpoints,checked_new_ticks=checked,
        through_blocks=blocks,factorial_columns=['common','visual','auditory','joint'],
        factorial_pair_order=PAIR_ORDER,
        columns=['time_s','angle_rad','velocity_rad_s','command','load_Nm','net_torque_Nm',
            'prediction_0','prediction_1','prediction_along_load','angle_along_load',
            'teaching_0','teaching_1','eta_0','eta_1','max_selected_q_change'],
        limits='Parent references are retained but excluded from checked_new_ticks. '
               'Stored-content and acquired-state diagnostics have different initial states. '
               'Each block contains all pairings, but its last pairing varies: duration is not isolated from recency.')
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(cases=len(cases),checked_new_ticks=checked,blocks=blocks)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--blocks',type=int,default=16)
    p.add_argument('roots',type=Path,nargs='+')
    a=p.parse_args();analyze(a.roots,a.output,a.blocks)
