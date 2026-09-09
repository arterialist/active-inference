"""Independently audit transient learned content and its physical consequences."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import context_organization as base
from .crossed_av_continuation_analysis import trajectory
from .crossed_av_analysis import verify_stimuli
from .opponent_context_analysis import read_record
from .temporal_verification import verify_learning


def analyze(roots, output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    traces={};cases=[];sources=[];seen=set();checked=0
    for root in map(lambda p:Path(p).resolve(),roots):
        m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text())
        if m['seed'] in seen:raise ValueError('Duplicate seed')
        seen.add(m['seed'])
        if base.digest(Path(__file__).with_name('crossed_av_transient.py'))!=m['producer_sha256']:
            raise ValueError('Transient producer changed')
        for p,h in {**m['source_hashes'],**m['physical_sources']}.items():
            if base.digest(p)!=h:raise ValueError('Source changed')
        if not s['parent_replay_exact'] or s['executed_ticks']!=1152:raise ValueError('Incomplete experiment')
        source=Path(m['source']);sm=json.loads((source/'manifest.json').read_text())
        parent=Path(sm['parent']);pm=json.loads((parent/'manifest.json').read_text())
        ps=json.loads((parent/'summary.json').read_text())
        for name,h in sm['parent_evidence'].items():
            if base.digest(parent/name)!=h:raise ValueError('Parent changed')
        selected=read_record(source,m['source_record'],sm,learning_auditor=verify_learning)
        features=[]
        for clip in (0,1):
            paths=[p for p in m['physical_sources'] if Path(p).name==f'sensory-{clip}.npz']
            if len(paths)!=1:raise ValueError('Ambiguous sensory input')
            with np.load(paths[0]) as z:features.append({k:z[k] for k in ('visual','auditory')})
        actual=set()
        for row in s['probes']:
            condition=(row['when'],row['video'],row['audio'])
            if condition in actual:raise ValueError('Duplicate probe')
            actual.add(condition)
            z=read_record(root,row,m,learning_auditor=verify_learning)
            verify_stimuli(z,features,row,False)
            if len(z['body'])!=96 or not np.array_equal(z['body_initial'],base.Arm().state()):
                raise ValueError('Wrong probe length or physical initial state')
            if not np.array_equal(z['weights_initial'],selected['weights'][row['when']]):
                raise ValueError('Recorded acquisition weights were not transferred')
            if any(np.any(z[k]) for k in ('delay_initial','context_initial','error_initial')):
                raise ValueError('Residual state in content assay')
            if np.any(z['weights'][:64]!=z['weights_initial']) or np.any(z['drive'][:64,194:198]):
                raise ValueError('Prefix reacquires memory or receives fresh somatic evidence')
            baseline_row=next(r for r in ps['probes'] if
                (r['kind'],r['weights'],r['presentation'],r['video'],r['audio'])==
                ('resting','birth','both',row['video'],row['audio']))
            baseline=read_record(parent,baseline_row,pm,learning_auditor=verify_learning)
            if not np.array_equal(z['drive'][:64],baseline['drive'][:64]):
                raise ValueError('Birth and learned probes receive different prefix drives')
            x=trajectory(z,m['groups']);base_angle=baseline['body'][:,1]
            improvement=abs(base_angle)-abs(z['body'][:,1])
            key=f's{m["seed"]}_q{row["when"]}_v{row["video"]}_a{row["audio"]}'
            traces[key]=np.column_stack((x,base_angle,improvement));checked+=96
            cases.append(dict(key=key,seed=m['seed'],when=row['when'],video=row['video'],audio=row['audio'],
                candidate=row['when']==m['candidate_tick'],
                min_prediction_16_63=float(x[16:64,8].min()),
                wrong_prediction_ticks=np.flatnonzero(x[:,8]<0).tolist(),
                worse_displacement_ticks=np.flatnonzero(improvement<0).tolist(),
                displacement_reduction_at_63=float(improvement[63]/abs(base_angle[63])),
                min_eta=float(z['eta'].min())))
        expected={(t,v,a) for t in (m['candidate_tick'],m['later_tick']) for v in (0,1) for a in (0,1)}
        if actual!=expected:raise ValueError('Missing comparison')
        sources.append(dict(root=str(root),manifest_sha256=base.digest(root/'manifest.json'),
                            summary_sha256=base.digest(root/'summary.json')))
    if not seen:raise ValueError('No transient evidence')
    output.mkdir();np.savez_compressed(output/'per-tick.npz',**traces)
    result=dict(cases=cases,sources=sources,checked_new_ticks=checked,
        columns=['time_s','angle_rad','velocity_rad_s','command','load_Nm','net_torque_Nm',
                 'prediction_0','prediction_1','prediction_along_load','angle_along_load',
                 'teaching_0','teaching_1','eta_0','eta_1','max_selected_q_change',
                 'birth_angle_rad','absolute_displacement_reduction_rad'],
        limits='Retrospective state selection, not autonomous retention or ordinary acquired-state recall. '
               'A small correct-sign response is not adequate compensation. '
               'Preserve onset reversals and later feedback-driven changes; no learning is frozen.')
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(cases=len(cases),checked_new_ticks=checked)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('roots',type=Path,nargs='+');a=p.parse_args();analyze(a.roots,a.output)
