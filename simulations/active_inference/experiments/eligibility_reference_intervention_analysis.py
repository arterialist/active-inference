"""Trace acquired-state learning interventions through neurons and physical action."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import context_organization as base
from .body_state_memory_analysis import intervals,mechanics
from .crossed_av_analysis import verify_stimuli
from .crossed_av_continuation_analysis import trajectory,PAIR_ORDER
from .eligibility_reference_probe import verify_learning
from .eligibility_reference_pool_audit import audit_pools
from .magnitude_feedback_analysis import verify_returns
from .opponent_context_analysis import read_record


def changed_ticks(a,b):
    a,b=map(np.asarray,(a,b))
    if a.shape!=b.shape or a.ndim<1:raise ValueError('Need matched trajectories')
    delta=a!=b
    return np.any(delta,axis=tuple(range(1,a.ndim))) if a.ndim>1 else delta


def onset(a,b):
    indices=np.flatnonzero(changed_ticks(a,b))
    return int(indices[0]) if len(indices) else None


def analyze(roots,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    arrays={};cases=[];seeds=set();checked=0;sources={}
    for root in map(lambda p:Path(p).resolve(),roots):
        m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text())
        seed=m['seed'];g=m['groups'];parent=Path(m['parent'])
        if seed in seeds:raise ValueError('Duplicate seed')
        seeds.add(seed)
        for p,h in {**m['source_hashes'],**m['physical_sources']}.items():
            if base.digest(p)!=h:raise ValueError('Source changed')
        pm=json.loads((parent/'manifest.json').read_text())
        if base.digest(parent/'manifest.json')!=m['parent_manifest_sha256']:raise ValueError('Parent changed')
        if base.digest(parent/f'completed-contrast-r0-b{m["block"]}.json')!=m['parent_progress_sha256']:
            raise ValueError('Acquired reference changed')
        cfg=json.loads((parent/'contrast.json').read_text())
        if base.digest(parent/'contrast.json')!=pm['config_hashes']['contrast']:raise ValueError('Source graph changed')
        if s['executed_ticks']!=1152 or s['exact_replay_ticks']!=384:raise ValueError('Incomplete intervention course')
        expected={(c,v,a) for c in ('intact','native','slow_native') for v,a in PAIR_ORDER}
        identities=[(r['condition'],r['video'],r['audio']) for r in s['rows']]
        if len(identities)!=len(expected) or set(identities)!=expected:raise ValueError('Incomplete conditions')
        features=[]
        for clip in (0,1):
            paths=[p for p in m['physical_sources'] if Path(p).name==f'sensory-{clip}.npz']
            if len(paths)!=1:raise ValueError('Ambiguous media')
            with np.load(paths[0]) as z:features.append({k:z[k] for k in ('visual','auditory')})
        for v,a in PAIR_ORDER:
            conditions={}
            for row in (r for r in s['rows'] if (r['video'],r['audio'])==(v,a)):
                c=row['condition'];z=read_record(root,row,m,learning_auditor=verify_learning)
                if len(z['body'])!=96:raise ValueError('Wrong duration')
                verify_returns(z);verify_stimuli(z,features,row,False);audit_pools(z,cfg);mech=mechanics(z)
                strength,scale=m['interventions'][c]
                if not np.array_equal(z['reference_strength'],[strength]*2) or not np.array_equal(z['basal_eta'],[1e-5*scale]*2):
                    raise ValueError('Intervention parameters differ')
                if c=='intact':
                    if base.digest(parent/row['reference_file'])!=row['reference_sha256']:raise ValueError('Replay source changed')
                    with np.load(parent/row['reference_file']) as old:
                        if set(old.files)!=set(z) or any(not np.array_equal(old[k],z[k]) for k in old.files):
                            raise ValueError('Independent intact replay check differs')
                conditions[c]=(z,trajectory(z,g),mech);checked+=96
                arrays[f's{seed}_{c}_v{v}_a{a}']=trajectory(z,g)
            intact,reference,rm=conditions['intact']
            for c in ('native','slow_native'):
                z,tr,mech=conditions[c]
                for field in ('weights_initial','body_initial','delay_initial','context_initial','error_initial',
                              'reference_initial','terminal_initial','pool_weight_initial','neuron_ids','context_source_ids'):
                    if not np.array_equal(z[field],intact[field]):raise ValueError('Initial states or column identities differ')
                name=f's{seed}_{c}_v{v}_a{a}';first={};event_intervals={}
                for field in ('weights','cells','terminal_info','body','drive','raw_afferents','errors','eta','reference_arrivals'):
                    first[field]=onset(z[field],intact[field]);event_intervals[field]=intervals(changed_ticks(z[field],intact[field]))
                ids=list(z['neuron_ids']);populations={}
                for role,members in g.items():
                    index=[ids.index(n) for n in members]
                    populations[role]={field:onset(z['cells'][:,index,j],intact['cells'][:,index,j])
                                       for j,field in enumerate(base.FIELDS)}
                difference=np.column_stack((abs(tr[:,1])-abs(reference[:,1]),tr[:,3]-reference[:,3],
                                             tr[:,8]-reference[:,8],mech[:,5]-rm[:,5]))
                arrays[name+'_effects']=difference
                arrays[name+'_cell_changed']=z['cells']!=intact['cells']
                cases.append(dict(seed=seed,condition=c,video=v,audio=a,trace=name,
                    first_difference=first,difference_intervals=event_intervals,populations=populations,
                    lesser_pose_error=intervals(difference[:,0]<0),greater_pose_error=intervals(difference[:,0]>0)))
        for name in ('manifest.json','summary.json'):sources[str(root/name)]=base.digest(root/name)
    if len(seeds)<4:raise ValueError('Need four graph seeds')
    output.mkdir();np.savez_compressed(output/'per-tick.npz',**arrays)
    result=dict(cases=cases,seeds=sorted(seeds),checked_ticks=checked,exact_replay_ticks=384*len(seeds),
        effects=['delta_absolute_angle','delta_command','delta_prediction_along_load','delta_motor_work'],
        sources=sources,producer_sha256=base.digest(__file__),
        limits='Matched acquired-state parameter intervention, not neural discovery of a regulatory policy. '
        'First divergences are factual times, not proof every possible route is absent. '
        'Each physical integration interval is 4 ms; actuator work is not metabolic cost. '
        'Reference-cell initial in-flight history has the pool auditor\'s declared coverage gap.')
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(seeds=result['seeds'],checked_ticks=checked,cases=len(cases))),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();analyze(a.roots,a.output)
