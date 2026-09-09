"""Audit complete reference-pathway courses and retain all favorable/failing ticks."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import context_organization as base
from .crossed_av_analysis import verify_continuity,verify_stimuli
from .crossed_av_continuation_analysis import trajectory,factorial_modes,PAIR_ORDER
from .eligibility_reference_probe import verify_learning
from .magnitude_feedback_analysis import verify_returns
from .opponent_context_analysis import read_record
from .body_state_memory_analysis import intervals
from .eligibility_reference_pool_audit import audit_pools


def opposed_writes(data):
    """Actual changes impossible under a positive native rate at THIS fixed state.

    Restrict to currently driven positive inputs. This is not a proof that a
    differently regulated native network cannot reach useful future behavior.
    """
    delta=np.diff(np.concatenate((data['weights_initial'][None],data['weights'])),axis=0)
    return (delta*data['errors'][:,:,0,None]<0)&(data['arrivals']>0)


def verify_checkpoint_family(manifest, checkpoints):
    expected={(condition,reverse,block) for condition in manifest['conditions']
              for reverse,count in ((False,manifest['normal_blocks']),(True,manifest['reversal_blocks']))
              for block in range(1,count+1)}
    observed=[(c['condition'],c['reverse'],c['block']) for c in checkpoints]
    if len(set(observed))!=len(observed) or set(observed)!=expected:
        raise ValueError('Incomplete or duplicated executable checkpoint family')


def analyze(roots,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    traces={};cases=[];sources={};seeds=set();checked=0;pool_observations=[]
    for root in map(lambda p:Path(p).resolve(),roots):
        m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text())
        seed=m['seed']
        if seed in seeds:raise ValueError('Duplicate seed')
        seeds.add(seed)
        for p,h in {**m['source_hashes'],**m['physical_sources']}.items():
            if base.digest(p)!=h:raise ValueError('Source changed')
        if not m['preflight_exact'] or base.digest(root/'preflight.npz')!=m['preflight_sha256']:
            raise ValueError('Missing exact disabled replay')
        features=[]
        for clip in (0,1):
            paths=[p for p in m['physical_sources'] if Path(p).name==f'sensory-{clip}.npz']
            if len(paths)!=1:raise ValueError('Ambiguous media')
            with np.load(paths[0]) as z:features.append({k:z[k] for k in ('visual','auditory')})
        verify_checkpoint_family(m,s['checkpoints'])
        for checkpoint in s['checkpoints']:
            for name in ('neural','physical'):
                if base.digest(root/checkpoint[name])!=checkpoint[name+'_sha256']:
                    raise ValueError('Checkpoint changed')
        for condition,(strength,scale) in m['conditions'].items():
            cfg_path=root/f'{condition}.json'
            if base.digest(cfg_path)!=m['config_hashes'][condition]:raise ValueError('Graph changed')
            cfg=json.loads(cfg_path.read_text());g=m['groups'][condition];nodes={n['id']:n for n in cfg['neurons']}
            for nid in g['prediction']:
                n=nodes[nid]
                if n['params']['eta_post']!=1e-5*scale or n['metadata']['prediction_reference_strength']!=strength:
                    raise ValueError('Unexpected learning condition')
                for name,default in [('prediction_tau_context',64),('prediction_tau_error',4),
                                     ('prediction_cap',1),('prediction_boost',499),('prediction_half',.01)]:
                    if n['metadata'].get(name,default)!=default:raise ValueError('Unexpected prediction constants')
            previous=None;weights={};seen=set();responses={};controls={}
            rows=[r for r in s['rows'] if r['condition']==condition]
            training_order=[(r['reverse'],r['block'],r['video'],r['audio']) for r in rows if r['kind']=='training']
            expected_order=[(reverse,b+1,v,a) for reverse,count in ((False,m['normal_blocks']),(True,m['reversal_blocks']))
                            for b,pairs in enumerate(m['schedule'][:count]) for v,a in pairs]
            if training_order!=expected_order:raise ValueError('Acquisition schedule incomplete or reordered')
            expected=set()
            for reverse,count in ((False,m['normal_blocks']),(True,m['reversal_blocks'])):
                for block in range(1,count+1):
                    expected|={(reverse,block,k,'learned',v,a) for k in ('training','resting') for v,a in PAIR_ORDER}
                expected|={(reverse,count,'acquired',w,v,a) for w in ('learned','reset') for v,a in PAIR_ORDER}
            for row in rows:
                identity=tuple(row[k] for k in ('reverse','block','kind','weights','video','audio'))
                if identity in seen:raise ValueError('Duplicate record')
                seen.add(identity)
                z=read_record(root,row,dict(groups=g),learning_auditor=verify_learning)
                verify_returns(z);verify_stimuli(z,features,row,row['reverse'])
                pool_audit=audit_pools(z,cfg)
                if len(z['body'])!=(364 if row['kind']=='training' else 96):raise ValueError('Wrong duration')
                if not np.array_equal(z['basal_eta'],[1e-5*scale]*2) or not np.array_equal(z['reference_strength'],[strength]*2):
                    raise ValueError('Recorded parameters differ from graph')
                for j,nid in enumerate(g['prediction']):
                    mapping=dict(nodes[nid]['metadata']['prediction_reference_map'])
                    ports=sorted(set(mapping.values()))
                    if not np.array_equal(z['reference_ports'][j],ports):raise ValueError('Reference port identity differs')
                    if not np.array_equal(z['reference_indices'][j],[ports.index(mapping[s]) for s in nodes[nid]['metadata']['prediction_ports']]):
                        raise ValueError('Reference assignment differs')
                    for port,source in zip(ports,z['reference_sources'][j]):
                        edges=[c for c in cfg['connections'] if c['target_neuron']==nid and c['target_synapse']==port]
                        if len(edges)!=1 or tuple(source)!=(edges[0]['source_neuron'],edges[0]['source_terminal']):
                            raise ValueError('Reference source identity differs')
                group=(row['reverse'],row['block'])
                if row['kind']=='training':
                    if previous is not None:
                        verify_continuity(previous,z)
                        if not np.array_equal(previous['reference_trace'][-1],z['reference_initial']):raise ValueError('Reference reset')
                        if not np.array_equal(previous['terminal_info'][-1],z['terminal_initial']):raise ValueError('Terminal reset')
                    weights[group]=z['weights'][-1];previous=z
                else:
                    target=weights[group] if row['weights']=='learned' else np.zeros_like(z['weights_initial'])
                    if not np.array_equal(z['weights_initial'],target):raise ValueError('Wrong selected memory')
                    if row['kind']=='resting':
                        if not np.array_equal(z['body_initial'],base.Arm().state()):raise ValueError('Body not at rest')
                        if any(np.any(z[k]) for k in ('context_initial','error_initial','delay_initial','reference_initial')):
                            raise ValueError('Non-weight memory in resting probe')
                    else:
                        key=(*group,row['video'],row['audio'])
                        state={k:z[k] for k in ('body_initial','delay_initial','context_initial','error_initial','reference_initial','terminal_initial')}
                        if key in controls and any(not np.array_equal(v,controls[key][k]) for k,v in state.items()):
                            raise ValueError('Unmatched full-state weight intervention')
                        controls[key]=state
                name=f's{seed}_{Path(row["file"]).stem}';tr=trajectory(z,g);traces[name]=tr
                traces[name+'_pool_residual']=pool_audit['residual']
                traces[name+'_pool_checked']=pool_audit['checked']
                opposed=opposed_writes(z);traces[name+'_opposed_writes']=opposed
                checked+=len(tr)
                traces[name+'_eligibility_extrema']=np.stack((z['effective_eligibility'].min(2),z['effective_eligibility'].max(2)),axis=2)
                pool_ids=list(z['pool_ids']);ids=list(z['neuron_ids'])
                pool=z['cells'][:,[ids.index(n) for n in pool_ids],base.FIELDS.index('O')]
                traces[name+'_pool_output']=pool
                pool_observations.append(dict(trace=name,negative_eligibility=intervals(np.any(z['effective_eligibility']<0,axis=(1,2))),
                    active_reference=intervals(np.any(z['reference_arrivals']>0,axis=(1,2))),
                    opposed_write_intervals=intervals(np.any(opposed,axis=(1,2))),
                    opposed_write_count=int(opposed.sum())))
                if row['kind']!='training':
                    cases.append(dict(seed=seed,condition=condition,identity=identity,trace=name,
                        correct_prediction=intervals(tr[:,8]>0),wrong_prediction=intervals(tr[:,8]<0)))
                    key=(*group,row['kind'],row['weights'])
                    responses.setdefault(key,{})[(row['video'],row['audio'])]=tr[:,6]-tr[:,7]
            if seen!=expected:raise ValueError('Incomplete diagnostic family')
            for group,pairs in responses.items():
                traces[f's{seed}_{condition}_'+ '_'.join(map(str,group))+'_modes']=factorial_modes(np.column_stack([pairs[p] for p in PAIR_ORDER]))
        if s['executed_ticks']!=364+sum(364 if r['kind']=='training' else 96 for r in s['rows']):
            raise ValueError('Incorrect executed tick count')
        for name in ('manifest.json','summary.json'):sources[str(root/name)]=base.digest(root/name)
    if len(seeds)<4:raise ValueError('Need four independent graph seeds')
    output.mkdir();np.savez_compressed(output/'per-tick.npz',**traces)
    result=dict(cases=cases,reference_pathways=pool_observations,checked_ticks=checked,seeds=sorted(seeds),sources=sources,
        producer_sha256=base.digest(__file__),limits='Complete declared course, not sufficient duration by itself. '
        'Selected updates and reference releases independently checked. Reference weights reconstructed '
        'from local tick 1 and soma/output from tick 2; earlier in-flight history explicitly unchecked. '
        'No learned supervision, order generalization, semantic recognition or consciousness established.')
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(seeds=result['seeds'],checked_ticks=checked,cases=len(cases))),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',type=Path,nargs='+');p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();analyze(a.roots,a.output)
