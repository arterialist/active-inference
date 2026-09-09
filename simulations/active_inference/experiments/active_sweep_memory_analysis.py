"""Full-trajectory audit of retained predictive weights in the resistive sweep.

Physical stroke advance is measured in the direction of the unchanged neural
motor rhythm, relative to each branch's own angle just before that stroke.
It is a diagnostic, not a neural input or a replacement for gate completion.
Weight reset retains other memory and permits immediate reacquisition.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import context_organization as base
from .active_sweep_memory import exact_prefix
from .active_sweep_probe import verify
from .active_sweep_analysis import mechanics, verify_predictive_arrivals
from .body_state_memory_analysis import intervals
from .eligibility_reference_intervention_analysis import onset, changed_ticks
from ..components.body.loaded_hinge import LoadedHinge


def stroke_effect(a, b, groups):
    ids = list(a['neuron_ids']); oi = base.FIELDS.index('O')
    clock_a = a['cells'][:,[ids.index(n) for n in groups['cpg']],oi]
    clock_b = b['cells'][:,[ids.index(n) for n in groups['cpg']],oi]
    if not np.array_equal(clock_a,clock_b):
        raise ValueError('Cannot phase-match different motor clocks')
    events = sorted([(int(t),direction) for col,direction in ((0,1),(2,-1))
                     for t in np.flatnonzero(clock_a[:,col]>0)])
    effect = np.full(len(clock_a),np.nan); rows=[]
    qa, qb = a['body'][:,1],b['body'][:,1]
    arm=LoadedHinge(.8);arm.restore(a['body_initial']);initial_a=float(arm.data.qpos[0])
    arm.restore(b['body_initial']);initial_b=float(arm.data.qpos[0])
    for i,(start,direction) in enumerate(events):
        end=events[i+1][0] if i+1<len(events) else len(clock_a)
        aa=qa[start-1] if start else initial_a;bb=qb[start-1] if start else initial_b
        effect[start:end]=direction*((qa[start:end]-aa)-(qb[start:end]-bb))
        rows.append(dict(start=start,end=end,direction=direction,complete=i+1<len(events),
                         intact_advance=float(direction*(qa[end-1]-aa)),
                         reset_advance=float(direction*(qb[end-1]-bb)),
                         endpoint_effect=float(effect[end-1])))
    return effect,rows


def analyze(roots, output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    sources={};arrays={};cases=[];seeds=set();checked=0
    for root in map(lambda p:Path(p).resolve(),roots):
        m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text())
        seed=m['seed'];parent=Path(m['parent']);groups=m['groups']
        if seed in seeds:raise ValueError('Duplicate seed')
        seeds.add(seed)
        for name in ('manifest','summary'):
            path=parent/f'{name}.json'
            if base.digest(path)!=m[f'parent_{name}_sha256']:raise ValueError('Acquisition changed')
            sources[str(path)]=base.digest(path)
        pm=json.loads((parent/'manifest.json').read_text())
        cfg=json.loads((parent/'loaded_fused.json').read_text())
        if base.digest(parent/'loaded_fused.json')!=pm['config_hashes']['loaded_fused']:
            raise ValueError('Graph changed')
        if base.digest(pm['physical_source'])!=pm['physical_sha256']:raise ValueError('Media changed')
        with np.load(pm['physical_source']) as f:features={k:f[k] for k in ('visual','auditory')}
        for p,h in m['source_hashes'].items():
            if base.digest(p)!=h:raise ValueError('Source changed: '+p)
            sources[p]=h
        identities=[r['condition'] for r in s['rows']]
        if len(identities)!=3 or set(identities)!={'intact','reset','replay'}:
            raise ValueError('Incomplete matched branches')
        data={}
        for row in s['rows']:
            path=root/row['file']
            if base.digest(path)!=row['sha256']:raise ValueError('Trace changed')
            sources[str(path)]=row['sha256']
            for key in ('checkpoint','physical'):
                if key in row and base.digest(root/row[key])!=row[key+'_sha256']:
                    raise ValueError('Final state changed')
            with np.load(path) as f:z={k:f[k] for k in f.files}
            if len(z['body'])!=row['ticks']:raise ValueError('Duration differs')
            if not np.array_equal(z['physical_parameters'],[.8,.15,.008,1.]):
                raise ValueError('Physical condition differs')
            verify(z,groups,features);verify_predictive_arrivals(z,cfg,groups);mech=mechanics(z)
            data[row['condition']]=z;checked+=len(z['body'])
            if row['condition']=='replay':continue
            ids=list(z['neuron_ids']);out=z['cells'][:,:,base.FIELDS.index('O')]
            pred=out[:,ids.index(groups['prediction'][0])]-out[:,ids.index(groups['prediction'][1])]
            key=f's{seed}_{row["condition"]}'
            arrays[key]=np.column_stack((z['body'],pred,z['gate'],mech))
        a,b=data['intact'],data['reset'];exact_prefix(a,data['replay'])
        if len(a['body'])!=m['ticks'] or len(b['body'])!=m['ticks'] or m['ticks']<512:
            raise ValueError('Need at least three intact and reset cycles')
        if s['executed_ticks']!=2*m['ticks']+96 or len(data['replay']['body'])!=96:
            raise ValueError('Incorrect replay count')
        for key in ('body_initial','delay_initial','context_initial','error_initial','terminal_initial','gate_initial'):
            if not np.array_equal(a[key],b[key]):raise ValueError('Unmatched initial field: '+key)
        if np.any(b['weights_initial']) or not np.any(a['weights_initial']):
            raise ValueError('Missing acquired-weight intervention')
        name=f's{seed}_effect';first={}
        for field in ('cells','weights','terminal_info','errors','eta','drive','raw_afferents','body'):
            first[field]=onset(a[field],b[field]);arrays[name+'_'+field]=changed_ticks(a[field],b[field])
        effect,strokes=stroke_effect(a,b,groups)
        arrays[name+'_stroke']=effect
        arrays[name+'_body_delta']=a['body']-b['body']
        ids=list(a['neuron_ids']);populations={}
        for role,members in groups.items():
            ix=[ids.index(n) for n in members]
            populations[role]=onset(a['cells'][:,ix,base.FIELDS.index('O')],b['cells'][:,ix,base.FIELDS.index('O')])
        cases.append(dict(seed=seed,first_difference=first,population_output=populations,strokes=strokes,
                         greater_stroke_advance=intervals(effect>0),lesser_stroke_advance=intervals(effect<0),
                         gate_ticks={c:np.flatnonzero(data[c]['gate'][:,2]).tolist() for c in ('intact','reset')},
                         crossings={c:int(data[c]['gate'][-1,0]-data[c]['gate_initial'][0]) for c in ('intact','reset')}))
        for name in ('manifest.json','summary.json'):sources[str(root/name)]=base.digest(root/name)
    if len(seeds)<4:raise ValueError('Need four seeds')
    output.mkdir();np.savez_compressed(output/'per-tick.npz',**arrays)
    result=dict(cases=cases,seeds=sorted(seeds),checked_ticks=checked,sources=sources,
                analyzer_sha256=base.digest(__file__),limits=__doc__,
                columns=['time','angle','velocity','command','environmental_torque','signed_prediction',
                         'crossings','next_gate','crossed','motor_torque','load_torque','passive_torque',
                         'motor_work','load_work','damping_work','kinetic_change','implicit_loss'])
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(checked_ticks=checked,cases=cases)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',nargs='+',type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();analyze(a.roots,a.output)
