"""Independent per-tick audit of the eligibility-kernel factorial experiment.

Reconstruct local cascades, selected learning, neural teaching arrivals,
movement-dependent force, physical integration, delayed sensory inputs, source
release and return events. Compare full acquisition and acquired-weight effects.
Unit-area kernels do not imply equal error-modulated learning exposure.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import context_organization as base
from .active_sweep_analysis import mechanics,verify_predictive_arrivals
from .active_sweep_memory_analysis import stroke_effect
from .active_sweep_probe import PhysicalDelay
from .magnitude_feedback_analysis import verify_returns
from .body_state_memory_analysis import intervals
from .eligibility_reference_intervention_analysis import onset
from ..components.body.loaded_hinge import LoadedHinge,DT


def audit(z,cfg,groups,features):
    for key,value in z.items():
        if not np.isfinite(value).all():raise ValueError('Nonfinite recorded field: '+key)
    nodes={n['id']:n for n in cfg['neurons']};ids=list(z['neuron_ids'])
    predictors=[nodes[n] for n in groups['prediction']]
    spec=cfg['metadata']['credit_kernel'];stages=spec['stages'];mean=spec['mean_age']
    expected_d=math.exp(-1/64) if spec['condition']=='old' else mean/(mean+stages)
    if z['credit_states'].shape!=(len(z['body']),2,stages,z['weights'].shape[-1]):
        raise ValueError('Unexpected cascade dimensions')
    if not np.array_equal(z['credit_mean'],[mean,mean]) or not np.array_equal(z['credit_decay'],[expected_d]*2):
        raise ValueError('Undeclared credit kernel')
    q=z['weights_initial'].copy();state=z['credit_initial'].copy();error=z['error_initial'].copy()
    for t,arrivals in enumerate(z['arrivals']):
        for j in range(2):
            incoming=arrivals[j]
            for k in range(stages):
                state[j,k]=expected_d*state[j,k]+(1-expected_d)*incoming
                incoming=state[j,k]
        eta=1e-5*(1+499*abs(error)/(.01+abs(error)))
        q=np.clip(q+eta[:,None]*error[:,None]*state[:,-1],0.,1.)
        for actual,expected in ((z['credit_states'][t],state),(z['weights'][t],q),
                                (z['eta'][t],eta),(z['errors'][t,:,0],error)):
            if np.max(abs(actual-expected))>2e-12:raise ValueError('Local credit/learning recurrence differs')
        de=math.exp(-1/4);error=de*error+(1-de)*z['errors'][t,:,1]
        if np.max(abs(error-z['errors'][t,:,2]))>2e-12 or np.any(eta<=0):
            raise ValueError('Neural error receptor recurrence differs')
    terms=list(map(tuple,z['terminal_ids']));edges={(e['target_neuron'],e['target_synapse']):
        (e['source_neuron'],e['source_terminal']) for e in cfg['connections']}
    # Tick zero's pending neural releases are preserved in executable state,
    # not inferred from an absent pre-recording source raster.
    for j,node in enumerate(predictors):
        incoming=[]
        for port,polarity in node['metadata']['prediction_error_ports']:
            source,terminal=edges[node['id'],port]
            out=z['cells'][:-1,ids.index(source),base.FIELDS.index('O')]
            info=z['terminal_info'][:-1,terms.index((source,terminal))]
            incoming.append(((out*info).astype(np.float32),polarity))
        expected=np.array([float(sum(a[t]*sign for a,sign in incoming)) for t in range(len(z['body'])-1)])
        if not np.array_equal(expected,z['errors'][1:,j,1]):raise ValueError('Teaching differs from neural release')
    if not np.array_equal(z['physical_parameters'],[.8,.15,.008,1.]):raise ValueError('Physical condition changed')
    body=LoadedHinge(.8);body.restore(z['body_initial'],crossings=int(z['gate_initial'][0]),next_gate=int(z['gate_initial'][1]))
    delay=PhysicalDelay(z['delay_initial']);start=round(body.data.time/DT)
    motor=z['cells'][:,[ids.index(n) for n in groups['muscle']],base.FIELDS.index('O')]
    if not np.array_equal(motor[:,0]-motor[:,1],z['neural_command']):raise ValueError('Motor command differs')
    for t,row in enumerate(z['body']):
        q,v=float(body.data.qpos[0]),float(body.data.qvel[0]);torque=-.15*q-.8*v
        raw=np.maximum([torque/.2,-torque/.2,q/.05,-q/.05,v/.2,-v/.2],0.)
        delivered=delay.step(raw);tick=start+t
        expected=np.r_[features['visual'][tick%300],features['auditory'][tick%300],1.,0.,delivered]
        if not np.array_equal(raw,z['raw_afferents'][t]) or not np.array_equal(expected,z['drive'][t]):
            raise ValueError('Physical sensory delivery differs')
        if z['birth_input'][t]!=(5. if tick==0 else 0.) or row[3]!=z['neural_command'][t]:
            raise ValueError('Undeclared actuator or birth drive')
        force,crossed=body.step(float(row[3]))
        if (not np.array_equal(row,[body.data.time,body.data.qpos[0],body.data.qvel[0],row[3],force])
                or not np.array_equal(z['physical_states'][t],body.state())
                or not np.array_equal(z['gate'][t],[body.crossings,body.next_gate,crossed])):
            raise ValueError('Body integration or gate differs')
    if not np.array_equal(delay.state(),z['delay_final']):raise ValueError('Delay endpoint differs')
    verify_predictive_arrivals(z,cfg,groups);verify_returns(z);mechanics(z)
    return 0.


def analyze(roots,output):
    from .active_sweep_credit import CONDITIONS,configure
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    cases=[];arrays={};sources={};seeds=set();checked=0;replayed=0
    for root in map(lambda p:Path(p).resolve(),roots):
        m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text())
        seed=m['seed'];g=m['groups'];parent=Path(m['parent'])
        if seed in seeds:raise ValueError('Duplicate seed')
        seeds.add(seed)
        if base.digest(parent/'manifest.json')!=m['parent_manifest_sha256']:raise ValueError('Parent changed')
        original=json.loads((parent/'loaded_fused.json').read_text())
        for p,h in {**m['source_hashes'],**m['references'],m['media']:m['media_sha256']}.items():
            if base.digest(p)!=h:raise ValueError('Evidence changed: '+p)
            sources[p]=h
        with np.load(m['media']) as f:features={k:f[k] for k in ('visual','auditory')}
        expected={(c,k) for c in CONDITIONS for k in ('train','intact','reset')}
        if len(s['rows'])!=12 or {(r['condition'],r['kind']) for r in s['rows']}!=expected:
            raise ValueError('Incomplete factorial cases')
        for condition in CONDITIONS:
            cfg=json.loads((root/f'{condition}.json').read_text())
            if base.encode(cfg)!=base.encode(configure(original,g,condition)) or base.digest(root/f'{condition}.json')!=m['config_hashes'][condition]:
                raise ValueError('Undeclared graph difference')
            data={}
            for row in [r for r in s['rows'] if r['condition']==condition]:
                for field,hkey in (('file','sha256'),('checkpoint','checkpoint_sha256'),('physical','physical_sha256')):
                    p=root/row[field]
                    if base.digest(p)!=row[hkey]:raise ValueError('Trace or checkpoint changed')
                    sources[str(p)]=row[hkey]
                with np.load(root/row['file']) as f:z={k:f[k] for k in f.files}
                kind=row['kind'];data[kind]=z
                if len(z['body'])!=(1024 if kind=='train' else 512):raise ValueError('Wrong duration')
                audit(z,cfg,g,features);checked+=len(z['body']);replayed+=row['exact_replay_ticks']
                ids=list(z['neuron_ids']);o=z['cells'][:,:,base.FIELDS.index('O')]
                pred=o[:,ids.index(g['prediction'][0])]-o[:,ids.index(g['prediction'][1])]
                name=f's{seed}_{condition}_{kind}'
                arrays[name]=np.column_stack((z['body'],pred,z['gate']))
                arrays[name+'_wrong_sign']=pred*z['body'][:,4]<0
                if condition=='old':
                    ref=next(p for p in m['references'] if Path(p).name==('loaded_fused.npz' if kind=='train' else kind+'.npz'))
                    with np.load(ref) as f:
                        for key in f.files:
                            if not np.array_equal(f[key],z[key]):raise ValueError('Baseline replay differs')
            a,b=data['intact'],data['reset'];train=data['train']
            for key in ('body_initial','delay_initial','context_initial','error_initial','terminal_initial','gate_initial','credit_initial'):
                if not np.array_equal(a[key],b[key]):raise ValueError('Unmatched acquired branches')
            for initial,previous in (('weights_initial','weights'),('credit_initial','credit_states'),('body_initial','physical_states')):
                if not np.array_equal(a[initial],train[previous][-1]):raise ValueError('Acquisition continuity differs')
            if np.any(b['weights_initial']):raise ValueError('Missing reset')
            effect,strokes=stroke_effect(a,b,g);arrays[f's{seed}_{condition}_stroke_effect']=effect
            cases.append(dict(seed=seed,condition=condition,kernel=m['kernels'][condition],strokes=strokes,
                first_memory_effect={field:onset(a[field],b[field]) for field in ('weights','cells','body','errors','drive')},
                crossings={k:int(z['gate'][-1,0]-z['gate_initial'][0]) for k,z in data.items()},
                wrong_sign_intervals={k:intervals(arrays[f's{seed}_{condition}_{k}_wrong_sign']) for k in data},
                greater_stroke_advance=intervals(effect>0),lesser_stroke_advance=intervals(effect<0)))
        for name in ('manifest.json','summary.json'):sources[str(root/name)]=base.digest(root/name)
    if len(seeds)<4:raise ValueError('Need four graph seeds')
    output.mkdir();np.savez_compressed(output/'per-tick.npz',**arrays)
    result=dict(cases=cases,checked_ticks=checked,exact_replay_ticks=replayed,seeds=sorted(seeds),sources=sources,
                columns=['time','angle','velocity','command','environmental_torque','signed_prediction','crossings','next_gate','crossed'],
                analyzer_sha256=base.digest(__file__),limits=__doc__)
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(checked_ticks=checked,exact_replay_ticks=replayed,
        cases=[{k:c[k] for k in ('seed','condition','crossings')} for c in cases])),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();analyze(a.roots,a.output)
