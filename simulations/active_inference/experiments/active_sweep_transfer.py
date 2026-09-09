"""Resistance removal from the acquired PAULA sweep, with continuing adaptation.

Branch the matched cascade at global tick 4304 into drag .8 or 0., retaining
spring, gates, full neural/body state and the 64-sample sensory history. In
each world run 1024 ticks with retained or zeroed selected predictive weights.
At the removal/intact endpoint, compare 328 ticks of continued acquired weights
against restored pre-removal weights, keeping every other state unchanged.
Reproduce 64 uninterrupted ticks by executable reload. The unchanged loaded
world must reproduce both previous 512-tick probes exactly.

Crossings after removing resistance are not evidence of learned compensation.
Prediction, physical action, local changes and their causal usefulness are
separate observations. No change flag, host policy or training switch enters
the brain. No neural equation or anatomy is altered.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import context_organization as base
from .active_sweep_acquisition import load, reserve, replay_prefix, continuation
from .active_sweep_analysis import mechanics, verify_predictive_arrivals
from .active_sweep_credit import record_credit
from .active_sweep_memory import restore, reset_selected
from .active_sweep_memory_analysis import stroke_effect
from .active_sweep_probe import PhysicalDelay
from .body_state_memory_analysis import intervals
from .crossed_av_continuation import isolated_rng
from .eligibility_reference_intervention_analysis import onset
from .magnitude_feedback_analysis import verify_returns
from ..components.body.loaded_hinge import LoadedHinge, DT


START, TRANSFER, PROBE, REPLAY = 4304, 1024, 328, 64
WORLDS = {'loaded': .8, 'released': 0.}
INITIALS = ('body_initial','delay_initial','context_initial','error_initial',
            'terminal_initial','gate_initial','credit_initial')


def match_initial(a,b,weights=True):
    for key in INITIALS+(('weights_initial',) if weights else ()):
        if not np.array_equal(a[key],b[key]):
            raise ValueError('Unmatched initial state: '+key)


def replace_weights(net,groups,weights):
    predictors=[net.network.neurons[n] for n in groups['prediction']]
    if weights.shape!=(2,len(predictors[0].prediction_ports)) or not np.isfinite(weights).all():
        raise ValueError('Invalid selected weight shape')
    for n,values in zip(predictors,weights):
        if np.any(values<0) or np.any(values>n.prediction_cap):raise ValueError('Invalid selected weights')
        for sid,value in zip(n.prediction_ports,values):
            n.postsynaptic_points[sid].u_i.info=float(value)


def audit(z,cfg,g,features,drag):
    """Fixed-cascade neural audit plus a declared, independently replayed world.

    Kept separate from the archived fixed-world auditor so previous executable
    checkpoints and source fingerprints remain valid. The original loaded
    records cross-check both auditors; this does not relax their tolerance.
    """
    for key,value in z.items():
        if not np.isfinite(value).all():raise ValueError('Nonfinite field: '+key)
    spec=cfg['metadata']['credit_kernel'];mean=spec['mean_age'];stages=spec['stages']
    if spec['condition']!='matched_cascade' or stages!=8:raise ValueError('Unexpected learning condition')
    d=mean/(mean+stages);de=math.exp(-1/4);ids=list(z['neuron_ids'])
    if not np.array_equal(z['physical_parameters'],[drag,.15,.008,1.]):raise ValueError('Undeclared world')
    if not np.array_equal(z['credit_mean'],[mean]*2) or not np.array_equal(z['credit_decay'],[d]*2):
        raise ValueError('Undeclared credit kernel')
    state=z['credit_initial'].copy();q=z['weights_initial'].copy();error=z['error_initial'].copy()
    if z['credit_states'].shape!=(len(z['body']),2,stages,q.shape[-1]):raise ValueError('Credit dimensions differ')
    for t,arrivals in enumerate(z['arrivals']):
        for j in range(2):
            incoming=arrivals[j]
            for k in range(stages):
                state[j,k]=d*state[j,k]+(1-d)*incoming;incoming=state[j,k]
        eta=1e-5*(1+499*abs(error)/(.01+abs(error)))
        q=np.clip(q+eta[:,None]*error[:,None]*state[:,-1],0.,1.)
        for actual,expected in ((z['credit_states'][t],state),(z['weights'][t],q),
                                (z['eta'][t],eta),(z['errors'][t,:,0],error)):
            if np.max(abs(actual-expected))>2e-12:raise ValueError('Learning recurrence differs')
        error=de*error+(1-de)*z['errors'][t,:,1]
        if np.max(abs(error-z['errors'][t,:,2]))>2e-12 or np.any(eta<=0):raise ValueError('Error receptor differs')
    nodes={n['id']:n for n in cfg['neurons']};terms=list(map(tuple,z['terminal_ids']))
    edges={(e['target_neuron'],e['target_synapse']):(e['source_neuron'],e['source_terminal']) for e in cfg['connections']}
    for j,nid in enumerate(g['prediction']):
        incoming=[]
        for port,polarity in nodes[nid]['metadata']['prediction_error_ports']:
            source,terminal=edges[nid,port]
            out=z['cells'][:-1,ids.index(source),base.FIELDS.index('O')]
            info=z['terminal_info'][:-1,terms.index((source,terminal))]
            incoming.append(((out*info).astype(np.float32),polarity))
        expected=np.array([float(sum(a[t]*sign for a,sign in incoming)) for t in range(len(z['body'])-1)])
        if not np.array_equal(expected,z['errors'][1:,j,1]):raise ValueError('Teaching differs from neural release')
    body=LoadedHinge(drag);body.restore(z['body_initial'],crossings=int(z['gate_initial'][0]),next_gate=int(z['gate_initial'][1]))
    delay=PhysicalDelay(z['delay_initial']);start=round(body.data.time/DT)
    motor=z['cells'][:,[ids.index(n) for n in g['muscle']],base.FIELDS.index('O')]
    if not np.array_equal(motor[:,0]-motor[:,1],z['neural_command']):raise ValueError('Motor command differs')
    for t,row in enumerate(z['body']):
        position,velocity=float(body.data.qpos[0]),float(body.data.qvel[0]);torque=-.15*position-drag*velocity
        raw=np.maximum([torque/.2,-torque/.2,position/.05,-position/.05,velocity/.2,-velocity/.2],0.)
        delivered=delay.step(raw);tick=start+t
        expected=np.r_[features['visual'][tick%300],features['auditory'][tick%300],1.,0.,delivered]
        if not np.array_equal(raw,z['raw_afferents'][t]) or not np.array_equal(expected,z['drive'][t]):
            raise ValueError('Physical sensory delivery differs')
        if z['birth_input'][t]!=(5. if tick==0 else 0.) or row[3]!=z['neural_command'][t]:raise ValueError('Undeclared input')
        force,crossed=body.step(float(row[3]))
        if (not np.array_equal(row,[body.data.time,body.data.qpos[0],body.data.qvel[0],row[3],force]) or
                not np.array_equal(z['physical_states'][t],body.state()) or
                not np.array_equal(z['gate'][t],[body.crossings,body.next_gate,crossed])):
            raise ValueError('Physical integration differs')
    if not np.array_equal(delay.state(),z['delay_final']):raise ValueError('Delay endpoint differs')
    verify_predictive_arrivals(z,cfg,g);verify_returns(z);mechanics(z)
    return 0.


def run(parent,output):
    parent,output=(Path(p).resolve() for p in (parent,output))
    if output.exists():raise FileExistsError(output)
    m=json.loads((parent/'manifest.json').read_text());g=m['groups'];seed=m['seed']
    kernel=Path(m['parent']);km=json.loads((kernel/'manifest.json').read_text())
    config=kernel/'matched_cascade.json';cfg=json.loads(config.read_text())
    row=next(r for r in m['rows'] if r['condition']=='matched_cascade' and r['kind']=='to-4304')
    sources=dict(m['sources']);sources[str(parent/'manifest.json')]=base.digest(parent/'manifest.json')
    for r in [r for r in m['rows'] if r['condition']=='matched_cascade' and r['kind'] in ('to-4304','intact','reset')]:
        for key,h in (('file','sha256'),('checkpoint','checkpoint_sha256'),('physical','physical_sha256')):
            sources[str(parent/r[key])]=r[h]
    sources[str(Path(__file__).resolve())]=base.digest(__file__)
    for p,h in sources.items():
        if base.digest(p)!=h:raise ValueError('Evidence changed: '+p)
    features=load(km['media']);acquired=load(parent/row['file']);before=acquired['weights'][-1].copy()
    refs={k:load(parent/next(r['file'] for r in m['rows'] if r['condition']=='matched_cascade' and r['kind']==k)) for k in ('intact','reset')}
    output.mkdir();reserve(output);rows=[]
    manifest=dict(parent=str(parent),seed=seed,groups=g,config=str(config),sources=sources,
                  media=km['media'],start=START,ticks=TRANSFER,probe_ticks=PROBE,worlds=WORLDS,limits=__doc__)
    (output/'protocol.json').write_text(base.encode(manifest)+'\n')

    def save(data,name,net,body,delay,replayed=0):
        audit(data,cfg,g,features,body.drag)
        path=output/f'{name}.npz';np.savez_compressed(path,**data)
        r=dict(name=name,file=path.name,sha256=base.digest(path),drag=body.drag,
               start=net.current_tick-len(data['body']),end=net.current_tick,exact_replay_ticks=replayed)
        if name!='readapt-replay':
            n=output/f'{name}.paula';p=output/f'{name}-body.npz'
            base.save_checkpoint(net,n,sources=list(sources))
            np.savez_compressed(p,state=body.state(),delay=delay.state(),gate=[body.crossings,body.next_gate])
            r.update(checkpoint=n.name,checkpoint_sha256=base.digest(n),physical=p.name,physical_sha256=base.digest(p))
        rows.append(r);reserve(output)
        print(base.encode(dict(seed=seed,name=name,tick=net.current_tick,crossings=body.crossings)),flush=True)

    for world,drag in WORLDS.items():
        for kind in ('intact','reset'):
            with isolated_rng():
                net,body,delay=restore(parent/row['checkpoint'],parent/row['physical']);body.drag=drag
                if net.current_tick!=START:raise ValueError('Wrong acquired clock')
                if kind=='reset':reset_selected(net,g)
                data=record_credit(net,body,delay,features,g,TRANSFER)
                match_initial(refs[kind],data)
                replayed=0
                if world=='loaded':replay_prefix(data,refs[kind]);replayed=512
                save(data,f'{world}-{kind}',net,body,delay,replayed)
                if world=='released' and kind=='intact':
                    late=record_credit(net,body,delay,features,g,PROBE)
                    save(late,'readapt-intact',net,body,delay)
    for name,ticks in (('readapt-replay',REPLAY),('readapt-restored',PROBE)):
        with isolated_rng():
            net,body,delay=restore(output/'released-intact.paula',output/'released-intact-body.npz');body.drag=0.
            if name=='readapt-restored':replace_weights(net,g,before)
            data=record_credit(net,body,delay,features,g,ticks)
            if name=='readapt-replay':replay_prefix(late,data)
            else:
                match_initial(late,data,weights=False)
                if not np.array_equal(data['weights_initial'],before):raise ValueError('Missing pre-removal restoration')
            save(data,name,net,body,delay,REPLAY if name=='readapt-replay' else 0)
    if any(base.digest(p)!=h for p,h in sources.items()):raise ValueError('Evidence changed during run')
    manifest['rows']=rows
    (output/'manifest.json').write_text(base.encode(manifest)+'\n')
    return manifest


def analyze(roots,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    seeds=set();sources={};arrays={};cases=[];checked=0;replayed=0
    for root in map(lambda p:Path(p).resolve(),roots):
        m=json.loads((root/'manifest.json').read_text());g=m['groups'];seed=m['seed'];parent=Path(m['parent'])
        if seed in seeds:raise ValueError('Duplicate seed')
        seeds.add(seed)
        for p,h in m['sources'].items():
            if base.digest(p)!=h:raise ValueError('Evidence changed: '+p)
            sources[p]=h
        cfg=json.loads(Path(m['config']).read_text());features=load(m['media'])
        expected={'loaded-intact','loaded-reset','released-intact','released-reset','readapt-intact','readapt-restored','readapt-replay'}
        if len(m['rows'])!=7 or {r['name'] for r in m['rows']}!=expected:raise ValueError('Incomplete transfer family')
        records={};parent_rows=json.loads((parent/'manifest.json').read_text())['rows']
        for r in m['rows']:
            name=r['name'];drag=.8 if name.startswith('loaded-') else 0.
            for k,h in (('file','sha256'),('checkpoint','checkpoint_sha256'),('physical','physical_sha256')):
                if k not in r:continue
                p=root/r[k]
                if base.digest(p)!=r[h]:raise ValueError('Artifact changed: '+str(p))
                sources[str(p)]=r[h]
            z=load(root/r['file']);n=len(z['body']);records[name]=z
            start=START+TRANSFER if name.startswith('readapt-') else START
            length=REPLAY if name=='readapt-replay' else PROBE if name.startswith('readapt-') else TRANSFER
            if (r['start'],r['end'],n)!=(start,start+length,length) or round(z['body'][0,0]/DT)-1!=start:
                raise ValueError('Wrong duration or clock')
            audit(z,cfg,g,features,drag);checked+=n;replayed+=r['exact_replay_ticks']
            ids=list(z['neuron_ids']);o=z['cells'][:,:,base.FIELDS.index('O')]
            pred=o[:,ids.index(g['prediction'][0])]-o[:,ids.index(g['prediction'][1])]
            key=f's{seed}_{name}';arrays[key]=np.column_stack((z['body'],pred,z['gate']))
            arrays[key+'_learning']=np.concatenate((z['errors'].reshape(n,-1),z['eta']),axis=1)
            if name.startswith('loaded-'):
                kind=name.split('-')[1];r0=next(r for r in parent_rows if r['condition']=='matched_cascade' and r['kind']==kind)
                replay_prefix(z,load(parent/r0['file']))
        replay_prefix(records['readapt-intact'],records['readapt-replay'])
        continuation(records['released-intact'],records['readapt-intact'])
        prior=load(parent/'matched_cascade-to-4304.npz')['weights'][-1]
        if not np.array_equal(prior,records['readapt-restored']['weights_initial']):raise ValueError('Wrong restored memory')
        contrasts={}
        for world in WORLDS:
            a,b=records[world+'-intact'],records[world+'-reset'];match_initial(a,b,weights=False)
            if np.any(b['weights_initial']):raise ValueError('Missing zero-weight intervention')
            effect,strokes=stroke_effect(a,b,g);arrays[f's{seed}_{world}_stroke']=effect
            contrasts[world]=dict(strokes=strokes,greater=intervals(effect>0),lesser=intervals(effect<0))
        a,b=records['released-intact'],records['loaded-intact'];match_initial(a,b)
        changed={k:onset(a[k],b[k]) for k in ('body','raw_afferents','drive','cells','weights','errors','eta','terminal_info')}
        a,b=records['readapt-intact'],records['readapt-restored'];match_initial(a,b,weights=False)
        effect,strokes=stroke_effect(a,b,g);arrays[f's{seed}_readapt_stroke']=effect
        residual={}
        for name in ('readapt-intact','readapt-restored','released-intact','released-reset'):
            t=arrays[f's{seed}_{name}'];residual[name]=t[:,5]-t[:,4]/.2
            arrays[f's{seed}_{name}_residual']=residual[name]
        change=abs(residual['readapt-intact'])-abs(residual['readapt-restored'])
        arrays[f's{seed}_readapt_error_change']=change
        cases.append(dict(seed=seed,world_change_onset=changed,contrasts=contrasts,
            readaptation=dict(strokes=strokes,smaller_current_error=intervals(change<0),larger_current_error=intervals(change>0),
                first_difference={k:onset(a[k],b[k]) for k in ('body','drive','cells','weights','errors')}),
            gates={name:np.flatnonzero(z['gate'][:,2]).tolist() for name,z in records.items()}))
        sources[str(root/'manifest.json')]=base.digest(root/'manifest.json')
    if seeds!={11,23,44,77}:raise ValueError('Need all four declared seeds')
    output.mkdir();np.savez_compressed(output/'per-tick.npz',**arrays)
    result=dict(cases=cases,seeds=sorted(seeds),sources=sources,checked_ticks=checked,
                exact_replay_ticks=replayed,analyzer_sha256=base.digest(__file__),limits=__doc__)
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(checked_ticks=checked,exact_replay_ticks=replayed,cases=cases)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('paths',nargs='+',type=Path)
    p.add_argument('--analyze',action='store_true');p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.analyze:analyze(a.paths,a.output)
    elif len(a.paths)==1:run(a.paths[0],a.output)
    else:p.error('One parent per worker')
