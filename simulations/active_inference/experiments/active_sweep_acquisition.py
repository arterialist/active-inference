"""Continue two retained 596-cell brains without changing their learning or world.

Resume global tick 1536 from the kernel experiment's intact, never reset, state.
Acquire through tick 4304, then branch 512 intact/reset ticks. This is twenty
164-tick motor cycles beyond the earlier tick-1024 intervention. Media continues
at its original phase, so acquisition age is not isolated from sensory history.
Keep full per-tick data and executable checkpoints at 2560, 3584 and 4304.
No parameters, gates or sensory schedules are adjusted to obtain a success.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from . import context_organization as base
from .active_sweep_credit import record_credit
from .active_sweep_credit_analysis import audit
from .active_sweep_memory import exact_prefix, reset_selected, restore
from .active_sweep_memory_analysis import stroke_effect
from .body_state_memory_analysis import intervals
from .crossed_av_continuation import isolated_rng
from .eligibility_reference_intervention_analysis import onset


CONDITIONS = ('old', 'matched_cascade')
START, END, PROBE, REPLAY = 1536, 4304, 512, 64
CHUNKS = (1024, 1024, 720)
SEED_BUDGET = 384 * 1024**2


def replay_prefix(full, short):
    constants = ('credit_initial', 'credit_decay', 'credit_mean')
    for key in constants:
        if not np.array_equal(full[key], short[key]):
            raise ValueError('Cascade replay differs: '+key)
    exact_prefix({k:v for k,v in full.items() if k not in constants},
                 {k:v for k,v in short.items() if k not in constants})


def continuation(previous, current, reset=False):
    pairs = [('body_initial','physical_states'), ('terminal_initial','terminal_info'),
             ('credit_initial','credit_states')]
    if not reset:
        pairs.append(('weights_initial','weights'))
    elif np.any(current['weights_initial']):
        raise ValueError('Missing selected-weight reset')
    for initial, final in pairs:
        if not np.array_equal(current[initial], previous[final][-1]):
            raise ValueError('Continuation differs: '+initial)
    for a,b in ((current['delay_initial'],previous['delay_final']),
                (current['gate_initial'],previous['gate'][-1,:2]),
                (current['error_initial'],previous['errors'][-1,:,2])):
        if not np.array_equal(a,b):
            raise ValueError('Continuation delay, gate or error differs')


def load(path):
    with np.load(path) as z:
        return {k:z[k] for k in z.files}


def reserve(output):
    if shutil.disk_usage(output).free < 3*1024**3:
        raise OSError('Three GiB reserve reached')
    if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > SEED_BUDGET:
        raise OSError('Declared per-seed artifact budget exceeded')


def run(parent, output):
    parent,output=(Path(p).resolve() for p in (parent,output))
    if output.exists():
        raise FileExistsError(output)
    m=json.loads((parent/'manifest.json').read_text())
    s=json.loads((parent/'summary.json').read_text());g=m['groups']
    sources={**m['source_hashes'],m['media']:m['media_sha256']}
    for name in ('manifest.json','summary.json'):
        sources[str(parent/name)]=base.digest(parent/name)
    for c in CONDITIONS:
        sources[str(parent/f'{c}.json')]=m['config_hashes'][c]
        r=next(r for r in s['rows'] if r['condition']==c and r['kind']=='intact')
        for k,h in (('file','sha256'),('checkpoint','checkpoint_sha256'),('physical','physical_sha256')):
            sources[str(parent/r[k])]=r[h]
    sources[str(Path(__file__).resolve())]=base.digest(__file__)
    for p,h in sources.items():
        if base.digest(p)!=h:
            raise ValueError('Parent evidence changed: '+p)
    features=load(m['media']);output.mkdir();reserve(output);rows=[]
    manifest=dict(parent=str(parent),seed=m['seed'],groups=g,sources=sources,
                  start=START,end=END,probe_ticks=PROBE,replay_ticks=REPLAY,
                  chunks=CHUNKS,conditions=CONDITIONS,byte_budget=SEED_BUDGET,limits=__doc__)
    (output/'protocol.json').write_text(base.encode(manifest)+'\n')

    def save(data, condition, kind, net, body, delay):
        tag=f'{condition}-{kind}';path=output/f'{tag}.npz'
        np.savez_compressed(path,**data)
        row=dict(condition=condition,kind=kind,file=path.name,sha256=base.digest(path),
                 start=net.current_tick-len(data['body']),end=net.current_tick)
        if kind!='replay':
            neural=output/f'{tag}.paula';physical=output/f'{tag}-body.npz'
            base.save_checkpoint(net,neural,sources=list(sources))
            np.savez_compressed(physical,state=body.state(),delay=delay.state(),gate=[body.crossings,body.next_gate])
            row.update(checkpoint=neural.name,checkpoint_sha256=base.digest(neural),
                       physical=physical.name,physical_sha256=base.digest(physical))
        rows.append(row);reserve(output)
        print(base.encode(dict(seed=m['seed'],condition=condition,kind=kind,
                               tick=net.current_tick,crossings=body.crossings)),flush=True)

    for condition in CONDITIONS:
        cfg=json.loads((parent/f'{condition}.json').read_text())
        row=next(r for r in s['rows'] if r['condition']==condition and r['kind']=='intact')
        previous=load(parent/row['file'])
        with isolated_rng():
            net,body,delay=restore(parent/row['checkpoint'],parent/row['physical'])
            if net.current_tick!=START:
                raise ValueError('Wrong acquired clock')
            if not np.array_equal(base.cellular(list(net.network.neurons.values())),previous['cells'][-1]):
                raise ValueError('Acquired cells differ from recorded endpoint')
            for ticks in CHUNKS:
                reserve(output);data=record_credit(net,body,delay,features,g,ticks)
                continuation(previous,data);audit(data,cfg,g,features)
                save(data,condition,f'to-{net.current_tick}',net,body,delay);previous=data
            if net.current_tick!=END:
                raise ValueError('Acquisition horizon differs')
            acquired=previous
            intact=record_credit(net,body,delay,features,g,PROBE)
            continuation(acquired,intact);audit(intact,cfg,g,features)
            save(intact,condition,'intact',net,body,delay)
            for kind,ticks in (('replay',REPLAY),('reset',PROBE)):
                net,body,delay=restore(output/f'{condition}-to-{END}.paula',output/f'{condition}-to-{END}-body.npz')
                if kind=='reset':
                    changed=reset_selected(net,g)
                    np.savez_compressed(output/f'{condition}-intervention.npz',selected_before=changed)
                data=record_credit(net,body,delay,features,g,ticks)
                continuation(acquired,data,reset=kind=='reset');audit(data,cfg,g,features)
                if kind=='replay':
                    replay_prefix(intact,data)
                else:
                    for key in ('context_initial','credit_initial','terminal_initial','error_initial','body_initial','delay_initial','gate_initial'):
                        if not np.array_equal(intact[key],data[key]):
                            raise ValueError('Unmatched reset branch: '+key)
                save(data,condition,kind,net,body,delay)
    if any(base.digest(p)!=h for p,h in sources.items()):
        raise ValueError('Evidence changed during run')
    manifest['rows']=rows
    (output/'manifest.json').write_text(base.encode(manifest)+'\n')
    return manifest


def analyze(roots,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    seeds=set();cases=[];arrays={};sources={};checked=0
    for root in map(lambda p:Path(p).resolve(),roots):
        m=json.loads((root/'manifest.json').read_text());seed=m['seed'];g=m['groups'];parent=Path(m['parent'])
        if seed in seeds:raise ValueError('Duplicate seed')
        seeds.add(seed)
        for p,h in m['sources'].items():
            if base.digest(p)!=h:raise ValueError('Evidence changed: '+p)
            sources[p]=h
        pm=json.loads((parent/'manifest.json').read_text());ps=json.loads((parent/'summary.json').read_text())
        features=load(pm['media'])
        expected={(c,k) for c in CONDITIONS for k in ('to-2560','to-3584','to-4304','intact','reset','replay')}
        if len(m['rows'])!=12 or {(r['condition'],r['kind']) for r in m['rows']}!=expected:
            raise ValueError('Incomplete acquisition family')
        for condition in CONDITIONS:
            cfg=json.loads((parent/f'{condition}.json').read_text())
            pr=next(r for r in ps['rows'] if r['condition']==condition and r['kind']=='intact')
            previous=load(parent/pr['file']);acquired=None;probes={};course=[]
            for row in [r for r in m['rows'] if r['condition']==condition]:
                for k,h in (('file','sha256'),('checkpoint','checkpoint_sha256'),('physical','physical_sha256')):
                    if k not in row:continue
                    p=root/row[k]
                    if base.digest(p)!=row[h]:raise ValueError('Artifact changed: '+str(p))
                    sources[str(p)]=row[h]
                data=load(root/row['file']);kind=row['kind'];n=len(data['body'])
                start=round(data['body'][0,0]/.004)-1
                if (start,start+n)!=(row['start'],row['end']):raise ValueError('Recorded clock differs')
                audit(data,cfg,g,features);checked+=n
                out=data['cells'][:,:,base.FIELDS.index('O')];ids=list(data['neuron_ids'])
                pred=out[:,ids.index(g['prediction'][0])]-out[:,ids.index(g['prediction'][1])]
                trace=np.column_stack((data['body'],pred,data['gate']))
                if kind.startswith('to-'):
                    continuation(previous,data);previous=data;acquired=data;course.append(trace)
                else:
                    continuation(acquired,data,reset=kind=='reset');probes[kind]=data
                    arrays[f's{seed}_{condition}_{kind}']=trace
            replay_prefix(probes['intact'],probes['replay'])
            a,b=probes['intact'],probes['reset']
            if len(a['body'])!=PROBE or len(b['body'])!=PROBE or len(probes['replay']['body'])!=REPLAY:
                raise ValueError('Probe duration differs')
            if sum(len(t) for t in course)!=END-START:raise ValueError('Course duration differs')
            effect,strokes=stroke_effect(a,b,g)
            arrays[f's{seed}_{condition}_course']=np.concatenate(course)
            arrays[f's{seed}_{condition}_stroke_effect']=effect
            cases.append(dict(seed=seed,condition=condition,strokes=strokes,
                greater=intervals(effect>0),lesser=intervals(effect<0),
                first_difference={k:onset(a[k],b[k]) for k in ('weights','cells','body','errors','drive')},
                gate_ticks={k:(END+np.flatnonzero(z['gate'][:,2])).tolist() for k,z in probes.items()},
                acquisition_gate_ticks=(START+np.flatnonzero(arrays[f's{seed}_{condition}_course'][:,-1])).tolist()))
        sources[str(root/'manifest.json')]=base.digest(root/'manifest.json')
    if seeds!={11,23,44,77}:raise ValueError('Need all four declared seeds')
    output.mkdir();np.savez_compressed(output/'per-tick.npz',**arrays)
    result=dict(cases=cases,seeds=sorted(seeds),sources=sources,checked_ticks=checked,
                exact_replay_ticks=8*REPLAY,analyzer_sha256=base.digest(__file__),limits=__doc__)
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(checked_ticks=checked,exact_replay_ticks=8*REPLAY,cases=cases)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('paths',nargs='+',type=Path)
    p.add_argument('--analyze',action='store_true');p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.analyze:analyze(a.paths,a.output)
    elif len(a.paths)==1:run(a.paths[0],a.output)
    else:p.error('One parent per worker')
