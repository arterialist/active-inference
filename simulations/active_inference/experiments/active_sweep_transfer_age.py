"""Same-age counterfactual for adaptation after resistance removal.

Use the removal/intact state at tick 5328, but transplant only predictive
weights acquired through the matched still-loaded continuation. Compare with
the already recorded removal-acquired weights in exactly that same later state.
This separates weight age from changed physical experience. All adaptation and
other neural/body state continue. The donor graph and seed must match.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import context_organization as base
from .active_sweep_acquisition import load, reserve
from .active_sweep_transfer import audit, replace_weights, match_initial, PROBE, START, TRANSFER
from .active_sweep_memory import restore
from .active_sweep_credit import record_credit
from .active_sweep_memory_analysis import stroke_effect
from .body_state_memory_analysis import intervals
from .crossed_av_continuation import isolated_rng
from .eligibility_reference_intervention_analysis import onset


def run(parent,output):
    parent,output=(Path(p).resolve() for p in (parent,output))
    if output.exists():raise FileExistsError(output)
    m=json.loads((parent/'manifest.json').read_text());g=m['groups']
    sources=dict(m['sources']);sources[str(parent/'manifest.json')]=base.digest(parent/'manifest.json')
    chosen={r['name']:r for r in m['rows'] if r['name'] in ('loaded-intact','released-intact','readapt-intact')}
    if len(chosen)!=3:raise ValueError('Incomplete parent family')
    for r in chosen.values():
        for k,h in (('file','sha256'),('checkpoint','checkpoint_sha256'),('physical','physical_sha256')):
            sources[str(parent/r[k])]=r[h]
    sources[str(Path(__file__).resolve())]=base.digest(__file__)
    for p,h in sources.items():
        if base.digest(p)!=h:raise ValueError('Evidence changed: '+p)
    cfg=json.loads(Path(m['config']).read_text());features=load(m['media'])
    donor=load(parent/chosen['loaded-intact']['file'])['weights'][-1]
    reference=load(parent/chosen['readapt-intact']['file'])
    output.mkdir();reserve(output)
    with isolated_rng():
        r=chosen['released-intact'];net,body,delay=restore(parent/r['checkpoint'],parent/r['physical']);body.drag=0.
        if net.current_tick!=START+TRANSFER:raise ValueError('Wrong acquired clock')
        replace_weights(net,g,donor)
        data=record_credit(net,body,delay,features,g,PROBE)
        match_initial(reference,data,weights=False)
        if not np.array_equal(data['weights_initial'],donor):raise ValueError('Wrong donor weights')
        audit(data,cfg,g,features,0.)
        path=output/'same-age.npz';np.savez_compressed(path,**data)
        neural=output/'same-age.paula';physical=output/'same-age-body.npz'
        base.save_checkpoint(net,neural,sources=list(sources))
        np.savez_compressed(physical,state=body.state(),delay=delay.state(),gate=[body.crossings,body.next_gate])
    reserve(output)
    if any(base.digest(p)!=h for p,h in sources.items()):raise ValueError('Evidence changed during run')
    result=dict(parent=str(parent),seed=m['seed'],sources=sources,groups=g,config=m['config'],media=m['media'],
                file=path.name,sha256=base.digest(path),checkpoint=neural.name,checkpoint_sha256=base.digest(neural),
                physical=physical.name,physical_sha256=base.digest(physical),reference=chosen['readapt-intact']['file'],
                donor=chosen['loaded-intact']['file'],ticks=PROBE,limits=__doc__)
    (output/'manifest.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(seed=m['seed'],ticks=PROBE,crossings=body.crossings)),flush=True)
    return result


def analyze(roots,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    sources={};arrays={};cases=[];seeds=set()
    for root in map(lambda p:Path(p).resolve(),roots):
        m=json.loads((root/'manifest.json').read_text());seed=m['seed'];g=m['groups'];parent=Path(m['parent'])
        if seed in seeds:raise ValueError('Duplicate seed')
        seeds.add(seed)
        for p,h in m['sources'].items():
            if base.digest(p)!=h:raise ValueError('Evidence changed: '+p)
            sources[p]=h
        for k,h in (('file','sha256'),('checkpoint','checkpoint_sha256'),('physical','physical_sha256')):
            p=root/m[k]
            if base.digest(p)!=m[h]:raise ValueError('Artifact changed: '+str(p))
            sources[str(p)]=m[h]
        a=load(parent/m['reference']);b=load(root/m['file']);match_initial(a,b,weights=False)
        donor=load(parent/m['donor'])['weights'][-1]
        if not np.array_equal(b['weights_initial'],donor):raise ValueError('Wrong same-age memory')
        cfg=json.loads(Path(m['config']).read_text());features=load(m['media'])
        for z in (a,b):
            if len(z['body'])!=PROBE or round(z['body'][0,0]/.004)-1!=START+TRANSFER:raise ValueError('Wrong duration')
            audit(z,cfg,g,features,0.)
        ids=list(a['neuron_ids']);pi=[ids.index(n) for n in g['prediction']]
        traces=[]
        for z in (a,b):
            pred=z['cells'][:,pi[0],base.FIELDS.index('O')]-z['cells'][:,pi[1],base.FIELDS.index('O')]
            traces.append(np.column_stack((z['body'],pred,z['gate'])))
        current,other=traces
        change=abs(current[:,5]-current[:,4]/.2)-abs(other[:,5]-other[:,4]/.2)
        effect,strokes=stroke_effect(a,b,g)
        arrays[f's{seed}_current']=current;arrays[f's{seed}_same-age']=other
        arrays[f's{seed}_error_change']=change;arrays[f's{seed}_stroke']=effect
        cases.append(dict(seed=seed,strokes=strokes,smaller_current_error=intervals(change<0),
                          larger_current_error=intervals(change>0),
                          first_difference={k:onset(a[k],b[k]) for k in ('weights','cells','body','drive','errors')},
                          gates={k:np.flatnonzero(z['gate'][:,2]).tolist() for k,z in (('current',a),('same-age',b))}))
        sources[str(root/'manifest.json')]=base.digest(root/'manifest.json')
    if seeds!={11,23,44,77}:raise ValueError('Need all four seeds')
    output.mkdir();np.savez_compressed(output/'per-tick.npz',**arrays)
    result=dict(cases=cases,seeds=sorted(seeds),sources=sources,new_ticks=4*PROBE,checked_ticks=8*PROBE,
                analyzer_sha256=base.digest(__file__),limits=__doc__)
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(new_ticks=4*PROBE,cases=cases)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('paths',nargs='+',type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--analyze',action='store_true');a=p.parse_args()
    if a.analyze:analyze(a.paths,a.output)
    elif len(a.paths)==1:run(a.paths[0],a.output)
    else:p.error('One completed parent per worker')
