"""Factorial timing/shape test of local eligibility in the unchanged sweep.

Two kernel shapes, one or eight leaky stages, crossed with the inherited mean
age or the nominal linear verification-path mean. Unit DC gain in all cases.
No forward edge, body, sensory schedule, basal rate or task criterion changes.
Each condition has 1024 acquisition ticks then matched intact/reset 512-tick
continuations with adaptation active. The old condition must replay retained
acquisition and intact continuation exactly on all pre-existing recorder fields.
"""
import argparse
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
import shutil

import numpy as np

from . import context_organization as base
from .active_sweep_probe import record, PhysicalDelay
from .active_sweep_memory import reset_selected, restore
from .crossed_av_continuation import isolated_rng
from ..components.body.loaded_hinge import LoadedHinge
from neuron.extensions.experimental.cascade_eligibility import CascadeEligibilityNeuron


CONDITIONS = {'old': (1,False), 'mean_only': (1,True),
              'shape_only': (8,False), 'matched_cascade': (8,True)}


def kernel_means(cfg, groups):
    nodes = {n['id']:n for n in cfg['neurons']}
    points = {(p['neuron_id'],p['synapse_id']):p for p in cfg['synaptic_points'] if p['type']=='postsynaptic'}
    edges = {(e['source_neuron'],e['target_neuron']):e['target_synapse'] for e in cfg['connections']}
    means=[];old=[]
    for predictor in groups['prediction']:
        p=nodes[predictor];md=p['metadata']
        dx=math.exp(-1/md['prediction_tau_context']);old.append(dx/(1-dx))
        de=math.exp(-1/md['prediction_tau_error'])
        for comparator in groups['error_positive']+groups['error_negative']:
            sid=edges[predictor,comparator]
            for port in md['prediction_ports']:
                # Context dendrite; predictor membrane; prediction cleft and
                # comparator dendrite/membrane; teaching cleft; receptor EMA;
                # previous-error convention. A held-linear-path estimate only.
                means.append(points[predictor,port]['distance_to_hillock']+
                    (p['params']['lambda_param']-1)+1+
                    points[comparator,sid]['distance_to_hillock']+
                    (nodes[comparator]['params']['lambda_param']-1)+1+de/(1-de)+1)
    if len(set(means))!=1 or len(set(old))!=1:
        raise ValueError('This preparation does not have one nominal verification age')
    return old[0],means[0]


def configure(original, groups, condition):
    stages,matched=CONDITIONS[condition];cfg=deepcopy(original)
    old,mean=kernel_means(cfg,groups)
    for n in cfg['neurons']:
        if n['id'] in groups['prediction']:
            n['metadata']['prediction_credit_stages']=stages
            if matched:n['metadata']['prediction_credit_mean']=mean
    cfg['metadata']['credit_kernel']=dict(condition=condition,stages=stages,mean_age=mean if matched else old,
        nominal_verification_mean=mean,unit_dc_gain=True,
        limits='One declared local eligibility hypothesis; no fitted lag or learned timing. '
        'The linear-path mean does not include nonlinear gating or closed-loop sensitivities.')
    return cfg


def record_credit(net,body,delay,features,groups,ticks):
    predictors=[net.network.neurons[n] for n in groups['prediction']]
    initial=np.array([n.credit_states.copy() for n in predictors]);states=[]
    original=net.run_tick
    def step():
        result=original();states.append(np.array([n.credit_states.copy() for n in predictors]));return result
    net.run_tick=step
    try:data=record(net,body,delay,features,groups,ticks=ticks)
    finally:net.run_tick=original
    data.update(credit_initial=initial,credit_states=np.asarray(states),
                credit_decay=np.array([n.credit_decay for n in predictors]),
                credit_mean=np.array([n.credit_mean for n in predictors]))
    return data


def run(parent,output):
    from .active_sweep_credit_analysis import audit
    parent,output=(Path(p).resolve() for p in (parent,output))
    if output.exists():raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free<3*1024**3:raise OSError('Need 3 GiB reserve')
    m=json.loads((parent/'manifest.json').read_text());s=json.loads((parent/'summary.json').read_text())
    oldrow=next(r for r in s['rows'] if r['condition']=='loaded_fused')
    original=json.loads((parent/'loaded_fused.json').read_text());g=m['groups']['loaded_fused'];seed=m['seed']
    if base.digest(parent/'loaded_fused.json')!=m['config_hashes']['loaded_fused']:raise ValueError('Parent graph changed')
    hashes=dict(m['source_hashes'])
    for obj in (run,audit,CascadeEligibilityNeuron):
        p=str(Path(inspect.getfile(obj)).resolve());hashes[p]=base.digest(p)
    for p,h in hashes.items():
        if base.digest(p)!=h:raise ValueError('Source changed: '+p)
    if base.digest(m['physical_source'])!=m['physical_sha256']:raise ValueError('Media changed')
    with np.load(m['physical_source']) as f:features={k:f[k] for k in ('visual','auditory')}
    memory=parent.parent/f'20260909_active_sweep_memory_seed{seed}'
    ms=json.loads((memory/'summary.json').read_text())
    refs={'train':(parent/oldrow['file'],oldrow['sha256'])}
    for kind in ('intact','reset'):
        r=next(r for r in ms['rows'] if r['condition']==kind);refs[kind]=(memory/r['file'],r['sha256'])
    for p,h in refs.values():
        if base.digest(p)!=h:raise ValueError('Baseline reference changed')
    output.mkdir();rows=[];configs={};specs={}
    for condition in CONDITIONS:
        cfg=configure(original,g,condition);config=output/f'{condition}.json'
        config.write_text(base.encode(cfg)+'\n');configs[condition]=base.digest(config)
        specs[condition]=cfg['metadata']['credit_kernel']
        net=base.fresh(config,seed,CascadeEligibilityNeuron)[0];body=LoadedHinge(.8);delay=PhysicalDelay()
        for kind in ('train','intact','reset'):
            if shutil.disk_usage(output).free<3*1024**3:raise OSError('Storage reserve reached')
            with isolated_rng():
                if kind!='train':
                    net,body,delay=restore(output/f'{condition}-acquired.paula',output/f'{condition}-acquired-body.npz')
                    if kind=='reset':reset_selected(net,g)
                data=record_credit(net,body,delay,features,g,1024 if kind=='train' else 512)
                audit(data,cfg,g,features)
                exact=0
                if condition=='old':
                    with np.load(refs[kind][0]) as z:
                        for key in z.files:
                            if not np.array_equal(z[key],data[key]):raise ValueError('Baseline replay differs: '+key)
                    exact=len(data['body'])
                path=output/f'{condition}-{kind}.npz';np.savez_compressed(path,**data)
                tag='acquired' if kind=='train' else kind+'-final'
                neural=output/f'{condition}-{tag}.paula';base.save_checkpoint(net,neural,sources=list(hashes))
                physical=output/f'{condition}-{tag}-body.npz'
                np.savez_compressed(physical,state=body.state(),delay=delay.state(),gate=[body.crossings,body.next_gate])
                rows.append(dict(condition=condition,kind=kind,file=path.name,sha256=base.digest(path),
                    ticks=len(data['body']),exact_replay_ticks=exact,checkpoint=neural.name,
                    checkpoint_sha256=base.digest(neural),physical=physical.name,physical_sha256=base.digest(physical)))
            print(base.encode(dict(seed=seed,condition=condition,kind=kind,crossings=body.crossings)),flush=True)
    if any(base.digest(p)!=h for p,h in hashes.items()):raise ValueError('Source changed during experiment')
    manifest=dict(parent=str(parent),parent_manifest_sha256=base.digest(parent/'manifest.json'),seed=seed,
        source_hashes=hashes,groups=g,config_hashes=configs,kernels=specs,media=m['physical_source'],
        media_sha256=m['physical_sha256'],references={str(p):h for p,h in refs.values()},limits=__doc__)
    (output/'manifest.json').write_text(base.encode(manifest)+'\n')
    result=dict(rows=rows,executed_ticks=8192,exact_replay_ticks=2048)
    (output/'summary.json').write_text(base.encode(result)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.parent,a.output)
