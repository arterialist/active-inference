"""Partial/corrupted physical cues in the established neural association preparation.

Replay acquired states exactly; vary only receptor stimulation during independent
probes. No cue identity reaches the brain except through the input mask. Neural
consumer outputs, not a fitted classifier, determine completion. This tests a
synthetic building block, not real-media recognition or the full brain goal.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import inspect
import json
from pathlib import Path
import time
import numpy as np
from .composition_probe import encode
from .association_route_probe import digest
from .eligibility_association_probe import run_trial,dynamic_snapshot
from .multimodal_pairing_probe import fresh
from neuron.extensions.experimental.eligibility_trace import EligibilityTraceNeuron


def cue_cases(masks,seed):
    rng=np.random.default_rng(seed+19571);cases=[]
    specifications=(('clean',16,0,1),('omit25',12,0,2),('omit50',8,0,2),
                    ('omit75',4,0,2),('replace25',12,4,2),('balanced',8,8,2),('silence',0,0,1))
    for cue in (0,1):
        own,other=masks['vision'][cue],masks['vision'][1-cue]
        if len(own)!=16 or len(other)!=16 or set(own)&set(other):raise ValueError('Need disjoint 16-receptor acquisition cues')
        for name,correct,foreign,repetitions in specifications:
            for repeat in range(repetitions):
                selected=sorted(map(int,list(rng.choice(own,correct,replace=False))+list(rng.choice(other,foreign,replace=False))))
                cases.append(dict(name=f'{name}-cue{cue}-sample{repeat}',kind=name,cue=cue,sample=repeat,
                                  own_count=correct,foreign_count=foreign,selected_receptors=selected))
    return cases


def run(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    m=json.loads((source/'manifest.json').read_text());cfg=json.loads((source/'config.json').read_text())
    if m['mode']!='eligibility' or m['mapping'] not in ('paired','swapped'):raise ValueError('Need acquired eligibility assignment')
    hashes={**m['source_hashes'],str(Path(__file__).resolve()):digest(__file__),
            str(Path(inspect.getfile(run_trial)).resolve()):digest(inspect.getfile(run_trial))}
    if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('Acquisition runtime changed')
    files={str(p):digest(p) for p in source.iterdir() if p.is_file()}
    cases=cue_cases(m['masks'],m['seed'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'manifest.json').write_text(encode(dict(source=str(source),seed=m['seed'],mapping=m['mapping'],
        source_hashes=hashes,source_files_sha256=files,cases=cases,states=['initial','trained','reset_selected'],
        scope='Synthetic nearest-template cue completion. Balanced mixtures have no assigned correct class. Weak positive adaptation continues.'))+'\n')
    net,_,_,_=fresh(source/'config.json',m['seed'],EligibilityTraceNeuron)
    initial=deepcopy(net);initial_state=dynamic_snapshot(initial);started=time.perf_counter()
    for i,trial in enumerate(m['trials']):
        data=run_trial(net,m['groups'],m['masks'],trial)
        with np.load(source/f'train-{i:03d}.npz') as z:
            if set(z.files)!=set(data) or any(not np.array_equal(data[k],z[k]) for k in data):raise AssertionError('Training replay differs')
    parent=dynamic_snapshot(net)
    with gzip.open(source/'trained-state.json.gz','rt') as f:
        if json.loads(parent)!=json.load(f):raise AssertionError('Full acquired state differs')
    print(encode(dict(stage='training_exact',seconds=time.perf_counter()-started)),flush=True)
    points={(p['neuron_id'],p['synapse_id']):p for p in cfg['synaptic_points'] if p['type']=='postsynaptic'}
    starts={};rows=[]
    for state in ('initial','trained','reset_selected'):
        base=deepcopy(initial if state=='initial' else net)
        expected=json.loads(initial_state if state=='initial' else parent)
        if state=='reset_selected':
            for nid in m['groups']['auditory']:
                for sid in range(32):
                    value=points[nid,sid]['u_i']['info'];base.network.neurons[nid].postsynaptic_points[sid].u_i.info=value
                    expected['neurons'][str(nid)]['synapses'][str(sid)][0]=value
        if json.loads(dynamic_snapshot(base))!=expected:raise AssertionError('Undeclared state intervention')
        start_file=f'{state}-start.json.gz'
        with gzip.open(output/start_file,'wt') as f:f.write(encode(expected)+'\n')
        starts[state]=dict(file=start_file,sha256=digest(output/start_file))
        for case in cases:
            branch=deepcopy(base)
            if json.loads(dynamic_snapshot(branch))!=expected:raise AssertionError('Wrong probe state')
            masks=deepcopy(m['masks']);masks['vision'][case['cue']]=case['selected_receptors']
            data=run_trial(branch,m['groups'],masks,dict(cue=case['cue'],sound=None,ticks=64))
            if case['kind']=='clean' and state in ('initial','trained'):
                with np.load(source/f'probe-{state}-{case["cue"]}.npz') as z:
                    if any(not np.array_equal(data[k],z[k]) for k in data):raise AssertionError('Clean control differs')
            name=f'{state}-{case["name"]}.npz';np.savez_compressed(output/name,**data)
            rows.append(dict(state=state,case=case['name'],file=name,sha256=digest(output/name)))
        if dynamic_snapshot(net)!=parent or dynamic_snapshot(initial)!=initial_state:raise AssertionError('Parent changed')
        print(encode(dict(stage=state,seconds=time.perf_counter()-started)),flush=True)
    if any(digest(p)!=h for p,h in {**hashes,**files}.items()):raise ValueError('Source changed')
    result=dict(training_exact=True,clean_controls_exact=True,starts=starts,probes=rows,
                ticks=sum(t['ticks'] for t in m['trials'])+64*len(rows),seconds=time.perf_counter()-started)
    (output/'summary.json').write_text(encode(result)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
