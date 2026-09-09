"""Diagnostic restoration of inhibitory gain with all learning left active.

Replay an acquired balanced network, retain its learned association and all
other state, and restore only inhibitory incoming weights to their birth values.
This is a causal intervention, not a proposed autonomous neural mechanism.
"""
import argparse
from copy import deepcopy
import gzip
import inspect
import json
from pathlib import Path
import time

import numpy as np

from .association_balance_probe import record_trial
from .association_route_probe import digest
from .composition_probe import encode
from .eligibility_association_probe import run_trial,dynamic_snapshot
from .multimodal_pairing_probe import fresh
from neuron.extensions.experimental.eligibility_trace import EligibilityTraceNeuron


def run(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    m=json.loads((source/'manifest.json').read_text());old=json.loads((source/'summary.json').read_text())
    cfg=json.loads((source/'config.json').read_text());g=m['groups']
    hashes=dict(m['source_hashes'])
    for fn in (run,record_trial):
        p=Path(inspect.getfile(fn)).resolve();hashes[str(p)]=digest(p)
    if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('Acquisition runtime changed')
    output.mkdir(parents=True,exist_ok=False)
    (output/'config.json').write_text(encode(cfg)+'\n')
    manifest={**m,'states':['trained','reset_inhibition'],'source':str(source),'source_hashes':hashes,
        'intervention':'Only auditory incoming inhibitory q values restored to config. No learning rate changes. Host intervention tests causation, not an autonomous repair.'}
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    net,*_=fresh(source/'config.json',m['seed'],EligibilityTraceNeuron);started=time.perf_counter()
    for i,trial in enumerate(m['trials']):
        data=record_trial(net,g,m['masks'],trial)
        with np.load(source/old['training'][i]['file']) as z:
            if set(data)!=set(z.files) or any(not np.array_equal(data[k],z[k]) for k in data):raise AssertionError('Acquisition changed')
    parent=dynamic_snapshot(net)
    with gzip.open(source/'trained-state.json.gz','rt') as f:
        if json.loads(parent)!=json.load(f):raise AssertionError('Full acquired state differs')
    with gzip.open(output/'trained-state.json.gz','wt') as f:f.write(parent+'\n')
    points={(p['neuron_id'],p['synapse_id']):p for p in cfg['synaptic_points'] if p['type']=='postsynaptic'}
    starts={};rows=[]
    for state in manifest['states']:
        base=deepcopy(net);expected=json.loads(parent)
        if state=='reset_inhibition':
            for n in g['auditory']:
                for sid in range(35,67):
                    q=points[n,sid]['u_i']['info'];base.network.neurons[n].postsynaptic_points[sid].u_i.info=q
                    expected['neurons'][str(n)]['synapses'][str(sid)][0]=q
        if json.loads(dynamic_snapshot(base))!=expected:raise AssertionError('Undeclared intervention')
        name=state+'-start.json.gz'
        params={str(n.id):dict(r=n.params.r_base,b=n.params.b_base) for n in base.network.neurons.values()}
        with gzip.open(output/name,'wt') as f:f.write(encode(dict(state=expected,threshold_parameters=params))+'\n')
        starts[state]=dict(file=name,sha256=digest(output/name))
        for case in m['cases']:
            branch=deepcopy(base);masks=deepcopy(m['masks']);masks['vision'][case['cue']]=case['selected_receptors']
            trial=dict(cue=case['cue'],sound=None,ticks=64);data=record_trial(branch,g,masks,trial)
            if state=='trained':
                with np.load(source/f'trained-{case["name"]}.npz') as z:
                    if any(not np.array_equal(data[k],z[k]) for k in data):raise AssertionError('Unchanged control differs')
            elif case['kind']=='clean':
                control=deepcopy(base);plain=run_trial(control,g,masks,trial)
                if any(not np.array_equal(data[k],plain[k]) for k in plain) or dynamic_snapshot(control)!=dynamic_snapshot(branch):raise AssertionError('Observer changed probe')
            name=f'{state}-{case["name"]}.npz';np.savez_compressed(output/name,**data)
            rows.append(dict(state=state,case=case['name'],file=name,sha256=digest(output/name)))
        if dynamic_snapshot(net)!=parent:raise AssertionError('Changed parent')
    if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('Runtime changed')
    result=dict(training=[dict(item,file=str(source/item['file'])) for item in old['training']],
        starts=starts,probes=rows,observer_controls_exact=True,training_exact=True,unchanged_controls_exact=True,
        ticks=3072+64*(len(rows)+2),seconds=time.perf_counter()-started)
    (output/'summary.json').write_text(encode(result)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
