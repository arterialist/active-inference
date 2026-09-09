"""Acquisition and continued learning with port-specific modulation sensitivity.

Fixed comparisons at 16 and 64 presentations per cue. No runtime resets in
the learning trajectory. Counterfactual selected-weight resets occur only in
independent diagnostic clones. Uses the existing 176-cell graph unchanged.
"""
import argparse
from copy import deepcopy
import gzip
import inspect
import json
from pathlib import Path
import time
import numpy as np
from .association_balance_probe import balanced_config,record_trial
from .association_cue_probe import cue_cases
from .association_route_probe import digest
from .composition_probe import encode,fingerprint
from .eligibility_association_probe import protocol,dynamic_snapshot
from .multimodal_pairing_probe import fresh
from neuron.extensions.experimental.port_modulation import PortModulationNeuron


def run(output,seed=11,mapping='paired',sensitivity=.25):
    if sensitivity not in (.25,1.):raise ValueError('Predeclared comparison only')
    output=Path(output).resolve();cfg,g=balanced_config(seed)
    for n in cfg['neurons']:
        if n['id'] in g['auditory']:
            n['metadata']['native_port_modulation']=[dict(port=s,sensitivity=sensitivity) for s in range(35,67)]
    masks,trials=protocol(g,seed,mapping,64);cases=cue_cases(masks,seed)
    hashes=fingerprint()
    for fn in (run,balanced_config,record_trial,protocol,cue_cases,fresh):
        p=Path(inspect.getfile(fn)).resolve();hashes[str(p)]=digest(p)
    output.mkdir(parents=True,exist_ok=False);path=output/'config.json';path.write_text(encode(cfg)+'\n')
    m=dict(seed=seed,mapping=mapping,sensitivity=sensitivity,groups=g,masks=masks,trials=trials,cases=cases,
           checkpoints=[32,128],source_hashes=hashes,
           limits='Synthetic association. Fourfold continued exposure, not indefinite stability. No cue labels in neural computation. Counterfactual resets never touch the acquisition parent.')
    (output/'manifest.json').write_text(encode(m)+'\n')
    net,*_=fresh(path,seed,PortModulationNeuron);initial=json.loads(dynamic_snapshot(net))
    training=[];probes=[];starts={};started=time.perf_counter()
    for i,trial in enumerate(trials):
        data=record_trial(net,g,masks,trial)
        if sensitivity==1. and i<32:
            old=Path(f'.live/research/20260909_association_balance_{mapping}_seed{seed}')/f'train-{i:03d}.npz'
            with np.load(old) as z:
                if set(data)!=set(z.files) or any(not np.array_equal(data[k],z[k]) for k in data):raise AssertionError('Unit-sensitivity control changed')
        name=f'train-{i:03d}.npz';np.savez_compressed(output/name,**data)
        training.append(dict(file=name,sha256=digest(output/name)))
        if i+1 not in m['checkpoints']:continue
        parent=dynamic_snapshot(net);checkpoint=i+1
        for state in ('trained','reset_selected'):
            base=deepcopy(net);expected=json.loads(parent)
            if state=='reset_selected':
                for n in g['auditory']:
                    for sid in range(32):
                        q=initial['neurons'][str(n)]['synapses'][str(sid)][0]
                        base.network.neurons[n].postsynaptic_points[sid].u_i.info=q
                        expected['neurons'][str(n)]['synapses'][str(sid)][0]=q
            if json.loads(dynamic_snapshot(base))!=expected:raise AssertionError('Undeclared branch change')
            key=f'{checkpoint}-{state}';name=key+'-start.json.gz'
            with gzip.open(output/name,'wt') as f:f.write(encode(expected)+'\n')
            starts[key]=dict(file=name,sha256=digest(output/name))
            for case in cases:
                branch=deepcopy(base);physical=deepcopy(masks);physical['vision'][case['cue']]=case['selected_receptors']
                data=record_trial(branch,g,physical,dict(cue=case['cue'],sound=None,ticks=64))
                name=f'{key}-{case["name"]}.npz';np.savez_compressed(output/name,**data)
                probes.append(dict(checkpoint=checkpoint,state=state,case=case['name'],file=name,sha256=digest(output/name)))
            if dynamic_snapshot(net)!=parent:raise AssertionError('Probe changed parent')
        print(encode(dict(checkpoint=checkpoint,seconds=time.perf_counter()-started)),flush=True)
    if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('Runtime changed')
    result=dict(training=training,starts=starts,probes=probes,control_exact=sensitivity==1.,
        ticks=sum(t['ticks'] for t in trials)+64*len(probes),seconds=time.perf_counter()-started)
    (output/'summary.json').write_text(encode(result)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=11);p.add_argument('--mapping',choices=('paired','swapped'),default='paired')
    p.add_argument('--sensitivity',type=float,choices=(.25,1.),default=.25)
    a=p.parse_args();run(a.output,a.seed,a.mapping,a.sensitivity)
