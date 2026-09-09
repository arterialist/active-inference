"""Real-media intensity coding with an equal-size receptor diversity control.

Only an isolated sensory population is simulated. Physical sampling phase and
the image/audio transfer functions are preserved. No clip label or fitted
normalization is available to a neuron. Downstream learning is not tested.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np

from .association_balance_audit import checked
from .association_route_probe import digest
from .composition_probe import encode,fingerprint
from .multimodal_pairing_probe import fresh
from .population_hierarchy import cellular
from neuron.extensions.experimental.bounded_plasticity import BoundedPlasticityNeuron

THRESHOLDS = (.1,.2,.4,.8,1.6,3.2)


def population_config(original,condition):
    if condition not in ('baseline','homogeneous','diverse'): raise ValueError(condition)
    levels = (.6,) if condition=='baseline' else (.6,)*6 if condition=='homogeneous' else THRESHOLDS
    cfg = {k:deepcopy(original[k]) for k in ('global_params','simulation_params')}
    cfg.update(neurons=[],synaptic_points=[],connections=[],external_inputs=[],
               metadata=dict(preparation='sensory-range-probe-v0',condition=condition))
    oldpoints = {}
    for p in original['synaptic_points']:
        if p['neuron_id']<=192: oldpoints.setdefault(p['neuron_id'],[]).append(p)
    channels = []
    for old in original['neurons'][:192]:
        for level,r in enumerate(levels):
            nid = len(cfg['neurons'])+1
            n = deepcopy(old); n['id'] = nid
            n['params'].update(r_base=r,b_base=r+.25)
            n['metadata'].update(physical_receptor=old['id'],sensitivity_level=level)
            cfg['neurons'].append(n); channels.append(old['id']-1)
            for oldp in oldpoints[old['id']]:
                p = deepcopy(oldp); p['neuron_id'] = nid; cfg['synaptic_points'].append(p)
            cfg['external_inputs'].append(dict(target_neuron=nid,target_synapse=0))
    return cfg,np.array(channels)


def physical_gain(feature,sense,gain):
    if sense not in ('visual','audio') or gain<=0: raise ValueError('Invalid physical perturbation')
    values = np.zeros((len(feature['visual']),192))
    if sense=='visual':
        # Change luminance before the darkness transfer. Preserve the declared
        # first three unavailable frames, not an invented early grey stimulus.
        values[:, :96] = 1-np.clip(gain*(1-feature['visual']),0,1)
        values[:3,:96] = 0
    else:
        db = feature['band_db']+20*np.log10(gain)
        values[:,96:] = np.clip((db[:,:,None]-np.array([-65.,-45.,-25.]))/20,0,1).reshape(-1,96)
    return values


def record(net,channels,values):
    neurons = list(net.network.neurons.values());states=[];weights=[]
    for t,row in enumerate(values):
        active = (t%4==(channels+1)%4) & (row[channels]>0)
        for index in np.flatnonzero(active): net.set_external_input(int(index)+1,0,float(2*row[channels[index]]))
        net.run_tick(); states.append(cellular(neurons))
        weights.append([n.postsynaptic_points[0].u_i.info for n in neurons])
    return dict(cells=np.asarray(states),weights=np.asarray(weights),physical_values=values)


def run(recording,output):
    root,output = Path(recording).resolve(),Path(output).resolve()
    m = json.loads((root/'manifest.json').read_text());s = json.loads((root/'summary.json').read_text())
    cfg = json.loads((root/'config.json').read_text());source = Path(m['source_recording'])
    features = []
    for clip in (0,1):
        path = source/f'sensory-{clip}.npz'
        if digest(path)!=m['source_files_sha256'][path.name]: raise ValueError('Sensory source changed')
        with np.load(path) as z: features.append({k:z[k] for k in z.files})
    output.mkdir(parents=True,exist_ok=False);hashes=fingerprint();hashes[str(Path(__file__).resolve())]=digest(__file__)
    manifest=dict(source=str(root),seed=m['seed'],source_hashes=hashes,thresholds=THRESHOLDS,
        limits='Isolated sensory coding. Six cells per receptor increase total afferent conductance sixfold; diverse/homogeneous controls have equal size and conductance. No downstream consumer, adaptation to illumination, learned invariance or semantic recognition is tested.')
    (output/'manifest.json').write_text(encode(manifest)+'\n');records=[];started=time.perf_counter()
    references={(p['sense'],p['clip']):p for p in s['probes'] if p['condition']=='initial'}
    for condition in ('baseline','homogeneous','diverse'):
        local,channels=population_config(cfg,condition);path=output/f'{condition}.json';path.write_text(encode(local)+'\n')
        parent,*_=fresh(path,m['seed'],BoundedPlasticityNeuron)
        for sense in ('visual','audio'):
            for clip in (0,1):
                for gain in (.5,1.,2.):
                    data=record(deepcopy(parent),channels,physical_gain(features[clip],sense,gain))
                    if gain==1:
                        with np.load(checked(root,references[sense,clip])) as z: expected=z['cells'][:,:192]
                        if condition=='baseline' and not np.array_equal(data['cells'],expected):raise ValueError('Baseline receptor replay differs')
                        if condition=='homogeneous' and not np.array_equal(data['cells'],np.repeat(expected,6,axis=1)):raise ValueError('Equal-threshold copies differ')
                    name=f'{condition}-{sense}-{clip}-gain{gain:g}.npz';np.savez_compressed(output/name,**data)
                    records.append(dict(condition=condition,sense=sense,clip=clip,gain=gain,file=name,sha256=digest(output/name)))
        print(condition,'complete',time.perf_counter()-started,flush=True)
    if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('Runtime changed')
    result=dict(records=records,ticks=36*len(features[0]['visual']),seconds=time.perf_counter()-started)
    (output/'summary.json').write_text(encode(result)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.recording,a.output)
