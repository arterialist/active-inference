"""Continuous sensory release in the existing 1152-cell audiovisual preparation.

Retains graph, source media, stimulus order and eligibility-selected association
ports. The receptor output hypothesis changes only the 192 sensory neurons.
The spiking control must reproduce the prior acquisition and neutral probes.
"""
import argparse
from copy import deepcopy
import gzip
import inspect
import json
from pathlib import Path
import time
import numpy as np

from .association_balance_audit import checked
from .association_route_probe import digest
from .composition_probe import encode,fingerprint
from .eligibility_association_probe import dynamic_snapshot
from .eligibility_media_probe import record
from .multimodal_pairing_probe import fresh,WeightObserver
from .population_state_branch import TickDriver
from .sensory_population_probe import physical_gain
from neuron.extensions.experimental.graded_eligibility import GradedEligibilityNeuron


def configure(original,groups,condition):
    if condition not in ('spiking','graded'):raise ValueError(condition)
    cfg=deepcopy(original);sensory=set(groups['vision']+groups['touch'])
    for n in cfg['neurons']:
        if n['id'] in sensory:
            # Initial external input 2*v, synaptic throughput 2 and dendritic
            # attenuation .99 yield .99*v after release gain .25. This is a
            # fixed scale, not data-dependent normalization or a class output.
            n['metadata'].update(graded_gain=.25 if condition=='graded' else 0.,graded_S0=0.,graded_max=0.)
    cfg['metadata']['sensory_release']=condition
    return cfg


def run(source,output,condition):
    source,output=Path(source).resolve(),Path(output).resolve()
    m=json.loads((source/'manifest.json').read_text());s=json.loads((source/'summary.json').read_text())
    if m['condition']!='eligibility':raise ValueError('Need existing eligibility-media source')
    if any(digest(p)!=h for p,h in m['source_hashes'].items()):raise ValueError('Source runtime changed')
    original=json.loads((source/'config.json').read_text());cfg=configure(original,m['groups'],condition)
    features=[]
    for clip in (0,1):
        path=Path(m['source_recording'])/f'sensory-{clip}.npz'
        if digest(path)!=m['source_files_sha256'][path.name]:raise ValueError('Sensory source changed')
        with np.load(path) as z:features.append({k:z[k] for k in z.files})
    output.mkdir(parents=True,exist_ok=False);path=output/'config.json';path.write_text(encode(cfg)+'\n')
    hashes=fingerprint()
    for fn in (run,record,physical_gain,GradedEligibilityNeuron):
        p=Path(inspect.getfile(fn)).resolve();hashes[str(p)]=digest(p)
    manifest=dict(source=str(source),seed=m['seed'],mapping=m['mapping'],condition=condition,
        groups=m['groups'],selected_ports=m['selected_ports'],trials=m['trials'],source_hashes=hashes,
        limits='Full 1152-cell neural graph, no body. Phenomenological graded sensory release, not a biophysical retina/cochlea. Positive native adaptation remains active, with non-spiking receptors retaining a negative native timing direction. Tests two recordings, not animal categories or general understanding.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    net,core,neurons,syns=fresh(path,m['seed'],GradedEligibilityNeuron)
    initial=deepcopy(net);initial_state=dynamic_snapshot(initial);health=WeightObserver(neurons,syns)
    training=[];probes=[];started=time.perf_counter()
    for i,trial in enumerate(m['trials']):
        data=record(net,core,neurons,syns,features,m['groups'],trial,m['selected_ports'],health)
        if condition=='spiking':
            with np.load(checked(source,s['episodes'][i])) as z:
                if any(not np.array_equal(data[k],z[k]) for k in z.files):raise ValueError('Control acquisition differs')
        name=f'experience-{i:03d}.npz';np.savez_compressed(output/name,**data)
        training.append(dict(episode=i,file=name,sha256=digest(output/name)))
    parent=dynamic_snapshot(net)
    for label,state in (('initial',initial_state),('trained',parent)):
        with gzip.open(output/f'{label}-state.json.gz','wt') as f:f.write(state+'\n')
    print('acquisition complete',condition,time.perf_counter()-started,flush=True)
    refs={(p['condition'],p['sense'],p['clip']):p for p in s['probes']}
    for state,parent_net in (('initial',initial),('continuation',net)):
        for sense in ('visual','audio'):
            for clip in (0,1):
                for gain in (.5,1.,2.):
                    altered=deepcopy(features);values=physical_gain(features[clip],sense,gain)
                    altered[clip]['visual']=values[:,:96];altered[clip]['auditory']=values[:,96:]
                    branch=deepcopy(parent_net);members=list(branch.network.neurons.values())
                    points=[p for n in members for p in n.postsynaptic_points.values()];t=branch.current_tick
                    trial=dict(start=t,stop=t+m['clip_ticks'],visual_clip=clip if sense=='visual' else None,
                               audio_clip=clip if sense=='audio' else None)
                    data=record(branch,TickDriver(branch),members,points,altered,m['groups'],trial,m['selected_ports'])
                    if condition=='spiking' and gain==1:
                        with np.load(checked(source,refs[state,sense,clip])) as z:
                            if any(not np.array_equal(data[k],z[k]) for k in z.files):raise ValueError('Control probe differs')
                    name=f'{state}-{sense}-{clip}-gain{gain:g}.npz';np.savez_compressed(output/name,**data)
                    probes.append(dict(state=state,sense=sense,clip=clip,gain=gain,file=name,sha256=digest(output/name),start_tick=t))
        print(state,'probes complete',condition,time.perf_counter()-started,flush=True)
    if dynamic_snapshot(net)!=parent or dynamic_snapshot(initial)!=initial_state:raise ValueError('Probe changed acquisition parent')
    if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('Runtime changed')
    result=dict(training=training,probes=probes,control_exact=condition=='spiking',
                ticks=m['trials'][-1]['stop']+24*m['clip_ticks'],seconds=time.perf_counter()-started)
    (output/'summary.json').write_text(encode(result)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--condition',choices=('spiking','graded'),required=True)
    a=p.parse_args();run(a.source,a.output,a.condition)
