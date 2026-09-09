"""Finish only missing probes from an intact predictive-bridge checkpoint.

For a recording-budget stop after acquisition. Never replays acquisition or
overwrites completed records. Global free-space and a separate 80 MiB limit
bound this continuation. Original progress and provenance remain unchanged.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .predictive_bridge_probe import record, audit_record
from .composition_probe import encode
from .association_route_probe import digest
from ..core.runtime_checkpoint import load_checkpoint
from neuron.neuron import setup_neuron_logger


def resume(source):
    source=Path(source).resolve()
    if (source/'completion.json').exists(): raise FileExistsError('Continuation already completed')
    m=json.loads((source/'manifest.json').read_text()); cfg=json.loads((source/'config.json').read_text())
    progress=json.loads((source/'progress.json').read_text())
    for p,h in {**m['source_hashes'],**m['physical_sources']}.items():
        if digest(p)!=h: raise ValueError('Source changed')
    for row in progress['entries']:
        if digest(source/row['file'])!=row['sha256']: raise ValueError('Existing trace changed')
    if sum(e['phase'] in ('experience','withdrawal') for e in progress['entries'])!=len(m['trials']):
        raise ValueError('Acquisition is incomplete; cannot resume probes')
    features=[]
    for p in sorted(m['physical_sources']):
        with np.load(p) as z: features.append({k:z[k] for k in z.files})
    completed={e['file'] for e in progress['entries']}; added=[]; used=0
    setup_neuron_logger('CRITICAL')
    birth={(p['neuron_id'],p['synapse_id']):p['u_i']['info'] for p in cfg['synaptic_points'] if p['type']=='postsynaptic'}
    for state in ('initial','trained','reset_selected'):
        for clip in (0,1,None):
            if state=='reset_selected' and clip is None: continue
            name=f'probe-{state}-{clip}.npz'
            if name in completed: continue
            if (source/name).exists(): raise FileExistsError('Unindexed output requires inspection')
            if shutil.disk_usage(source).free<650*1024**2 or used>50*1024**2:
                raise OSError('Insufficient continuation recording budget')
            branch=load_checkpoint(source/('initial.neural-checkpoint' if state=='initial' else 'trained.neural-checkpoint'),trusted=True)
            net=branch.network
            if state=='reset_selected':
                for n,s,_ in m['selected']: net.network.neurons[n].postsynaptic_points[s].u_i.info=birth[n,s]
            t=net.current_tick; trial=dict(start=t,stop=t+300,visual_clip=clip,audio_clip=None)
            data=record(net,features,m['groups'],m['bridge'],trial)
            residual=audit_record(data,cfg,m['bridge'])
            np.savez_compressed(source/name,**data); used+=(source/name).stat().st_size
            added.append(dict(file=name,sha256=digest(source/name),trial=trial,phase='probe',
                              state=state,clip=clip,audit_residual=residual))
            (source/'continuation-progress.json').write_text(encode(dict(entries=added,bytes=used))+'\n')
            print(encode(added[-1]),flush=True)
            if used>80*1024**2: raise OSError('Continuation exceeded 80 MiB')
    for p,h in m['source_hashes'].items():
        if digest(p)!=h: raise ValueError('Runtime changed')
    result=dict(entries=progress['entries']+added,bytes=progress['bytes']+used,
                original_progress_sha256=digest(source/'progress.json'),
                continuation_source_sha256=digest(__file__),
                reason='Finish missing probes after bounded recorder stopped; no acquisition replay or deleted traces')
    assert len([e for e in result['entries'] if e['phase']=='probe'])==8
    (source/'completion.json').write_text(encode(result)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path)
    resume(p.parse_args().source)
