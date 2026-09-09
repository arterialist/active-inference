"""Timing changes applied to the same acquired physical association cues."""
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
from .eligibility_association_probe import dynamic_snapshot
from .multimodal_pairing_probe import fresh
from .sensory_schedule import TIMINGS,receptor_schedule,ScheduledReceptors
from neuron.extensions.experimental.port_modulation import PortModulationNeuron


def run(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    m=json.loads((source/'manifest.json').read_text());old=json.loads((source/'summary.json').read_text());g=m['groups']
    if m['sensitivity']!=.25:raise ValueError('Requires the working quarter-sensitivity preparation')
    hashes=dict(m['source_hashes'])
    for fn in (run,receptor_schedule,ScheduledReceptors):
        p=Path(inspect.getfile(fn)).resolve();hashes[str(p)]=digest(p)
    if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('Acquisition runtime changed')
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(source=str(source),seed=m['seed'],mapping=m['mapping'],timings=TIMINGS,
        checkpoints=m['checkpoints'],cases=m['cases'],source_hashes=hashes,
        criterion='Report full consumer timing. Early candidate activity is distinguished from post-completion activity; balanced mixtures have no correct class. A post-completion silence is not proof of absence of earlier recall.',
        limits='Inference-time receptor asynchrony only; acquisition remains synchronous. Synthetic input scheduling, not new neural computation.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    net,*_=fresh(source/'config.json',m['seed'],PortModulationNeuron)
    started=time.perf_counter();rows=[];starts={};schedules={}
    for i,trial in enumerate(m['trials']):
        data=record_trial(net,g,m['masks'],trial)
        with np.load(source/old['training'][i]['file']) as z:
            if any(not np.array_equal(data[k],z[k]) for k in data):raise AssertionError('Acquisition replay differs')
        checkpoint=i+1
        if checkpoint not in m['checkpoints']:continue
        parent=dynamic_snapshot(net)
        with gzip.open(source/f'{checkpoint}-trained-start.json.gz','rt') as f:
            if json.loads(parent)!=json.load(f):raise AssertionError('Acquired full state differs')
        starts[str(checkpoint)]=dict(file=str(source/f'{checkpoint}-trained-start.json.gz'),sha256=digest(source/f'{checkpoint}-trained-start.json.gz'))
        for case in m['cases']:
            for timing in TIMINGS:
                schedule=receptor_schedule(case,m['masks'],m['seed'],timing)
                schedules[case['name']+'/'+timing]=schedule
                branch=deepcopy(net);driver=ScheduledReceptors(branch,schedule)
                data=record_trial(driver,g,m['masks'],dict(cue=None,sound=None,ticks=len(schedule)))
                if timing=='synchronous':
                    path=source/f'{checkpoint}-trained-{case["name"]}.npz'
                    with np.load(path) as z:
                        if any(not np.array_equal(data[k],z[k]) for k in data):raise AssertionError('Scheduled control differs')
                name=f'{checkpoint}-{case["name"]}-{timing}.npz';np.savez_compressed(output/name,**data)
                rows.append(dict(checkpoint=checkpoint,case=case['name'],timing=timing,file=name,sha256=digest(output/name)))
                if dynamic_snapshot(net)!=parent:raise AssertionError('Probe changed acquisition')
        print(encode(dict(checkpoint=checkpoint,seconds=time.perf_counter()-started)),flush=True)
    np.savez_compressed(output/'schedules.npz',**schedules)
    if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('Runtime changed')
    result=dict(training_exact=True,synchronous_controls_exact=True,starts=starts,probes=rows,
                schedule_sha256=digest(output/'schedules.npz'),ticks=12288+64*len(rows),seconds=time.perf_counter()-started)
    (output/'summary.json').write_text(encode(result)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
