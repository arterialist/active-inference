"""Paired sixteen-block embodied acquisition with local magnitude feedback.

Reuses the exact original birth graph, physical world and acquisition schedule.
Only negative-input retrograde information errors change. No terminal reset,
learning freeze, fitted weight or host teaching signal is installed. Every
block has an executable state; diagnostic probes never modify acquisition.
"""
import argparse
import json
import random
from pathlib import Path
import shutil
import time

import numpy as np

from . import context_organization as base
from . import crossed_av_world as world
from .crossed_av_continuation import isolated_rng,restore_acquired
from .magnitude_feedback_probe import ReleaseObserver
from .opponent_context import AfferentDelay,verify_afferents
from .temporal_memory_transplant import transplant_selected
from .temporal_verification import verify_learning
from neuron.extensions.experimental.magnitude_retrograde import MagnitudeRetrogradeNeuron


def observed_course(net,arm,delay,features,groups,selected,video,audio,ticks=364):
    with ReleaseObserver(net,groups['context'][0]) as observer:
        data=world.course(net,arm,features,groups,selected,video,audio,ticks=ticks,delay=delay)
    return dict(**data,**observer.arrays())


def run(reference,output,blocks=16):
    reference=Path(reference).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if type(blocks) is not int or not 1<=blocks<=16:raise ValueError('Need 1..16 blocks')
    if shutil.disk_usage(output.parent).free<3*1024**3:raise OSError('Need 3 GiB reserve')
    cm=json.loads((reference/'manifest.json').read_text())
    parent=Path(cm['parent']);pm=json.loads((parent/'manifest.json').read_text())
    ps=json.loads((parent/'summary.json').read_text())
    if pm['reverse'] or not ps['birth_replay_exact']:raise ValueError('Need verified normal parent')
    for p,h in {**cm['source_hashes'],**cm['physical_sources']}.items():
        if base.digest(p)!=h:raise ValueError(f'Parent source changed: {p}')
    if base.digest(parent/'config.json')!=pm['config_sha256']:raise ValueError('Parent graph changed')
    features=[]
    for clip in (0,1):
        paths=[p for p in cm['physical_sources'] if Path(p).name==f'sensory-{clip}.npz']
        if len(paths)!=1:raise ValueError('Ambiguous media')
        with np.load(paths[0]) as z:features.append({k:z[k] for k in ('visual','auditory')})
    output.mkdir();began=time.perf_counter();seed=pm['seed']
    groups,selected=pm['groups'],pm['selected'];order=world.schedule(blocks)
    # Disabled extension plus observer must reproduce the old full first episode.
    with isolated_rng():
        original=base.fresh(parent/'config.json',seed,MagnitudeRetrogradeNeuron)[0]
        data=observed_course(original,base.Arm(),AfferentDelay(64),features,groups,selected,*order[0][0])
    ref=ps['training'][0]
    if base.digest(parent/ref['file'])!=ref['sha256']:raise ValueError('Replay evidence changed')
    with np.load(parent/ref['file']) as z:
        if any(not np.array_equal(data[k],z[k]) for k in z.files):raise ValueError('Native replay differs')
    np.savez_compressed(output/'preflight.npz',**data)
    cfg=json.loads((parent/'config.json').read_text())
    for n in cfg['neurons']:n['metadata']['retrograde_magnitude_error']=True
    (output/'config.json').write_text(base.encode(cfg)+'\n')
    hashes={**cm['source_hashes'],**base.fingerprint()}
    from . import magnitude_feedback_probe as probe
    for p in (Path(__file__).resolve(),Path(probe.__file__).resolve()):hashes[str(p)]=base.digest(p)
    manifest=dict(seed=seed,reverse=False,groups=groups,selected=selected,schedule=order,
        source_hashes=hashes,physical_sources=cm['physical_sources'],parent=str(parent),
        native_continuation=str(reference),config_sha256=base.digest(output/'config.json'),
        parent_manifest_sha256=base.digest(parent/'manifest.json'),
        native_continuation_manifest_sha256=base.digest(reference/'manifest.json'),
        parent_first_episode_sha256=ref['sha256'],cell_fields=base.FIELDS,
        preflight_sha256=base.digest(output/'preflight.npz'),preflight_exact=True,
        physical_delay=64,xml=base.XML,neural_tick_seconds=base.DT,
        context_columns=['before','after','return_count','arriving_release'],
        return_columns=['tick','source_neuron','source_port','info','plast','mod0','mod1'],
        protocol='Continuous 364-tick episodes, four pairings per block; four birth-state weight '
          'expression probes every block, eight acquired learned/reset probes at blocks 4/8/12/16. '
          'Probes last 96 ticks and keep learning positive.',
        limits='Magnitude feedback changes ALL inhibitory incoming ports, including sensory and '
          'comparison pathways, not only the context gate. Native comparator timing retained. '
          'One joint and two real recordings, not semantic understanding or a full autonomous agent. '
          'Fixed acquisition order; no asymptotic stability or consciousness claim.')
    (output/'manifest.json').write_text(base.encode(manifest)+'\n')
    net=base.fresh(output/'config.json',seed,MagnitudeRetrogradeNeuron)[0]
    arm=base.Arm();delay=AfferentDelay(64)
    base.save_checkpoint(net,output/'initial.paula',sources=list(hashes))
    training=[];probes=[];checkpoints=[]

    def record(data,name,**metadata):
        if shutil.disk_usage(output).free<3*1024**3:raise OSError('Storage reserve reached')
        residuals=dict(learning=verify_learning(data),physics=base.verify_physics(data),
                       afferents=verify_afferents(data))
        if any(not np.isfinite(a).all() for a in data.values()):raise ValueError('Nonfinite trace')
        np.savez_compressed(output/name,**data)
        return dict(file=name,sha256=base.digest(output/name),residuals=residuals,**metadata)

    for block,pairs in enumerate(order):
        for v,a in pairs:
            data=observed_course(net,arm,delay,features,groups,selected,v,a)
            training.append(record(data,f'train-b{block}-v{v}-a{a}.npz',block=block,video=v,audio=a))
        count=block+1;checkpoint=output/f'block-{count}.paula';body=output/f'body-{count}.npz'
        base.save_checkpoint(net,checkpoint,sources=list(hashes))
        np.savez_compressed(body,state=arm.state(),delay=delay.state())
        checkpoints.append(dict(blocks=count,neural=checkpoint.name,physical=body.name,
            neural_sha256=base.digest(checkpoint),physical_sha256=base.digest(body)))
        for v,a in ((0,0),(0,1),(1,0),(1,1)):
            with isolated_rng():
                saved=base.load_checkpoint(output/'initial.paula',trusted=True)
                branch=saved.network
                random.setstate(saved.python_rng);np.random.set_state(saved.numpy_rng)
                transplant_selected(branch,net,selected)
                data=observed_course(branch,base.Arm(),AfferentDelay(64),features,groups,selected,v,a,96)
            probes.append(record(data,f'resting-b{count}-v{v}-a{a}.npz',blocks=count,
                kind='resting',weights='learned',video=v,audio=a))
        if count in (4,8,12,16):
            for weights in ('learned','reset'):
                for v,a in ((0,0),(0,1),(1,0),(1,1)):
                    with isolated_rng():
                        branch,body_branch,history=restore_acquired(checkpoint,body)
                        if weights=='reset':
                            for nid,sid,_ in selected:
                                branch.network.neurons[nid].postsynaptic_points[sid].u_i.info=0.
                        data=observed_course(branch,body_branch,history,features,groups,selected,v,a,96)
                    probes.append(record(data,f'acquired-{weights}-b{count}-v{v}-a{a}.npz',
                        blocks=count,kind='acquired',weights=weights,video=v,audio=a))
        progress=dict(blocks=count,training=training,probes=probes,checkpoints=checkpoints,
            executed_ticks=364+len(training)*364+len(probes)*96,preflight_exact=True,
            seconds=time.perf_counter()-began)
        (output/f'completed-block-{count}.json').write_text(base.encode(progress)+'\n')
        terminal=next(iter(net.network.neurons[groups['context'][0]].presynaptic_points.values()))
        print(base.encode(dict(seed=seed,blocks=count,ticks=progress['executed_ticks'],
            seconds=progress['seconds'],context_release=float(terminal.u_o.info))),flush=True)
    if any(base.digest(p)!=h for p,h in hashes.items()):raise ValueError('Sources changed during run')
    (output/'summary.json').write_text(base.encode(progress)+'\n')
    return progress


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference',type=Path)
    p.add_argument('output',type=Path);p.add_argument('--blocks',type=int,default=16)
    a=p.parse_args();run(a.reference,a.output,a.blocks)
