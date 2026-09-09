"""Continuous embodied acquisition from a verified feedback-screen graph.

Uses the established sixteen-block physical course with fresh-state weight
expression and matched acquired/reset probes. A source-screen episode must
replay exactly before acquisition. This runner accepts any of the three screen
conditions; it does not optimize wiring, select successful seeds or install
offline fitted weights. All diagnostic branches are excluded from acquisition.
"""
import argparse
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np

from . import context_organization as base
from .crossed_av_continuation import isolated_rng,restore_acquired
from .magnitude_feedback_learning import observed_course
from .opponent_context import AfferentDelay,verify_afferents
from .temporal_memory_transplant import transplant_selected
from .temporal_verification import verify_learning
from neuron.extensions.experimental.magnitude_retrograde import MagnitudeRetrogradeNeuron


def run(reference,output,condition='feedback',blocks=16):
    reference=Path(reference).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if type(blocks) is not int or not 1<=blocks<=16:raise ValueError('Need 1..16 blocks')
    if condition not in ('none','wired_zero','feedback'):raise ValueError('Unknown screen condition')
    if shutil.disk_usage(output.parent).free<3*1024**3:raise OSError('Need 3 GiB reserve')
    m=json.loads((reference/'manifest.json').read_text());s=json.loads((reference/'summary.json').read_text())
    parent=Path(m['reference']);pm=json.loads((parent/'manifest.json').read_text())
    if base.digest(parent/'manifest.json')!=m['reference_manifest_sha256']:
        raise ValueError('Acquisition reference changed')
    hashes=dict(m['source_hashes']);hashes[str(Path(__file__).resolve())]=base.digest(__file__)
    for p,h in {**hashes,**m['physical_sources']}.items():
        if base.digest(p)!=h:raise ValueError(f'Source changed: {p}')
    cfg_path=reference/f'{condition}.json'
    if base.digest(cfg_path)!=m['config_hashes'][condition] or not s['unchanged_replay_exact']:
        raise ValueError('Screen graph or replay missing')
    groups=m['groups'][condition];selected=m['selected'];order=pm['schedule'][:blocks];seed=m['seed']
    features=[]
    for clip in (0,1):
        paths=[p for p in m['physical_sources'] if Path(p).name==f'sensory-{clip}.npz']
        if len(paths)!=1:raise ValueError('Ambiguous media')
        with np.load(paths[0]) as z:features.append({k:z[k] for k in ('visual','auditory')})
    output.mkdir();began=time.perf_counter()
    # Copy exact configuration bytes, never synthesize a task-specific readout.
    shutil.copyfile(cfg_path,output/'config.json')
    v,a=order[0][0]
    old=next(r for r in s['records'] if (r['condition'],r['video'],r['audio'])==(condition,v,a))
    if base.digest(reference/old['file'])!=old['sha256']:raise ValueError('Screen record changed')
    with isolated_rng():
        net=base.fresh(output/'config.json',seed,MagnitudeRetrogradeNeuron)[0]
        data=observed_course(net,base.Arm(),AfferentDelay(64),features,groups,selected,v,a)
    with np.load(reference/old['file']) as z:
        if set(z.files)!=set(data) or any(not np.array_equal(z[k],data[k]) for k in z.files):
            raise ValueError('Screen episode replay differs')
    np.savez_compressed(output/'preflight.npz',**data)
    manifest=dict(seed=seed,reverse=False,groups=groups,selected=selected,schedule=order,
        source_hashes=hashes,physical_sources=m['physical_sources'],
        config_sha256=base.digest(output/'config.json'),preflight_exact=True,
        preflight_sha256=base.digest(output/'preflight.npz'),cell_fields=base.FIELDS,
        screen=str(reference),screen_manifest_sha256=base.digest(reference/'manifest.json'),
        screen_summary_sha256=base.digest(reference/'summary.json'),condition=condition,
        matched_control=str(parent),physical_delay=64,xml=base.XML,neural_tick_seconds=base.DT,
        hypothesis='Does local mean feedback change continued learning interference and '
          'retained embodied expression, beyond changing representation contrast?',
        limits='Four fixed crossed recordings and a single physical joint. No task labels '
          'reach the brain. Same prior schedule; not an order-generalization assay. '
          'Ablation probes are diagnostic and never feed the acquired state.')
    (output/'manifest.json').write_text(base.encode(manifest)+'\n')
    net=base.fresh(output/'config.json',seed,MagnitudeRetrogradeNeuron)[0]
    arm=base.Arm();delay=AfferentDelay(64)
    base.save_checkpoint(net,output/'initial.paula',sources=list(hashes))
    training=[];probes=[];checkpoints=[]

    def record(data,name,**metadata):
        if shutil.disk_usage(output).free<3*1024**3:raise OSError('Storage reserve reached')
        residuals=dict(learning=verify_learning(data),physics=base.verify_physics(data),
                       afferents=verify_afferents(data))
        if any(not np.isfinite(x).all() for x in data.values()):raise ValueError('Nonfinite record')
        np.savez_compressed(output/name,**data)
        return dict(file=name,sha256=base.digest(output/name),residuals=residuals,**metadata)

    for block,pairs in enumerate(order):
        for v,a in pairs:
            data=observed_course(net,arm,delay,features,groups,selected,v,a)
            training.append(record(data,f'train-b{block}-v{v}-a{a}.npz',block=block,video=v,audio=a))
        count=block+1;checkpoint=output/f'block-{count}.paula';physical=output/f'body-{count}.npz'
        base.save_checkpoint(net,checkpoint,sources=list(hashes))
        np.savez_compressed(physical,state=arm.state(),delay=delay.state())
        checkpoints.append(dict(blocks=count,neural=checkpoint.name,physical=physical.name,
            neural_sha256=base.digest(checkpoint),physical_sha256=base.digest(physical)))
        for v,a in ((0,0),(0,1),(1,0),(1,1)):
            with isolated_rng():
                saved=base.load_checkpoint(output/'initial.paula',trusted=True);branch=saved.network
                random.setstate(saved.python_rng);np.random.set_state(saved.numpy_rng)
                transplant_selected(branch,net,selected)
                data=observed_course(branch,base.Arm(),AfferentDelay(64),features,groups,selected,v,a,96)
            probes.append(record(data,f'resting-b{count}-v{v}-a{a}.npz',blocks=count,
                                 kind='resting',weights='learned',video=v,audio=a))
        if count in (4,8,12,16):
            for weights in ('learned','reset'):
                for v,a in ((0,0),(0,1),(1,0),(1,1)):
                    with isolated_rng():
                        branch,body_branch,history=restore_acquired(checkpoint,physical)
                        if weights=='reset':
                            for nid,sid,_ in selected:branch.network.neurons[nid].postsynaptic_points[sid].u_i.info=0.
                        data=observed_course(branch,body_branch,history,features,groups,selected,v,a,96)
                    probes.append(record(data,f'acquired-{weights}-b{count}-v{v}-a{a}.npz',blocks=count,
                        kind='acquired',weights=weights,video=v,audio=a))
        progress=dict(blocks=count,training=training,probes=probes,checkpoints=checkpoints,
            executed_ticks=364+len(training)*364+len(probes)*96,preflight_exact=True,seconds=time.perf_counter()-began)
        (output/f'completed-block-{count}.json').write_text(base.encode(progress)+'\n')
        print(base.encode(dict(seed=seed,condition=condition,blocks=count,ticks=progress['executed_ticks'],
                              seconds=progress['seconds'])),flush=True)
    if any(base.digest(p)!=h for p,h in hashes.items()):raise ValueError('Sources changed during acquisition')
    (output/'summary.json').write_text(base.encode(progress)+'\n')
    return progress


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--condition',choices=('none','wired_zero','feedback'),default='feedback')
    p.add_argument('--blocks',type=int,default=16)
    a=p.parse_args();run(a.reference,a.output,a.condition,a.blocks)
