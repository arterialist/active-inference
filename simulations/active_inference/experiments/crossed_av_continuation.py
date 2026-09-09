"""Continue the same embodied learner, measuring stored content between blocks.

No new neuron, fitted weight, rate change or state reset in acquisition. Resume
the previous four-block brain/body/delay state and continue through block 16.
Diagnostic branches never write back to the acquiring brain. Each balanced
block saves an executable checkpoint, body/afferent history and probe records.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np

from . import context_organization as base
from . import crossed_av_world as world
from .opponent_context import AfferentDelay,verify_afferents
from .temporal_memory_transplant import transplant_selected
from .temporal_verification import verify_learning


def restore_acquired(checkpoint,body_path):
    saved=base.load_checkpoint(checkpoint,trusted=True)
    net=saved.network
    random.setstate(saved.python_rng);np.random.set_state(saved.numpy_rng)
    body=base.Arm()
    with np.load(body_path) as data:
        body.restore(data['state']);delay=AfferentDelay(64,data['delay'])
    return net,body,delay


@contextmanager
def isolated_rng():
    """Diagnostic branches must not advance the acquiring process's RNG streams."""
    python,numpy=random.getstate(),np.random.get_state()
    try:yield
    finally:
        random.setstate(python);np.random.set_state(numpy)


def expression(birth,learned,features,groups,selected,video,audio,reverse):
    with isolated_rng():
        saved=base.load_checkpoint(birth,trusted=True);branch=saved.network
        random.setstate(saved.python_rng);np.random.set_state(saved.numpy_rng)
        transplant_selected(branch,learned,selected)
        return world.course(branch,base.Arm(),features,groups,selected,video,audio,
            reverse=reverse,ticks=96,delay=AfferentDelay(64))


def run(parent,output):
    parent=Path(parent).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free<3*1024**3:raise OSError('Need 3 GiB reserve')
    pm=json.loads((parent/'manifest.json').read_text());ps=json.loads((parent/'summary.json').read_text())
    for p,h in {**pm['source_hashes'],**pm['physical_sources']}.items():
        if base.digest(p)!=h:raise ValueError('Parent source changed')
    order=world.schedule(16)
    if pm['schedule']!=[[list(pair) for pair in block] for block in order[:4]]:
        raise ValueError('Parent is not the declared acquisition prefix')
    if not ps['birth_replay_exact'] or ps['recorded_ticks']!=9280:
        raise ValueError('Parent course is incomplete')
    features=[]
    for clip in (0,1):
        paths=[p for p in pm['physical_sources'] if Path(p).name==f'sensory-{clip}.npz']
        if len(paths)!=1:raise ValueError('Ambiguous media source')
        with np.load(paths[0]) as z:features.append({k:z[k] for k in ('visual','auditory')})
    net,body,delay=restore_acquired(parent/'final.paula',parent/'final-body.npz')
    groups,selected=pm['groups'],pm['selected'];began=time.perf_counter()
    hashes={**pm['source_hashes'],str(Path(__file__).resolve()):base.digest(__file__)}
    output.mkdir()
    manifest=dict(parent=str(parent),seed=pm['seed'],reverse=pm['reverse'],groups=groups,selected=selected,
        source_hashes=hashes,physical_sources=pm['physical_sources'],schedule=order,
        parent_evidence={name:base.digest(parent/name) for name in
            ('manifest.json','summary.json','initial.paula','final.paula','final-body.npz')},
        first_new_block=4,last_block=15,full_state_probe_blocks=[8,12,16],
        limits='Twelve additional balanced blocks in the unchanged neural/body state. '
               'Stored-content probes reset only diagnostic branches to birth, not acquisition. '
               'Normal-assignment four-seed continuation, not a repeat of both assignment directions. '
               'Independent snapshots at each block, no automatic resumability claimed.')
    (output/'manifest.json').write_text(base.encode(manifest)+'\n')
    for video in (0,1):
        for audio in (0,1):
            replay=expression(parent/'initial.paula',net,features,groups,selected,video,audio,pm['reverse'])
            ref=next(r for r in ps['probes'] if (r['kind'],r['weights'],r['presentation'],r['video'],r['audio'])
                     ==('resting','learned','both',video,audio))
            if base.digest(parent/ref['file'])!=ref['sha256']:raise ValueError('Parent reference changed')
            with np.load(parent/ref['file']) as original:
                if set(original.files)!=set(replay) or any(not np.array_equal(original[k],replay[k]) for k in original.files):
                    raise ValueError('Parent stored-content replay differs')
    (output/'preflight.json').write_text(base.encode(dict(parent_expression_replay_exact=True,ticks=384))+'\n')
    training=[];probes=[];checkpoints=[]
    def record(data,name,**metadata):
        if shutil.disk_usage(output).free<3*1024**3:raise OSError('Storage reserve reached')
        residuals=dict(learning_residual=verify_learning(data),physics_residual=base.verify_physics(data),
                       afferent_residual=verify_afferents(data))
        np.savez_compressed(output/name,**data)
        return dict(file=name,sha256=base.digest(output/name),**metadata,**residuals)
    for block in range(4,16):
        for video,audio in order[block]:
            data=world.course(net,body,features,groups,selected,video,audio,
                              reverse=pm['reverse'],delay=delay)
            training.append(record(data,f'train-b{block}-v{video}-a{audio}.npz',block=block,video=video,audio=audio))
        count=block+1;checkpoint=output/f'block-{count}.paula';physical=output/f'body-{count}.npz'
        base.save_checkpoint(net,checkpoint,sources=list(hashes))
        np.savez_compressed(physical,state=body.state(),delay=delay.state())
        checkpoints.append(dict(blocks=count,neural=checkpoint.name,physical=physical.name,
                                neural_sha256=base.digest(checkpoint),physical_sha256=base.digest(physical)))
        for video in (0,1):
            for audio in (0,1):
                data=expression(parent/'initial.paula',net,features,groups,selected,video,audio,pm['reverse'])
                probes.append(record(data,f'resting-b{count}-v{video}-a{audio}.npz',blocks=count,
                    kind='resting',weights='learned',presentation='both',video=video,audio=audio,offset=0))
        if count in (8,12,16):
            for weights in ('learned','reset'):
                for video in (0,1):
                    for audio in (0,1):
                        with isolated_rng():
                            branch,arm,history=restore_acquired(checkpoint,physical)
                            if weights=='reset':
                                for nid,sid,_ in selected:branch.network.neurons[nid].postsynaptic_points[sid].u_i.info=0.
                            data=world.course(branch,arm,features,groups,selected,video,audio,
                                reverse=pm['reverse'],ticks=96,delay=history)
                        probes.append(record(data,f'acquired-{weights}-b{count}-v{video}-a{audio}.npz',
                            blocks=count,kind='acquired',weights=weights,presentation='both',video=video,audio=audio,offset=0))
        # A completed block manifest provides evidence even if a later block fails.
        progress=dict(blocks=count,training=training,probes=probes,checkpoints=checkpoints,
            executed_ticks=384+len(training)*364+len(probes)*96,seconds=time.perf_counter()-began)
        (output/f'completed-block-{count}.json').write_text(base.encode(progress)+'\n')
        print(base.encode(dict(stage='block-complete',blocks=count,ticks=progress['executed_ticks'],
                               seconds=progress['seconds'])),flush=True)
    if any(base.digest(p)!=h for p,h in hashes.items()):raise ValueError('Source changed during run')
    (output/'summary.json').write_text(base.encode(progress)+'\n')
    print(base.encode(dict(stage='complete',ticks=progress['executed_ticks'],seconds=progress['seconds'])),flush=True)
    return progress


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('parent',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.parent,a.output)
