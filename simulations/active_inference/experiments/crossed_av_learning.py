"""Bounded embodied conjunctive-learning experiment with causal sensory probes.

No architecture change: 590-cell delayed-verification preparation. Four crossed
real-media pairings require both senses before somatic feedback. Paired physical
assignment reversal shares brain birth configuration and acquisition order.
"""
import argparse
import inspect
from pathlib import Path
import shutil
import time

import numpy as np

from . import context_organization as base
from . import crossed_av_world as world
from .opponent_context import AfferentDelay,verify_afferents,course as recorded_course
from .temporal_verification import configure,verify_learning
from .temporal_memory_transplant import transplant_selected


def run(output,seed=11,reverse=False):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free<3*1024**3:raise OSError('Need 3 GiB reserve')
    media=Path(__file__).resolve().parents[3]/'.live/research/20260908_bounded_learning_paired_seed11'
    features=[];physical_sources={}
    for clip in (0,1):
        p=media/f'sensory-{clip}.npz';physical_sources[str(p)]=base.digest(p)
        with np.load(p) as z:features.append({k:z[k] for k in ('visual','auditory')})
    cfg,groups,selected=configure(seed,aligned=True)
    output.mkdir();(output/'config.json').write_text(base.encode(cfg)+'\n')
    hashes=base.fingerprint()
    for obj in (run,configure,world.course,recorded_course,base.run,base.append_predictive_bridge,
                base.cellular,base.fresh,transplant_selected):
        p=Path(inspect.getfile(obj)).resolve();hashes[str(p)]=base.digest(p)
    # Include the construction helpers explicitly, not just their caller.
    from ..components.learning.temporal_verification import configure_verification
    from ..components.learning.opponent_prediction import couple_opponent_predictions
    for obj in (configure_verification,couple_opponent_predictions):
        p=Path(inspect.getfile(obj)).resolve();hashes[str(p)]=base.digest(p)
    order=world.schedule()
    manifest=dict(seed=seed,reverse=reverse,groups=groups,selected=selected,
        source_hashes=hashes,physical_sources=physical_sources,cell_fields=base.FIELDS,
        neural_tick_seconds=base.DT,xml=base.XML,schedule=order,
        config_sha256=base.digest(output/'config.json'),physical_delay=64,
        assignment='positive load iff video XOR audio XOR reverse == 0',
        protocol='16 continuous embodied acquisition episodes; 8 full-state 96-tick probes; '
                 '24 birth-state sensory/weight probes plus 4 shifted-onset learned probes, all 96 ticks.',
        limits='One joint, two recordings crossed. Neither semantic recognition nor a full agent. '
               'Context lamp is constant 0; second mixed bank remains suppressed as in prior anatomy. '
               'Learning stays positive. Sensory deletion changes drive, not a neural training flag. '
               'Shifted-onset probes reuse recorded samples, not unseen semantic examples. '
               'Acquisition order is fixed across seeds and reversed assignments, not an order factorial.')
    (output/'manifest.json').write_text(base.encode(manifest)+'\n')
    net,_,_,_=base.fresh(output/'config.json',seed,base.PredictiveReceptorNeuron)
    base.save_checkpoint(net,output/'initial.paula',sources=list(hashes))
    arm=base.Arm();delay=AfferentDelay(64);began=time.perf_counter();training=[];probes=[]
    def record(data,name,**metadata):
        if shutil.disk_usage(output).free<3*1024**3:raise OSError('Storage reserve reached')
        residuals=dict(learning_residual=verify_learning(data),
            physics_residual=base.verify_physics(data),afferent_residual=verify_afferents(data))
        np.savez_compressed(output/name,**data)
        return dict(file=name,sha256=base.digest(output/name),**metadata,**residuals)
    for block,pairs in enumerate(order):
        for video,audio in pairs:
            z=world.course(net,arm,features,groups,selected,video,audio,reverse=reverse,delay=delay)
            training.append(record(z,f'train-b{block}-v{video}-a{audio}.npz',block=block,video=video,audio=audio))
        print(base.encode(dict(stage='train',block=block,seconds=time.perf_counter()-began)),flush=True)
    base.save_checkpoint(net,output/'final.paula',sources=list(hashes))
    np.savez_compressed(output/'final-body.npz',state=arm.state(),delay=delay.state())
    replay_net=base.load_checkpoint(output/'initial.paula',trusted=True).network
    first_video,first_audio=order[0][0]
    replay=world.course(replay_net,base.Arm(),features,groups,selected,first_video,first_audio,
                        reverse=reverse,delay=AfferentDelay(64))
    with np.load(output/training[0]['file']) as original:
        if set(original.files)!=set(replay) or any(not np.array_equal(original[k],replay[k]) for k in original.files):
            raise ValueError('First acquisition episode does not replay exactly')
    for kind in ('acquired','resting'):
        for weights in ('birth','learned'):
            for presentation in (('both',) if kind=='acquired' else world.PRESENTATIONS):
                for video in (0,1):
                    for audio in (0,1):
                        branch=base.load_checkpoint(output/f'{"final" if kind=="acquired" else "initial"}.paula',trusted=True).network
                        body=base.Arm();history=AfferentDelay(64)
                        if kind=='acquired':
                            body.restore(arm.state());history=AfferentDelay(64,delay.state())
                            if weights=='birth':
                                for nid,sid,_ in selected:branch.network.neurons[nid].postsynaptic_points[sid].u_i.info=0.
                        elif weights=='learned':transplant_selected(branch,net,selected)
                        z=world.course(branch,body,features,groups,selected,video,audio,
                            reverse=reverse,presentation=presentation,ticks=96,delay=history)
                        name=f'{kind}-{weights}-{presentation}-v{video}-a{audio}.npz'
                        probes.append(record(z,name,kind=kind,weights=weights,presentation=presentation,
                                             video=video,audio=audio,offset=0))
        print(base.encode(dict(stage='probe',kind=kind,seconds=time.perf_counter()-began)),flush=True)
    for video in (0,1):
        for audio in (0,1):
            branch=base.load_checkpoint(output/'initial.paula',trusted=True).network
            transplant_selected(branch,net,selected)
            z=world.course(branch,base.Arm(),features,groups,selected,video,audio,
                reverse=reverse,offset=64,ticks=96,delay=AfferentDelay(64))
            probes.append(record(z,f'shifted-learned-both-v{video}-a{audio}.npz',kind='shifted',
                weights='learned',presentation='both',video=video,audio=audio,offset=64))
    if any(base.digest(p)!=h for p,h in hashes.items()):raise ValueError('Sources changed during run')
    summary=dict(training=training,probes=probes,ticks=17*364+36*96,
                 recorded_ticks=16*364+36*96,birth_replay_exact=True,
                 seconds=time.perf_counter()-began,neurons=len(net.network.neurons))
    (output/'summary.json').write_text(base.encode(summary)+'\n')
    print(base.encode(dict(stage='complete',ticks=summary['ticks'],seconds=summary['seconds'])),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=11)
    p.add_argument('--reverse',action='store_true')
    a=p.parse_args();run(a.output,a.seed,a.reverse)
