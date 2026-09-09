"""Continuous normal/reversed physical contingencies with neural learning controls.

Three matched graphs retain identical reference wiring: native eligibility,
quarter-rate native eligibility, and neural-reference contrast. No adaptation
is frozen, and neither phase nor pair identity reaches the brain. The initial
short course is a structural preflight, not evidence of retained association.
"""
import argparse
import inspect
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np

from . import context_organization as base
from . import crossed_av_world as world
from .crossed_av_analysis import verify_continuity,verify_stimuli
from .crossed_av_continuation import isolated_rng,restore_acquired
from .eligibility_reference_probe import record,verify_learning
from .magnitude_feedback_learning import observed_course
from .magnitude_feedback_analysis import verify_returns
from .opponent_context import AfferentDelay,verify_afferents
from .temporal_memory_transplant import transplant_selected
from ..components.learning.eligibility_reference import append_eligibility_reference
from neuron.extensions.experimental.contrast_eligibility import ContrastEligibilityNeuron


CONDITIONS={'wired':(0.,1.),'slow':(0.,.25),'contrast':(1.,1.)}


def run(reference,output,normal_blocks=1,reversal_blocks=1):
    reference,output=map(lambda p:Path(p).resolve(),(reference,output))
    if output.exists():raise FileExistsError(output)
    if any(type(n) is not int or not 1<=n<=16 for n in (normal_blocks,reversal_blocks)):
        raise ValueError('Each phase needs 1..16 balanced blocks')
    if shutil.disk_usage(output.parent).free<3*1024**3:raise OSError('Need 3 GiB reserve')
    m=json.loads((reference/'manifest.json').read_text());s=json.loads((reference/'summary.json').read_text())
    hashes=dict(m['source_hashes'])
    for obj in (run,record,verify_learning,append_eligibility_reference,ContrastEligibilityNeuron):
        p=Path(inspect.getfile(obj)).resolve();hashes[str(p)]=base.digest(p)
    for p,h in {**hashes,**m['physical_sources']}.items():
        if base.digest(p)!=h:raise ValueError('Source changed')
    if base.digest(reference/'config.json')!=m['config_sha256'] or s['blocks']!=16:
        raise ValueError('Need completed sixteen-block reference with original birth graph')
    features=[]
    for clip in (0,1):
        paths=[p for p in m['physical_sources'] if Path(p).name==f'sensory-{clip}.npz']
        if len(paths)!=1:raise ValueError('Ambiguous media')
        with np.load(paths[0]) as z:features.append({k:z[k] for k in ('visual','auditory')})
    original=json.loads((reference/'config.json').read_text());seed=m['seed'];selected=m['selected']
    output.mkdir();began=time.perf_counter();order=world.schedule(max(normal_blocks,reversal_blocks))
    old=s['training'][0]
    with isolated_rng():
        net=base.fresh(reference/'config.json',seed,ContrastEligibilityNeuron)[0]
        pre=observed_course(net,base.Arm(),AfferentDelay(64),features,m['groups'],selected,old['video'],old['audio'])
    if base.digest(reference/old['file'])!=old['sha256']:raise ValueError('Reference replay record changed')
    with np.load(reference/old['file']) as z:
        if set(pre)!=set(z.files) or any(not np.array_equal(pre[k],z[k]) for k in pre):
            raise ValueError('Disabled extension differs from original body course')
    np.savez_compressed(output/'preflight.npz',**pre)
    configs={};groups={};rows=[];checkpoints=[];ticks=364
    for condition,(strength,rate) in CONDITIONS.items():
        cfg,added=append_eligibility_reference(original,{k:m['groups'][k] for k in ('mixed_0','mixed_1')},
            m['groups']['prediction'],enabled=True,strength=strength,rate_scale=rate)
        path=output/f'{condition}.json';path.write_text(base.encode(cfg)+'\n')
        configs[condition]=base.digest(path);groups[condition]=dict(m['groups'],**added)
    manifest=dict(seed=seed,reference=str(reference),reference_manifest_sha256=base.digest(reference/'manifest.json'),
        reference_summary_sha256=base.digest(reference/'summary.json'),source_hashes=hashes,
        physical_sources=m['physical_sources'],groups=groups,selected=selected,config_hashes=configs,
        normal_blocks=normal_blocks,reversal_blocks=reversal_blocks,schedule=order,conditions=CONDITIONS,
        preflight_exact=True,preflight_sha256=base.digest(output/'preflight.npz'),
        cell_fields=base.FIELDS,xml=base.XML,neural_tick_seconds=base.DT,physical_delay=64,
        limits=__doc__+' Quarter-rate is a predetermined control, not fitted or exposure-matched. '
        'Only two recordings, one joint and one order; a short prefix cannot establish retention. '
        'All reference incoming weights, cell states and terminal releases are recorded.')
    (output/'manifest.json').write_text(base.encode(manifest)+'\n')

    def persist(data,name,**meta):
        nonlocal ticks
        if shutil.disk_usage(output).free<3*1024**3:raise OSError('Storage reserve reached')
        if any(not np.isfinite(v).all() for v in data.values()):raise ValueError('Nonfinite trace')
        residuals=dict(learning=verify_learning(data),physics=base.verify_physics(data),afferents=verify_afferents(data))
        verify_returns(data);verify_stimuli(data,features,meta,meta['reverse'])
        np.savez_compressed(output/name,**data);ticks+=len(data['body'])
        rows.append(dict(file=name,sha256=base.digest(output/name),residuals=residuals,**meta))

    for condition in CONDITIONS:
        g=groups[condition];net=base.fresh(output/f'{condition}.json',seed,ContrastEligibilityNeuron)[0]
        arm=base.Arm();delay=AfferentDelay(64);previous=None
        birth=output/f'{condition}-initial.paula';base.save_checkpoint(net,birth,sources=list(hashes))
        for reverse,count in ((False,normal_blocks),(True,reversal_blocks)):
            for block,pairs in enumerate(order[:count]):
                tag=f'{condition}-r{int(reverse)}-b{block+1}'
                for v,a in pairs:
                    data=record(net,arm,delay,features,g,selected,v,a,reverse=reverse)
                    if previous is not None:
                        verify_continuity(previous,data)
                        if not np.array_equal(previous['reference_trace'][-1],data['reference_initial']):
                            raise ValueError('Reference state reset between episodes')
                    persist(data,f'{tag}-train-v{v}-a{a}.npz',condition=condition,reverse=reverse,
                            block=block+1,kind='training',weights='learned',video=v,audio=a)
                    previous=data
                checkpoint=output/f'{tag}.paula';physical=output/f'{tag}-body.npz'
                base.save_checkpoint(net,checkpoint,sources=list(hashes))
                np.savez_compressed(physical,state=arm.state(),delay=delay.state())
                checkpoints.append(dict(condition=condition,reverse=reverse,block=block+1,
                    neural=checkpoint.name,neural_sha256=base.digest(checkpoint),
                    physical=physical.name,physical_sha256=base.digest(physical)))
                for v,a in ((0,0),(0,1),(1,0),(1,1)):
                    with isolated_rng():
                        saved=base.load_checkpoint(birth,trusted=True);branch=saved.network
                        random.setstate(saved.python_rng);np.random.set_state(saved.numpy_rng)
                        transplant_selected(branch,net,selected)
                        data=record(branch,base.Arm(),AfferentDelay(64),features,g,selected,v,a,ticks=96,reverse=reverse)
                    persist(data,f'{tag}-rest-v{v}-a{a}.npz',condition=condition,reverse=reverse,
                        block=block+1,kind='resting',weights='learned',video=v,audio=a)
                if block+1==count:
                    for weight_state in ('learned','reset'):
                        for v,a in ((0,0),(0,1),(1,0),(1,1)):
                            with isolated_rng():
                                branch,body,history=restore_acquired(checkpoint,physical)
                                if weight_state=='reset':
                                    for nid,sid,_ in selected:branch.network.neurons[nid].postsynaptic_points[sid].u_i.info=0.
                                data=record(branch,body,history,features,g,selected,v,a,ticks=96,reverse=reverse)
                            persist(data,f'{tag}-acquired-{weight_state}-v{v}-a{a}.npz',condition=condition,
                                reverse=reverse,block=block+1,kind='acquired',weights=weight_state,video=v,audio=a)
                progress=dict(rows=rows,checkpoints=checkpoints,executed_ticks=ticks,seconds=time.perf_counter()-began)
                (output/f'completed-{tag}.json').write_text(base.encode(progress)+'\n')
                print(base.encode(dict(seed=seed,condition=condition,reverse=reverse,block=block+1,ticks=ticks)),flush=True)
    if any(base.digest(p)!=h for p,h in hashes.items()):raise ValueError('Source changed during course')
    (output/'summary.json').write_text(base.encode(progress)+'\n')
    return progress


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--normal-blocks',type=int,default=1);p.add_argument('--reversal-blocks',type=int,default=1)
    a=p.parse_args();run(a.reference,a.output,a.normal_blocks,a.reversal_blocks)
