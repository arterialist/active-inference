"""Test acquired prediction weights from a common resting body and birth state.

This is a causal memory-content assay, not a replacement for full-state embodied
acceptance. The previous acquisition's body displacement and restoring reflex
can obscure the sign of learned disturbance compensation. Move only selected
weights into the original executable brain; keep basal plasticity positive.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from . import context_organization as base
from .opponent_context import AfferentDelay,course,verify_afferents
from .temporal_verification import verify_learning


def transplant_selected(destination,source,selected):
    for nid,sid,_ in selected:
        before=source.network.neurons[nid].postsynaptic_points[sid].u_i.info
        destination.network.neurons[nid].postsynaptic_points[sid].u_i.info=float(before)


def run(source,output):
    source=Path(source).resolve();output=Path(output).resolve()
    m=json.loads((source/'manifest.json').read_text())
    summary=json.loads((source/'summary.json').read_text())
    for p,h in {**m['source_hashes'],**m['physical_sources']}.items():
        if base.digest(p)!=h:raise ValueError('Source changed')
    features=[]
    for clip in (0,1):
        paths=[Path(p) for p in m['physical_sources'] if Path(p).name==f'sensory-{clip}.npz']
        if len(paths)!=1:raise ValueError('Ambiguous sensory source')
        with np.load(paths[0]) as z:features.append({k:z[k] for k in ('visual','auditory')})
    reference=summary['training'][0]
    if base.digest(source/reference['file'])!=reference['sha256']:raise ValueError('Reference changed')
    output.mkdir(exist_ok=False);began=time.perf_counter()
    manifest=dict(source=str(source),source_manifest_sha256=base.digest(source/'manifest.json'),
        producer_sha256=base.digest(__file__),seed=m['seed'],order=m['order'],aligned=m['aligned'],
        groups=m['groups'],selected=m['selected'],
        initial_checkpoint_sha256=base.digest(source/'initial.paula'),
        learned_checkpoint_sha256=base.digest(source/'final.paula'),
        intervention='Transfer selected learned prediction q only into original executable birth state and resting body.',
        limits='96-tick causal content probe, not complete embodied acceptance or full engram transfer. '
               'Plasticity remains active; first 64 ticks have no newly measured bodily load.')
    (output/'manifest.json').write_text(base.encode(manifest)+'\n')
    net=base.load_checkpoint(source/'initial.paula',trusted=True).network
    replay=course(net,base.Arm(),features,m['groups'],m['selected'],0,m['order'],delay=AfferentDelay(64))
    with np.load(source/reference['file']) as z:
        if set(z.files)!=set(replay) or any(not np.array_equal(z[k],replay[k]) for k in z.files):
            raise ValueError('Birth replay differs from original acquisition')
    learned=base.load_checkpoint(source/'final.paula',trusted=True).network
    results=[]
    for weights in ('birth','learned'):
        for context in (0,1):
            for clip in (0,1):
                net=base.load_checkpoint(source/'initial.paula',trusted=True).network
                if weights=='learned':transplant_selected(net,learned,m['selected'])
                data=course(net,base.Arm(),features,m['groups'],m['selected'],context,clip,
                            ticks=96,delay=AfferentDelay(64))
                name=f'{weights}-c{context}-v{clip}.npz';np.savez_compressed(output/name,**data)
                results.append(dict(file=name,sha256=base.digest(output/name),weights=weights,
                    context=context,clip=clip,physics_residual=base.verify_physics(data),
                    learning_residual=verify_learning(data),afferent_residual=verify_afferents(data),
                    min_eta=float(data['eta'].min())))
    result=dict(results=results,birth_replay_exact=True,ticks=364+8*96,seconds=time.perf_counter()-began)
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(stage='complete',ticks=result['ticks'],seconds=result['seconds'])),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.source,a.output)
