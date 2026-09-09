"""One declared feedback-gain intervention on an acquired opponent network.

Scale existing comparator input weights by one half, preserving their signs,
delays, active learning and the complete acquired state elsewhere. This tests
the unstable correction loop, not whether a differently trained circuit learns
a repertoire. The multiplier is not fitted: the new circuit lets two learning
branches respond to the same signed residual. Half is a structural control for
that doubled participation, though rectification, bounds and nonlinear rates
prevent exact gain equivalence. Unchanged replay must match all original fields.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from . import context_organization as base
from .opponent_context import AfferentDelay, course, verify_afferents


def attenuate_comparators(net, groups):
    changes=[]
    for nid in groups['error_positive']+groups['error_negative']:
        n=net.network.neurons[nid]
        for sid,p in n.postsynaptic_points.items():
            before=float(p.u_i.info)
            p.u_i.info=.5*before
            changes.append([nid,sid,before,float(p.u_i.info)])
    return changes


def run(source,output):
    source=Path(source).resolve();output=Path(output).resolve()
    m=json.loads((source/'manifest.json').read_text())
    old=json.loads((source/'summary.json').read_text())
    if not m['opponent']:
        raise ValueError('This probe targets the coupled opponent architecture')
    for p,h in {**m['source_hashes'],**m['physical_sources']}.items():
        if base.digest(p)!=h:
            raise ValueError('Source changed')
    features=[]
    for clip in (0,1):
        matches=[Path(p) for p in m['physical_sources'] if Path(p).name==f'sensory-{clip}.npz']
        if len(matches)!=1:
            raise ValueError('Ambiguous media source')
        with np.load(matches[0]) as z:
            features.append({k:z[k] for k in ('visual','auditory')})
    with np.load(source/'final-body.npz') as z:
        state=z['state'];history=z['afferent_history']
    output.mkdir(exist_ok=False);results=[];began=time.perf_counter()
    manifest=dict(source=str(source),source_manifest_sha256=base.digest(source/'manifest.json'),
        producer_sha256=base.digest(__file__),seed=m['seed'],order=m['order'],groups=m['groups'],
        checkpoint_sha256=base.digest(source/'final.paula'),
        intervention='Multiply every incoming comparator info weight by 0.5 at branch start; all learning continues.',
        limits='Expression/stability intervention, not a new acquisition or pure loop-gain equivalence.')
    (output/'manifest.json').write_text(base.encode(manifest)+'\n')
    for delay in (0,64):
        for context in (0,1):
            for clip in (0,1):
                original=f'probe-d{delay}-r0-c{context}-v{clip}.npz'
                row=next(r for r in old['probes'] if r['file']==original)
                if base.digest(source/original)!=row['sha256']:
                    raise ValueError('Reference changed')
                changes=[]
                for changed in (False,True):
                    net=base.load_checkpoint(source/'final.paula',trusted=True).network
                    arm=base.Arm();arm.restore(state)
                    if changed:
                        changes=attenuate_comparators(net,m['groups'])
                    data=course(net,arm,features,m['groups'],m['selected'],context,clip,ticks=192,
                        delay=AfferentDelay(delay,history if delay else None))
                    if not changed:
                        with np.load(source/original) as z:
                            if set(z.files)!=set(data) or any(not np.array_equal(z[k],data[k]) for k in z.files):
                                raise ValueError('Unchanged full-field replay differs')
                        continue
                    name=f'half-d{delay}-c{context}-v{clip}.npz'
                    np.savez_compressed(output/name,**data)
                    results.append(dict(file=name,sha256=base.digest(output/name),delay=delay,
                        context=context,clip=clip,changes=changes,unchanged_replay_exact=True,
                        physics_residual=base.verify_physics(data),learning_residual=base.verify_learning(data),
                        afferent_residual=verify_afferents(data),min_eta=float(data['eta'].min())))
            print(base.encode(dict(stage='feedback-probe',delay=delay,context=context,
                                  seconds=time.perf_counter()-began)),flush=True)
    result=dict(results=results,ticks=16*192,seconds=time.perf_counter()-began)
    (output/'summary.json').write_text(base.encode(result)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.source,a.output)
