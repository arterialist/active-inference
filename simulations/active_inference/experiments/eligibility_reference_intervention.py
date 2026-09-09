"""Matched acquired-state interventions on ongoing learning, never a host policy.

Restore one contrast-acquired checkpoint, retaining its weights, body, queues,
reference/error traces and RNG. Compare intact contrast, native eligibility,
and quarter-rate native eligibility. All rates remain positive. The physical
stimulus is unchanged. Exact intact replay against all four old acquired probes
is required; no intervention branch writes back to the acquiring organism.
"""
import argparse
import inspect
import json
from pathlib import Path
import shutil

import numpy as np

from . import context_organization as base
from .crossed_av_analysis import verify_stimuli
from .crossed_av_continuation import isolated_rng,restore_acquired
from .eligibility_reference_probe import record,verify_learning
from .eligibility_reference_pool_audit import audit_pools
from .magnitude_feedback_analysis import verify_returns
from .opponent_context import verify_afferents


INTERVENTIONS={'intact':(1.,1.),'native':(0.,1.),'slow_native':(0.,.25)}


def run(parent,output):
    parent,output=map(lambda p:Path(p).resolve(),(parent,output))
    if output.exists():raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free<3*1024**3:raise OSError('Need 3 GiB reserve')
    m=json.loads((parent/'manifest.json').read_text());block=m['normal_blocks']
    progress=parent/f'completed-contrast-r0-b{block}.json';s=json.loads(progress.read_text())
    hashes=dict(m['source_hashes'])
    for obj in (run,audit_pools):
        p=Path(inspect.getfile(obj)).resolve();hashes[str(p)]=base.digest(p)
    for p,h in {**hashes,**m['physical_sources']}.items():
        if base.digest(p)!=h:raise ValueError('Source changed')
    cfg_path=parent/'contrast.json'
    if base.digest(cfg_path)!=m['config_hashes']['contrast']:raise ValueError('Contrast graph changed')
    cfg=json.loads(cfg_path.read_text());g=m['groups']['contrast'];selected=m['selected']
    checkpoint=next(c for c in s['checkpoints'] if (c['condition'],c['reverse'],c['block'])==('contrast',False,block))
    for name in ('neural','physical'):
        if base.digest(parent/checkpoint[name])!=checkpoint[name+'_sha256']:raise ValueError('Checkpoint changed')
    features=[]
    for clip in (0,1):
        paths=[p for p in m['physical_sources'] if Path(p).name==f'sensory-{clip}.npz']
        if len(paths)!=1:raise ValueError('Ambiguous media')
        with np.load(paths[0]) as z:features.append({k:z[k] for k in ('visual','auditory')})
    output.mkdir();rows=[];events=[]
    for v,a in ((0,0),(0,1),(1,0),(1,1)):
        original=next(r for r in s['rows'] if
            (r['condition'],r['reverse'],r['block'],r['kind'],r['weights'],r['video'],r['audio'])
            ==('contrast',False,block,'acquired','learned',v,a))
        if base.digest(parent/original['file'])!=original['sha256']:raise ValueError('Source probe changed')
        intact=None
        for condition,(strength,scale) in INTERVENTIONS.items():
            with isolated_rng():
                net,arm,delay=restore_acquired(parent/checkpoint['neural'],parent/checkpoint['physical'])
                for nid in g['prediction']:
                    n=net.network.neurons[nid]
                    n.prediction_reference_strength=strength
                    n.metadata['prediction_reference_strength']=strength
                    n.params.eta_post*=scale
                data=record(net,arm,delay,features,g,selected,v,a,ticks=96)
            residuals=dict(learning=verify_learning(data),physics=base.verify_physics(data),afferents=verify_afferents(data))
            verify_returns(data);verify_stimuli(data,features,dict(video=v,audio=a),False)
            pools=audit_pools(data,cfg)
            if condition=='intact':
                with np.load(parent/original['file']) as z:
                    if set(z.files)!=set(data) or any(not np.array_equal(z[k],data[k]) for k in z.files):
                        raise ValueError('Intact acquired reference replay differs')
                intact=data
            else:
                for key in ('weights_initial','body_initial','delay_initial','context_initial','error_initial',
                            'reference_initial','terminal_initial','pool_weight_initial'):
                    if not np.array_equal(data[key],intact[key]):raise ValueError('Unmatched acquired initial state')
                if not np.array_equal(data['cells'][:2],intact['cells'][:2]):
                    raise ValueError('Learning-path change altered early forward state')
                divergence={}
                for key in ('weights','cells','terminal_info','body'):
                    changed=np.flatnonzero(np.any(data[key]!=intact[key],axis=tuple(range(1,data[key].ndim))))
                    divergence[key]=int(changed[0]) if len(changed) else None
                events.append(dict(condition=condition,video=v,audio=a,first_difference=divergence))
            name=f'{condition}-v{v}-a{a}.npz';np.savez_compressed(output/name,**data)
            np.savez_compressed(output/f'{condition}-v{v}-a{a}-pool-audit.npz',**pools)
            rows.append(dict(file=name,sha256=base.digest(output/name),condition=condition,video=v,audio=a,
                residuals=residuals,reference_file=original['file'],reference_sha256=original['sha256']))
    manifest=dict(parent=str(parent),seed=m['seed'],block=block,groups=g,selected=selected,
        parent_manifest_sha256=base.digest(parent/'manifest.json'),parent_progress_sha256=base.digest(progress),
        checkpoint=checkpoint,source_hashes=hashes,physical_sources=m['physical_sources'],
        interventions=INTERVENTIONS,limits=__doc__)
    if any(base.digest(p)!=h for p,h in hashes.items()):raise ValueError('Source changed')
    (output/'manifest.json').write_text(base.encode(manifest)+'\n')
    result=dict(rows=rows,divergences=events,executed_ticks=1152,exact_replay_ticks=384)
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(seed=m['seed'],executed_ticks=1152,divergences=events)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.parent,a.output)
