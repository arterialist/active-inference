"""Screen a local inhibitory population in the same 590-cell embodied circuit.

Three conditions: unchanged graph, added pools with zero output weight, and
functional mean-activity feedback. The wired zero control keeps return paths,
so it is not claimed to be identical to no added cells. Every condition gets
four fresh-state crossed-media body episodes with ongoing learning. This
screens representation and coupling, not multi-episode memory retention.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np

from . import context_organization as base
from .magnitude_feedback_learning import observed_course
from .opponent_context import AfferentDelay,verify_afferents
from .temporal_verification import verify_learning
from ..components.learning.feedback_competition import append_feedback_competition
from neuron.extensions.experimental.magnitude_retrograde import MagnitudeRetrogradeNeuron


def run(reference,output):
    reference=Path(reference).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free<3*1024**3:raise OSError('Need 3 GiB reserve')
    m=json.loads((reference/'manifest.json').read_text())
    progress=json.loads((reference/'completed-block-4.json').read_text())
    for p,h in {**m['source_hashes'],**m['physical_sources']}.items():
        if base.digest(p)!=h:raise ValueError('Reference changed')
    original=json.loads((reference/'config.json').read_text())
    if base.digest(reference/'config.json')!=m['config_sha256']:raise ValueError('Graph changed')
    features=[]
    for clip in (0,1):
        paths=[p for p in m['physical_sources'] if Path(p).name==f'sensory-{clip}.npz']
        if len(paths)!=1:raise ValueError('Ambiguous media')
        with np.load(paths[0]) as z:features.append({k:z[k] for k in ('visual','auditory')})
    output.mkdir();hashes=dict(m['source_hashes'])
    from ..components.learning import feedback_competition as component
    for p in (__file__,component.__file__):hashes[str(Path(p).resolve())]=base.digest(p)
    manifest=dict(seed=m['seed'],reference=str(reference),source_hashes=hashes,
        physical_sources=m['physical_sources'],reference_manifest_sha256=base.digest(reference/'manifest.json'),
        groups={},config_hashes={},selected=m['selected'],cell_fields=base.FIELDS,limits=__doc__,
        hypothesis='Local feedback decreases shared representation relative to joint structure '
                   'without silencing receptors or consumers. No claim about learned memory yet.',
        frozen_linear_common_mode_radius=float(abs(np.roots([1.,-1.,.25,0.,.25*.99**2])).max()))
    records=[];replay=False
    for condition,strength in (('none',None),('wired_zero',0.),('feedback',1.)):
        groups=deepcopy(m['groups'])
        cfg,extra=append_feedback_competition(original,
            {key:groups[key] for key in ('mixed_0','mixed_1')},enabled=strength is not None,
            strength=0. if strength is None else strength)
        groups.update(extra);manifest['groups'][condition]=groups
        cfg_path=output/f'{condition}.json';cfg_path.write_text(base.encode(cfg)+'\n')
        manifest['config_hashes'][condition]=base.digest(cfg_path)
        for v,a in ((0,0),(0,1),(1,0),(1,1)):
            net=base.fresh(cfg_path,m['seed'],MagnitudeRetrogradeNeuron)[0]
            data=observed_course(net,base.Arm(),AfferentDelay(64),features,groups,m['selected'],v,a)
            if condition=='none' and [v,a]==m['schedule'][0][0]:
                old=progress['training'][0]
                if base.digest(reference/old['file'])!=old['sha256']:raise ValueError('Replay evidence changed')
                with np.load(reference/old['file']) as z:
                    if set(z.files)!=set(data) or any(not np.array_equal(z[k],data[k]) for k in z.files):
                        raise ValueError('Unchanged whole-loop replay differs')
                replay=True
            residuals=dict(learning=verify_learning(data),physics=base.verify_physics(data),afferents=verify_afferents(data))
            name=f'{condition}-v{v}-a{a}.npz';np.savez_compressed(output/name,**data)
            records.append(dict(condition=condition,video=v,audio=a,file=name,
                                sha256=base.digest(output/name),residuals=residuals))
            if shutil.disk_usage(output).free<3*1024**3:raise OSError('Storage reserve reached')
        print(base.encode(dict(seed=m['seed'],completed=condition)),flush=True)
    if not replay:raise ValueError('Missing original exact replay')
    if any(base.digest(p)!=h for p,h in hashes.items()):raise ValueError('Source changed during run')
    (output/'manifest.json').write_text(base.encode(manifest)+'\n')
    result=dict(records=records,unchanged_replay_exact=True,executed_ticks=12*364)
    (output/'summary.json').write_text(base.encode(result)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.reference,a.output)
