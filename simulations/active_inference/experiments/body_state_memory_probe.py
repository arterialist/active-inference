"""Separate physical displacement from retained neural memory and delayed history.

At the previously identified block-12 checkpoint, cross selected-weight reset
with a physical intervention that puts the arm at rest. Brain state, in-flight
neural events and delayed afferent history stay acquired in every branch. The
physical intervention is an experimental displacement, not an agent action.
Each branch continues for 192 ticks with learning active. Original acquired
branches must replay every field of the existing 96-tick probes exactly.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from . import context_organization as base
from .crossed_av_continuation import isolated_rng,restore_acquired
from .magnitude_feedback_learning import observed_course
from .magnitude_feedback_analysis import verify_returns
from .opponent_context import verify_afferents
from .temporal_verification import verify_learning


def check_prefix(data,original,ticks=96):
    """Explicit identities for initial, terminal and variable-length arrays."""
    initial={'neuron_ids','body_initial','weights_initial','context_initial','error_initial',
             'delay_initial','prediction_ids','context_source_ids','terminal_ids','terminal_initial'}
    for key in original.files:
        old=original[key]
        if key in initial:new=data[key]
        elif key=='delay_final':
            # The final delay queue belongs to tick 96, not to the new endpoint.
            n=len(data['delay_initial']);new=data['raw_afferents'][ticks-n:ticks]
        elif key=='retrograde_offsets':new=data[key][:ticks+1]
        elif key=='retrograde_events':new=data[key][:data['retrograde_offsets'][ticks]]
        else:new=data[key][:ticks]
        if not np.array_equal(new,old):raise ValueError(f'Acquired replay differs: {key}')


def run(reference,output,blocks=12):
    reference=Path(reference).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if blocks not in (4,8,12,16):raise ValueError('Need an acquired/reset checkpoint')
    if shutil.disk_usage(output.parent).free<3*1024**3:raise OSError('Need 3 GiB reserve')
    m=json.loads((reference/'manifest.json').read_text())
    s=json.loads((reference/f'completed-block-{blocks}.json').read_text())
    for p,h in {**m['source_hashes'],**m['physical_sources']}.items():
        if base.digest(p)!=h:raise ValueError('Reference source changed')
    checkpoint=next(c for c in s['checkpoints'] if c['blocks']==blocks)
    for key in ('neural','physical'):
        if base.digest(reference/checkpoint[key])!=checkpoint[key+'_sha256']:raise ValueError('Checkpoint changed')
    features=[]
    for clip in (0,1):
        paths=[p for p in m['physical_sources'] if Path(p).name==f'sensory-{clip}.npz']
        if len(paths)!=1:raise ValueError('Ambiguous media')
        with np.load(paths[0]) as z:features.append({k:z[k] for k in ('visual','auditory')})
    output.mkdir();hashes=dict(m['source_hashes']);hashes[str(Path(__file__).resolve())]=base.digest(__file__)
    manifest=dict(seed=m['seed'],blocks=blocks,reference=str(reference),
        reference_manifest_sha256=base.digest(reference/'manifest.json'),checkpoint=checkpoint,
        source_hashes=hashes,physical_sources=m['physical_sources'],groups=m['groups'],selected=m['selected'],
        cell_fields=base.FIELDS,limits=__doc__,
        hypothesis='Does the effect of selected memory on pose regulation change when physical '
          'displacement is removed while acquired neural and delayed afferent state remain?')
    (output/'manifest.json').write_text(base.encode(manifest)+'\n')
    rows=[]
    for physical in ('acquired','rest'):
        for weights in ('learned','reset'):
            for video,audio in ((0,0),(0,1),(1,0),(1,1)):
                with isolated_rng():
                    net,arm,history=restore_acquired(reference/checkpoint['neural'],reference/checkpoint['physical'])
                    if physical=='rest':
                        time=arm.data.time;arm=base.Arm();arm.data.time=time
                    if weights=='reset':
                        for nid,sid,_ in m['selected']:net.network.neurons[nid].postsynaptic_points[sid].u_i.info=0.
                    data=observed_course(net,arm,history,features,m['groups'],m['selected'],video,audio,192)
                residuals=dict(learning=verify_learning(data),physics=base.verify_physics(data),
                               afferents=verify_afferents(data),returns=verify_returns(data))
                if physical=='acquired':
                    old=next(r for r in s['probes'] if (r['blocks'],r['kind'],r['weights'],r['video'],r['audio'])
                        ==(blocks,'acquired',weights,video,audio))
                    if base.digest(reference/old['file'])!=old['sha256']:raise ValueError('Replay record changed')
                    with np.load(reference/old['file']) as original:check_prefix(data,original)
                name=f'{physical}-{weights}-v{video}-a{audio}.npz';np.savez_compressed(output/name,**data)
                rows.append(dict(physical=physical,weights=weights,video=video,audio=audio,file=name,
                    sha256=base.digest(output/name),residuals=residuals,acquired_prefix_exact=physical=='acquired'))
                if shutil.disk_usage(output).free<3*1024**3:raise OSError('Storage reserve reached')
        print(base.encode(dict(seed=m['seed'],physical=physical,completed_ticks=len(rows)*192)),flush=True)
    if any(base.digest(p)!=h for p,h in hashes.items()):raise ValueError('Source changed')
    result=dict(records=rows,executed_ticks=len(rows)*192,exact_reference_ticks=8*96)
    (output/'summary.json').write_text(base.encode(result)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--blocks',type=int,default=12)
    a=p.parse_args();run(a.reference,a.output,a.blocks)
