"""Probe acquired feature associations under changed sensory timing.

No retraining. Only learned or within-cell shuffled contextual weights transfer
into the original executable state. The neural equations and ongoing plasticity
are unchanged. Continuous input uses current physical values, never future
samples. For a changing movie, dose is recorded, NOT assumed equal to the
four-phase encoder's subsampled dose. Two timing interventions are exploratory.
"""
import argparse
import inspect
import json
from pathlib import Path
import random
import shutil

import numpy as np

from .association_route_probe import digest
from .composition_probe import encode
from . import predictive_bridge_probe as recorder
from .contrast_prediction_probe import record_contrast
from .multimodal_pairing_probe import inputs as native_inputs
from .predictive_weight_transplant import install_weights,recording_contrast_groups
from ..core.runtime_checkpoint import load_checkpoint
from neuron.neuron import setup_neuron_logger


def timed_inputs(features,groups,trial,tick,mode):
    if mode=='native':return native_inputs(features,groups,trial,tick)
    if mode not in ('phase_shift_1','continuous'):raise ValueError('Unknown sensory timing')
    rel=tick-trial['start'];result=[]
    for key,role,field in (('visual_clip','vision','visual'),('audio_clip','touch','auditory')):
        clip=trial[key]
        if clip is None:continue
        values=features[clip][field][rel%features[clip]['ticks']]
        for nid,value in zip(groups[role],values):
            if value>0 and (mode=='continuous' or (rel+1)%4==nid%4):
                result.append((nid,float(value*(.5 if mode=='continuous' else 2.))))
    return result


def record_timing(net,features,groups,bridge,contrast,trial,mode):
    """Compatibility adapter for the immutable historical recorder.

    Only its external stimulus callback is temporarily replaced. No neural
    method or parameter is changed here. One process per recording, no threads.
    The callback and all actual applied inputs are checked/retained.
    """
    ids=groups['vision']+groups['touch'];index={n:i for i,n in enumerate(ids)}
    if len(index)!=len(ids):raise ValueError('Sensory channel IDs overlap')
    rows=[];old=recorder.inputs
    if old is not native_inputs:raise RuntimeError('Another stimulus adapter is active')

    def drive(fs,gs,tr,t):
        signals=timed_inputs(fs,gs,tr,t,mode);row=np.zeros(len(ids))
        for n,v in signals:row[index[n]]=v
        rows.append(row);return signals

    recorder.inputs=drive
    try:data=record_contrast(net,features,groups,bridge,contrast,trial)
    finally:recorder.inputs=old
    data['external_information']=np.array(rows)
    return data


def run(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    m=json.loads((source/'manifest.json').read_text())
    parent=Path(m['source']);pm=json.loads((parent/'manifest.json').read_text())
    cfg=json.loads((parent/'config.json').read_text())
    done=json.loads((source/'completion.json').read_text())
    for p,h in {**m['source_hashes'],**pm['source_hashes'],**pm['physical_sources']}.items():
        if digest(p)!=h:raise ValueError('Source runtime, checkpoint or media changed')
    reference={}
    for row in done['entries']:
        if row['condition'] not in ('learned','shuffled'):continue
        path=source/row['file']
        if digest(path)!=row['sha256']:raise ValueError('Reference recording changed')
        reference[row['condition'],row['clip']]=path
    if len(reference)!=4:raise ValueError('Incomplete reference transplant course')
    contrast=recording_contrast_groups(pm['contrast']);features=[]
    for p in sorted(pm['physical_sources']):
        with np.load(p) as z:features.append({k:z[k] for k in z.files})
    hashes={str(path):digest(path) for path in reference.values()}
    for obj in (run,record_contrast,recorder.record,install_weights,load_checkpoint):
        path=Path(inspect.getfile(obj)).resolve();hashes[str(path)]=digest(path)
    if shutil.disk_usage(output.parent).free<750*1024**2:raise OSError('Free-space reserve reached')
    setup_neuron_logger('CRITICAL');output.mkdir(exist_ok=False)
    manifest=dict(source=str(source),parent=str(parent),source_manifest_sha256=digest(source/'manifest.json'),
        source_hashes=hashes,physical_sources=pm['physical_sources'],mapping=m['mapping'],order=m['order'],
        bridge=pm['bridge'],external_neuron_ids=pm['groups']['vision']+pm['groups']['touch'],
        contrast_weight_neuron_order=[n for ids in contrast.values() for n in ids],
        modes=['native','phase_shift_1','continuous'],
        scope='Acquired weights in birth state; active adaptation. Changed input timing, no retraining. '
        'Native-prefix exact controls. Physical movie dose can differ under temporal sampling; '
        'actual inputs are recorded. One graph seed, two familiar clips; no category or action claim.')
    (output/'manifest.json').write_text(encode(manifest)+'\n');entries=[];used=0
    for mode in manifest['modes']:
        length=32 if mode=='native' else 300
        for condition in ('learned','shuffled'):
            for clip in (0,1):
                if used>200*1024**2 or shutil.disk_usage(output).free<650*1024**2:
                    raise OSError('Timing-transfer recording reserve reached')
                branch=load_checkpoint(parent/'initial.neural-checkpoint',trusted=True)
                net=branch.network
                if net.current_tick!=0:raise ValueError('Need birth executable state')
                with np.load(reference[condition,clip]) as z:q=z['start_weights'].copy()
                install_weights(net,pm['bridge'],q)
                ambient=random.getstate(),np.random.get_state()
                random.setstate(branch.python_rng);np.random.set_state(branch.numpy_rng)
                trial=dict(start=0,stop=length,visual_clip=clip,audio_clip=None)
                try:data=record_timing(net,features,pm['groups'],pm['bridge'],contrast,trial,mode)
                finally:random.setstate(ambient[0]);np.random.set_state(ambient[1])
                residual=recorder.audit_record(data,cfg,pm['bridge'])
                with np.load(reference[condition,clip]) as z:
                    for key,value in data.items():
                        if key.startswith('start_'):expected=z[key]
                        elif mode=='native' and key not in ('end_incoming_info','external_information'):
                            expected=z[key][:length]
                        else:continue
                        if not np.array_equal(value,expected):raise ValueError(f'Unexpected reference change: {key}')
                name=f'{mode}-{condition}-cue{clip}.npz';path=output/name
                np.savez_compressed(path,**data);used+=path.stat().st_size
                entries.append(dict(file=name,sha256=digest(path),mode=mode,condition=condition,clip=clip,
                                    ticks=length,equation_residual=residual))
                (output/'progress.json').write_text(encode(dict(entries=entries,bytes=used))+'\n')
                print(encode(entries[-1]),flush=True)
                if used>230*1024**2:raise OSError('Hard per-course raw cap reached')
    if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('Source changed during probe')
    result=dict(entries=entries,bytes=used,ticks=sum(e['ticks'] for e in entries))
    (output/'completion.json').write_text(encode(result)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','output'):p.add_argument('--'+key,type=Path,required=True)
    run(**vars(p.parse_args()))
