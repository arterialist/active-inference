"""Test weight-carried associations in the original unexperienced neural state.

Transfer only contextual input weights from a trusted learned checkpoint. A
within-predictor permutation preserves the full weight multiset and fan-in but
changes feature assignment. No activity, error/context trace, queue, modulator,
terminal or other learned weight transfers. All plasticity continues at test.
This is a causal preparation, not a proposed biological memory reset mechanism.
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
from .predictive_bridge_probe import record, audit_record
from .contrast_prediction_probe import record_contrast
from ..core.runtime_checkpoint import load_checkpoint
from neuron.neuron import setup_neuron_logger


def recording_contrast_groups(groups):
    """Recover the creator's ID-allocation order, not JSON's sorted key order.

    Existing contrast records allocate contiguous ascending IDs by population,
    then flatten weights in that order. The JSON encoder sorts dictionary keys.
    New manifests also save the explicit ID sequence to avoid inference later.
    """
    ordered=dict(sorted(groups.items(),key=lambda item:min(item[1])))
    ids=[n for population in ordered.values() for n in population]
    if not ids or len(ids)!=len(set(ids)) or ids!=sorted(ids):
        raise ValueError('Cannot recover the contrast recording layout')
    return ordered


def arrange_weights(weights, *, shuffled=False, seed=23):
    weights=np.asarray(weights,dtype=float)
    if weights.ndim!=2 or not np.isfinite(weights).all() or np.any(weights<0):
        raise ValueError('Expected a finite nonnegative receptor weight matrix')
    result=weights.copy()
    if shuffled:
        rng=np.random.default_rng(seed)
        for row in result: rng.shuffle(row)
    if not np.array_equal(np.sort(result,axis=1),np.sort(weights,axis=1)):
        raise ValueError('Weight distribution changed')
    return result


def install_weights(net, bridge, weights):
    predictors=[net.network.neurons[n] for n in bridge['prediction']]
    if weights.shape!=(len(predictors),len(predictors[0].prediction_ports)):
        raise ValueError('Transplant shape differs')
    # Validate the entire intervention before mutating any target.
    if not np.isfinite(weights).all() or np.any(weights<0) or any(
            np.any(row>n.prediction_cap) for n,row in zip(predictors,weights)):
        raise ValueError('Transplant violates the local receptor cap')
    for n,row in zip(predictors,weights):
        for sid,value in zip(n.prediction_ports,row):
            n.postsynaptic_points[sid].u_i.info=float(value)


def run(source,output,*,shuffle_seed=23):
    source,output=Path(source).resolve(),Path(output).resolve()
    m=json.loads((source/'manifest.json').read_text());cfg=json.loads((source/'config.json').read_text())
    # Checkpoints are local executable data. Never use this entrypoint on uploads.
    for p,h in {**m['source_hashes'],**m['physical_sources']}.items():
        if digest(p)!=h:raise ValueError('Source runtime or media changed')
    checkpoints={name:source/f'{name}.neural-checkpoint' for name in ('initial','trained')}
    hashes={str(p):digest(p) for p in checkpoints.values()}
    for obj in (run,record,record_contrast,load_checkpoint):
        p=Path(inspect.getfile(obj)).resolve();hashes[str(p)]=digest(p)
    setup_neuron_logger('CRITICAL')
    trained=load_checkpoint(checkpoints['trained'],trusted=True).network
    weights=np.array([[trained.network.neurons[n].postsynaptic_points[s].u_i.info
                       for s in trained.network.neurons[n].prediction_ports] for n in m['bridge']['prediction']])
    del trained
    features=[]
    for p in sorted(m['physical_sources']):
        with np.load(p) as z:features.append({k:z[k] for k in z.files})
    if shutil.disk_usage(output.parent).free<750*1024**2:raise OSError('Free-space reserve reached')
    output.mkdir(exist_ok=False)
    contrast=recording_contrast_groups(m['contrast']) if 'contrast' in m else None
    manifest=dict(source=str(source),source_manifest_sha256=digest(source/'manifest.json'),
        source_config_sha256=digest(source/'config.json'),source_hashes=hashes,
        shuffle_seed=shuffle_seed,mapping=m['mapping'],order=m['order'],bridge=m['bridge'],
        contrast_weight_neuron_order=[n for ids in contrast.values() for n in ids] if contrast else [],
        intervention='Selected contextual information weights only, in initial executable state',
        limits='One base graph and one weight-permutation seed; not independent replication. '
        'Same two physical clips, no category generalization or functional action claim.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    entries=[];used=0

    def measure(condition,clip,length):
        nonlocal used
        if used>120*1024**2 or shutil.disk_usage(output).free<650*1024**2:
            raise OSError('Transplant recording budget reached')
        branch=load_checkpoint(checkpoints['initial'],trusted=True);net=branch.network
        if net.current_tick!=0:raise ValueError('Initial checkpoint is not birth state')
        if condition!='birth_control':
            q=arrange_weights(weights,shuffled=condition=='shuffled',seed=shuffle_seed)
            install_weights(net,m['bridge'],q)
        # Recorder uses net.run_tick; restore this branch's own RNG state before
        # its uninterrupted recording and restore ambient RNG afterward.
        ambient=random.getstate(),np.random.get_state()
        random.setstate(branch.python_rng);np.random.set_state(branch.numpy_rng)
        trial=dict(start=0,stop=length,visual_clip=clip,audio_clip=None)
        try:
            if contrast:
                data=record_contrast(net,features,m['groups'],m['bridge'],contrast,trial)
            else:data=record(net,features,m['groups'],m['bridge'],trial)
        finally:
            random.setstate(ambient[0]);np.random.set_state(ambient[1])
        residual=audit_record(data,cfg,m['bridge'])
        golden_path=source/f'probe-initial-{clip}.npz'
        progress=json.loads((source/'progress.json').read_text())
        golden_entry=next(e for e in progress['entries'] if e['file']==golden_path.name)
        if digest(golden_path)!=golden_entry['sha256']:raise ValueError('Initial reference changed')
        with np.load(golden_path) as golden:
            if condition=='birth_control':
                for key,value in data.items():
                    if key.startswith('start_'):expected=golden[key]
                    elif key=='end_incoming_info':continue  # Different stop time.
                    else:expected=golden[key][:length]
                    if not np.array_equal(value,expected):raise ValueError(f'Birth replay differs: {key}')
            else:
                for key,value in data.items():
                    if key.startswith('start_') and key not in ('start_weights','start_incoming_info'):
                        if not np.array_equal(value,golden[key]):raise ValueError(f'Undeclared starting-state change: {key}')
                all_points=[p for n in cfg['neurons'] for p in cfg['synaptic_points']
                            if p['type']=='postsynaptic' and p['neuron_id']==n['id']]
                expected=golden['start_incoming_info'].copy()
                chosen={(n,s):float(value) for n,row in zip(m['bridge']['prediction'],q)
                        for s,value in zip(net.network.neurons[n].prediction_ports,row)}
                for i,p in enumerate(all_points):
                    if (p['neuron_id'],p['synapse_id']) in chosen:
                        expected[i]=chosen[p['neuron_id'],p['synapse_id']]
                if not np.array_equal(data['start_incoming_info'],expected):
                    raise ValueError('Incoming weights changed beyond the transplant')
        path=output/f'{condition}-cue{clip}.npz';np.savez_compressed(path,**data);used+=path.stat().st_size
        entries.append(dict(file=path.name,sha256=digest(path),condition=condition,clip=clip,
                            ticks=length,equation_residual=residual,initial_reference_sha256=digest(golden_path)))
        (output/'progress.json').write_text(encode(dict(entries=entries,bytes=used))+'\n')
        print(encode(entries[-1]),flush=True)
        if used>150*1024**2:raise OSError('Hard transplant recording budget reached')

    for clip in (0,1):measure('birth_control',clip,32)
    for condition in ('learned','shuffled'):
        for clip in (0,1):measure(condition,clip,300)
    if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('Source changed during probing')
    result=dict(entries=entries,bytes=used,ticks=sum(e['ticks'] for e in entries))
    (output/'completion.json').write_text(encode(result)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--shuffle-seed',type=int,default=23)
    run(**vars(p.parse_args()))
