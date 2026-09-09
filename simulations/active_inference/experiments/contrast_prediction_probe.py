"""Compose the neural contrast population with the locally learning predictor.

Same physical course, learning constants, 32 contextual inputs and unit birth
conductance as the raw-input bridge. Contrast adds delay and changes amplitude;
there is no fitted normalization. This is a representation intervention, not a
scale-only control. No body/live server, labels in neurons, or frozen plasticity.
"""
import argparse
import inspect
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .composition_probe import encode, fingerprint
from .association_route_probe import digest
from .multimodal_pairing_probe import fresh
from .media_order_control import protocol
from .population_hierarchy import FIELDS
from .predictive_bridge_probe import record, audit_record
from ..components.sensory.population_contrast import append_population_contrast
from ..components.learning.predictive_bridge import append_predictive_bridge
from ..core.runtime_checkpoint import save_checkpoint, load_checkpoint
from neuron.extensions.experimental.predictive_receptor import PredictiveReceptorNeuron


def record_contrast(net, features, groups, bridge, contrast, trial):
    """Passive additional recording of every contrast input weight."""
    added = [n for ids in contrast.values() for n in ids]
    rows = {nid: [] for nid in added}
    initial = np.array([p.u_i.info for nid in added
                        for p in net.network.neurons[nid].postsynaptic_points.values()])
    original = PredictiveReceptorNeuron.tick

    def observed(n, external_inputs, current_tick, dt=1.):
        events = original(n, external_inputs, current_tick, dt)
        if n.id in rows:
            rows[n.id].append([p.u_i.info for p in n.postsynaptic_points.values()])
        return events

    PredictiveReceptorNeuron.tick = observed
    try:
        data = record(net, features, groups, bridge, trial)
    finally:
        PredictiveReceptorNeuron.tick = original
    data['start_contrast_weights'] = initial
    data['contrast_weights'] = np.concatenate([np.array(rows[n]) for n in added], axis=1)
    return data


def run(source, output, *, mapping='paired', order=0, seed=11, repeats=2):
    source, output = Path(source).resolve(), Path(output).resolve()
    old = json.loads((source/'manifest.json').read_text())
    original = json.loads((source/'config.json').read_text())
    if len(original['neurons']) != 1152:
        raise ValueError('Expected the original population graph')
    if any(digest(p) != h for p,h in old['source_hashes'].items()):
        raise ValueError('Source runtime changed')
    sensory, contrast, sensory_edges = append_population_contrast(original, old['groups']['vision'])
    context = contrast['contrast_above'] + contrast['contrast_below']
    cfg, bridge, edges, selected = append_predictive_bridge(sensory, context, old['groups']['touch'], seed=seed)
    features = []
    for p,h in sorted(old['physical_sources'].items()):
        if digest(p) != h: raise ValueError('Physical media changed')
        with np.load(p) as z: features.append({k:z[k] for k in z.files})
    if len(features) != 2 or any(int(f['ticks']) != 300 for f in features):
        raise ValueError('Expected two original 300-tick recordings')
    if shutil.disk_usage(output.parent).free < 1024**3:
        raise OSError('Less than 1 GiB free before acquisition')
    trials = protocol(300, repeats, mapping, order)
    hashes = fingerprint()
    for obj in (run, record, append_population_contrast, append_predictive_bridge, fresh, protocol):
        p = Path(inspect.getfile(obj)).resolve(); hashes[str(p)] = digest(p)
    output.mkdir(exist_ok=False)
    cfg_path = output/'config.json'; cfg_path.write_text(encode(cfg)+'\n')
    manifest = dict(source=str(source), source_config_sha256=digest(source/'config.json'),
        base_graph_seed=original['metadata'].get('seed'), bridge_seed=seed, mapping=mapping,
        order=order, repeats=repeats, groups=old['groups'], contrast=contrast, bridge=bridge,
        selected=selected, edges=sensory_edges+edges, fields=FIELDS, trials=trials,
        source_hashes=hashes, physical_sources=old['physical_sources'],
        limitations='1761 cells, one base graph, two physical recordings; a mechanistic screen, '
        'not accepted recall, hierarchy, action, embodiment or consciousness. Context comes from '
        'neural contrast. Same fan-in and birth conductance as the raw bridge, different delays '
        'and amplitudes. All learning and native return pathways remain active.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    net,_,members,_ = fresh(cfg_path, seed, PredictiveReceptorNeuron)
    assert len(members) == 1761
    save_checkpoint(net, output/'initial.neural-checkpoint', sources=(__file__,))
    entries=[]; used=0; started=time.perf_counter()

    def perform(network, trial, name, **tags):
        nonlocal used
        if shutil.disk_usage(output).free < 750*1024**2:
            raise OSError('Free-space reserve reached')
        data = record_contrast(network, features, old['groups'], bridge, contrast, trial)
        residual = audit_record(data, cfg, bridge)
        path = output/(name+'.npz'); np.savez_compressed(path, **data)
        used += path.stat().st_size
        entries.append(dict(file=path.name, sha256=digest(path), trial=trial, audit_residual=residual, **tags))
        progress = dict(entries=entries, bytes=used, seconds=time.perf_counter()-started,
                        ticks=sum(e['trial']['stop']-e['trial']['start'] for e in entries))
        (output/'progress.json').write_text(encode(progress)+'\n')
        print(encode(dict(file=path.name,tick=network.current_tick,bytes=used,residual=residual)),flush=True)
        if used > 320*1024**2: raise OSError('Per-run raw budget reached; records retained')

    for i,trial in enumerate(trials): perform(net,trial,f'experience-{i:02d}',phase=trial['phase'])
    save_checkpoint(net, output/'trained.neural-checkpoint', sources=(__file__,))
    birth={(p['neuron_id'],p['synapse_id']):p['u_i']['info'] for p in cfg['synaptic_points'] if p['type']=='postsynaptic'}
    for state in ('initial','trained','reset_selected'):
        for clip in (0,1,None):
            if state=='reset_selected' and clip is None: continue
            parent='initial' if state=='initial' else 'trained'
            branch=load_checkpoint(output/f'{parent}.neural-checkpoint',trusted=True)
            probe=branch.network
            if state=='reset_selected':
                for nid,sid,_ in selected:
                    probe.network.neurons[nid].postsynaptic_points[sid].u_i.info=birth[nid,sid]
            t=probe.current_tick
            trial=dict(start=t,stop=t+300,visual_clip=clip,audio_clip=None)
            perform(probe,trial,f'probe-{state}-{clip}',phase='probe',state=state,clip=clip)
    if any(digest(p)!=h for p,h in hashes.items()): raise ValueError('Runtime changed during experiment')
    result=json.loads((output/'progress.json').read_text())
    (output/'completion.json').write_text(encode(result)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--mapping',choices=('paired','swapped'),default='paired')
    p.add_argument('--order',type=int,choices=(0,1),default=0)
    p.add_argument('--seed',type=int,default=11);p.add_argument('--repeats',type=int,default=2)
    run(**vars(p.parse_args()))
