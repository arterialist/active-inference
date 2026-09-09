"""Separate acquired input contrast from birth placement and target-level gain.

Uses existing trusted neural checkpoints, without acquisition replay. All
interventions are offline experimental manipulations, never a brain controller.
Every branch retains ongoing positive adaptation and its coupled return paths.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from simulations.active_inference.core.runtime_checkpoint import load_checkpoint
from .association_route_probe import digest
from .composition_probe import encode
from .eligibility_association_probe import dynamic_snapshot
from .eligibility_media_probe import record
from .media_order_audit import load_state
from .media_weight_identity import cycle_weights, set_selected_weights, PathObserver
from .runtime_checkpoint_probe import CheckpointDriver


def growth_factors(ports, initial, learned):
    initial, learned = np.asarray(initial), np.asarray(learned)
    if initial.shape != (len(ports),) or learned.shape != initial.shape:
        raise ValueError('Wrong weight shape')
    if any(not np.isfinite(q).all() or (q < 0).any() or (q > 1).any() for q in (initial,learned)):
        raise ValueError('Invalid initial or learned selected weights')
    factors = {f'birth_cycle{i}': cycle_weights(ports, initial, i)[0] for i in (1,2,3)}
    mean_growth = initial.copy()
    for nid in sorted({n for n,_,_ in ports}):
        indices = [i for i,(n,_,_) in enumerate(ports) if n == nid]
        mean_growth[indices] += (learned[indices]-initial[indices]).mean()
    factors['mean_growth'] = mean_growth
    if any(not np.isfinite(q).all() or (q < 0).any() or (q > 1).any() for q in factors.values()):
        raise ValueError('Intervention lies outside the selected rule; do not silently clip')
    return factors


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    m = json.loads((source/'manifest.json').read_text())
    prior = json.loads((source/'summary.json').read_text())
    course = Path(m['source']); cm = json.loads((course/'manifest.json').read_text())
    checkpoint = next(c for c in prior['checkpoints'] if c['repeats']==16)
    path = source/checkpoint['file']
    if digest(path) != checkpoint['sha256']:
        raise ValueError('Checkpoint changed')
    if shutil.disk_usage(output.parent).free < 1536*1024**2:
        raise OSError('Need 1.5 GiB free; no acquisition data is copied')
    hashes = {str(source/f): digest(source/f) for f in ('manifest.json','summary.json',checkpoint['file'])}
    hashes.update({str(Path(__file__).resolve()): digest(__file__), **m['source_hashes']})
    if any(digest(p) != h for p,h in hashes.items()):
        raise ValueError('Changed sources')
    ports = m['selected_ports']; parent = load_state(course/'checkpoint-16-state.json.gz')
    birth = load_state(course/'initial-state.json.gz')
    initial = np.array([birth['neurons'][str(n)]['synapses'][str(sid)][0] for n,sid,_ in ports])
    learned = np.array([parent['neurons'][str(n)]['synapses'][str(sid)][0] for n,sid,_ in ports])
    factors = growth_factors(ports, initial, learned)
    features = []
    for clip in (0,1):
        p = next(Path(p) for p in cm['physical_sources'] if Path(p).name==f'sensory-{clip}.npz')
        with np.load(p) as z:
            features.append({k:z[k] for k in z.files})
    output.mkdir(parents=True,exist_ok=False)
    manifest = dict(source=str(source), course=str(course), groups=m['groups'], selected_ports=ports,
        mapping=m['mapping'], order=m['order'], seed=m['seed'], source_hashes=hashes,
        conditions=list(factors), cues=[None,0,1], checkpoint=checkpoint,
        limits='Diagnostic weight interventions from the same learned fast state. '
               'Birth cycles preserve birth weights per target; mean growth preserves each target learned sum '
               'and its birth weight differences. No frozen adaptation, stimulus labels or decoded outputs enter neurons.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    np.savez_compressed(output/'weights.npz', initial=initial, learned=learned, **factors)
    rows = []; began = time.perf_counter()
    for condition,q in [('control',learned),*factors.items()]:
        for cue in ([0] if condition=='control' else [None,0,1]):
            if shutil.disk_usage(output).free < 1024**3:
                raise OSError('Less than 1 GiB free; evidence retained')
            restored = load_checkpoint(path,trusted=True); net = restored.network
            if json.loads(dynamic_snapshot(net)) != parent:
                raise ValueError('Checkpoint state differs')
            set_selected_weights(net,ports,q)
            cells = list(net.network.neurons.values()); syns = [s for n in cells for s in n.postsynaptic_points.values()]
            if any(n.params.eta_post <= 0 or n.params.eta_retro <= 0 for n in cells):
                raise ValueError('Frozen adaptation')
            observer = PathObserver(cells,syns,net,ports)
            trial = dict(start=net.current_tick,stop=net.current_tick+300,visual_clip=cue,audio_clip=None)
            d = record(net,CheckpointDriver(restored),cells,syns,features,m['groups'],trial,ports,observer)
            d.update(selected_potential=np.asarray(observer.potentials),terminals=np.asarray(observer.releases))
            d['selected_local_current'] = np.where(d['arrivals']>0,d['selected_potential'],0.)
            if condition=='control':
                with np.load(source/'intact-cue-0.npz') as z:
                    if set(d)!=set(z.files) or any(not np.array_equal(d[k],z[k]) for k in d):
                        raise ValueError('Cached intact control differs')
            else:
                name=f'{condition}-cue-{cue}.npz'; np.savez_compressed(output/name,**d)
                rows.append(dict(condition=condition,cue=cue,trial=trial,file=name,sha256=digest(output/name)))
            a=d['cells'][:,np.array(m['groups']['tactile_core'])-1,1]
            print(encode(dict(condition=condition,cue=cue,events=int((a>0).sum()),seconds=round(time.perf_counter()-began,2))),flush=True)
            del restored,net,cells,syns,observer,d
    if any(digest(p)!=h for p,h in hashes.items()):
        raise ValueError('Source changed during execution')
    result=dict(probes=rows,intact_control_exact=True,acquisition_ticks=0,ticks=3900,
                seconds=time.perf_counter()-began)
    (output/'summary.json').write_text(encode(result)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
