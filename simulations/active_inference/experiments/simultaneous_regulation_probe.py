"""Same physical audiovisual inputs under crossed learned histories.

All nine combinations of two visual clips/silence and two sounds/silence are
represented. Six new branches per checkpoint supplement three existing exact
visual/silence controls. A repeated visual control must reproduce every field.
Training and probes are simultaneous, unlike earlier delayed-context probes.
No input labels, familiarity values or observer signals enter the network.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from neuron.neuron import setup_neuron_logger
from ..core.runtime_checkpoint import load_checkpoint
from .association_route_probe import digest
from .composition_probe import encode
from .eligibility_association_probe import dynamic_snapshot
from .eligibility_media_probe import record
from .media_order_audit import load_state
from .media_weight_identity import PathObserver
from .runtime_checkpoint_probe import CheckpointDriver


def run(source, output):
    setup_neuron_logger('CRITICAL')
    source, output = Path(source).resolve(), Path(output).resolve()
    m = json.loads((source/'manifest.json').read_text())
    prior = json.loads((source/'summary.json').read_text())
    course = Path(m['source'])
    cm = json.loads((course/'manifest.json').read_text())
    checkpoint = next(c for c in prior['checkpoints'] if c['repeats'] == 16)
    path = source/checkpoint['file']
    if digest(path) != checkpoint['sha256']:
        raise ValueError('Changed checkpoint')
    hashes = {**m['source_hashes'], **cm['physical_sources'],
              **{str(source/p): digest(source/p) for p in ('manifest.json', 'summary.json', checkpoint['file'])},
              str(Path(__file__).resolve()): digest(__file__)}
    if any(digest(p) != h for p, h in hashes.items()):
        raise ValueError('Changed sources')
    if shutil.disk_usage(output.parent).free < 1280*1024**2:
        raise OSError('Need 1.25 GiB free before this bounded probe')
    parent = load_state(course/'checkpoint-16-state.json.gz')
    features = []
    for clip in (0, 1):
        p = next(Path(p) for p in cm['physical_sources'] if Path(p).name == f'sensory-{clip}.npz')
        with np.load(p) as z:
            features.append({k: z[k] for k in z.files})
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(source=str(source), course=str(course), groups=m['groups'],
                    selected_ports=m['selected_ports'], mapping=m['mapping'], order=m['order'],
                    seed=m['seed'], source_hashes=hashes, checkpoint=checkpoint,
                    cases=[[v, a] for v in (None, 0, 1) for a in (None, 0, 1)],
                    limits='One graph seed. Identical physical pair across learned histories. '
                           'Simultaneous 300-tick stimulation, no decoded familiarity input. '
                           'Both native and selected learning remain active.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    began = time.perf_counter()
    rows = []
    for v in (None, 0, 1):
        p = next(p for p in prior['probes'] if p['condition'] == 'intact' and p['cue'] == v)
        if digest(source/p['file']) != p['sha256']:
            raise ValueError('Changed reusable control')
        rows.append(dict(visual=v, audio=None, trial=p['trial'], file=str(source/p['file']),
                         sha256=p['sha256'], reused=True))
    for visual, audio in [(0, None), *[(v, a) for v in (None, 0, 1) for a in (0, 1)]]:
        if shutil.disk_usage(output).free < 1024**3:
            raise OSError('Less than 1 GiB free; stopping without deleting evidence')
        branch = load_checkpoint(path, trusted=True)
        net = branch.network
        if json.loads(dynamic_snapshot(net)) != parent:
            raise ValueError('Restored state differs')
        cells = list(net.network.neurons.values())
        syns = [p for n in cells for p in n.postsynaptic_points.values()]
        if any(n.params.eta_post <= 0 or n.params.eta_retro <= 0 for n in cells):
            raise ValueError('Frozen adaptation')
        observer = PathObserver(cells, syns, net, m['selected_ports'])
        trial = dict(start=net.current_tick, stop=net.current_tick+300,
                     visual_clip=visual, audio_clip=audio)
        d = record(net, CheckpointDriver(branch), cells, syns, features, m['groups'],
                   trial, m['selected_ports'], observer)
        d.update(selected_potential=np.asarray(observer.potentials), terminals=np.asarray(observer.releases))
        d['selected_local_current'] = np.where(d['arrivals'] > 0, d['selected_potential'], 0.)
        if audio is None:
            with np.load(source/'intact-cue-0.npz') as z:
                if set(d) != set(z.files) or any(not np.array_equal(d[k], z[k]) for k in d):
                    raise ValueError('Intact control differs')
        else:
            name = f'visual-{visual}-audio-{audio}.npz'
            np.savez_compressed(output/name, **d)
            rows.append(dict(visual=visual, audio=audio, trial=trial, file=name,
                             sha256=digest(output/name), reused=False))
        print(encode(dict(visual=visual, audio=audio, seconds=round(time.perf_counter()-began, 2))), flush=True)
        del branch, net, cells, syns, observer, d
    if any(digest(p) != h for p, h in hashes.items()):
        raise ValueError('Sources changed during execution')
    result = dict(probes=rows, intact_control_exact=True, acquisition_ticks=0,
                  new_probe_ticks=1800, validation_ticks=300, seconds=time.perf_counter()-began)
    (output/'summary.json').write_text(encode(result)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    run(a.source, a.output)
