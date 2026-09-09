"""Continue real-media acquisition and test sensory context at fixed checkpoints.

The source is an existing, order-controlled graded PAULA recording. Its entire
acquisition prefix and trained state must replay exactly. Continued acquisition
changes experience duration only. Probe clones remain plastic and cannot alter
the parent. No clip identity, prediction score or observer enters a neuron.

Context probes present 64 ticks of visual input or silence, then 300 ticks of
sound or silence, with vision withdrawn. The full 3x3 physical factorial is
recorded. A delayed sound is a new test condition, not the original simultaneous
training stimulus. A contextual interaction alone is not predictive learning.
"""
import argparse
from copy import deepcopy
import gzip
import inspect
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .association_balance_audit import checked
from .association_route_probe import digest
from .composition_probe import encode, fingerprint
from .eligibility_association_probe import dynamic_snapshot
from .eligibility_media_probe import record
from .eligibility_media_audit import verify_ledger
from .graded_media_audit import verify_rates
from .media_drive_audit import ReceptorAudit, physical_values
from .media_order_control import protocol, ledger_start
from .multimodal_pairing_probe import fresh, WeightObserver
from .population_state_branch import TickDriver
from neuron.extensions.experimental.graded_eligibility import GradedEligibilityNeuron


def probe_segments(start, visual, audio, *, prefix=64, length=300):
    if visual not in (None, 0, 1) or audio not in (None, 0, 1):
        raise ValueError('Unknown physical recording')
    if prefix < 1 or prefix % 4 or length < 1 or length % 4:
        raise ValueError('Durations must preserve the four-phase physical transducer')
    return [dict(start=start, stop=start+prefix, visual_clip=visual, audio_clip=None),
            dict(start=start+prefix, stop=start+prefix+length, visual_clip=None, audio_clip=audio)]


def validate_checkpoints(checkpoints, source_repeats):
    if (not checkpoints or any(type(n) is not int or n < 1 for n in checkpoints)
            or list(checkpoints) != sorted(set(checkpoints)) or checkpoints[0] != source_repeats):
        raise ValueError('Need increasing checkpoints starting at the source exposure count')


def save_state(path, state):
    with gzip.open(path, 'wt') as f:
        f.write(state+'\n')


def run(source, output, checkpoints=(4, 16)):
    source, output = Path(source).resolve(), Path(output).resolve()
    old = json.loads((source/'manifest.json').read_text())
    prior = json.loads((source/'summary.json').read_text())
    validate_checkpoints(checkpoints, old['repeats'])
    if any(digest(p) != h for p, h in old['source_hashes'].items()):
        raise ValueError('Source runtime changed')
    if shutil.disk_usage(output.parent).free < 1500*1024**2:
        raise OSError('Need at least 1.5 GiB free')
    cfg = json.loads((source/'config.json').read_text())
    groups, ports = old['groups'], old['selected_ports']
    features = []
    for clip in (0, 1):
        path = next(Path(p) for p in old['physical_sources'] if Path(p).name == f'sensory-{clip}.npz')
        if digest(path) != old['physical_sources'][str(path)]:
            raise ValueError('Physical source changed')
        with np.load(path) as z:
            features.append({k: z[k] for k in z.files})
    length = len(features[0]['visual'])
    if length != 300 or any(len(f[k]) != length for f in features for k in ('visual', 'auditory')):
        raise ValueError('This continuation uses the declared two 300-tick recordings')
    trials = protocol(length, max(checkpoints), old['mapping'], old['order'])
    if trials[:len(old['trials'])] != old['trials']:
        raise ValueError('Acquisition prefix changed')
    hashes = fingerprint()
    for obj in (run, protocol, ledger_start, record, verify_ledger, verify_rates,
                ReceptorAudit, fresh, TickDriver, dynamic_snapshot):
        p = Path(inspect.getfile(obj)).resolve(); hashes[str(p)] = digest(p)
    sources = {str(source/name): digest(source/name) for name in ('config.json', 'manifest.json', 'summary.json', 'trained-state.json.gz')}
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.json').write_text(encode(cfg)+'\n')
    m = dict(source=str(source), seed=old['seed'], mapping=old['mapping'], order=old['order'],
             groups=groups, selected_ports=ports, physical_sources=old['physical_sources'], source_hashes=hashes,
             source_files=sources, repeats=max(checkpoints), checkpoints=list(checkpoints),
             trials=trials, prefix_ticks=64, sound_ticks=length,
             limits='One graph seed and two clips. Repeated deterministic experience, not independent samples. '
                    'Context probes introduce a 64-tick audiovisual delay; initial training was simultaneous. '
                    'All learning remains active. No observer or clip identity enters the neural network.')
    (output/'manifest.json').write_text(encode(m)+'\n')
    net, core, members, points = fresh(output/'config.json', old['seed'], GradedEligibilityNeuron)
    birth = dynamic_snapshot(net)
    save_state(output/'initial-state.json.gz', birth)
    receptor = ReceptorAudit(cfg, groups, graded_gain=.25)
    start = ledger_start(json.loads(birth), cfg, ports)
    health = WeightObserver(members, points)
    training, probes, states = [], [], []
    residual = 0.; began = time.perf_counter()

    def verify(data, trial, current, sensory):
        nonlocal residual
        result = verify_ledger(data, cfg, ports, *current[:4])
        verify_rates(data['cells'], current[4], cfg, ports)
        sensory.check(data['cells'], physical_values(features, trial))
        if not np.allclose(data['incoming_info_after'][:384:2], sensory.q, atol=2e-12, rtol=0):
            raise ValueError('Receptor weight reconstruction differs')
        residual = max(residual, result[4])
        return (*result[:4], data['cells'][-1, :, 3].copy())

    for i, trial in enumerate(trials):
        data = record(net, core, members, points, features, groups, trial, ports, health)
        start = verify(data, trial, start, receptor)
        if i < len(prior['training']):
            with np.load(checked(source, prior['training'][i])) as z:
                if set(z.files) != set(data) or any(not np.array_equal(z[k], data[k]) for k in z.files):
                    raise ValueError('Original acquisition prefix did not replay exactly')
        path = output/f'experience-{i:03d}.npz'
        np.savez_compressed(path, **data)
        training.append(dict(file=path.name, sha256=digest(path), trial=trial))
        del data
        if (i+1) % 4:
            continue
        repeats = (i+1)//4
        print(encode(dict(stage='acquisition', repeats=repeats, ticks=net.current_tick,
                          seconds=round(time.perf_counter()-began, 2))), flush=True)
        if repeats not in checkpoints:
            continue
        parent = dynamic_snapshot(net)
        if repeats == old['repeats']:
            with gzip.open(source/'trained-state.json.gz', 'rt') as f:
                if json.loads(parent) != json.load(f):
                    raise ValueError('Original trained state did not replay exactly')
        state_name = f'checkpoint-{repeats}-state.json.gz'
        save_state(output/state_name, parent)
        states.append(dict(repeats=repeats, file=state_name, sha256=digest(output/state_name)))
        cases = [('recall', v, None) for v in (0, 1)] + [('context', v, a) for v in (None, 0, 1) for a in (None, 0, 1)]
        for kind, visual, audio in cases:
            if shutil.disk_usage(output).free < 1024**3:
                raise OSError('Free space fell below 1 GiB; stopped without deleting evidence')
            branch = deepcopy(net)
            if dynamic_snapshot(branch) != parent:
                raise ValueError('Probe clone does not preserve complete recorded state')
            cells = list(branch.network.neurons.values())
            syns = [p for n in cells for p in n.postsynaptic_points.values()]
            observer = WeightObserver(cells, syns)
            sensory = deepcopy(receptor)
            current = ledger_start(json.loads(parent), cfg, ports)
            t = branch.current_tick
            segments = probe_segments(t, visual, audio, length=length) if kind == 'context' else [dict(start=t, stop=t+length, visual_clip=visual, audio_clip=None)]
            parts = []
            for segment in segments:
                data = record(branch, TickDriver(branch), cells, syns, features, groups, segment, ports, observer)
                current = verify(data, segment, current, sensory)
                parts.append(data)
            combined = {k: np.concatenate([d[k] for d in parts]) for k in parts[0]
                        if k not in ('incoming_info_before', 'incoming_info_after')}
            combined['incoming_info_before'] = parts[0]['incoming_info_before']
            combined['incoming_info_after'] = parts[-1]['incoming_info_after']
            if repeats == old['repeats'] and kind == 'recall':
                ref = next(p for p in prior['probes'] if p['state'] == 'trained' and p['expression'] == 'graded' and p['sense'] == 'visual' and p['clip'] == visual)
                with np.load(checked(source, ref)) as z:
                    if any(not np.array_equal(z[k], combined[k]) for k in z.files):
                        raise ValueError('Original graded recall did not replay exactly')
            name = f'checkpoint-{repeats}-{kind}-v{visual}-a{audio}.npz'
            path = output/name
            np.savez_compressed(path, **combined)
            probes.append(dict(checkpoint=repeats, kind=kind, visual=visual, audio=audio,
                               segments=segments, file=name, sha256=digest(path), parent_state=state_name))
            if dynamic_snapshot(net) != parent:
                raise ValueError('Probe altered the acquisition parent')
            print(encode(dict(stage='probe', checkpoint=repeats, kind=kind, visual=visual, audio=audio)), flush=True)
            del branch, cells, syns, observer, sensory, parts, data, combined
    if any(digest(p) != h for p, h in {**hashes, **sources, **old['physical_sources']}.items()):
        raise ValueError('Sources changed during execution')
    summary = dict(training=training, probes=probes, states=states, original_acquisition_exact=True,
                   original_trained_state_exact=True, original_graded_recall_exact=True, parent_unchanged=True,
                   max_selected_update_residual=residual, seconds=time.perf_counter()-began,
                   ticks=net.current_tick+sum(s['stop']-s['start'] for p in probes for s in p['segments']))
    (output/'summary.json').write_text(encode(summary)+'\n')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoints', type=int, nargs='+', default=[4, 16])
    a = p.parse_args(); run(a.source, a.output, tuple(a.checkpoints))
