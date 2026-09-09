"""Transfer the controlled eligibility rule to the unchanged real-media graph.

Only visual-core to auditory-core incoming learning changes. All other native
learning, recurrent wiring, delays, modulation and physical stimuli remain.
Empty-port control must reproduce the original full training record exactly.
This is a transfer screen, not an association or embodiment acceptance test.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import gzip
import inspect
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .association_route_probe import digest
from .composition_probe import encode, fingerprint, snapshot, network_module
from .eligibility_association_probe import dynamic_snapshot
from .multimodal_pairing_probe import fresh, episode, WeightObserver
from .population_hierarchy import weight_values
from .population_state_branch import TickDriver
from neuron.extensions.experimental.eligibility_trace import EligibilityTraceNeuron


def configure(original, manifest, condition):
    if condition not in ('eligibility', 'control'):
        raise ValueError(condition)
    visual, audio = set(manifest['groups']['visual_core']), set(manifest['groups']['tactile_core'])
    ports = sorted((t, sid, s) for s, t, sid, family, present in manifest['edges']
                   if present and family == 'crossmodal' and s in visual and t in audio)
    if not ports or len({(n, s) for n, s, _ in ports}) != len(ports):
        raise ValueError('Missing or duplicate selected ports')
    cfg = deepcopy(original)
    for n in cfg['neurons']:
        selected = [sid for nid, sid, _ in ports if nid == n['id']]
        if selected:
            n['metadata'].update(eligibility_ports=selected if condition == 'eligibility' else [],
                eligibility_tau_pre=4., eligibility_tau_post=4., eligibility_alpha=1., eligibility_cap=1.)
    return cfg, ports


class EligibilityRecorder:
    """Bounded, read-only full-tick ledger of arbitrary declared input ports."""
    def __init__(self, net, ports, ticks):
        self.net, self.ports, self.count = net, ports, 0
        self.data = {k: np.zeros((ticks, len(ports))) for k in ('weights', 'pre', 'post', 'arrivals', 'eta')}
        self.by_neuron = {}
        for index, (nid, sid, _) in enumerate(ports):
            self.by_neuron.setdefault(nid, []).append((index, sid))

    @contextmanager
    def observe(self):
        original = EligibilityTraceNeuron.tick
        recorder = self
        def tick(n, external_inputs, current_tick, dt=1.):
            selected = recorder.by_neuron.get(n.id)
            if not selected or recorder.net.network.neurons[n.id] is not n:
                return original(n, external_inputs, current_tick, dt)
            if recorder.count >= len(recorder.data['weights']):
                raise ValueError('Ledger capacity exceeded')
            indices, sids = map(list, zip(*selected))
            row = recorder.count
            recorder.data['arrivals'][row, indices] = n.input_buffer[sids, 0]
            recorder.data['eta'][row, indices] = n.params.eta_post*n.rate_multiplier()
            result = original(n, external_inputs, current_tick, dt)
            recorder.data['weights'][row, indices] = [n.postsynaptic_points[s].u_i.info for s in sids]
            if n.eligibility_ports:
                recorder.data['pre'][row, indices] = [n.eligibility_pre[n.eligibility_ports.index(s)] for s in sids]
            recorder.data['post'][row, indices] = n.eligibility_post
            return result
        EligibilityTraceNeuron.tick = tick
        try:
            yield self
        finally:
            EligibilityTraceNeuron.tick = original

    def __call__(self):
        self.count += 1

    def finish(self):
        if self.count != len(self.data['weights']) or any(not np.isfinite(v).all() for v in self.data.values()):
            raise ValueError('Incomplete or nonfinite ledger')
        return self.data


def record(net, core, neurons, syns, features, groups, trial, ports, health=None):
    health = health or WeightObserver(neurons, syns)
    health.rows = []
    before = weight_values(syns)
    ledger = EligibilityRecorder(net, ports, trial['stop']-trial['start'])
    def observe():
        ledger()
        health()
    with ledger.observe():
        cells = episode(net, core, neurons, features, groups, trial, observe)
    return dict(cells=cells, incoming_info_before=before, incoming_info_after=weight_values(syns),
                weight_health=np.asarray(health.rows), **ledger.finish())


def run(source, output, condition='eligibility'):
    source, output = Path(source).resolve(), Path(output).resolve()
    m = json.loads((source/'manifest.json').read_text())
    if m.get('weight_dynamics') != 'bounded' or m.get('architecture') != 'regional':
        raise ValueError('Requires bounded regional source')
    if network_module.MIN_CONNECTION_SIGNAL_TRAVEL_TICKS != 1 or network_module.MAX_CONNECTION_SIGNAL_TRAVEL_TICKS != 1:
        raise ValueError('Requires deterministic one-tick cleft delays')
    if shutil.disk_usage(output.parent).free < 1500*1024**2:
        raise OSError('Need 1.5 GiB free')
    hashes = dict(m['source_hashes'])
    if any(digest(p) != h for p, h in hashes.items()):
        raise ValueError('Original runtime sources changed')
    hashes.update(fingerprint())
    for name in (__file__, inspect.getfile(dynamic_snapshot), inspect.getfile(TickDriver), inspect.getfile(episode)):
        p = Path(name)
        hashes[str(p.resolve())] = digest(p)
    cfg, ports = configure(json.loads((source/'config.json').read_text()), m, condition)
    output.mkdir(parents=True, exist_ok=False)
    path = output/'config.json'; path.write_text(encode(cfg)+'\n')
    manifest = {**m, 'condition': condition, 'selected_ports': ports, 'source_recording': str(source),
        'source_hashes': hashes, 'source_files_sha256': {name: digest(source/name) for name in
        ('manifest.json', 'config.json', 'sensory-0.npz', 'sensory-1.npz', 'training-final-state.json.gz')},
        'scope': 'Only selected incoming learning changes. No added consumer, no semantic claim. Two-seed transfer screen is not acceptance.',
        'health_caveat': 'Inherited bounded counters include provisional updates subsequently replaced on selected ports.'}
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    features = []
    for clip in (0, 1):
        with np.load(source/f'sensory-{clip}.npz') as z:
            features.append({key: z[key] for key in z.files})
    net, core, neurons, syns = fresh(path, m['seed'], EligibilityTraceNeuron)
    with gzip.open(output/'initial-state.json.gz', 'wt') as f: f.write(dynamic_snapshot(net)+'\n')
    initial, health = weight_values(syns), WeightObserver(neurons, syns)
    started, episodes = time.perf_counter(), []
    for i, trial in enumerate(m['trials']):
        data = record(net, core, neurons, syns, features, m['groups'], trial, ports, health)
        if condition == 'control':
            with np.load(source/f'experience-{i:03d}.npz') as old:
                if any(not np.array_equal(data[k], old[k]) for k in ('cells', 'incoming_info_before', 'incoming_info_after', 'weight_health')):
                    raise AssertionError('Unchanged control differs from original')
        name = f'experience-{i:03d}.npz'
        np.savez_compressed(output/name, **data)
        episodes.append({'episode': i, 'file': name, 'sha256': digest(output/name)})
        print(encode({'episode': i, 'seconds': round(time.perf_counter()-started, 2)}), flush=True)
        del data
    np.savez_compressed(output/'parameters.npz', initial_info=initial, learned_info=weight_values(syns))
    parent = dynamic_snapshot(net)
    if condition == 'control':
        with gzip.open(source/'training-final-state.json.gz', 'rt') as f:
            if encode(snapshot(net)) != encode(json.load(f)): raise AssertionError('Control full state differs')
    with gzip.open(output/'training-final-state.json.gz', 'wt') as f: f.write(parent+'\n')
    probes = []
    for state in ('initial', 'continuation'):
        for sense in ('visual', 'audio'):
            for clip in (0, 1):
                if state == 'initial':
                    branch, _, members, points = fresh(path, m['seed'], EligibilityTraceNeuron)
                else:
                    branch = deepcopy(net)
                    if dynamic_snapshot(branch) != parent: raise AssertionError('Full-state branch differs')
                    members = list(branch.network.neurons.values())
                    points = [p for n in members for p in n.postsynaptic_points.values()]
                t = branch.current_tick
                trial = {'start': t, 'stop': t+m['clip_ticks'], 'visual_clip': clip if sense == 'visual' else None,
                         'audio_clip': clip if sense == 'audio' else None}
                data = record(branch, TickDriver(branch), members, points, features, m['groups'], trial, ports)
                name = f'probe-{state}-{sense}-{clip}.npz'
                np.savez_compressed(output/name, **data)
                if dynamic_snapshot(net) != parent: raise AssertionError('Probe changed parent')
                probes.append({'condition': state, 'sense': sense, 'clip': clip, 'file': name,
                               'start_tick': t, 'sha256': digest(output/name), 'parent_unchanged': True})
                print(encode({'probe': name, 'seconds': round(time.perf_counter()-started, 2)}), flush=True)
                del branch, members, points, data
    if any(digest(p) != h for p, h in hashes.items()): raise ValueError('Runtime changed during run')
    summary = {'seed': m['seed'], 'mapping': m['mapping'], 'condition': condition, 'episodes': episodes, 'probes': probes,
        'control_replay_exact': True if condition == 'control' else None,
        'ticks': m['trials'][-1]['stop']+8*m['clip_ticks'], 'seconds': time.perf_counter()-started}
    (output/'summary.json').write_text(encode(summary)+'\n')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--condition', choices=('eligibility', 'control'), default='eligibility')
    a = p.parse_args()
    run(a.source, a.output, a.condition)
