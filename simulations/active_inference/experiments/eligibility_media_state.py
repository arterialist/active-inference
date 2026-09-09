"""Distinguish selected synaptic storage from its coupled neural expression.

Exactly replay a real-media eligibility record, then branch its complete state.
Only declared visual-to-auditory weights are reset or exchanged with the other
experience assignment. Keep all traces, modulation, other weights and learning.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import inspect
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .eligibility_media_probe import record
from .eligibility_association_probe import dynamic_snapshot
from .composition_probe import encode
from .association_route_probe import digest
from .multimodal_pairing_probe import fresh, WeightObserver
from .population_state_branch import TickDriver
from neuron.extensions.experimental.eligibility_trace import EligibilityTraceNeuron


def assignment_values(parent, opposite, ports, condition):
    """Offline causal intervention, never a runtime learning/controller rule.

Assignment controls preserve each target's exact raw-weight multiset. Rotations cover all other
owners once across the three four-input controls. Rank transfer uses opposite
experience only to order the recipient's own weights. Neither preserves the
time-varying input current or attenuation-weighted conductance. Factor controls
instead exchange the target's raw mean or its zero-mean input differences.
These are weight-space factors, not identical current or excitability factors.
"""
    if condition not in ('rotate_1', 'rotate_2', 'rotate_3', 'opposite_rank', 'opposite_mean', 'opposite_residual'):
        raise ValueError(condition)
    by_target = {}
    for nid, sid, _ in ports: by_target.setdefault(nid, []).append(sid)
    values = {}
    for nid, sids in by_target.items():
        if len(sids) != 4 or len(set(sids)) != 4: raise ValueError('This balanced null requires four inputs per target')
        own = np.array([parent['neurons'][str(nid)]['synapses'][str(sid)][0] for sid in sids])
        if condition in ('opposite_mean', 'opposite_residual'):
            donor = np.array([opposite['neurons'][str(nid)]['synapses'][str(sid)][0] for sid in sids])
            moved = own + (donor.mean()-own.mean()) if condition == 'opposite_mean' else donor + (own.mean()-donor.mean())
            # Additive target means and zero-mean input differences form a
            # complete two-factor decomposition. No clipping: it would change
            # the intervention. These selected eligibility ports have cap 1.
            if not np.isfinite(moved).all() or np.any(moved < 0) or np.any(moved > 1):
                raise ValueError('Factor transplant exceeds selected weight bounds')
        elif condition.startswith('rotate_'):
            moved = np.roll(own, int(condition[-1]))
        else:
            donor = [opposite['neurons'][str(nid)]['synapses'][str(sid)][0] for sid in sids]
            moved = np.empty_like(own)
            moved[np.argsort(donor, kind='stable')] = np.sort(own)
        if condition not in ('opposite_mean', 'opposite_residual') and not np.array_equal(np.sort(own), np.sort(moved)):
            raise AssertionError('Weight distribution changed')
        values.update({(nid, sid): float(value) for sid, value in zip(sids, moved)})
    return values


def probe_schedule(mode, clip_ticks, trained_tick):
    """Physical stimuli only. Congruence labels never enter a neuron."""
    if mode == 'exchange':
        conditions = ('unchanged', 'reset_selected', 'opposite_selected')
    elif mode == 'assignment':
        conditions = ('unchanged', 'rotate_1', 'rotate_2', 'rotate_3', 'opposite_rank')
    elif mode == 'congruence':
        conditions = ('initial', 'unchanged', 'reset_selected')
    elif mode == 'factors':
        conditions = ('unchanged', 'opposite_mean', 'opposite_residual', 'opposite_selected')
    else:
        raise ValueError(mode)
    if clip_ticks <= 0 or trained_tick < 0: raise ValueError('Invalid probe time')
    schedule = []
    for condition in conditions:
        for visual in (0, 1):
            for audio in (0, 1) if mode == 'congruence' else (None,):
                start = 0 if condition == 'initial' else trained_tick
                stem = f'{condition}-v{visual}-a{audio}' if mode == 'congruence' else f'{condition}-{visual}'
                schedule.append(dict(condition=condition, stem=stem, trial=dict(start=start, stop=start+clip_ticks,
                                     visual_clip=visual, audio_clip=audio)))
    return schedule


def run(source, opposite, output, mode='exchange'):
    source, opposite, output = map(lambda p: Path(p).resolve(), (source, opposite, output))
    m, other = [json.loads((p/'manifest.json').read_text()) for p in (source, opposite)]
    if (m['condition'] != 'eligibility' or other['condition'] != 'eligibility' or m['seed'] != other['seed']
            or {m['mapping'], other['mapping']} != {'paired', 'swapped'}
            or (source/'config.json').read_bytes() != (opposite/'config.json').read_bytes()
            or m['selected_ports'] != other['selected_ports']): raise ValueError('Unmatched source pair')
    hashes = dict(m['source_hashes']); hashes[str(Path(__file__).resolve())] = digest(__file__)
    hashes[str(Path(inspect.getfile(record)).resolve())] = digest(inspect.getfile(record))
    if any(digest(p) != h for p, h in hashes.items()): raise ValueError('Runtime changed')
    if shutil.disk_usage(output.parent).free < 1000*1024**2: raise OSError('Need 1 GiB free')
    schedule = probe_schedule(mode, m['clip_ticks'], m['trials'][-1]['stop'])
    conditions = list(dict.fromkeys(s['condition'] for s in schedule))
    output.mkdir(parents=True, exist_ok=False)
    cfg = source/'config.json'; config = json.loads(cfg.read_text()); ports = m['selected_ports']
    source_files = {str(p/name): digest(p/name) for p in (source, opposite)
                    for name in ('manifest.json', 'config.json', 'training-final-state.json.gz')}
    if mode == 'congruence': source_files[str(source/'initial-state.json.gz')] = digest(source/'initial-state.json.gz')
    (output/'manifest.json').write_text(encode(dict(source_recording=str(source), opposite_recording=str(opposite),
        source_hashes=hashes, source_files_sha256=source_files, mode=mode, conditions=conditions, probe_schedule=schedule,
        scope='Full trained-state selected-weight interventions; never freeze adaptation. Assignment mode preserves raw per-target weight distributions, not actual currents. Factors mode exchanges per-target raw means and zero-mean input differences separately and together, not intrinsic excitability.'))+'\n')
    features = []
    for clip in (0, 1):
        with np.load(Path(m['source_recording'])/f'sensory-{clip}.npz') as z: features.append({k: z[k] for k in z.files})
    net, core, members, points = fresh(cfg, m['seed'], EligibilityTraceNeuron)
    initial_state = dynamic_snapshot(net)
    if mode == 'congruence':
        with gzip.open(source/'initial-state.json.gz', 'rt') as f:
            if initial_state != encode(json.load(f)): raise AssertionError('Initial state differs')
    health = WeightObserver(members, points); started = time.perf_counter()
    for i, trial in enumerate(m['trials']):
        data = record(net, core, members, points, features, m['groups'], trial, ports, health)
        with np.load(source/f'experience-{i:03d}.npz') as z:
            if any(not np.array_equal(data[k], z[k]) for k in data): raise AssertionError('Training replay differs')
        del data
        if i % 8 == 7: print(encode({'training': i+1, 'seconds': time.perf_counter()-started}), flush=True)
    parent = dynamic_snapshot(net)
    with gzip.open(source/'training-final-state.json.gz', 'rt') as f:
        if parent != encode(json.load(f)): raise AssertionError('Complete trained state differs')
    with gzip.open(opposite/'training-final-state.json.gz', 'rt') as f: other_state = json.load(f)
    initial = {(p['neuron_id'], p['synapse_id']): p['u_i']['info'] for p in config['synaptic_points'] if p['type'] == 'postsynaptic'}
    branches = []
    for spec in schedule:
            condition, trial, stem = spec['condition'], spec['trial'], spec['stem']
            clip = trial['visual_clip']
            if condition == 'initial':
                branch, _, _, _ = fresh(cfg, m['seed'], EligibilityTraceNeuron)
                expected = json.loads(initial_state)
            else:
                branch = deepcopy(net); expected = json.loads(parent)
            assigned = assignment_values(expected, other_state, ports, condition) if condition not in (
                'initial', 'unchanged', 'reset_selected', 'opposite_selected') else None
            if condition not in ('initial', 'unchanged'):
                for nid, sid, _ in ports:
                    value = assigned[nid, sid] if assigned is not None else initial[nid, sid] if condition == 'reset_selected' else other_state['neurons'][str(nid)]['synapses'][str(sid)][0]
                    branch.network.neurons[nid].postsynaptic_points[sid].u_i.info = value
                    expected['neurons'][str(nid)]['synapses'][str(sid)][0] = value
            start = dynamic_snapshot(branch)
            if start != encode(expected): raise AssertionError('Undeclared branch change')
            with gzip.open(output/f'{stem}-start.json.gz', 'wt') as f: f.write(start+'\n')
            cells = list(branch.network.neurons.values()); syns = [p for n in cells for p in n.postsynaptic_points.values()]
            if branch.current_tick != trial['start']: raise AssertionError('Branch clock differs')
            data = record(branch, TickDriver(branch), cells, syns, features, m['groups'], trial, ports)
            if condition == 'unchanged' and mode != 'congruence':
                with np.load(source/f'probe-continuation-visual-{clip}.npz') as z:
                    if any(not np.array_equal(data[k], z[k]) for k in data): raise AssertionError('Control probe differs')
            np.savez_compressed(output/f'{stem}.npz', **data)
            if dynamic_snapshot(net) != parent: raise AssertionError('Parent mutated')
            branches.append(dict(condition=condition, clip=clip, file=f'{stem}.npz', start_file=f'{stem}-start.json.gz',
                                 trial=trial, sha256=digest(output/f'{stem}.npz'), start_sha256=digest(output/f'{stem}-start.json.gz')))
            print(encode({'branch': stem, 'seconds': time.perf_counter()-started}), flush=True)
            del branch, cells, syns, data
    if any(digest(p) != h for p, h in {**hashes, **source_files}.items()): raise ValueError('Sources changed')
    result = dict(seed=m['seed'], mapping=m['mapping'], full_training_replay_exact=True,
                  unchanged_probe_replay_exact=True if mode != 'congruence' else None,
                  branches=branches, mode=mode, ticks=m['trials'][-1]['stop']+len(schedule)*m['clip_ticks'], seconds=time.perf_counter()-started)
    (output/'summary.json').write_text(encode(result)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--opposite', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('exchange', 'assignment', 'congruence', 'factors'), default='exchange')
    a = p.parse_args(); run(a.source, a.opposite, a.output, a.mode)
