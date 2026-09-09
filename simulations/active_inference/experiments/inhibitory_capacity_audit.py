"""Data-only structural, update-chain and response audit of capacity trials.

No acceptance score is computed. Export all tick rasters/credit totals and
fixed-reference trajectories. Silence, changed activity and a positive readout
direction are not, by themselves, evidence of learned association.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .association_credit_audit import verify_events
from .multimodal_pairing_audit import readout, contrast, weight_health
from .population_state_audit import KINDS, tick_projection, reference_comparison


def check_capacity(config, original, targets, condition):
    """Independent current-conservation and exact permitted-change check."""
    restored = deepcopy(config)
    ns = {n['id']: n for n in original['neurons']}
    old = {(p['neuron_id'], p['synapse_id']): p for p in original['synaptic_points'] if p['type'] == 'postsynaptic'}
    new = {(p['neuron_id'], p['synapse_id']): p for p in config['synaptic_points'] if p['type'] == 'postsynaptic'}
    terms = {(p['neuron_id'], p['terminal_id']): p for p in original['synaptic_points'] if p['type'] == 'presynaptic'}
    targets = set(targets)
    changed = 0
    for p in restored['synaptic_points']:
        if p['type'] == 'postsynaptic' and p['neuron_id'] in targets:
            before = old[p['neuron_id'], p['synapse_id']]['u_i']['info']
            if before < 0 and condition == 'balanced':
                if not p['u_i']['info'] < 0:
                    raise ValueError('Inhibitory sign lost')
                changed += p['u_i']['info'] != before
                p['u_i']['info'] = before
    if restored != original or condition not in ('balanced', 'control'):
        raise ValueError('Unexpected config change')
    currents = {t: [0., 0.] for t in targets}
    for c in original['connections']:
        t, sid = c['target_neuron'], c['target_synapse']
        if t not in targets:
            continue
        p = new[t, sid]
        gain = terms[c['source_neuron'], c['source_terminal']]['u_o']['info'] / ns[c['source_neuron']]['params']['c']
        current = gain * (p['u_i']['info']+p['u_i']['plast']) * ns[t]['params']['delta_decay'] ** p['distance_to_hillock']
        currents[t][0 if current >= 0 else 1] += current
    if condition == 'balanced' and any(a <= 0 or b >= 0 or abs(a+b) > 1e-12 for a, b in currents.values()):
        raise ValueError('Initial current capacities do not balance')
    return changed


def audit(recording, output):
    root, output = Path(recording).resolve(), Path(output).resolve()
    m = json.loads((root/'manifest.json').read_text())
    s = json.loads((root/'summary.json').read_text())
    source = Path(m['source_recording'])
    source_manifest = json.loads((source/'manifest.json').read_text())
    if any(m[key] != source_manifest[key] for key in ('seed', 'mapping', 'groups', 'edges', 'trials', 'fields', 'clip_ticks')):
        raise ValueError('Source protocol, indexing or topology changed')
    for name, expected in m['source_files_sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Source input changed')
    cfg = json.loads((root/'config.json').read_text())
    original = json.loads((source/'config.json').read_text())
    changed = check_capacity(cfg, original, m['groups']['upper_core'], m['condition'])
    if any(n['params']['eta_post'] <= 0 or n['params']['eta_retro'] <= 0 for n in cfg['neurons']):
        raise ValueError('Frozen learning is not this experiment')
    health = weight_health(root, m)
    if s['seed'] != m['seed'] or s['mapping'] != m['mapping'] or s['condition'] != m['condition']:
        raise ValueError('Record metadata differs')
    ns = {n['id']: n for n in cfg['neurons']}
    if list(ns) != list(range(1, len(ns)+1)):
        raise ValueError('Cell array indexing is unspecified')
    points = {(p['neuron_id'], p['synapse_id']): p for p in cfg['synaptic_points'] if p['type'] == 'postsynaptic'}
    global_order = [(n, p['synapse_id']) for n in ns for p in cfg['synaptic_points'] if p['type'] == 'postsynaptic' and p['neuron_id'] == n]
    lookup = {pair: i for i, pair in enumerate(global_order)}
    ports = m['credit_ports']
    auditory, upper = set(m['groups']['tactile_core']), set(m['groups']['upper_core'])
    expected = {(t, i, src, f) for src, t, i, f, present in m['edges'] if present and
                ((t in auditory and f in ('crossmodal', 'descending')) or (t in upper and f == 'ascending'))}
    if set(map(tuple, ports)) != expected or len(ports) != len(expected):
        raise ValueError('Credit port set differs')
    weights = np.array([points[n, i]['u_i']['info'] for n, i, _, _ in ports])
    take = [lookup[n, i] for n, i, _, _ in ports]
    eta = np.array([ns[n]['params']['eta_post'] for n, *_ in ports])
    cap = np.array([ns[n]['metadata'].get('plasticity_magnitude_cap', 10.) for n, *_ in ports])
    decay = np.array([ns[n]['metadata'].get('plasticity_magnitude_decay', .02) for n, *_ in ports])
    last = np.full(len(ns), -1, dtype=np.int32)
    families = sorted({p[3] for p in ports})
    family = np.array([families.index(p[3]) for p in ports])
    floor = np.array([2*ns[n]['params']['c'] for n, *_ in ports])
    fields = ('events', 'positive_direction', 'negative_direction', 'weight_increases', 'weight_decreases', 'delta',
              'positive_beyond_minimum_window')
    tick_credit = np.zeros((m['trials'][-1]['stop'], len(families), len(fields)))
    tick_spikes = np.zeros((len(tick_credit), len(ns)), dtype=bool)
    max_residual = 0.
    if len(s['episodes']) != len(m['trials']):
        raise ValueError('Missing trial')
    for i, (e, trial) in enumerate(zip(s['episodes'], m['trials'])):
        if e['episode'] != i:
            raise ValueError('Episode order differs')
        for name, digest in ((e['file'], e['sha256']), (e['credit_file'], e['credit_sha256'])):
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
                raise ValueError('Artifact changed')
        with np.load(root/e['file']) as raw, np.load(root/e['credit_file']) as ledger:
            cells, events = raw['cells'], ledger['events']
            if cells.shape != (trial['stop']-trial['start'], len(ns), 8) or not np.isfinite(cells).all() or len(events) != e['events']:
                raise ValueError('Incomplete cellular or credit record')
            if not np.array_equal(weights, raw['incoming_info_before'][take]):
                raise ValueError('Initial endpoint differs')
            weights, residual, _ = verify_events(events, ports, trial, cells, last, weights, eta, cap, decay)
            if not np.array_equal(weights, raw['incoming_info_after'][take]):
                raise ValueError('Update chain does not reconstruct weights')
            max_residual = max(max_residual, residual)
        tick_spikes[trial['start']:trial['stop']] = cells[:, :, 1] > 0
        delta = events['after']-events['before']
        values = (np.ones(len(events)), events['direction'] > 0, events['direction'] < 0, delta > 0, delta < 0, delta,
                  (events['direction'] > 0) & (events['age'] > floor[events['port']]))
        for k, v in enumerate(values):
            np.add.at(tick_credit[:, :, k], (events['tick'], family[events['port']]), v)
    with np.load(root/'parameters.npz') as z:
        initial, learned = z['initial_info'], z['learned_info']
    if not np.array_equal(initial, [points[p]['u_i']['info'] for p in global_order]) or not np.array_equal(weights, learned[take]):
        raise ValueError('Saved parameters differ')
    probes = {}
    required = {(c, sense, clip) for c in ('initial', 'continuation') for sense in ('visual', 'audio') for clip in (0, 1)}
    for p in s['probes']:
        key = p['condition'], p['sense'], p['clip']
        if key not in required or key in probes or not p['parent_unchanged']:
            raise ValueError('Undeclared/duplicate probe or changed parent')
        if p['start_tick'] != (0 if key[0] == 'initial' else m['trials'][-1]['stop']):
            raise ValueError('Probe start tick differs')
        if hashlib.sha256((root/p['file']).read_bytes()).hexdigest() != p['sha256']:
            raise ValueError('Probe artifact changed')
        with np.load(root/p['file']) as z:
            cells = z['cells']
            if cells.shape != (m['clip_ticks'], len(ns), 8) or not np.isfinite(cells).all():
                raise ValueError('Invalid probe cells')
            if not np.array_equal(z['incoming_info_before'], initial if key[0] == 'initial' else learned):
                raise ValueError('Probe initial weights differ')
            probes[key] = cells
    if set(probes) != required:
        raise ValueError('Missing probes')
    readouts, trajectories, counts = {}, {}, {}
    for role in ('tactile_core', 'upper_core'):
        ids = m['groups'][role]
        counts[role] = {f'{state}/{sense}/{clip}': int((v[:, np.array(ids)-1, 1] > 0).sum()) for (state, sense, clip), v in probes.items()}
        readouts[role] = {}
        for kind in KINDS:
            value = lambda state, sense: np.stack([readout(probes[state, sense, clip], ids, kind) for clip in (0, 1)])
            ref = value('initial', 'audio')
            before, after = contrast(value('initial', 'visual'), ref), contrast(value('continuation', 'visual'), ref)
            readouts[role][kind] = {'before': before, 'after': after, 'change': after-before if before is not None and after is not None else None,
                                    **reference_comparison(value('continuation', 'visual'), ref, value('continuation', 'audio'))}
            if kind == 'population_centered_rate':
                for state in ('initial', 'continuation'):
                    curve = tick_projection([probes[state, 'visual', clip] for clip in (0, 1)], ids, ref)
                    if curve is not None:
                        trajectories[f'{role}/{state}'] = curve
                curve = tick_projection([probes['continuation', 'visual', clip] for clip in (0, 1)], ids, value('continuation', 'audio'))
                if curve is not None:
                    trajectories[f'{role}/continuation_current_reference'] = curve
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'ticks.npz', credit=tick_credit, fields=fields, families=families, spikes=tick_spikes, **trajectories)
    result = {'structurally_valid': True, 'recording': str(root), 'seed': m['seed'], 'mapping': m['mapping'],
              'condition': m['condition'], 'changed_initial_weights': changed, 'max_update_residual': max_residual,
              'events': sum(e['events'] for e in s['episodes']), 'counts': counts, 'readouts': readouts,
              'weight_health': health,
              'family_totals': {f: dict(zip(fields, tick_credit[:, j].sum(axis=0).tolist())) for j, f in enumerate(families)},
              'auditor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limits': 'Two clips and two seeds are not acceptance. Initial references are condition-specific. No neural recall consumer or unpaired training control yet. Update chains cover selected incoming routes, not all possible zero-effect arrivals. Capacity conservation is initial mean-current arithmetic, not actual runtime E/I balance.'}
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    audit(args.recording, args.output)
