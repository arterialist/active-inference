"""Independent full-tick eligibility audit and unselected response comparisons.

Data only: no neuron implementation, simulator, fitted decoder or acceptance
score. Reconstruct local traces and learning from actual receptor arrivals and
somatic spikes, checking the graph's source spikes at the one-tick cleft delay.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .multimodal_pairing_audit import readout, contrast, weight_health, sensory_marginals
from .population_state_audit import KINDS, reference_comparison, tick_projection


def check_config(cfg, original, manifest):
    restored = deepcopy(cfg)
    visual, audio = set(manifest['groups']['visual_core']), set(manifest['groups']['tactile_core'])
    expected = sorted((t, sid, s) for s, t, sid, family, present in manifest['edges']
                      if present and family == 'crossmodal' and s in visual and t in audio)
    ports = list(map(tuple, manifest['selected_ports']))
    if ports != expected or not ports or manifest['condition'] not in ('control', 'eligibility'):
        raise ValueError('Selected port declaration differs')
    for n in restored['neurons']:
        selected = [sid for nid, sid, _ in ports if nid == n['id']]
        if selected:
            allowed = dict(eligibility_ports=selected if manifest['condition'] == 'eligibility' else [],
                           eligibility_tau_pre=4., eligibility_tau_post=4., eligibility_alpha=1., eligibility_cap=1.)
            for key, value in allowed.items():
                if n['metadata'].pop(key, None) != value: raise ValueError('Eligibility parameters differ')
    if restored != original: raise ValueError('Undeclared config change')
    connections = {(c['target_neuron'], c['target_synapse']): c['source_neuron'] for c in cfg['connections']}
    if any(connections[n, sid] != src for n, sid, src in ports): raise ValueError('Graph and selected ports differ')
    if any(n['params']['eta_post'] <= 0 or n['params']['eta_retro'] <= 0 for n in cfg['neurons']):
        raise ValueError('Frozen plasticity')
    return ports


def verify_ledger(data, cfg, ports, initial_weights, initial_pre, initial_post, previous_spikes, enabled=True):
    ns = {n['id']: n for n in cfg['neurons']}
    ids = {nid: i for i, nid in enumerate(ns)}
    target = np.array([ids[n] for n, _, _ in ports]); source = np.array([ids[s] for _, _, s in ports])
    basal = np.array([ns[n]['params']['eta_post'] for n, _, _ in ports])
    cells = data['cells']; ticks = len(cells)
    if cells.shape != (ticks, len(ns), 8) or not np.isfinite(cells).all(): raise ValueError('Malformed cells')
    if any(data[k].shape != (ticks, len(ports)) or not np.isfinite(data[k]).all()
           for k in ('weights', 'pre', 'post', 'arrivals', 'eta')): raise ValueError('Malformed ledger')
    spikes = cells[:, :, 1] > 0
    arriving = np.concatenate((previous_spikes[None, source], spikes[:-1, source]), axis=0)
    if not np.array_equal(arriving, data['arrivals'] > 0): raise ValueError('Source arrival masks differ')
    if not np.array_equal(cells[:, target, 7]*basal, data['eta']): raise ValueError('Local rate ledger differs')
    q, pre, post = [np.array(v, dtype=float, copy=True) for v in (initial_weights, initial_pre, initial_post)]
    changes = np.zeros((ticks, 4)); residual = 0.
    for t in range(ticks):
        if enabled:
            x, y = pre*np.exp(-.25), post*np.exp(-.25)
            plus, minus = spikes[t, target]*x, arriving[t]*y
            total = plus+minus
            equilibrium = np.divide(plus, total, out=np.zeros_like(q), where=total > 0)
            decay = np.exp(-data['eta'][t]*total)
            expected = q*decay+equilibrium*(1-decay)
            residual = max(residual, float(abs(expected-data['weights'][t]).max()))
            pre, post = x+arriving[t], y+spikes[t, target]
            residual = max(residual, float(abs(pre-data['pre'][t]).max()), float(abs(post-data['post'][t]).max()))
            if residual > 2e-12: raise ValueError('Local eligibility equation differs')
        elif np.any(data['pre'][t]) or np.any(data['post'][t]):
            raise ValueError('Empty-port control has eligibility state')
        delta = data['weights'][t]-q
        changes[t] = (np.count_nonzero(delta > 0), np.count_nonzero(delta < 0), delta.sum(), np.sum(abs(delta)))
        q = data['weights'][t].copy()
    return q, pre, post, spikes[-1], residual, changes


def audit(recording, output):
    root, output = Path(recording).resolve(), Path(output).resolve()
    m = json.loads((root/'manifest.json').read_text()); s = json.loads((root/'summary.json').read_text())
    source = Path(m['source_recording']); old = json.loads((source/'manifest.json').read_text())
    if any(m[k] != old[k] for k in ('seed', 'mapping', 'groups', 'edges', 'trials', 'fields', 'clip_ticks', 'media')):
        raise ValueError('Protocol or sensory sources changed')
    if any(hashlib.sha256((source/name).read_bytes()).hexdigest() != h for name, h in m['source_files_sha256'].items()):
        raise ValueError('Source record changed')
    cfg = json.loads((root/'config.json').read_text())
    ports = check_config(cfg, json.loads((source/'config.json').read_text()), m)
    health = weight_health(root, m)
    order = [(n['id'], p['synapse_id']) for n in cfg['neurons'] for p in cfg['synaptic_points']
             if p['type'] == 'postsynaptic' and p['neuron_id'] == n['id']]
    lookup = {pair: i for i, pair in enumerate(order)}; take = [lookup[n, sid] for n, sid, _ in ports]
    with np.load(root/'parameters.npz') as z: initial, learned = z['initial_info'], z['learned_info']
    points = {(p['neuron_id'], p['synapse_id']): p for p in cfg['synaptic_points'] if p['type'] == 'postsynaptic'}
    if not np.array_equal(initial, [points[key]['u_i']['info'] for key in order]): raise ValueError('Initial weights differ')
    q = initial[take]; pre, post = np.zeros(len(ports)), np.zeros(len(ports))
    previous = np.zeros(len(cfg['neurons']), bool)
    traces, updates, residual = [], [], 0.
    if len(s['episodes']) != len(m['trials']): raise ValueError('Missing trial')
    for i, (e, trial) in enumerate(zip(s['episodes'], m['trials'])):
        if e['episode'] != i or hashlib.sha256((root/e['file']).read_bytes()).hexdigest() != e['sha256']:
            raise ValueError('Episode differs')
        with np.load(root/e['file']) as z:
            if len(z['cells']) != trial['stop']-trial['start'] or not np.array_equal(q, z['incoming_info_before'][take]):
                raise ValueError('Episode boundary differs')
            q, pre, post, previous, error, change = verify_ledger(z, cfg, ports, q, pre, post, previous, m['condition'] == 'eligibility')
            if not np.array_equal(q, z['incoming_info_after'][take]): raise ValueError('Selected endpoint differs')
            if m['condition'] == 'control':
                with np.load(source/f'experience-{i:03d}.npz') as original:
                    if any(not np.array_equal(z[k], original[k]) for k in ('cells', 'incoming_info_before', 'incoming_info_after', 'weight_health')):
                        raise ValueError('Control differs from original')
            traces.append(z['cells'][:, :, 1] > 0); updates.append(change); residual = max(residual, error)
    if not np.array_equal(q, learned[take]): raise ValueError('Learned endpoint differs')
    with gzip.open(root/'training-final-state.json.gz', 'rt') as f: final = json.load(f)
    for i, (nid, sid, _) in enumerate(ports):
        entry = final['eligibility'][str(nid)]
        n = next(n for n in cfg['neurons'] if n['id'] == nid)
        if final['neurons'][str(nid)]['synapses'][str(sid)][0] != q[i]: raise ValueError('Snapshot weight differs')
        if m['condition'] == 'eligibility':
            if (entry['pre'][n['metadata']['eligibility_ports'].index(sid)] != pre[i] or entry['post'] != post[i]
                    or entry['last_tick'] != m['trials'][-1]['stop']-1): raise ValueError('Snapshot trace differs')
    probes = {}; probe_updates = {}; branch_start = (q.copy(), pre.copy(), post.copy(), previous.copy())
    required = {(state, sense, clip) for state in ('initial', 'continuation') for sense in ('visual', 'audio') for clip in (0, 1)}
    for p in s['probes']:
        key = p['condition'], p['sense'], p['clip']
        if key not in required or key in probes or not p['parent_unchanged']: raise ValueError('Probe declaration differs')
        is_initial = key[0] == 'initial'
        if p['start_tick'] != (0 if is_initial else m['trials'][-1]['stop']): raise ValueError('Probe time differs')
        if hashlib.sha256((root/p['file']).read_bytes()).hexdigest() != p['sha256']: raise ValueError('Probe file differs')
        start = (initial[take], np.zeros(len(ports)), np.zeros(len(ports)), np.zeros(len(cfg['neurons']), bool)) if is_initial else branch_start
        with np.load(root/p['file']) as z:
            if len(z['cells']) != m['clip_ticks'] or not np.array_equal(z['incoming_info_before'], initial if is_initial else learned):
                raise ValueError('Probe start differs')
            end, _, _, _, error, change = verify_ledger(z, cfg, ports, *start, m['condition'] == 'eligibility')
            if not np.array_equal(end, z['incoming_info_after'][take]): raise ValueError('Probe endpoint differs')
            probes[key] = z['cells']; probe_updates['/'.join(map(str, key))] = change
            residual = max(residual, error)
            if key[1] == 'visual' and np.any(z['cells'][:, np.array(m['groups']['touch'])-1, 1] > 0):
                raise ValueError('Auditory receptors active during visual-only probe')
    if set(probes) != required: raise ValueError('Missing probes')
    readouts, curves, counts, windows = {}, {}, {}, {}
    for role in ('tactile_core', 'upper_core'):
        ids = m['groups'][role]; readouts[role] = {}
        counts[role] = {'/'.join(map(str, k)): int((v[:, np.array(ids)-1, 1] > 0).sum()) for k, v in probes.items()}
        for kind in KINDS:
            value = lambda state, sense: np.stack([readout(probes[state, sense, clip], ids, kind) for clip in (0, 1)])
            ref, current = value('initial', 'audio'), value('continuation', 'audio')
            before, after = contrast(value('initial', 'visual'), ref), contrast(value('continuation', 'visual'), ref)
            readouts[role][kind] = dict(before=before, after=after, change=after-before if before is not None and after is not None else None,
                **reference_comparison(value('continuation', 'visual'), ref, current))
            if kind == 'population_centered_rate':
                for state, label, basis in (('initial', 'original', ref), ('continuation', 'original', ref), ('continuation', 'current', current)):
                    key = f'{role}/{state}/{label}'
                    curve = tick_projection([probes[state, 'visual', clip] for clip in (0, 1)], ids, basis)
                    if curve is not None:
                        curves[key] = curve
                        windows[key] = [{'start': a, 'stop': min(a+32, len(curve)), 'contrast': float(curve[a:a+32].mean())}
                                        for a in range(0, len(curve), 32)]
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'ticks.npz', spikes=np.concatenate(traces), updates=np.concatenate(updates),
                        **curves, **{f'updates/{k}': v for k, v in probe_updates.items()})
    result = dict(structurally_valid=True, recording=str(root), seed=m['seed'], mapping=m['mapping'], condition=m['condition'],
                  max_update_residual=residual, counts=counts, readouts=readouts, windows=windows, weight_health=health,
                  limits='Data-only selected-input audit. Native updates elsewhere not reconstructed. Projections are observers, not neural consumers. Two clips/two seeds cannot establish general recall, hierarchy or embodiment.')
    (output/'summary.json').write_text(json.dumps(result, allow_nan=False, indent=2)+'\n')
    return result


def check_branch_start(start, parent, opposite, original, ports, condition):
    """Reject any undeclared change, including hidden traces and queued events."""
    expected = deepcopy(parent)
    if condition in ('opposite_mean', 'opposite_residual'):
        grouped = {}
        for nid, sid, _ in ports: grouped.setdefault(nid, []).append(sid)
        for nid, sids in grouped.items():
            if len(sids) != 4 or len(set(sids)) != 4: raise ValueError('Factor test requires four distinct inputs')
            old = np.array([parent['neurons'][str(nid)]['synapses'][str(sid)][0] for sid in sids])
            donor = np.array([opposite['neurons'][str(nid)]['synapses'][str(sid)][0] for sid in sids])
            got = np.array([start['neurons'][str(nid)]['synapses'][str(sid)][0] for sid in sids])
            wanted = old-old.mean()+donor.mean() if condition == 'opposite_mean' else donor-donor.mean()+old.mean()
            if not np.isfinite(got).all() or np.any(got<0) or np.any(got>1) or not np.allclose(got,wanted,atol=3e-16,rtol=0):
                raise ValueError('Selected mean/residual factor differs')
            for sid,value in zip(sids,got): expected['neurons'][str(nid)]['synapses'][str(sid)][0]=float(value)
    elif condition in ('rotate_1', 'rotate_2', 'rotate_3', 'opposite_rank'):
        grouped = {}
        for nid, sid, _ in ports: grouped.setdefault(nid, []).append(sid)
        for nid, sids in grouped.items():
            if len(sids) != 4 or len(set(sids)) != 4: raise ValueError('Assignment requires four distinct inputs')
            old = [parent['neurons'][str(nid)]['synapses'][str(sid)][0] for sid in sids]
            got = [start['neurons'][str(nid)]['synapses'][str(sid)][0] for sid in sids]
            if sorted(old) != sorted(got): raise ValueError('Per-target weight distribution differs')
            if condition.startswith('rotate_'):
                shift = int(condition[-1]); wanted = [old[(j-shift) % 4] for j in range(4)]
            else:
                ordering = sorted(range(4), key=lambda j: (opposite['neurons'][str(nid)]['synapses'][str(sids[j])][0], j))
                wanted = [0.]*4
                for rank, j in enumerate(ordering): wanted[j] = sorted(old)[rank]
            for sid, value in zip(sids, wanted): expected['neurons'][str(nid)]['synapses'][str(sid)][0] = value
    elif condition not in ('unchanged', 'reset_selected', 'opposite_selected'):
        raise ValueError('Undeclared condition')
    elif condition != 'unchanged':
        for nid, sid, _ in ports:
            value = original[nid, sid] if condition == 'reset_selected' else opposite['neurons'][str(nid)]['synapses'][str(sid)][0]
            expected['neurons'][str(nid)]['synapses'][str(sid)][0] = value
    if start != expected: raise ValueError('Undeclared branch state change')


def factor_response(samples):
    """Nonadditivity of two interventions, not an information or memory score."""
    required={'unchanged','opposite_mean','opposite_residual','opposite_selected'}
    if set(samples)!=required: raise ValueError('Need all four weight-factor conditions')
    arrays=list(samples.values())
    if any(v.shape!=arrays[0].shape or not np.isfinite(v).all() for v in arrays):
        raise ValueError('Incompatible factor responses')
    return samples['opposite_selected']-samples['opposite_mean']-samples['opposite_residual']+samples['unchanged']


def conditional_factor_effects(samples, fields):
    """Compare each factor on both parameter backgrounds, retaining tick counts."""
    factor_response(samples)  # Validate the complete factorial and finite arrays.
    pairs=dict(residual_on_own_mean=('unchanged','opposite_residual'),
               residual_on_opposite_mean=('opposite_mean','opposite_selected'),
               mean_on_own_residual=('unchanged','opposite_mean'),
               mean_on_opposite_residual=('opposite_residual','opposite_selected'))
    result={}
    for label,(before,after) in pairs.items():
        a,b=samples[before],samples[after]
        if a.ndim!=3 or a.shape[2]!=len(fields): raise ValueError('Expected tick-cell-field arrays')
        result[label]={}
        for index,field in enumerate(fields):
            difference=b[:,:,index]-a[:,:,index]; changed=difference!=0
            ticks=np.flatnonzero(changed.any(axis=1))
            result[label][field]=dict(different_cell_ticks=int(changed.sum()),
                first_tick=int(ticks[0]) if len(ticks) else None,max_absolute_difference=float(abs(difference).max()),
                different_cells_per_tick=changed.sum(axis=1).tolist())
    return result


def audit_state(recording, output):
    root, output = Path(recording).resolve(), Path(output).resolve()
    meta = json.loads((root/'manifest.json').read_text()); summary = json.loads((root/'summary.json').read_text())
    source, other = Path(meta['source_recording']), Path(meta['opposite_recording'])
    m, alt = [json.loads((p/'manifest.json').read_text()) for p in (source, other)]
    if (summary.get('full_training_replay_exact') is not True or summary.get('unchanged_probe_replay_exact') is not True
            or summary.get('seed')!=m['seed'] or summary.get('mapping')!=m['mapping']):
        raise ValueError('Training or control replay declaration differs')
    if (m['seed'] != alt['seed'] or {m['mapping'], alt['mapping']} != {'paired', 'swapped'}
            or (source/'config.json').read_bytes() != (other/'config.json').read_bytes()
            or m['selected_ports'] != alt['selected_ports']): raise ValueError('Unmatched assignment pair')
    if sensory_marginals(Path(m['source_recording']), m) != sensory_marginals(Path(alt['source_recording']), alt):
        raise ValueError('Unmatched sensory marginals')
    if any(hashlib.sha256(Path(name).read_bytes()).hexdigest() != h for name, h in meta['source_files_sha256'].items()):
        raise ValueError('Source record changed')
    cfg = json.loads((source/'config.json').read_text()); ports = m['selected_ports']
    original = {(p['neuron_id'], p['synapse_id']): p['u_i']['info'] for p in cfg['synaptic_points'] if p['type'] == 'postsynaptic'}
    def state(path):
        with gzip.open(path, 'rt') as f: return json.load(f)
    parent, opposite = [state(p/'training-final-state.json.gz') for p in (source, other)]
    ns = {n['id']: n for n in cfg['neurons']}
    order = [(n, p['synapse_id']) for n in ns for p in cfg['synaptic_points'] if p['type'] == 'postsynaptic' and p['neuron_id'] == n]
    lookup = {pair: i for i, pair in enumerate(order)}
    take = [lookup[n, sid] for n, sid, _ in ports]
    mode = meta.get('mode', 'exchange')
    conditions_by_mode = dict(exchange=('unchanged', 'reset_selected', 'opposite_selected'),
        assignment=('unchanged', 'rotate_1', 'rotate_2', 'rotate_3', 'opposite_rank'),
        factors=('unchanged', 'opposite_mean', 'opposite_residual', 'opposite_selected'))
    if mode not in conditions_by_mode: raise ValueError('Undeclared experiment mode')
    conditions = conditions_by_mode[mode]
    if tuple(meta.get('conditions', conditions)) != conditions: raise ValueError('Condition list differs')
    required = {(c, clip) for c in conditions for clip in (0, 1)}
    samples = {}; residual = 0.; trajectories = {}; effects = {}; comparisons = []; projection_windows = {}
    selected_series = {}; selected_start = {}
    for row in summary['branches']:
        key = row['condition'], row['clip']
        if key not in required or key in samples: raise ValueError('Undeclared branch')
        for field, digest_field in (('file', 'sha256'), ('start_file', 'start_sha256')):
            if hashlib.sha256((root/row[field]).read_bytes()).hexdigest() != row[digest_field]: raise ValueError('Branch artifact changed')
        start = state(root/row['start_file'])
        check_branch_start(start, parent, opposite, original, ports, key[0])
        if 'trial' in row or mode == 'factors':
            if row.get('trial')!=dict(start=parent['tick'],stop=parent['tick']+m['clip_ticks'],visual_clip=key[1],audio_clip=None):
                raise ValueError('Physical probe declaration differs')
        q = np.array([start['neurons'][str(n)]['synapses'][str(sid)][0] for n, sid, _ in ports])
        pre = np.array([start['eligibility'][str(n)]['pre'][ns[n]['metadata']['eligibility_ports'].index(sid)] for n, sid, _ in ports])
        post = np.array([start['eligibility'][str(n)]['post'] for n, _, _ in ports])
        previous = np.array([start['neurons'][str(n)]['O'] > 0 for n in ns])
        with np.load(root/row['file']) as z:
            if len(z['cells']) != m['clip_ticks'] or not np.array_equal(z['incoming_info_before'],
                    [start['neurons'][str(n)]['synapses'][str(sid)][0] for n, sid in order]): raise ValueError('Branch weights or length differ')
            end, _, _, _, error, changes = verify_ledger(z, cfg, ports, q, pre, post, previous)
            if not np.array_equal(end, z['incoming_info_after'][take]):
                raise ValueError('Branch final weights differ')
            residual = max(residual, error); samples[key] = z['cells']
            if mode == 'factors':
                selected_series[key] = z['weights']; selected_start[key] = q
            if np.any(z['cells'][:, np.array(m['groups']['touch'])-1, 1] > 0): raise ValueError('Auditory receptors active')
            if key[0] == 'unchanged':
                with np.load(source/f'probe-continuation-visual-{key[1]}.npz') as old:
                    if set(z.files) != set(old.files) or any(not np.array_equal(z[k], old[k]) for k in z.files):
                        raise ValueError('Unchanged probe replay differs')
            trajectories[f'updates/{key[0]}/{key[1]}'] = changes
    if set(samples) != required: raise ValueError('Missing branch')
    factor_summaries = {}; factor_context = {}
    if mode == 'factors':
        grouped = {}
        for index,(nid,_,_) in enumerate(ports): grouped.setdefault(nid,[]).append(index)
        if any(len(row)!=4 for row in grouped.values()): raise ValueError('Unequal factor fan-in')
        rows = np.array(list(grouped.values()))
        for clip in (0,1):
            factor_context[str(clip)]=conditional_factor_effects({c:samples[c,clip] for c in conditions},m['fields'])
            for condition in conditions:
                if condition == 'unchanged': continue
                delta = selected_series[condition,clip]-selected_series['unchanged',clip]
                initial_norm = float(np.linalg.norm(selected_start[condition,clip]-selected_start['unchanged',clip]))
                row_delta = delta[:,rows]
                means = np.broadcast_to(row_delta.mean(axis=2,keepdims=True),row_delta.shape)
                norms = np.stack([np.linalg.norm(delta,axis=1),np.linalg.norm(means,axis=(1,2)),
                                  np.linalg.norm(row_delta-means,axis=(1,2))],axis=1)
                key=f'factor_weights/{condition}/{clip}'
                trajectories[key]=norms
                factor_summaries[key]=dict(initial_total_norm=initial_norm,
                    minimum_total_ratio=float(norms[:,0].min()/initial_norm) if initial_norm else None,
                    maximum_total_ratio=float(norms[:,0].max()/initial_norm) if initial_norm else None)
            for role,ids in m['groups'].items():
                values={condition:samples[condition,clip][:,np.array(ids)-1].copy() for condition in conditions}
                for v in values.values(): v[:,:,1]=v[:,:,1]>0
                trajectories[f'factor_response/{role}/{clip}']=factor_response(values)
    for role in ('tactile_core', 'upper_core'):
        ids = m['groups'][role]; effects[role] = {}; auditory = {}
        for state_label in ('initial', 'continuation'):
            auditory[state_label] = []
            for clip in (0, 1):
                with np.load(source/f'probe-{state_label}-audio-{clip}.npz') as z: auditory[state_label].append(z['cells'])
        for kind in KINDS:
            effects[role][kind] = {}
            for reference, cells in auditory.items():
                basis = np.stack([readout(c, ids, kind) for c in cells])
                for condition in conditions:
                    value = np.stack([readout(samples[condition, clip], ids, kind) for clip in (0, 1)])
                    effects[role][kind][f'{reference}/{condition}'] = contrast(value, basis)
                    if kind == 'population_centered_rate':
                        curve = tick_projection([samples[condition, clip] for clip in (0, 1)], ids, basis)
                        if curve is not None:
                            key = f'{role}/{reference}/{condition}'
                            trajectories[key] = curve
                            projection_windows[key] = {
                                'all_ticks': float(curve.mean()), 'onset_0_32': float(curve[:32].mean()),
                                'after_32': float(curve[32:].mean()),
                                'windows': [{'start': t, 'stop': min(t+32, len(curve)), 'contrast': float(curve[t:t+32].mean())}
                                            for t in range(0, len(curve), 32)]}
        for clip in (0, 1):
            old = samples['unchanged', clip][:, np.array(ids)-1, 1] > 0
            for condition in conditions:
                spikes = samples[condition, clip][:, np.array(ids)-1, 1] > 0
                differing = spikes != old; times = np.flatnonzero(differing.any(axis=1))
                comparisons.append(dict(role=role, clip=clip, condition=condition, spikes=int(spikes.sum()),
                    first_spike_difference=int(times[0]) if len(times) else None, different_spike_entries=int(differing.sum()),
                    counts_32ticks=[int(spikes[t:t+32].sum()) for t in range(0, len(spikes), 32)]))
                trajectories[f'spikes/{role}/{condition}/{clip}'] = spikes
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'ticks.npz', **trajectories)
    result = dict(structurally_valid=True, recording=str(root), seed=m['seed'], mapping=m['mapping'],
        max_update_residual=residual, effects=effects, comparisons=comparisons, projection_windows=projection_windows,
        mode=mode, factor_weight_summaries=factor_summaries, factor_context=factor_context,
        factor_definitions='Weight columns: total, target-mean and within-target L2 distance from unchanged at each tick. Factor response: both-minus-mean-minus-residual-plus-unchanged for each cell and field; O is a spike flag. Nonadditivity is not memory or consciousness.',
        limits='Selected-weight interventions in two seeds. Assignment controls preserve raw per-target weight distributions, not actual input currents. Factor controls exchange target means and input differences, not intrinsic excitability. A small counterfactual shift is not full sound reinstatement, a learned consumer, generalization or consciousness.')
    (output/'summary.json').write_text(json.dumps(result, allow_nan=False, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.recording/'manifest.json').read_text())
    (audit_state if 'opposite_recording' in manifest else audit)(a.recording, a.output)
