"""Independent acquisition-course validation and full-tick contextual effects.

Factorial subtraction removes additive visual, auditory and blank responses.
Its residual is a nonlinear contextual interaction, not automatically a learned
prediction. Counterbalancing assignment/order removes some history confounds;
history-dependent gain and other nonlinear adaptation can still contribute.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .association_balance_audit import checked
from .association_route_probe import digest
from .eligibility_media_audit import verify_ledger
from .eligibility_exposure_audit import affine_step
from .graded_media_audit import verify_rates
from .media_drive_audit import ReceptorAudit, physical_values
from .media_order_audit import load_state, verify_protocol, crossed_effect, temporal_summary, reference_projections


def context_interactions(samples):
    expected = {(v, a) for v in (None, 0, 1) for a in (None, 0, 1)}
    if set(samples) != expected:
        raise ValueError('Need the entire physical context factorial')
    shape = samples[None, None].shape
    if any(x.shape != shape or not np.isfinite(x).all() for x in samples.values()):
        raise ValueError('Mismatched contextual records')
    interactions = {(v, a): samples[v, a]-samples[v, None]-samples[None, a]+samples[None, None]
                    for v in (0, 1) for a in (0, 1)}
    diagonal = (interactions[0, 0]+interactions[1, 1]-interactions[0, 1]-interactions[1, 0])/2
    return interactions, diagonal


def check_segments(probe, parent_tick):
    v, a = probe['visual'], probe['audio']
    if v not in (None, 0, 1) or a not in (None, 0, 1):
        raise ValueError('Unknown physical clip')
    if probe['kind'] == 'recall' and v in (0, 1) and a is None:
        expected = [dict(start=parent_tick, stop=parent_tick+300, visual_clip=v, audio_clip=None)]
    elif probe['kind'] == 'context':
        expected = [dict(start=parent_tick, stop=parent_tick+64, visual_clip=v, audio_clip=None),
                    dict(start=parent_tick+64, stop=parent_tick+364, visual_clip=None, audio_clip=a)]
    else:
        raise ValueError('Unknown probe')
    if probe['segments'] != expected:
        raise ValueError('Undeclared sensory phase or duration')


def recruitment_trajectories(cue, withdrawn, blank, prefix=64):
    """Separate driven recruitment from ongoing activity and withdrawal tails.

    Inputs are actual cell-by-field traces from matched parent-state branches.
    This measures output persistence, not persistence of every hidden state.
    """
    arrays = [np.asarray(a) for a in (cue, withdrawn, blank)]
    if (any(a.ndim != 3 or a.shape[2] < 2 or not np.isfinite(a).all() for a in arrays)
            or withdrawn.shape != blank.shape or cue.shape[1:] != blank.shape[1:]
            or not 0 < prefix < len(cue) <= len(blank)):
        raise ValueError('Incompatible cue/withdrawal/blank traces')
    if not np.array_equal(cue[:prefix], withdrawn[:prefix]):
        raise ValueError('Cue branches differ before their physical inputs diverge')
    return dict(cue_minus_blank=cue[:, :, 1]-blank[:len(cue), :, 1],
                withdrawn_minus_blank=withdrawn[:, :, 1]-blank[:, :, 1])


def compare_recruitment(recordings, output):
    """Small data-only supplement to the complete course audit; no simulation."""
    roots = [Path(p).resolve() for p in recordings]
    manifests = {tuple((m['mapping'], m['order'])): m for m in
                 (json.loads((p/'manifest.json').read_text()) for p in roots)}
    verify_protocol(manifests)
    if len(roots) != 4:
        raise ValueError('Need four distinct histories')
    traces, rows, health = {}, {}, {}
    for root in roots:
        m = json.loads((root/'manifest.json').read_text())
        s = json.loads((root/'summary.json').read_text())
        samples = {}
        for p in s['probes']:
            if p['audio'] is not None:
                continue
            with np.load(checked(root, p)) as z:
                samples[p['checkpoint'], p['kind'], p['visual']] = z['cells'].copy()
        for cp in m['checkpoints']:
            blank = samples[cp, 'context', None]
            for v in (0, 1):
                cue, withdrawn = samples[cp, 'recall', v], samples[cp, 'context', v]
                ds = recruitment_trajectories(cue, withdrawn, blank)
                for role in ('tactile_core', 'upper_core', 'mismatch_candidate'):
                    ids = np.array(m['groups'][role])-1
                    name = f"{m['mapping']}/{m['order']}/{cp}/v{v}/{role}"
                    for label, value in ds.items():
                        traces[name+'/'+label] = value[:, ids]
                    def activity(data, start=0):
                        events = data[start:, ids, 1] > 0
                        ticks = np.flatnonzero(events.any(axis=1))+start
                        return dict(events=int(events.sum()), active_ticks=int(len(ticks)),
                                    first=int(ticks[0]) if len(ticks) else None,
                                    last=int(ticks[-1]) if len(ticks) else None)
                    rows[name] = dict(cue=activity(cue), cue_after32=activity(cue, 32),
                        blank=activity(blank), withdrawn=activity(withdrawn),
                        withdrawal_tail=activity(withdrawn, 64),
                        tail_limit='Tick indices start at probe onset; visual input ends after tick 63. '
                                   'No post-withdrawal spikes does not mean all hidden state has vanished.')
        # Read every training health row; column definitions belong to WeightObserver.
        minima, maxima = None, None
        for item in s['training']:
            with np.load(checked(root, item)) as z:
                h = z['weight_health']
                if not np.isfinite(h).all():
                    raise ValueError('Nonfinite weight health')
                minima = h.min(axis=0) if minima is None else np.minimum(minima, h.min(axis=0))
                maxima = h.max(axis=0) if maxima is None else np.maximum(maxima, h.max(axis=0))
        health[f"{m['mapping']}/{m['order']}"] = dict(minima=minima.tolist(), maxima=maxima.tolist())
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'recruitment-trajectories.npz', **traces)
    result = dict(rows=rows, training_weight_health=health, recordings=[str(p) for p in roots],
                  identical_cue_prefixes=True,
                  limits='Data-only supplement; use with the full course protocol audit. '
                         'Cue-driven spikes are not a certificate of content-specific recall or hidden-state erasure.')
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def compare(recordings, output):
    roots, manifests, summaries = {}, {}, {}
    for path in recordings:
        root = Path(path).resolve(); m = json.loads((root/'manifest.json').read_text())
        key = m['mapping'], m['order']
        if key in roots:
            raise ValueError('Duplicate learning history')
        roots[key], manifests[key] = root, m
        summaries[key] = json.loads((root/'summary.json').read_text())
    verify_protocol(manifests)
    base = manifests['paired', 0]
    features = []
    for clip in (0, 1):
        path = next(Path(p) for p in base['physical_sources'] if Path(p).name == f'sensory-{clip}.npz')
        if digest(path) != base['physical_sources'][str(path)]:
            raise ValueError('Physical source changed')
        with np.load(path) as z:
            features.append({k: z[k] for k in z.files})
    cfg_text = (roots['paired', 0]/'config.json').read_bytes()
    cfg = json.loads(cfg_text); ports = base['selected_ports']
    ns = {n['id']: n for n in cfg['neurons']}
    if any(n['params']['eta_post'] <= 0 or n['params']['eta_retro'] <= 0 for n in ns.values()):
        raise ValueError('Frozen adaptation')
    syn_order = [(n['id'], p['synapse_id']) for n in cfg['neurons'] for p in cfg['synaptic_points']
                 if p['type'] == 'postsynaptic' and p['neuron_id'] == n['id']]
    index = {key: i for i, key in enumerate(syn_order)}
    take = np.array([index[n, sid] for n, sid, _ in ports])
    target = np.array([list(ns).index(n) for n, _, _ in ports])
    def weights(state):
        return np.array([state['neurons'][str(n)]['synapses'][str(sid)][0] for n, sid in syn_order])
    def start_values(state):
        return (weights(state)[take],
                np.array([state['eligibility'][str(n)]['pre'][ns[n]['metadata']['eligibility_ports'].index(sid)] for n, sid, _ in ports]),
                np.array([state['eligibility'][str(n)]['post'] for n, _, _ in ports]),
                np.array([state['neurons'][str(n)]['O'] > 0 for n in ns]),
                np.array([state['neurons'][str(n)]['M'][0] for n in ns]))
    birth = load_state(roots['paired', 0]/'initial-state.json.gz')
    checkpoints = base['checkpoints']
    if checkpoints != [4, 16] or base['prefix_ticks'] != 64 or base['sound_ticks'] != 300:
        raise ValueError('This comparison covers the declared 4/16 exposure course')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    observed, exposure_rows, traces, counts = {}, {}, {}, {}
    residual = 0.; affine_residual = 0.

    def verify_data(data, values, sensory, start, full_weights):
        nonlocal residual
        if not np.array_equal(data['incoming_info_before'], full_weights):
            raise ValueError('Broken input-weight continuity')
        result = verify_ledger(data, cfg, ports, *start[:4])
        verify_rates(data['cells'], start[4], cfg, ports)
        sensory.check(data['cells'], values)
        if not np.allclose(data['incoming_info_after'][:384:2], sensory.q, atol=2e-12, rtol=0):
            raise ValueError('Sensory weight reconstruction differs')
        if not np.array_equal(data['incoming_info_after'][take], result[0]):
            raise ValueError('Selected weight endpoint differs')
        residual = max(residual, result[4])
        return (*result[:4], data['cells'][-1, :, 3].copy())

    for key, m in manifests.items():
        root, summary = roots[key], summaries[key]; source = Path(m['source'])
        old = json.loads((source/'manifest.json').read_text())
        if any(m[f] != base[f] for f in ('seed', 'groups', 'selected_ports', 'physical_sources', 'checkpoints')):
            raise ValueError('Unmatched architecture, inputs or checkpoints')
        if (root/'config.json').read_bytes() != cfg_text or (source/'config.json').read_bytes() != cfg_text:
            raise ValueError('Configuration changed')
        if load_state(root/'initial-state.json.gz') != birth:
            raise ValueError('Initial neural state changed')
        if any(digest(p) != h for p, h in {**m['source_files'], **m['source_hashes']}.items()):
            raise ValueError('Changed source')
        if any(m[f] != old[f] for f in ('seed', 'mapping', 'order', 'groups', 'selected_ports', 'physical_sources')):
            raise ValueError('Source protocol mismatch')
        if len(summary['training']) != len(m['trials']):
            raise ValueError('Missing acquisition record')
        states = {item['repeats']: item for item in summary['states']}
        if set(states) != set(checkpoints) or len(states) != len(summary['states']):
            raise ValueError('Missing or repeated state checkpoint')
        checkpoint_states, sensory_states = {}, {}
        start = start_values(birth); full_weights = weights(birth)
        sensory = ReceptorAudit(cfg, m['groups'], graded_gain=.25)
        a, b, q0 = np.ones(len(ports)), np.zeros(len(ports)), start[0].copy()
        cumulative_dose = np.zeros(len(ports))
        for i, (item, trial) in enumerate(zip(summary['training'], m['trials'], strict=True)):
            if item['trial'] != trial:
                raise ValueError('Changed acquisition declaration')
            with np.load(checked(root, item)) as z:
                data = {k: z[k] for k in z.files}
            previous_pre, previous_post = start[1].copy(), start[2].copy()
            start = verify_data(data, physical_values(features, trial), sensory, start, full_weights)
            full_weights = data['incoming_info_after']
            coeff = np.empty_like(data['weights']); exposure = np.empty_like(coeff)
            for t in range(len(data['cells'])):
                plus = (data['cells'][t, target, 1] > 0)*previous_pre*np.exp(-.25)
                minus = (data['arrivals'][t] > 0)*previous_post*np.exp(-.25)
                a, b = affine_step(a, b, plus, minus, data['eta'][t])
                cumulative_dose += data['eta'][t]*(plus+minus)
                coeff[t], exposure[t] = a, cumulative_dose
                affine_residual = max(affine_residual, float(np.max(abs(a*q0+b-data['weights'][t]))))
                previous_pre, previous_post = data['pre'][t], data['post'][t]
            if affine_residual > 2e-12:
                raise ValueError('Learning exposure decomposition differs')
            name = f'{key[0]}-order{key[1]}-exposure-{i:03d}.npz'
            np.savez_compressed(output/name, initial_coefficient=coeff, cumulative_local_exposure=exposure)
            repeats = (i+1)//4
            if (i+1) % 4 == 0 and repeats in checkpoints:
                cp = states[repeats]
                if digest(root/cp['file']) != cp['sha256']:
                    raise ValueError('Changed checkpoint state')
                state = load_state(root/cp['file'])
                if state['tick'] != trial['stop'] or not np.array_equal(weights(state), full_weights):
                    raise ValueError('Checkpoint weight state differs')
                if any(not np.array_equal(x, y) for x, y in zip(start_values(state), start, strict=True)):
                    raise ValueError('Checkpoint eligibility state differs')
                if repeats == 4 and state != load_state(source/'trained-state.json.gz'):
                    raise ValueError('Original trained-state checkpoint changed')
                checkpoint_states[repeats], sensory_states[repeats] = state, deepcopy(sensory)
                exposure_rows[f'{key[0]}/{key[1]}/{repeats}'] = dict(
                    initial_coefficient_quantiles=np.quantile(a, [0, .1, .5, .9, 1]).tolist(),
                    exposure_quantiles=np.quantile(cumulative_dose, [0, .1, .5, .9, 1]).tolist())
        required = {(cp, 'recall', v, None) for cp in checkpoints for v in (0, 1)} | {
            (cp, 'context', v, a) for cp in checkpoints for v in (None, 0, 1) for a in (None, 0, 1)}
        seen = set()
        for p in summary['probes']:
            case = p['checkpoint'], p['kind'], p['visual'], p['audio']
            if case not in required or case in seen:
                raise ValueError('Unknown or duplicated probe')
            seen.add(case); cp = p['checkpoint']; parent = checkpoint_states[cp]
            if p['parent_state'] != states[cp]['file']:
                raise ValueError('Incorrect probe parent')
            check_segments(p, parent['tick'])
            with np.load(checked(root, p)) as z:
                data = {k: z[k] for k in z.files}
            values = np.concatenate([physical_values(features, segment) for segment in p['segments']])
            verify_data(data, values, deepcopy(sensory_states[cp]), start_values(parent), weights(parent))
            observed[key+case] = data['cells'][:, :, 1].copy()
            for role in ('tactile_core', 'upper_core', 'mismatch_candidate'):
                ids = np.array(m['groups'][role])-1
                spikes = (data['cells'][:, ids, 1] > 0).sum(axis=1)
                name = '/'.join(map(str, (*key, *case, role)))
                traces[name+'/spikes'] = spikes
                counts[name] = dict(total=int(spikes.sum()), first=int(np.flatnonzero(spikes)[0]) if spikes.any() else None,
                                    prefix=int(spikes[:64].sum()) if p['kind'] == 'context' else None,
                                    after_sound_onset=int(spikes[64:].sum()) if p['kind'] == 'context' else None)
        if seen != required:
            raise ValueError('Missing probe')
    interactions, diagonal = {}, {}
    rows, recall_rows = {}, {}
    original_audio = []
    for clip in (0, 1):
        source = Path(base['source'])
        with np.load(source/f'initial-graded-audio-{clip}.npz') as z:
            original_audio.append(z['cells'][:, :, 1].copy())
    for cp in checkpoints:
        for mapping, order in manifests:
            samples = {(v, a): observed[mapping, order, cp, 'context', v, a]
                       for v in (None, 0, 1) for a in (None, 0, 1)}
            for v in (None, 0, 1):
                if any(not np.array_equal(samples[v, None][:64], samples[v, a][:64]) for a in (0, 1)):
                    raise ValueError('Future sound influenced its identical visual prefix')
            inter, diag = context_interactions(samples)
            interactions[mapping, order, cp], diagonal[mapping, order, cp] = inter, diag
            for (v, a), value in inter.items():
                if value[:64].any():
                    raise ValueError('Context interaction appears before sound availability')
                traces[f'{mapping}/{order}/{cp}/interaction/v{v}/a{a}'] = value
            recall = [observed[mapping, order, cp, 'recall', v, None] for v in (0, 1)]
            for role in ('tactile_core', 'upper_core', 'mismatch_candidate'):
                ids = np.array(base['groups'][role])-1
                diff = recall[0][:, ids]-recall[1][:, ids]
                refs = dict(original=[(a[32:, ids] > 0).mean(axis=0) for a in original_audio],
                            current=[(samples[None, a][96:, ids] > 0).mean(axis=0) for a in (0, 1)])
                name = f'{mapping}/{order}/{cp}/recall/{role}'
                traces[name+'/cue_difference'] = diff
                for basis, reference in refs.items():
                    for channel, projection in reference_projections(diff, reference).items():
                        if projection is not None:
                            traces[name+f'/{basis}/{channel}'] = projection
                            recall_rows[name+f'/{basis}/{channel}'] = temporal_summary(projection)
        main, order_interaction = crossed_effect(*(diagonal[m, o, cp] for m, o in (
            ('paired', 0), ('swapped', 0), ('paired', 1), ('swapped', 1))))
        traces[f'{cp}/assignment_effect'] = main
        traces[f'{cp}/assignment_by_order'] = order_interaction
        for role in ('tactile_core', 'upper_core', 'mismatch_candidate'):
            ids = np.array(base['groups'][role])-1
            v = main[64:, ids].sum(axis=1)
            rows[f'{cp}/{role}'] = dict(**temporal_summary(v),
                nonzero_cell_ticks=int(np.count_nonzero(main[64:, ids])),
                interpretation='Signed population interaction; neither prediction accuracy nor content decoding.')
    np.savez_compressed(output/'context-trajectories.npz', **traces)
    result = dict(protocol_valid=True, identical_prefixes=True, max_selected_update_residual=residual,
                  max_affine_reconstruction_residual=affine_residual, exposure=exposure_rows,
                  counts=counts, interactions=rows, recall_projections=recall_rows, recordings=[str(p) for p in roots.values()],
                  limits='All declared graded sensory and selected learning trajectories reconstructed. '
                  'Other neural currents and full hidden probe-start state are not independently reconstructed. '
                  'One graph seed; repeated clips are not independent samples. Nonlinear history-dependent gain can '
                  'produce an assignment interaction without a content-specific predictive circuit. '
                  'Delayed context probes test a new temporal condition, not the original simultaneous experience.')
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', type=Path, nargs=4, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); compare(a.recording, a.output)
