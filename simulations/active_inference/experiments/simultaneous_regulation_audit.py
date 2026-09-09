"""Independent full-tick tests of learned pairing and regulatory response.

Untrained-pair minus trained-pair contrasts compare the same physical input
across histories. A second contrast removes each history's unimodal and blank
responses. This removes additive effects, not nonlinear history/gain confounds.
No expected sign is fed into the simulator or used as a fitted decoder.
"""
import json
from pathlib import Path

import numpy as np

from .association_balance_audit import checked
from .association_route_probe import digest
from .eligibility_media_audit import verify_ledger
from .graded_media_audit import verify_rates
from .media_drive_audit import physical_values
from .media_order_audit import load_state, temporal_summary, verify_protocol
from .media_order_control import ledger_start
from .media_weight_identity_audit import quiet_receptor_state, verify_local_current


def interaction(samples, visual, audio):
    required = {(v, a) for v in (None, 0, 1) for a in (None, 0, 1)}
    if set(samples) != required or visual not in (0, 1) or audio not in (0, 1):
        raise ValueError('Need the complete physical factorial')
    shape = samples[None, None].shape
    if any(x.shape != shape or not np.isfinite(x).all() for x in samples.values()):
        raise ValueError('Invalid factorial trajectories')
    return samples[visual, audio] - samples[visual, None] - samples[None, audio] + samples[None, None]


def pairing_contrast(paired, swapped, visual, audio):
    """Same physical pair: history without that pairing minus history with it."""
    if visual not in (0, 1) or audio not in (0, 1):
        raise ValueError('Need two present sensory streams')
    a, b = np.asarray(paired), np.asarray(swapped)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid matched histories')
    return b-a if visual == audio else a-b


def compare(recordings, output):
    roots = [Path(p).resolve() for p in recordings]
    if len(roots) != 4 or len(set(roots)) != 4:
        raise ValueError('Need four distinct histories')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifests = [json.loads((p/'manifest.json').read_text()) for p in roots]
    courses = [json.loads((Path(m['course'])/'manifest.json').read_text()) for m in manifests]
    verify_protocol({(m['mapping'], m['order']): m for m in courses})
    base = manifests[0]
    cfg_bytes = (Path(base['course'])/'config.json').read_bytes()
    cfg = json.loads(cfg_bytes)
    ports = base['selected_ports']
    birth = load_state(Path(base['course'])/'initial-state.json.gz')
    responses, rows, sources = {}, {}, {}
    residual = 0.
    for root, m, cm in zip(roots, manifests, courses, strict=True):
        course = Path(m['course'])
        s = json.loads((root/'summary.json').read_text())
        for name in ('manifest.json', 'summary.json'):
            sources[str(root/name)] = digest(root/name)
        if not s['intact_control_exact'] or s['acquisition_ticks'] != 0:
            raise ValueError('Replay control failed')
        if (any(m[k] != base[k] for k in ('seed', 'groups', 'selected_ports'))
                or cm['physical_sources'] != courses[0]['physical_sources']
                or any(m[k] != cm[k] for k in ('seed', 'mapping', 'order', 'groups', 'selected_ports'))
                or (course/'config.json').read_bytes() != cfg_bytes
                or load_state(course/'initial-state.json.gz') != birth):
            raise ValueError('Unmatched birth or configuration')
        if any(digest(p) != h for p, h in m['source_hashes'].items()):
            raise ValueError('Sources changed')
        parent = load_state(course/'checkpoint-16-state.json.gz')
        initial = ledger_start(parent, cfg, ports)
        features = []
        for clip in (0, 1):
            p = next(Path(p) for p in cm['physical_sources'] if Path(p).name == f'sensory-{clip}.npz')
            if digest(p) != cm['physical_sources'][str(p)]:
                raise ValueError('Changed sensory data')
            with np.load(p) as z:
                features.append({k: z[k] for k in z.files})
        full_order = [(n['id'], p['synapse_id']) for n in cfg['neurons'] for p in cfg['synaptic_points']
                      if p['type'] == 'postsynaptic' and p['neuron_id'] == n['id']]
        full = np.array([parent['neurons'][str(n)]['synapses'][str(s)][0] for n, s in full_order])
        take = [full_order.index((n, sid)) for n, sid, _ in ports]
        seen = set()
        for p in s['probes']:
            key = p['visual'], p['audio']
            if key in seen or key not in {(v, a) for v in (None, 0, 1) for a in (None, 0, 1)}:
                raise ValueError('Duplicate or invalid physical condition')
            seen.add(key)
            trial = dict(start=parent['tick'], stop=parent['tick']+300, visual_clip=key[0], audio_clip=key[1])
            if p['trial'] != trial:
                raise ValueError('Changed timing or input')
            with np.load(checked(root, p)) as z:
                d = {k: z[k] for k in z.files}
            if not np.array_equal(d['incoming_info_before'], full):
                raise ValueError('Unintended initial weight change')
            result = verify_ledger(d, cfg, ports, *initial[:4])
            verify_rates(d['cells'], initial[4], cfg, ports)
            verify_local_current(d['arrivals'], d['weights'], initial[0], d['selected_local_current'])
            sensory = quiet_receptor_state(cfg, m['groups'], parent)
            sensory.check(d['cells'], physical_values(features, trial))
            if not np.allclose(d['incoming_info_after'][:384:2], sensory.q, atol=2e-12, rtol=0):
                raise ValueError('Wrong sensory endpoint')
            if not np.array_equal(d['incoming_info_after'][take], result[0]):
                raise ValueError('Wrong selected weight endpoint')
            residual = max(residual, result[4])
            for role in ('mismatch_candidate', 'activity_regulator', 'tactile_core', 'visual_core', 'upper_core'):
                ids = np.array(m['groups'][role])-1
                for field, column in (('O', 1), ('S', 0), ('M0', 3), ('rate', 7)):
                    x = d['cells'][:, ids, column].copy()
                    responses[m['mapping'], m['order'], role, field, *key] = x
                    label = f"{m['mapping']}/{m['order']}/{role}/{field}/{key[0]}/{key[1]}"
                    rows[label] = temporal_summary(x.mean(axis=1))
    traces, contrasts = {}, {}
    for role in ('mismatch_candidate', 'activity_regulator', 'tactile_core', 'visual_core', 'upper_core'):
        for field in ('O', 'S', 'M0', 'rate'):
            per_order = {kind: [] for kind in ('raw', 'interaction')}
            for order in (0, 1):
                samples = {mapping: {(v, a): responses[mapping, order, role, field, v, a]
                                    for v in (None, 0, 1) for a in (None, 0, 1)}
                           for mapping in ('paired', 'swapped')}
                values = {kind: [] for kind in per_order}
                for v in (0, 1):
                    for a in (0, 1):
                        pairs = dict(raw=(samples['paired'][v, a], samples['swapped'][v, a]),
                                     interaction=tuple(interaction(samples[m], v, a) for m in ('paired', 'swapped')))
                        for kind, pair in pairs.items():
                            x = pairing_contrast(*pair, v, a)
                            label = f'{role}/{field}/{kind}/order{order}/v{v}/a{a}'
                            traces[label] = x
                            contrasts[label] = temporal_summary(x.mean(axis=1))
                            values[kind].append(x)
                for kind in values:
                    x = np.mean(values[kind], axis=0)
                    label = f'{role}/{field}/{kind}/order{order}/balanced'
                    traces[label] = x
                    contrasts[label] = temporal_summary(x.mean(axis=1))
                    per_order[kind].append(x)
            for kind, values in per_order.items():
                for name, x in (('balanced', (values[0]+values[1])/2),
                                ('order_interaction', (values[0]-values[1])/2)):
                    label = f'{role}/{field}/{kind}/{name}'
                    traces[label] = x
                    contrasts[label] = temporal_summary(x.mean(axis=1))
    # Cellwise contrasts remain available, not only the displayed population means.
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'trajectories.npz', **traces)
    result = dict(physically_matched=True, selected_current_exact=True,
                  max_selected_update_residual=residual, rows=rows, contrasts=contrasts,
                  recordings=[str(r) for r in roots], source_records=sources,
                  audit_source_sha256=digest(__file__),
                  limits='One graph seed. Same-input learned-history contrasts, not isolated causal regulator lesions. '
                         'Interaction subtraction removes additive unimodal contributions, not nonlinear gain. '
                         'Balanced raw and interaction aggregates are algebraically equal; they are not independent evidence. '
                         'O is spike output; S resets at spikes and is not an activity measure.')
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result
