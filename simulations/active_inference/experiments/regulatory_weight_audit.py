"""Audit selected-weight interventions on simultaneous regulatory responses."""
from copy import deepcopy
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
from .media_weight_identity_audit import quiet_receptor_state, verify_local_current, first_difference
from .simultaneous_regulation_audit import pairing_contrast


def compare(recordings, output):
    roots = [Path(p).resolve() for p in recordings]
    if len(roots) != 4 or len(set(roots)) != 4:
        raise ValueError('Need four histories')
    manifests = [json.loads((r/'manifest.json').read_text()) for r in roots]
    upstream = [json.loads((Path(m['source'])/'manifest.json').read_text()) for m in manifests]
    courses = [json.loads((Path(u['course'])/'manifest.json').read_text()) for u in upstream]
    verify_protocol({(m['mapping'], m['order']): m for m in courses})
    samples, traces, rows, firsts, sources = {}, {}, {}, {}, {}
    max_error = 0.
    cfg_bytes = (Path(upstream[0]['course'])/'config.json').read_bytes()
    cfg = json.loads(cfg_bytes)
    ports = upstream[0]['selected_ports']
    base_birth = load_state(Path(upstream[0]['course'])/'initial-state.json.gz')
    for root, m, u, cm in zip(roots, manifests, upstream, courses, strict=True):
        source, course = Path(m['source']), Path(u['course'])
        s = json.loads((root/'summary.json').read_text())
        old = json.loads((source/'summary.json').read_text())
        if not s['intact_control_exact'] or s['acquisition_ticks'] != 0:
            raise ValueError('Missing exact intact control')
        if m['intervention'] != 'selected_info_to_birth' or m['pairs'] != [[0, 1], [1, 1]]:
            raise ValueError('Wrong intervention')
        if any(digest(p) != h for p, h in m['source_hashes'].items()):
            raise ValueError('Source changed')
        for name in ('manifest.json', 'summary.json'):
            sources[str(root/name)] = digest(root/name)
        birth = load_state(course/'initial-state.json.gz')
        if (birth != base_birth or (course/'config.json').read_bytes() != cfg_bytes
                or cm['physical_sources'] != courses[0]['physical_sources']
                or u['selected_ports'] != ports or u['groups'] != upstream[0]['groups']):
            raise ValueError('Unmatched physical source or graph')
        parent = load_state(course/'checkpoint-16-state.json.gz')
        start = deepcopy(parent)
        for n, sid, _ in ports:
            start['neurons'][str(n)]['synapses'][str(sid)][0] = birth['neurons'][str(n)]['synapses'][str(sid)][0]
        initial = ledger_start(start, cfg, ports)
        features = []
        for clip in (0, 1):
            p = next(Path(p) for p in cm['physical_sources'] if Path(p).name == f'sensory-{clip}.npz')
            with np.load(p) as z:
                features.append({k: z[k] for k in z.files})
        full_order = [(n['id'], p['synapse_id']) for n in cfg['neurons'] for p in cfg['synaptic_points']
                      if p['type'] == 'postsynaptic' and p['neuron_id'] == n['id']]
        full = np.array([start['neurons'][str(n)]['synapses'][str(sid)][0] for n, sid in full_order])
        take = [full_order.index((n, sid)) for n, sid, _ in ports]
        if {(p['visual'], p['audio']) for p in s['probes']} != {(0, 1), (1, 1)} or len(s['probes']) != 2:
            raise ValueError('Missing intervention pair')
        for p in s['probes']:
            visual = p['visual']
            if p['trial'] != dict(start=parent['tick'], stop=parent['tick']+300, visual_clip=visual, audio_clip=1):
                raise ValueError('Wrong sensory trial')
            with np.load(checked(root, p)) as z:
                d = {k: z[k] for k in z.files}
            if not np.array_equal(d['incoming_info_before'], full):
                raise ValueError('Nonselected initial weights changed')
            result = verify_ledger(d, cfg, ports, *initial[:4])
            verify_rates(d['cells'], initial[4], cfg, ports)
            verify_local_current(d['arrivals'], d['weights'], initial[0], d['selected_local_current'])
            sensory = quiet_receptor_state(cfg, u['groups'], parent)
            sensory.check(d['cells'], physical_values(features, p['trial']))
            if not np.allclose(d['incoming_info_after'][:384:2], sensory.q, atol=2e-12, rtol=0):
                raise ValueError('Wrong sensory endpoint')
            if not np.array_equal(d['incoming_info_after'][take], result[0]):
                raise ValueError('Wrong selected endpoint')
            max_error = max(max_error, result[4])
            prior = next(q for q in old['probes'] if q['visual'] == visual and q['audio'] == 1)
            with np.load(checked(source, prior)) as z:
                a = {k: z[k] for k in ('cells', 'selected_local_current', 'terminals')}
            history = f"{u['mapping']}/{u['order']}/v{visual}"
            for field in ('selected_local_current', 'terminals'):
                firsts[f'{history}/{field}'] = first_difference(a[field], d[field])
            for role, fields in (('mismatch_candidate', (('O', 1),)),
                                 ('tactile_core', (('O', 1), ('rate', 7))),
                                 ('upper_core', (('O', 1),))):
                ids = np.array(u['groups'][role])-1
                for field, col in fields:
                    av, dv = a['cells'][:, ids, col], d['cells'][:, ids, col]
                    label = f'{history}/{role}/{field}'
                    firsts[label] = first_difference(av, dv)
                    traces[label+'/intact_minus_reset'] = av-dv
                    rows[label+'/intact_minus_reset'] = temporal_summary((av-dv).mean(axis=1))
                    samples[u['mapping'], u['order'], visual, role, field] = av, dv
    for role, field in (('mismatch_candidate', 'O'), ('tactile_core', 'O'), ('tactile_core', 'rate'), ('upper_core', 'O')):
        for visual in (0, 1):
            values = {'intact': [], 'reset': [], 'selected_contribution': []}
            for order in (0, 1):
                paired = samples['paired', order, visual, role, field]
                swapped = samples['swapped', order, visual, role, field]
                intact = pairing_contrast(paired[0], swapped[0], visual, 1)
                reset = pairing_contrast(paired[1], swapped[1], visual, 1)
                for kind, x in (('intact', intact), ('reset', reset), ('selected_contribution', intact-reset)):
                    label = f'{role}/{field}/v{visual}/order{order}/{kind}'
                    traces[label] = x
                    rows[label] = temporal_summary(x.mean(axis=1))
                    values[kind].append(x)
            for kind, pair in values.items():
                label = f'{role}/{field}/v{visual}/balanced/{kind}'
                traces[label] = (pair[0]+pair[1])/2
                rows[label] = temporal_summary(traces[label].mean(axis=1))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'trajectories.npz', **traces)
    report = dict(intervention_valid=True, selected_current_exact=True, max_selected_update_residual=max_error,
                  rows=rows, first_effects=firsts, source_records=sources, audit_source_sha256=digest(__file__),
                  limits='One graph seed. Two sound-1 pairs chosen after the full factorial. Conditional pathway '
                         'localization, not independent confirmation, complete memory erasure or a beneficial-regulation test.')
    (output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
