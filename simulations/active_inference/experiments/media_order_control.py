"""Counterbalance audiovisual assignment against presentation order.

Four matched histories cross paired/swapped assignment with visual orders 01
and 10. Each order is repeated equally. Across orders, each assignment has the
same visual and auditory sequences as multisets, including final-item identity.
This controls those sequence marginals, not all nonlinear history interactions.
No condition or clip label is sent to a neuron.
"""
import argparse
from copy import deepcopy
import gzip
import inspect
import json
from pathlib import Path
import time

import numpy as np

from .association_route_probe import digest
from .composition_probe import encode, fingerprint
from .eligibility_association_probe import dynamic_snapshot
from .eligibility_media_probe import record
from .eligibility_media_audit import verify_ledger
from .graded_media_audit import verify_rates
from .graded_media_probe import configure
from .graded_recall_factors import intervene
from .media_drive_audit import ReceptorAudit, physical_values
from .multimodal_pairing_probe import fresh, WeightObserver
from .population_state_branch import TickDriver
from neuron.extensions.experimental.graded_eligibility import GradedEligibilityNeuron


def protocol(length, repeats, mapping, order):
    if length <= 0 or length % 4 or repeats < 1 or mapping not in ('paired', 'swapped') or order not in (0, 1):
        raise ValueError('Invalid counterbalanced protocol')
    trials = []; t = 0
    for _ in range(repeats):
        for visual in (order, 1-order):
            trials.append(dict(start=t, stop=t+length, visual_clip=visual,
                               audio_clip=visual if mapping=='paired' else 1-visual, phase='experience'))
            t += length
            trials.append(dict(start=t, stop=t+96, visual_clip=None, audio_clip=None, phase='withdrawal'))
            t += 96
    return trials


def ledger_start(state, cfg, ports):
    ns = {n['id']: n for n in cfg['neurons']}
    q = np.array([state['neurons'][str(n)]['synapses'][str(sid)][0] for n, sid, _ in ports])
    pre = np.array([state['eligibility'][str(n)]['pre'][ns[n]['metadata']['eligibility_ports'].index(sid)] for n, sid, _ in ports])
    post = np.array([state['eligibility'][str(n)]['post'] for n, _, _ in ports])
    previous = np.array([state['neurons'][str(n['id'])]['O'] > 0 for n in cfg['neurons']])
    modulation = np.array([state['neurons'][str(n['id'])]['M'][0] for n in cfg['neurons']])
    return q, pre, post, previous, modulation


def run(source, output, mapping, order, repeats=4):
    source, output = Path(source).resolve(), Path(output).resolve()
    old = json.loads((source/'manifest.json').read_text())
    if old['condition'] != 'eligibility':raise ValueError('Requires the real-media eligibility graph')
    if any(digest(p) != h for p, h in old['source_hashes'].items()):raise ValueError('Source runtime changed')
    cfg = configure(json.loads((source/'config.json').read_text()), old['groups'], 'graded')
    trials = protocol(old['clip_ticks'], repeats, mapping, order)
    features = []; physical_sources = {}
    for clip in (0, 1):
        path = Path(old['source_recording'])/f'sensory-{clip}.npz'
        if digest(path) != old['source_files_sha256'][path.name]:raise ValueError('Physical source changed')
        physical_sources[str(path)] = digest(path)
        with np.load(path) as z:features.append({k: z[k] for k in z.files})
    hashes = fingerprint()
    for obj in (run, configure, record, ledger_start, verify_ledger, verify_rates, intervene,
                ReceptorAudit, fresh, TickDriver, dynamic_snapshot):
        path = Path(inspect.getfile(obj)).resolve();hashes[str(path)] = digest(path)
    output.mkdir(parents=True, exist_ok=False)
    path = output/'config.json';path.write_text(encode(cfg)+'\n')
    m = dict(source=str(source), seed=old['seed'], mapping=mapping, order=order,
             repeats=repeats, trials=trials, groups=old['groups'], selected_ports=old['selected_ports'],
             source_hashes=hashes, physical_sources=physical_sources,
             limits='Fixed graph, two recordings. Order is a crossed experimental factor. '
                    'Spiking visual release at test is a diagnostic intervention, not neural self-regulation. '
                    'All plasticity remains active. No labels or observer values enter neurons.')
    (output/'manifest.json').write_text(encode(m)+'\n')
    net, core, members, points = fresh(path, old['seed'], GradedEligibilityNeuron)
    initial = deepcopy(net); birth = dynamic_snapshot(initial)
    ports = m['selected_ports']; health = WeightObserver(members, points)
    start = ledger_start(json.loads(birth), cfg, ports); q_initial = start[0].copy()
    receptor = ReceptorAudit(cfg, m['groups'], graded_gain=.25)
    training = []; residual = 0.; began = time.perf_counter()
    for i, trial in enumerate(trials):
        data = record(net, core, members, points, features, m['groups'], trial, ports, health)
        result = verify_ledger(data, cfg, ports, *start[:4])
        verify_rates(data['cells'], start[4], cfg, ports)
        receptor.check(data['cells'], physical_values(features, trial))
        if not np.allclose(data['incoming_info_after'][:384:2], receptor.q, atol=2e-12, rtol=0):
            raise ValueError('Sensory adaptation differs')
        start = (*result[:4], data['cells'][-1, :, 3].copy());residual = max(residual, result[4])
        path = output/f'experience-{i:03d}.npz';np.savez_compressed(path, **data)
        training.append(dict(file=path.name, sha256=digest(path), trial=trial))
    parent = dynamic_snapshot(net); learned_q = start[0].copy()
    for name, state in (('initial', birth), ('trained', parent)):
        with gzip.open(output/f'{name}-state.json.gz', 'wt') as f:f.write(state+'\n')
    print('acquired', mapping, order, round(time.perf_counter()-began, 2), flush=True)
    cases = [('initial', 'spiking', 'visual'), ('initial', 'graded', 'audio'),
             ('trained', 'graded', 'visual'), ('trained', 'spiking', 'visual'),
             ('reset_selected', 'spiking', 'visual'), ('trained', 'graded', 'audio')]
    probes = []
    for state, expression, sense in cases:
        for clip in (0, 1):
            original = initial if state=='initial' else net
            branch = deepcopy(original)
            q = q_initial if state in ('initial', 'reset_selected') else learned_q
            before, expected = intervene(branch, ports, m['groups']['vision'], q, expression)
            if before != json.loads(birth if state=='initial' else parent):raise ValueError('Parent differs')
            name = f'{state}-{expression}-{sense}-{clip}'
            path = output/f'{name}-start.json.gz'
            with gzip.open(path, 'wt') as f:f.write(encode(expected)+'\n')
            start_sha = digest(path)
            cells = list(branch.network.neurons.values()); syns = [p for n in cells for p in n.postsynaptic_points.values()]
            t = branch.current_tick
            trial = dict(start=t, stop=t+old['clip_ticks'], visual_clip=clip if sense=='visual' else None,
                         audio_clip=clip if sense=='audio' else None)
            data = record(branch, TickDriver(branch), cells, syns, features, m['groups'], trial, ports)
            start = ledger_start(expected, cfg, ports)
            result = verify_ledger(data, cfg, ports, *start[:4]);verify_rates(data['cells'], start[4], cfg, ports)
            residual = max(residual, result[4])
            if expression=='graded':
                r = ReceptorAudit(cfg, m['groups'], graded_gain=.25) if state=='initial' else deepcopy(receptor)
                r.check(data['cells'], physical_values(features, trial))
            silent = m['groups']['touch'] if sense=='visual' else m['groups']['vision']
            if data['cells'][:, np.array(silent)-1, 1].any():raise ValueError('Absent sense emitted output')
            path = output/f'{name}.npz';np.savez_compressed(path, **data)
            probes.append(dict(state=state, expression=expression, sense=sense, clip=clip,
                trial=trial, file=path.name, sha256=digest(path), start_file=f'{name}-start.json.gz', start_sha256=start_sha))
            if dynamic_snapshot(net)!=parent or dynamic_snapshot(initial)!=birth:raise ValueError('Probe mutated parent')
            print('probe', mapping, order, name, flush=True)
    if any(digest(p)!=h for p,h in {**hashes, **physical_sources}.items()):raise ValueError('Sources changed during run')
    summary = dict(training=training, probes=probes, max_selected_update_residual=residual,
                   parent_unchanged=True, ticks=trials[-1]['stop']+len(probes)*old['clip_ticks'],
                   seconds=time.perf_counter()-began)
    (output/'summary.json').write_text(encode(summary)+'\n')
    return summary


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mapping', choices=('paired', 'swapped'), required=True)
    p.add_argument('--order', type=int, choices=(0, 1), required=True)
    p.add_argument('--repeats', type=int, default=4)
    a = p.parse_args();run(a.source, a.output, a.mapping, a.order, a.repeats)
