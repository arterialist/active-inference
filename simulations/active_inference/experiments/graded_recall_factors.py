"""Separate association weights from cue expression in a trained media graph.

Diagnostic clones only. The trained graded parent is reconstructed by exact
acquisition replay. All adaptation continues during every probe. Donor weights
are an experimental intervention, not a neural learning or recall mechanism.
"""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import time

import numpy as np

from .association_balance_audit import checked
from .association_route_probe import digest
from .composition_probe import encode
from .eligibility_association_probe import dynamic_snapshot
from .eligibility_media_probe import record
from .eligibility_media_audit import verify_ledger
from .graded_media_audit import verify_rates
from .multimodal_pairing_probe import fresh
from .population_state_branch import TickDriver
from neuron.extensions.experimental.graded_eligibility import GradedEligibilityNeuron


def intervene(net, ports, vision, weights, expression):
    """Change only selected incoming information weights and visual release."""
    if expression not in ('graded', 'spiking'):
        raise ValueError('Unknown cue expression')
    if len(weights) != len(ports) or not np.isfinite(weights).all():
        raise ValueError('Malformed intervention weights')
    if len({(n, sid) for n, sid, _ in ports}) != len(ports):
        raise ValueError('Duplicate intervention port')
    before = json.loads(dynamic_snapshot(net))
    expected = deepcopy(before)
    for (nid, sid, _), weight in zip(ports, weights, strict=True):
        net.network.neurons[nid].postsynaptic_points[sid].u_i.info = float(weight)
        expected['neurons'][str(nid)]['synapses'][str(sid)][0] = float(weight)
    for nid in vision:
        n = net.network.neurons[nid]
        if n.eligibility_ports:
            raise ValueError('Visual release switch has undefined spike eligibility')
        n._gg = .25 if expression == 'graded' else 0.
        n.metadata['graded_gain'] = n._gg
    if json.loads(dynamic_snapshot(net)) != expected:
        raise ValueError('Undeclared dynamic-state intervention')
    return before, expected


def run(source, donor, output):
    source, donor, output = map(lambda p: Path(p).resolve(), (source, donor, output))
    m = json.loads((source/'manifest.json').read_text())
    dm = json.loads((donor/'manifest.json').read_text())
    summary = json.loads((source/'summary.json').read_text())
    if m['condition'] != 'graded' or dm['condition'] != 'spiking':
        raise ValueError('Requires matched graded and spiking acquisitions')
    if any(m[k] != dm[k] for k in ('seed', 'mapping', 'groups', 'selected_ports', 'trials', 'source')):
        raise ValueError('Donor protocol differs')
    hashes = {**m['source_hashes'], str(Path(__file__).resolve()): digest(__file__)}
    if any(digest(p) != h for p, h in hashes.items()):
        raise ValueError('Recorded runtime changed')
    cfg = json.loads((source/'config.json').read_text())
    if any(n['params']['eta_post'] <= 0 or n['params']['eta_retro'] <= 0 for n in cfg['neurons']):
        raise ValueError('Plasticity must remain positive')
    old = json.loads((Path(m['source'])/'manifest.json').read_text())
    features = []
    for clip in (0, 1):
        path = Path(old['source_recording'])/f'sensory-{clip}.npz'
        if digest(path) != old['source_files_sha256'][path.name]:
            raise ValueError('Sensory input changed')
        with np.load(path) as z:
            features.append({k: z[k] for k in z.files})
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    net, core, members, points = fresh(source/'config.json', m['seed'], GradedEligibilityNeuron)
    for item, trial in zip(summary['training'], m['trials'], strict=True):
        data = record(net, core, members, points, features, m['groups'], trial, m['selected_ports'])
        with np.load(checked(source, item)) as z:
            if set(data) != set(z.files) or any(not np.array_equal(data[k], z[k]) for k in z.files):
                raise ValueError('Acquisition replay differs')
    parent = dynamic_snapshot(net)
    with gzip.open(source/'trained-state.json.gz', 'rt') as f:
        if json.loads(parent) != json.load(f):
            raise ValueError('Trained replay endpoint differs')
    print('exact acquisition replay complete', round(time.perf_counter()-started, 2), flush=True)
    states = {}
    for name, path in (('graded', source/'trained-state.json.gz'),
                       ('spiking', donor/'trained-state.json.gz'),
                       ('initial', source/'initial-state.json.gz')):
        with gzip.open(path, 'rt') as f:
            states[name] = json.load(f)
    ports = m['selected_ports']
    qsets = {name: np.array([s['neurons'][str(n)]['synapses'][str(sid)][0]
                            for n, sid, _ in ports]) for name, s in states.items()}
    manifest = dict(source=str(source), donor=str(donor), seed=m['seed'], mapping=m['mapping'],
        source_hashes=hashes, selected_ports=ports, groups=m['groups'],
        factors=dict(weights=list(qsets), visual_release=['graded', 'spiking']),
        limits='One-seed diagnostic. Only selected incoming weights and visual receptor release change. '
               'All other graded-trained state, including eligibility, queues and terminals, is retained. '
               'Release switch also changes subsequent receptor spike-dependent adaptation. '
               'This is not a complete acquisition-state transplant or a semantic recall test.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    rows = []
    for weights, q in qsets.items():
        for expression in ('graded', 'spiking'):
            for clip in (0, 1):
                branch = deepcopy(net)
                before, state = intervene(branch, ports, m['groups']['vision'], q, expression)
                if before != json.loads(parent):
                    raise ValueError('Clone differs from parent')
                name = f'{weights}-weights-{expression}-cue-{clip}'
                with gzip.open(output/f'{name}-start.json.gz', 'wt') as f:
                    f.write(encode(state)+'\n')
                members = list(branch.network.neurons.values())
                points = [p for n in members for p in n.postsynaptic_points.values()]
                t = branch.current_tick
                trial = dict(start=t, stop=t+len(features[clip]['visual']), visual_clip=clip, audio_clip=None)
                data = record(branch, TickDriver(branch), members, points, features, m['groups'], trial, ports)
                if weights == 'graded' and expression == 'graded':
                    p = next(p for p in summary['probes'] if p['state']=='continuation'
                             and p['sense']=='visual' and p['clip']==clip and p['gain']==1.)
                    with np.load(checked(source, p)) as z:
                        if any(not np.array_equal(data[k], z[k]) for k in z.files):
                            raise ValueError('No-op branch differs')
                pre = np.array([branch_cfg['metadata']['eligibility_ports'].index(sid)
                                for n, sid, _ in ports
                                for branch_cfg in cfg['neurons'] if branch_cfg['id']==n])
                p0 = np.array([state['eligibility'][str(n)]['pre'][i] for (n, _, _), i in zip(ports, pre)])
                y0 = np.array([state['eligibility'][str(n)]['post'] for n, _, _ in ports])
                previous = np.array([state['neurons'][str(n['id'])]['O'] > 0 for n in cfg['neurons']])
                result = verify_ledger(data, cfg, ports, q, p0, y0, previous)
                previous_m = np.array([state['neurons'][str(n['id'])]['M'][0] for n in cfg['neurons']])
                verify_rates(data['cells'], previous_m, cfg, ports)
                audio = data['cells'][:, np.array(m['groups']['tactile_core'])-1]
                if np.any(data['cells'][:, np.array(m['groups']['touch'])-1, 1]):
                    raise ValueError('Sound was supplied during silent cue')
                path = output/f'{name}.npz'; np.savez_compressed(path, **data)
                spikes = (audio[:, :, 1] > 0).sum(axis=1)
                row = dict(weights=weights, expression=expression, clip=clip, file=path.name,
                    sha256=digest(path), selected_update_residual=result[4],
                    auditory_spikes_by_tick=spikes.tolist(),
                    first_auditory_spike=int(np.flatnonzero(spikes)[0]) if spikes.any() else None,
                    auditory_max_s_minus_r_by_tick=(audio[:, :, 0]-audio[:, :, 5]).max(axis=1).tolist())
                rows.append(row)
                if dynamic_snapshot(net) != parent:
                    raise ValueError('Diagnostic changed acquisition parent')
                print(name, 'auditory spikes', int(spikes.sum()), 'first', row['first_auditory_spike'], flush=True)
    if any(digest(p) != h for p, h in hashes.items()):
        raise ValueError('Runtime changed during experiment')
    result = dict(exact_acquisition_replay=True, no_op_probes_exact=True, parent_unchanged=True,
        ticks=m['trials'][-1]['stop']+12*len(features[0]['visual']),
        seconds=time.perf_counter()-started, probes=rows)
    (output/'summary.json').write_text(encode(result)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'donor', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); run(a.source, a.donor, a.output)
