"""Connect the learned evidence bank and consumer in one PAULA network.

The original 368 cells retain their classes and incoming wiring. The 65-cell
consumer uses existing graded and ordinary PAULA cells. Actual synaptic
connections replace the isolated replay ports, including native retrograde
adaptation of the source terminals. This is neural integration, not embodiment.
"""
import argparse
from copy import deepcopy
import gzip
import inspect
import json
from pathlib import Path
import time

import numpy as np

from . import eligibility_association_probe as acquisition
from .association_balance_audit import checked
from .association_evidence_population import evidence_config, record as record_bank, test_schedules
from .association_route_probe import digest
from .composition_probe import encode, fingerprint, snapshot, k
from .evidence_consumer import consumer_config
from .evidence_consumer_audit import ConsumerAudit
from .multimodal_pairing_probe import fresh
from .sensory_schedule import ScheduledReceptors
from neuron.extensions.graded import GradedNeuron
from neuron.extensions.experimental.port_modulation import PortModulationNeuron

OFFSET = 368


def embedded_config(seed):
    cfg, groups = evidence_config(seed, 'diverse')
    consumer = consumer_config('evidence_slow')
    for n in consumer['neurons']:
        n['id'] += OFFSET
    for p in consumer['synaptic_points']:
        p['neuron_id'] += OFFSET
    for e in consumer['connections']:
        e['source_neuron'] += OFFSET
        e['target_neuron'] += OFFSET
    cfg['neurons'].extend(consumer['neurons'])
    cfg['synaptic_points'].extend(consumer['synaptic_points'])
    cfg['connections'].extend(consumer['connections'])
    for coordinate, sources in enumerate(groups['evidence_coordinates']):
        for port, src in enumerate(sources):
            cfg['connections'].append(k.conn(src, OFFSET+coordinate+1, port))
    # No external consumer inputs and no new instruction/label channels.
    groups['temporal_integrators'] = list(range(369, 401))
    groups['mean_inhibition'] = [401]
    groups['contrast'] = list(range(402, 434))
    cfg['metadata']['preparation'] = 'connected-evidence-consumer-v0'
    return cfg, groups


def cell_factory(neuron_id, params, **kwargs):
    cls = PortModulationNeuron if neuron_id <= OFFSET else GradedNeuron
    return cls(neuron_id, params, **kwargs)


def full_snapshot(net):
    value = snapshot(net)
    value['eligibility'] = {
        str(n.id): dict(pre=n.eligibility_pre, post=n.eligibility_post,
                       last_tick=n.eligibility_last_tick, updates=n.eligibility_updates)
        for n in net.network.neurons.values() if isinstance(n, PortModulationNeuron)
    }
    return encode(value)


def record(net, groups, masks, trial):
    """Passive taps retain the old bank schema and a separate consumer ledger."""
    original_tick = GradedNeuron.tick
    original_cellular = acquisition.cellular
    captures, order, states, terminals, bank_terminals = [], [], [], [], []
    def observed(n, ext, tick, dt=1.):
        if net.network.neurons.get(n.id) is not n:
            return original_tick(n, ext, tick, dt)
        ports = list(n.postsynaptic_points.values())
        before = [p.u_i.info for p in ports]
        incoming = n.input_buffer[:, 0].copy()
        events = original_tick(n, ext, tick, dt)
        captures.append((incoming, before, [p.u_i.info for p in ports]))
        order.append(n.id)
        return events
    def cellular(neurons):
        lower, upper = neurons[:OFFSET], neurons[OFFSET:]
        states.append([[n.S, n.O, n.F_avg, *n.M_vector, n.r, n.b, n.t_ref] for n in upper])
        terminals.append([n.presynaptic_points[900].u_o.info for n in upper])
        bank_terminals.append([n.presynaptic_points[900].u_o.info for n in lower[176:]])
        return original_cellular(lower)
    GradedNeuron.tick = observed
    acquisition.cellular = cellular
    try:
        lower = record_bank(net, groups, masks, trial)
    finally:
        GradedNeuron.tick = original_tick
        acquisition.cellular = original_cellular
    if order != list(range(369, 434))*trial['ticks']:
        raise AssertionError('Unexpected consumer observation order')
    upper = dict(states=np.asarray(states), terminal_info=np.asarray(terminals))
    for j, key in enumerate(('incoming', 'before', 'after')):
        upper[key] = np.array([np.concatenate([x[j] for x in captures[t:t+65]])
                               for t in range(0, len(captures), 65)])
    upper['source'] = (lower['states'][:, 176:, 1] > 0)*np.asarray(bank_terminals)
    lower['bank_terminal_info'] = np.asarray(bank_terminals)
    return lower, upper


def compare_lower(actual, source):
    for key in source:
        if not np.array_equal(actual[key], source[key]):
            raise AssertionError(f'Connected source population changed: {key}')


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    m = json.loads((source/'manifest.json').read_text())
    s = json.loads((source/'summary.json').read_text())
    if m['architecture'] != 'diverse' or not s['baseline_exact']:
        raise ValueError('Requires completed diverse source')
    if any(digest(p) != h for p, h in m['source_hashes'].items()):
        raise ValueError('Source runtime changed')
    cfg, groups = embedded_config(m['seed'])
    output.mkdir(parents=True, exist_ok=False)
    path = output/'config.json'; path.write_text(encode(cfg)+'\n')
    hashes = fingerprint()
    for fn in (run, record_bank, consumer_config, ConsumerAudit, acquisition.run_trial):
        p = Path(inspect.getfile(fn)).resolve(); hashes[str(p)] = digest(p)
    manifest = dict(source=str(source), seed=m['seed'], mapping=m['mapping'], groups=groups,
        masks=m['masks'], source_hashes=hashes,
        limits='433-cell neural integration only, not body integration. Coordinate-preserving fixed consumer, not a learned upper association. Graded cells retain weak native adaptation with its non-spiking limitations. Actual source-terminal back-action remains active.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    net, *_ = fresh(path, m['seed'], cell_factory)
    auditor = ConsumerAudit(consumer_config('evidence_slow'))
    initial = json.loads(full_snapshot(net)); schedules = test_schedules(m['masks'], m['seed'])
    training, probes, starts = [], [], {}; started = time.perf_counter()
    def save(name, lower, upper, audit):
        audit.check(upper)
        np.savez_compressed(output/name, **lower, **{'consumer_'+k: v for k, v in upper.items()})
        return dict(file=name, sha256=digest(output/name), audit_residual=audit.max_error)
    for i, trial in enumerate(m['trials']):
        lower, upper = record(net, groups, m['masks'], trial)
        with np.load(checked(source, s['training'][i])) as z: compare_lower(lower, z)
        training.append(save(f'train-{i:03d}.npz', lower, upper, auditor))
        cp = i+1
        if cp not in (32, 128): continue
        parent = full_snapshot(net); name = f'{cp}-start.json.gz'
        with gzip.open(output/name, 'wt') as f: f.write(parent+'\n')
        starts[str(cp)] = dict(file=name, sha256=digest(output/name))
        for p in s['probes']:
            if p['checkpoint'] != cp: continue
            branch = deepcopy(net)
            if p['state'] == 'reset_evidence':
                for nid in groups['evidence']:
                    for sid in range(32):
                        branch.network.neurons[nid].postsynaptic_points[sid].u_i.info = initial['neurons'][str(nid)]['synapses'][str(sid)][0]
            schedule = schedules[p['case']]
            lower, upper = record(ScheduledReceptors(branch, schedule), groups, m['masks'],
                                  dict(cue=None, sound=None, ticks=len(schedule)))
            with np.load(checked(source, p)) as z: compare_lower(lower, z)
            item = save(f'{cp}-{p["state"]}-{p["case"].replace("/", "--")}.npz', lower, upper, deepcopy(auditor))
            probes.append(dict(**item, checkpoint=cp, state=p['state'], case=p['case']))
        if full_snapshot(net) != parent: raise AssertionError('Probe changed acquisition parent')
        print(encode(dict(checkpoint=cp, seconds=time.perf_counter()-started)), flush=True)
    if any(digest(p) != h for p, h in hashes.items()): raise ValueError('Runtime changed')
    result = dict(training=training, probes=probes, starts=starts, lower_exact=True,
                  ticks=s['ticks'], seconds=time.perf_counter()-started)
    (output/'summary.json').write_text(encode(result)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.source, a.output)
