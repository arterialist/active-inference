"""Test learned temporal placement in the existing embodied feedback loop.

Only incoming predictor weights are permuted, within each motor source and
membrane-timescale band. This preserves their distribution within those groups,
not the actual input current, which depends on timing and delay attenuation.
Birth weights were uniform. Thus this intervention rearranges acquired
deviations without changing source identity or adding/removing conductance.
The intervention is an offline causal experiment, never an agent controller.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .association_route_probe import digest
from .associative_mismatch_audit import contextual_weight_mask
from .composition_probe import encode
from .hierarchical_body_perturbation import DisturbedRower, audit_intervention
from .predictive_bridge_probe import audit_record
from .predictive_weight_transplant import install_weights
from .proprioceptive_learning_probe import BranchNetwork
from .sensory_motor_probe import record_correction, audit_motor, body_measures
from .temporal_body_probe import audit_history
from ..core.runtime_checkpoint import load_checkpoint, save_checkpoint, _serializer
from neuron.neuron import setup_neuron_logger


def temporal_permutation(cfg, bridge, weights, seed):
    """Return rearranged weights and an explicit source-column map per row."""
    nodes = {n['id']: n for n in cfg['neurons']}
    edges = {(e['target_neuron'], e['target_synapse']): e['source_neuron']
             for e in cfg['connections']}
    q = np.asarray(weights, dtype=float)
    if q.ndim != 2 or len(q) != len(bridge['prediction']) or not np.isfinite(q).all():
        raise ValueError('Invalid predictor matrix')
    rng = np.random.default_rng(seed)
    mapping = np.tile(np.arange(q.shape[1]), (q.shape[0], 1))
    for row, nid in enumerate(bridge['prediction']):
        ports = nodes[nid]['metadata']['prediction_ports']
        if len(ports) != q.shape[1]:
            raise ValueError('Prediction column mismatch')
        groups = {}
        for col, sid in enumerate(ports):
            src = nodes[edges[nid, sid]]
            md = src['metadata']
            key = md['history_source'], src['params']['lambda_param']
            groups.setdefault(key, []).append(col)
        for cols in groups.values():
            if len(cols) != 4:
                raise ValueError('Expected four delayed histories per source/timescale')
            delays = [nodes[edges[nid, ports[c]]]['metadata']['history_delay'] for c in cols]
            if len(set(delays)) != 4:
                raise ValueError('Delay identities are not unique')
            order = rng.permutation(cols)
            while np.array_equal(order, cols):
                order = rng.permutation(cols)
            mapping[row, cols] = order
    out = np.take_along_axis(q, mapping, axis=1)
    np.testing.assert_array_equal(np.sort(out, axis=1), np.sort(q, axis=1))
    return out, mapping


def payload_hash(branch):
    _, pickler = _serializer()
    stream = io.BytesIO()
    pickler(stream, protocol=5).dump(dict(network=branch.network,
        python_rng=branch.python_rng, numpy_rng=branch.numpy_rng))
    return hashlib.sha256(stream.getvalue()).hexdigest()


def intervene(branch, cfg, bridge, seed):
    net = branch.network
    points = [[net.network.neurons[n].postsynaptic_points[s]
               for s in net.network.neurons[n].prediction_ports] for n in bridge['prediction']]
    original_objects = [[p.u_i.info for p in row] for row in points]
    original = np.array(original_objects)
    q, mapping = temporal_permutation(cfg, bridge, original, seed)
    before = payload_hash(branch)
    install_weights(net, bridge, q)
    # Restoring only the original weight objects must recover the complete
    # serialized runtime, including queues, extension traces, aliases and RNG.
    for row, values in zip(points, original_objects):
        for p, value in zip(row, values):
            p.u_i.info = value
    if payload_hash(branch) != before:
        raise ValueError('Intervention changed another runtime field')
    install_weights(net, bridge, q)
    return original, q, mapping, before


def audit(data, cfg, m):
    return dict(motor=audit_motor(data, cfg, m['motor']),
        prediction=audit_record(data, cfg, m['bridge']),
        history=audit_history(data, cfg, m['basis']),
        physics=audit_intervention(data, cfg, cut=False, torque=m['torque'],
            gain=m['gain'], start=m['force_start'], stop=m['force_stop']))


def reference_prefix(key, value, ticks):
    static = {'neuron_ids', 'history_neuron_ids', 'feedback_sources',
              'feedback_targets', 'feedback_ports'}
    return value if key.startswith('start_') or key in static else value[:ticks]


def run(source, output, *, seed=23):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or shutil.disk_usage(output.parent).free < 2*1024**3:
        raise ValueError('Need a new output and 2 GiB free')
    m = json.loads((source/'manifest.json').read_text())
    cfg = json.loads((source/'config.json').read_text())
    summary = json.loads((source/'summary.json').read_text())
    if m['mode'] != 'expectation' or m['reset_prediction'] or m['ticks'] != 640:
        raise ValueError('Expected the unchanged acquired 640-tick expectation course')
    if any(digest(p) != h for p, h in m['source_hashes'].items()):
        raise ValueError('Parent source changed')
    raw = source/'closed-loop.npz'
    if digest(raw) != summary['raw_sha256']:
        raise ValueError('Reference changed')
    hashes = {str(source/f): digest(source/f) for f in
              ('manifest.json', 'config.json', 'summary.json', 'initial.neural-checkpoint', 'closed-loop.npz')}
    hashes[str(Path(__file__).resolve())] = digest(__file__)
    setup_neuron_logger('CRITICAL')
    with np.load(raw) as z:
        physical = z['physical_before'][0].copy()
        original_start = {k: z[k] for k in z.files if k.startswith('start_')}

        # Verify a real unmodified replay prefix before any new causal result.
        control = load_checkpoint(source/'initial.neural-checkpoint', trusted=True)
        body = DisturbedRower(m['torque'], start=m['force_start'], stop=m['force_stop'])
        body.restore(physical)
        check = record_correction(BranchNetwork(control), body, m, cfg, 32)
        for k, a in check.items():
            if k == 'end_incoming_info':
                continue  # Its endpoint is deliberately earlier.
            expected = reference_prefix(k, z[k], 32)
            np.testing.assert_array_equal(a, expected, err_msg='Replay '+k)
        audit(check, cfg, m)
    del control, check
    branch = load_checkpoint(source/'initial.neural-checkpoint', trusted=True)
    before, after, mapping, original_hash = intervene(branch, cfg, m['bridge'], seed)
    output.mkdir(exist_ok=False)
    (output/'config.json').write_text(encode(cfg)+'\n')
    manifest = dict(m, source=str(source), source_hashes=hashes,
        placement_seed=seed, initial_payload_before=original_hash,
        intervention='Incoming prediction q only, permuted within motor-source/lambda groups across four delays.',
        limits='One acquired graph/state. Four permutation seeds are NOT graph replications. '
               'Weight distributions match within each motor source and timescale, but effective input currents need not. '
               'No hierarchy, consciousness or general learned-control acceptance claim.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    np.savez_compressed(output/'intervention.npz', before=before, after=after, mapping=mapping)
    save_checkpoint(branch, output/'initial.neural-checkpoint', sources=(__file__,))
    body = DisturbedRower(m['torque'], start=m['force_start'], stop=m['force_stop']); body.restore(physical)
    started = time.perf_counter()
    data = record_correction(BranchNetwork(branch), body, m, cfg, m['ticks'])
    mask = contextual_weight_mask(cfg, m['bridge'])
    for k, a in original_start.items():
        if k == 'start_weights':
            np.testing.assert_array_equal(data[k], after)
        elif k == 'start_incoming_info':
            np.testing.assert_array_equal(data[k][~mask], a[~mask])
        else:
            np.testing.assert_array_equal(data[k], a, err_msg=k)
    metrics = body_measures(data)
    residuals = audit(data, cfg, m)
    path = output/'closed-loop.npz'; np.savez_compressed(path, **data)
    save_checkpoint(branch, output/'final.neural-checkpoint', sources=(__file__,))
    if any(digest(p) != h for p, h in hashes.items()):
        raise ValueError('Source changed while running')
    result = dict(placement_seed=seed, torque=m['torque'], neurons=len(cfg['neurons']),
        ticks=m['ticks'], replay_ticks=32, residuals=residuals, metrics=metrics,
        raw_bytes=path.stat().st_size, raw_sha256=digest(path), seconds=time.perf_counter()-started)
    (output/'summary.json').write_text(encode(result)+'\n')
    print(encode(result), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=23)
    run(**vars(p.parse_args()))
