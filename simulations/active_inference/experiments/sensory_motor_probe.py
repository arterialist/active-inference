"""Bounded closed-loop test of direct and learned-expectation motor feedback."""
import argparse
import inspect
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from .association_route_probe import digest
from .composition_probe import encode
from .multimodal_pairing_probe import fresh
from .hierarchical_body_perturbation import DisturbedRower, record_feedback, audit_intervention
from .temporal_body_probe import audit_history
from .predictive_bridge_probe import audit_record
from .predictive_weight_transplant import install_weights
from .proprioceptive_learning_probe import BranchNetwork
from ..components.motor.sensory_correction import append_sensory_correction, install_on_runtime
from ..components.body.radian_research_rower import RadianResearchRower
from ..core.runtime_checkpoint import load_checkpoint, save_checkpoint
from neuron.extensions.experimental.predictive_receptor import PredictiveReceptorNeuron
from neuron.neuron import setup_neuron_logger


def record_correction(net, body, m, cfg, ticks):
    muscles = [net.network.neurons[n] for n in m['motor']['muscles']]
    ids = {n.id for n in muscles}; last = muscles[-1].id
    start_due = np.array([sum(v*n.params.delta_decay**n.distances[s]
        for t, _, v, s in n.propagation_queue if t <= net.current_tick) for n in muscles])
    rows = {k: [] for k in ('motor_input_vectors', 'motor_weights_before', 'motor_weights_after', 'motor_scheduled')}
    observed = {}; original = PredictiveReceptorNeuron.tick
    def observe(n, external, t, dt=1.):
        if n.id not in ids:
            return original(n, external, t, dt)
        arriving = n.input_buffer.copy()
        weights = [p.u_i.info for p in n.postsynaptic_points.values()]
        events = original(n, external, t, dt)
        after = [p.u_i.info for p in n.postsynaptic_points.values()]
        scheduled = [p.potential if arriving[s, 0] > 0 else 0. for s, p in n.postsynaptic_points.items()]
        observed[n.id] = arriving, weights, after, scheduled
        if n.id == last:
            for column, key in enumerate(rows):
                rows[key].append([observed[cell.id][column] for cell in muscles])
        return events
    PredictiveReceptorNeuron.tick = observe
    try:
        data = record_feedback(net, body, m, cfg, ticks)
    finally:
        PredictiveReceptorNeuron.tick = original
    data.update({key: np.asarray(value) for key, value in rows.items()})
    data['start_motor_due'] = start_due
    return data


def audit_motor(data, cfg, motor):
    nodes = {n['id']: n for n in cfg['neurons']}
    index = {int(n): i for i, n in enumerate(data['neuron_ids'])}
    edges = {(e['target_neuron'], e['target_synapse']): e['source_neuron'] for e in cfg['connections']}
    muscles = motor['muscles']; positions = [index[n] for n in muscles]
    count = nodes[muscles[0]]['params']['num_inputs']
    if any(nodes[n]['params']['num_inputs'] != count for n in muscles):
        raise ValueError('Expected equal motor port counts')
    for p in cfg['synaptic_points']:
        if p['type'] == 'postsynaptic' and p['neuron_id'] in muscles:
            if p['distance_to_hillock'] != 1 or p['u_i']['plast'] != 0:
                raise ValueError('Motor auditor requires unit-delay information ports')
    prev, terminal = data['start_cells'], data['start_terminals']
    due = data['start_motor_due'].astype(np.float32)
    lam = np.array([nodes[n]['params']['lambda_param'] for n in muscles], dtype=np.float32)
    residual = 0.
    def same(a, b, name):
        nonlocal residual
        error = float(np.max(np.abs(a-b))); residual = max(residual, error)
        if not np.allclose(a, b, atol=2e-12, rtol=0):
            raise ValueError(f'{name}: {error}')
    for t, cells in enumerate(data['cells']):
        release = prev[:, 1]*terminal
        arriving = np.zeros((4, count), dtype=np.float32)
        for j, n in enumerate(muscles):
            for sid in range(count):
                if (n, sid) in edges:
                    arriving[j, sid] = release[index[edges[n, sid]]]
        vectors = data['motor_input_vectors'][t].astype(np.float32)
        same(vectors[:, :, 0], arriving, 'motor transmission')
        same(vectors[:, :, 1:], np.zeros_like(vectors[:, :, 1:]), 'motor other channels')
        q = data['motor_weights_before'][t]
        if t:
            same(q, data['motor_weights_after'][t-1], 'motor weight continuity')
        scheduled = arriving*q.astype(np.float32)
        same(data['motor_scheduled'][t], scheduled, 'motor synaptic current')
        s = prev[positions, 0].astype(np.float32)
        expected = s+(1/lam)*(-s+due)
        same(cells[positions, 0], expected, 'motor membrane')
        same(cells[positions, 1], np.maximum(expected, 0), 'graded motor release')
        after = q.copy()
        for j, n in enumerate(muscles):
            for sid in np.flatnonzero(arriving[j] > 0):
                v = vectors[j, sid].copy(); v[0] -= np.float32(q[j, sid])
                e = float(np.linalg.norm(v))
                after[j, sid] = q[j, sid]*math.exp(-nodes[n]['params']['eta_post']*(e+.02))
        same(data['motor_weights_after'][t], after, 'motor ongoing plasticity')
        # Reproduce native heap order rather than numpy's reassociated sum.
        due = np.array([sum(sorted(row)) for row in scheduled], dtype=np.float32)
        prev, terminal = cells, data['terminals'][t]
    return residual


def body_measures(data):
    body = RadianResearchRower()
    body.restore(data['physical_before'][0]); initial = body.data.qpos[:3].copy()
    forward = -np.array([np.cos(initial[2]), np.sin(initial[2])])
    pose, velocity, power = [], [], []
    for state, ctrl in zip(data['physical_after'], data['actuator_ctrl']):
        body.restore(state)
        v = body.data.qvel[body.model.jnt_dofadr[body.joint_ids]].copy()
        pose.append(body.data.qpos[:3].copy()); velocity.append(v)
        power.append(ctrl*body.model.actuator_gear[body.actuators, 0]*v[[0, 0, 1, 1]])
    pose, velocity, power = map(np.asarray, (pose, velocity, power))
    data.update(torso_pose=pose, joint_velocity=velocity, sampled_actuator_power=power,
                forward_progress=(pose[:, :2]-initial[:2])@forward)
    return dict(forward_endpoint=float(data['forward_progress'][-1]),
        yaw_change=float(pose[-1, 2]-initial[2]),
        last164_joint_range=np.ptp(data['joint_position'][-164:], axis=0),
        max_joint_abs=np.max(np.abs(data['joint_position']), axis=0),
        sampled_positive_work=float(np.maximum(power, 0).sum()*body.model.opt.timestep),
        absolute_control_integral=float(np.abs(data['actuator_ctrl']).sum()*body.model.opt.timestep),
        work_limit='Endpoint-sampled mechanical work estimate, not exact RK4 work or metabolic expenditure.')


def run(source, output, *, mode='position', torque=0., reset_prediction=False, ticks=640):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not 384 <= ticks <= 1024 or torque not in (0., .5) or (reset_prediction and mode != 'expectation'):
        raise ValueError('Invalid bounded comparison')
    if shutil.disk_usage(output.parent).free < 2*1024**3:
        raise OSError('Keep 2 GiB free')
    oldm = json.loads((source/'manifest.json').read_text())
    original = json.loads((source/'config.json').read_text())
    for p, h in oldm['source_hashes'].items():
        if digest(p) != h:
            raise ValueError('Parent source changed: '+p)
    cfg, correction = append_sensory_correction(original, oldm['motor'], oldm['bridge'], mode=mode)
    output.mkdir(exist_ok=False); cfg_path = output/'config.json'; cfg_path.write_text(encode(cfg)+'\n')
    setup_neuron_logger('CRITICAL')
    assembled, _, _, _ = fresh(cfg_path, oldm['seed'], PredictiveReceptorNeuron)
    branch = load_checkpoint(source/'final.neural-checkpoint', trusted=True)
    install_on_runtime(branch, assembled, original, cfg)
    del assembled
    body = DisturbedRower(torque, start=256, stop=304)
    with np.load(source/'closed-loop.npz') as z:
        body.restore(z['physical_after'][-1])
        if reset_prediction:
            install_weights(branch.network, oldm['bridge'], z['start_weights'])
    hashes = {str(source/f): digest(source/f) for f in
              ('manifest.json', 'config.json', 'closed-loop.npz', 'final.neural-checkpoint')}
    for obj in (run, append_sensory_correction, record_feedback, audit_record, audit_history, RadianResearchRower):
        p = Path(inspect.getfile(obj)).resolve(); hashes[str(p)] = digest(p)
    m = dict(source=str(source), source_hashes=hashes, mode=mode, torque=torque,
        reset_prediction=reset_prediction, correction=correction, fields=oldm['fields'],
        motor=oldm['motor'], bridge=oldm['bridge'], basis=oldm['basis'], gain=oldm['gain'],
        force_start=256, force_stop=304, ticks=ticks, neuron_ids=list(branch.network.network.neurons),
        hypothesis='Sensory-dependent motor current can reject load while preserving rhythmic movement; '
                   'learned expectations may preserve movement better than a centering reflex.',
        decision='No parameter search in this screen. Report baseline movement, load response, work and all failures. '
                 'If direct reflex suffices, no hierarchical necessity is inferred.',
        limits='Same acquired parent; graph delta changes motor input count and return paths in every mode. '
               'No new neuron equation, audiovisual playback or learned upper controller. One graph/state only.')
    (output/'manifest.json').write_text(encode(m)+'\n')
    save_checkpoint(branch, output/'initial.neural-checkpoint', sources=(__file__,))
    started = time.perf_counter()
    data = record_correction(BranchNetwork(branch), body, m, cfg, ticks)
    metrics = body_measures(data)
    raw = output/'closed-loop.npz'; np.savez_compressed(raw, **data)
    result = dict(mode=mode, torque=torque, reset_prediction=reset_prediction,
        neurons=len(cfg['neurons']), ticks=ticks, metrics=metrics,
        motor_residual=audit_motor(data, cfg, m['motor']),
        physical_residual=audit_intervention(data, cfg, cut=False, torque=torque, gain=m['gain'], start=256, stop=304),
        history_residual=audit_history(data, cfg, m['basis']),
        prediction_residual=audit_record(data, cfg, m['bridge']),
        raw_sha256=digest(raw), raw_bytes=raw.stat().st_size, seconds=time.perf_counter()-started)
    save_checkpoint(branch, output/'final.neural-checkpoint', sources=(__file__,))
    if any(digest(p) != h for p, h in hashes.items()):
        raise ValueError('Source changed')
    (output/'summary.json').write_text(encode(result)+'\n'); print(encode(result), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--mode', choices=('none', 'position', 'expectation'), default='position')
    p.add_argument('--torque', type=float, default=0.)
    p.add_argument('--reset-prediction', action='store_true')
    p.add_argument('--ticks', type=int, default=640)
    run(**vars(p.parse_args()))
