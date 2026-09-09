"""Physical disturbance x descending communication in the acquired population brain.

This is an intervention instrument, not a cognitive controller. The force
schedule is an external experimental disturbance. The optional cut removes
arriving information at declared upper-to-sensory ports, before native input
processing. It leaves modulation and already queued dendritic currents alone.
Information-triggered learning and retrograde messages consequently change too;
this is NOT a forward-current-only lesion. All basal adaptation stays positive.
"""
import argparse
from collections import defaultdict
import inspect
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .association_route_probe import digest
from .composition_probe import encode
from .predictive_bridge_probe import audit_record
from .proprioceptive_learning_probe import BranchNetwork
from .temporal_body_probe import record_history, audit_history
from ..components.body.radian_research_rower import RadianResearchRower
from ..core.runtime_checkpoint import load_checkpoint, save_checkpoint
from neuron.extensions.experimental.predictive_receptor import PredictiveReceptorNeuron
from neuron.neuron import setup_neuron_logger


def feedback_edges(cfg):
    roles = {n['id']: n['metadata'].get('role') for n in cfg['neurons']}
    order = {n['id']: i for i, n in enumerate(cfg['neurons'])}
    edges = [e for e in cfg['connections'] if roles[e['source_neuron']] == 'upper_core'
             and roles[e['target_neuron']] in ('visual_core', 'tactile_core')]
    edges.sort(key=lambda e: (order[e['target_neuron']], e['target_synapse']))
    if not edges or len({(e['target_neuron'], e['target_synapse']) for e in edges}) != len(edges):
        raise ValueError('Need unambiguous descending ports')
    if any(e['source_terminal'] != 900 for e in edges):
        raise ValueError('Recorder requires the declared standard terminal')
    return edges


class DisturbedRower(RadianResearchRower):
    """A scheduled external joint torque, independent of all neural signals."""
    def __init__(self, torque=0., start=64, stop=112):
        super().__init__()
        if not np.isfinite(torque) or abs(torque) > 1 or not 0 <= start < stop:
            raise ValueError('Invalid bounded force intervention')
        self.torque, self.start, self.stop = torque, start, stop
        self.elapsed = 0
        self.forces = []
        self.force_dof = int(self.model.jnt_dofadr[self.joint_ids[0]])

    def step(self, muscle_state, *, gain=.08):
        self.data.qfrc_applied[:] = 0.
        if self.start <= self.elapsed < self.stop:
            self.data.qfrc_applied[self.force_dof] = self.torque
        self.forces.append(self.data.qfrc_applied.copy())
        super().step(muscle_state, gain=gain)
        self.elapsed += 1


def record_feedback(net, body, manifest, cfg, ticks, *, cut=False):
    edges = feedback_edges(cfg)
    by_target = defaultdict(list)
    for j, e in enumerate(edges):
        by_target[e['target_neuron']].append((j, e['target_synapse']))
    last = edges[-1]['target_neuron']
    row_before = np.zeros((len(edges), 4), dtype=np.float32)
    row_after = row_before.copy()
    before, after = [], []
    original = PredictiveReceptorNeuron.tick

    def observe(cell, external_inputs, tick, dt=1.):
        if cell.id in by_target:
            for j, sid in by_target[cell.id]:
                row_before[j] = cell.input_buffer[sid]
                if cut:
                    cell.input_buffer[sid, 0] = 0.
                row_after[j] = cell.input_buffer[sid]
            if cell.id == last:
                before.append(row_before.copy()); after.append(row_after.copy())
        return original(cell, external_inputs, tick, dt)

    PredictiveReceptorNeuron.tick = observe
    try:
        data = record_history(net, body, manifest['motor'], manifest['bridge'],
                              manifest['basis'], ticks, gain=manifest['gain'])
    finally:
        PredictiveReceptorNeuron.tick = original
    data.update(feedback_before=np.asarray(before), feedback_after=np.asarray(after),
        feedback_sources=np.array([e['source_neuron'] for e in edges]),
        feedback_targets=np.array([e['target_neuron'] for e in edges]),
        feedback_ports=np.array([e['target_synapse'] for e in edges]),
        applied_force=np.asarray(body.forces))
    return data


def audit_intervention(data, cfg, *, cut, torque, gain, start=64, stop=112):
    """Verify delivered information, the precise cut and every physical step."""
    edges = feedback_edges(cfg)
    for name, key in (('sources', 'source_neuron'), ('targets', 'target_neuron'),
                      ('ports', 'target_synapse')):
        np.testing.assert_array_equal(data['feedback_'+name], [e[key] for e in edges])
    index = {int(n): i for i, n in enumerate(data['neuron_ids'])}
    sources = [index[e['source_neuron']] for e in edges]
    prev, terminal = data['start_cells'], data['start_terminals']
    # Independently use the plain body, not DisturbedRower.step.
    body = RadianResearchRower(); body.restore(data['physical_before'][0])
    dof = int(body.model.jnt_dofadr[body.joint_ids[0]])
    for t, cells in enumerate(data['cells']):
        expected = (prev[:, 1]*terminal)[sources].astype(np.float32)
        np.testing.assert_array_equal(data['feedback_before'][t, :, 0], expected)
        delivered = data['feedback_before'][t].copy()
        if cut:
            delivered[:, 0] = 0.
        np.testing.assert_array_equal(data['feedback_after'][t], delivered)
        np.testing.assert_array_equal(body.state(), data['physical_before'][t])
        np.testing.assert_array_equal(body.data.qpos[body.positions], data['joint_position'][t])
        scales = np.max(np.abs(body.model.jnt_range[body.joint_ids]), axis=1)
        v = np.clip(body.data.qpos[body.positions]/scales, -1., 1.)
        np.testing.assert_array_equal(data['joint_input'][t], np.maximum(0., [v[0], -v[0], v[1], -v[1]]))
        force = np.zeros(body.model.nv)
        if start <= t < stop:
            force[dof] = torque
        np.testing.assert_array_equal(data['applied_force'][t], force)
        body.data.qfrc_applied[:] = force
        muscle = np.asarray(data['muscle_state'][t], dtype=float)
        np.testing.assert_array_equal(data['actuator_ctrl'][t], gain*np.maximum(0., muscle))
        body.step(muscle, gain=gain)
        np.testing.assert_array_equal(body.state(), data['physical_after'][t])
        prev, terminal = cells, data['terminals'][t]
    return 0.


def run(source, output, *, cut=False, torque=0., ticks=384):
    source, output = Path(source).resolve(), Path(output).resolve()
    if ticks < 192 or ticks > 512:
        raise ValueError('Bounded diagnostic requires 192..512 ticks')
    if shutil.disk_usage(output.parent).free < 2*1024**3:
        raise OSError('Keep 2 GiB free')
    m = json.loads((source/'manifest.json').read_text())
    cfg = json.loads((source/'config.json').read_text())
    for p, h in m['source_hashes'].items():
        if digest(p) != h:
            raise ValueError('Parent source changed: '+p)
    setup_neuron_logger('CRITICAL')
    branch = load_checkpoint(source/'final.neural-checkpoint', trusted=True)
    for n in branch.network.network.neurons.values():
        if min(n.params.eta_post, n.params.eta_retro) <= 0:
            raise ValueError('Adaptation must remain positive')
    body = DisturbedRower(torque)
    with np.load(source/'closed-loop.npz') as z:
        body.restore(z['physical_after'][-1])
    if np.any(body.data.qfrc_applied):
        raise ValueError('Unexpected parent external force')
    hashes = {str(source/f): digest(source/f) for f in
              ('manifest.json', 'config.json', 'closed-loop.npz', 'final.neural-checkpoint')}
    for obj in (run, record_history, audit_record, RadianResearchRower, BranchNetwork):
        p = Path(inspect.getfile(obj)).resolve(); hashes[str(p)] = digest(p)
    manifest = dict(source=str(source), cut=cut, torque=torque, ticks=ticks, force_start=64,
        force_stop=112, source_hashes=hashes, neuron_ids=list(branch.network.network.neurons),
        fields=m['fields'], motor=m['motor'], bridge=m['bridge'], basis=m['basis'], gain=m['gain'],
        hypothesis='Upper-to-sensory information affects the response to a physical disturbance.',
        competing_explanation='Permissive gain, silent descending route, or no action influence.',
        decision='If no useful pathway effect appears, stop this diagnostic without gain tuning. '
                 'Activity differences alone cannot establish content-preserving regulation.',
        limits='One acquired graph and physical state. No audiovisual playback. No learned action task. '
               'Cut includes information-dependent return signals; modulation and queued dendritic current remain. '
               'Per-tick fields are not a complete intracellular record; complete endpoint runtimes are retained.')
    output.mkdir(exist_ok=False)
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    save_checkpoint(branch, output/'initial.neural-checkpoint', sources=(__file__,))
    started = time.perf_counter()
    data = record_feedback(BranchNetwork(branch), body, m, cfg, ticks, cut=cut)
    raw = output/'closed-loop.npz'; np.savez_compressed(raw, **data)
    result = dict(cut=cut, torque=torque, ticks=ticks, neurons=len(cfg['neurons']),
        intervention_residual=audit_intervention(data, cfg, cut=cut, torque=torque, gain=m['gain']),
        history_residual=audit_history(data, cfg, m['basis']),
        prediction_residual=audit_record(data, cfg, m['bridge']),
        raw_bytes=raw.stat().st_size, raw_sha256=digest(raw), seconds=time.perf_counter()-started)
    save_checkpoint(branch, output/'final.neural-checkpoint', sources=(__file__,))
    if any(digest(p) != h for p, h in hashes.items()):
        raise ValueError('Source changed during experiment')
    (output/'summary.json').write_text(encode(result)+'\n')
    print(encode(result), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--cut', action='store_true')
    p.add_argument('--torque', type=float, default=0.)
    p.add_argument('--ticks', type=int, default=384)
    run(**vars(p.parse_args()))
