"""Can predictive motor dynamics survive neural resource regulation and refeeding?

One acquired, continually plastic brain, three interacting timescales: motor
rhythm, learned sensorimotor prediction and slower physiological feedback.
No host rest/awake selector. Refeeding is a declared external intervention,
not autonomous food seeking. Energy failure is logged even if MuJoCo moves on.
"""
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
from .proprioceptive_learning_probe import BranchNetwork
from .sensory_motor_probe import record_correction, audit_motor, body_measures
from .hierarchical_body_perturbation import audit_intervention
from .predictive_bridge_probe import audit_record
from .temporal_body_probe import audit_history
from ..components.body.radian_research_rower import RadianResearchRower
from ..components.body.energy_budget import EnergyBudget, EnergyBudgetParameters
from ..components.body.metabolism import normalized_afferents
from ..components.arbitration.energy_feedback import append_energy_feedback, install_energy_feedback
from ..core.runtime_checkpoint import load_checkpoint, save_checkpoint
from neuron.extensions.experimental.predictive_receptor import PredictiveReceptorNeuron
from neuron.neuron import setup_neuron_logger


class BudgetRower(RadianResearchRower):
    def __init__(self):
        super().__init__()
        self.organs = EnergyBudget(energy_j=9., gut_j=0.)
        self.forces = []
        self.budget_rows = {k: [] for k in ('energy_after', 'positive_work_j', 'activation_squared_dt')}

    def step(self, muscle_state, *, gain=.08):
        self.data.qfrc_applied[:] = 0.
        self.forces.append(self.data.qfrc_applied.copy())
        super().step(muscle_state, gain=gain)
        velocity = self.data.qvel[self.model.jnt_dofadr[self.joint_ids]]
        ctrl = self.data.ctrl[self.actuators]
        power = ctrl*self.model.actuator_gear[self.actuators, 0]*velocity[[0, 0, 1, 1]]
        dt = self.model.opt.timestep
        work = float(np.maximum(power, 0).sum()*dt)
        activation = float(np.sum(np.maximum(0., np.asarray(muscle_state, dtype=float))**2)*dt)
        self.organs.advance(dt, work, activation)
        for key, value in zip(self.budget_rows, (self.organs.state(), work, activation)):
            self.budget_rows[key].append(value)


class EnergyInputs:
    def __init__(self, net, body, meta, *, fed):
        self.net, self.body, self.meta, self.fed = net, body, meta, fed
        self.elapsed = 0
        self.rows = {k: [] for k in ('energy_before', 'meal_j', 'energy_afferents')}

    @property
    def network(self):
        return self.net.network

    @property
    def current_tick(self):
        return self.net.current_tick

    def set_external_input(self, *args):
        return self.net.set_external_input(*args)

    def run_tick(self):
        organs = self.body.organs
        before = organs.state()
        meal = organs.ingest(48.) if self.fed and self.elapsed == 1024 else 0.
        _, energy, low, _ = normalized_afferents(organs)
        for key, value in (('energy', energy), ('deficit', low)):
            self.net.set_external_input(self.meta[key], 0, value)
        for key, value in zip(self.rows, (before, meal, [energy, low])):
            self.rows[key].append(value)
        self.elapsed += 1
        return self.net.run_tick()


def record_energy(net, body, m, cfg, ticks, *, fed):
    ids = m['energy_feedback']['neurons']; chosen = set(ids)
    original = PredictiveReceptorNeuron.tick
    rows = {k: [] for k in ('energy_cell_inputs', 'energy_cell_q_before', 'energy_cell_q_after', 'energy_cell_scheduled')}
    observed = {}
    def observe(n, ext, tick, dt=1.):
        if n.id not in chosen:
            return original(n, ext, tick, dt)
        vectors = n.input_buffer.copy()
        before = [p.u_i.info for p in n.postsynaptic_points.values()]
        events = original(n, ext, tick, dt)
        after = [p.u_i.info for p in n.postsynaptic_points.values()]
        scheduled = [p.potential if vectors[s, 0] > 0 else 0. for s, p in n.postsynaptic_points.items()]
        observed[n.id] = vectors, before, after, scheduled
        if n.id == ids[-1]:
            for col, key in enumerate(rows):
                rows[key].append([observed[nid][col] for nid in ids])
        return events
    wrapper = EnergyInputs(net, body, m['energy_feedback'], fed=fed)
    PredictiveReceptorNeuron.tick = observe
    try:
        data = record_correction(wrapper, body, m, cfg, ticks)
    finally:
        PredictiveReceptorNeuron.tick = original
    for values in (rows, wrapper.rows, body.budget_rows):
        data.update({key: np.asarray(value) for key, value in values.items()})
    return data


def audit_energy(data, cfg, m):
    """Reconstruct energetic accounting and all three new cell equations."""
    nodes = {n['id']: n for n in cfg['neurons']}
    idx = {int(n): i for i, n in enumerate(data['neuron_ids'])}
    ids = m['energy_feedback']['neurons']; loc = [idx[n] for n in ids]
    prev, terminal = data['start_cells'], data['start_terminals']
    due = np.zeros(3, dtype=np.float32)
    # New interoceptive cells have no pre-existing queued currents.
    np.testing.assert_array_equal(prev[loc, :2], 0.)
    p = m['energy_parameters']; dt = m['timestep']
    last = data['energy_before'][0].copy()
    qprev = None
    for t, cells in enumerate(data['cells']):
        np.testing.assert_array_equal(data['energy_before'][t], last)
        E, G, dig, cost, unmet, spill, ingested = last
        meal = min(48., p['gut_capacity_j']-G) if m['fed'] and t == 1024 else 0.
        np.testing.assert_array_equal(data['meal_j'][t], meal)
        G += meal; ingested += meal
        sensory = np.array([E/p['capacity_j'], max(0., 1.-E/p['capacity_j'])])
        np.testing.assert_array_equal(data['energy_afferents'][t], sensory)
        expected = np.zeros((3, 2, 4), dtype=np.float32)
        expected[:2, 0, 0] = sensory
        expected[2, :, 0] = (prev[:, 1]*terminal)[[loc[1], loc[0]]]
        np.testing.assert_array_equal(data['energy_cell_inputs'][t], expected)
        q = data['energy_cell_q_before'][t]
        if qprev is not None:
            np.testing.assert_array_equal(q, qprev)
        scheduled = expected[:, :, 0]*q.astype(np.float32)
        np.testing.assert_array_equal(data['energy_cell_scheduled'][t], scheduled)
        lam = np.array([nodes[n]['params']['lambda_param'] for n in ids], dtype=np.float32)
        s = prev[loc, 0].astype(np.float32)
        predicted = s+(1/lam)*(-s+due)
        np.testing.assert_array_equal(cells[loc, 0], predicted)
        np.testing.assert_array_equal(cells[loc, 1], np.maximum(0., predicted))
        qnext = q.copy()
        for j, n in enumerate(ids):
            for sid in np.flatnonzero(expected[j, :, 0] > 0):
                v = expected[j, sid].copy(); v[0] -= np.float32(q[j, sid])
                error = float(np.linalg.norm(v))
                qnext[j, sid] *= math.exp(-nodes[n]['params']['eta_post']*(error+.02))
        np.testing.assert_allclose(data['energy_cell_q_after'][t], qnext, rtol=0, atol=2e-12)
        # Native current multiplication and heap ordering, not a matrix sum.
        due = np.array([sum(v*nodes[n]['params']['delta_decay'] for v in sorted(row))
                        for n, row in zip(ids, scheduled)], dtype=np.float32)
        qprev = data['energy_cell_q_after'][t]
        work = float(np.maximum(data['sampled_actuator_power'][t], 0).sum()*dt)
        activation = float(np.sum(np.maximum(0., np.asarray(data['muscle_state'][t], dtype=float))**2)*dt)
        np.testing.assert_array_equal(data['positive_work_j'][t], work)
        np.testing.assert_array_equal(data['activation_squared_dt'][t], activation)
        d = min(G, p['digestion_w']*dt)
        demand = p['basal_w']*dt+work/p['efficiency']+p['activation_w']*activation
        available = E+d; deficit = max(0., demand-available)
        remainder = max(0., available-demand); overflow = max(0., remainder-p['capacity_j'])
        last = np.array([min(p['capacity_j'], remainder), G-d, dig+d, cost+demand,
                         unmet+deficit, spill+overflow, ingested])
        np.testing.assert_array_equal(data['energy_after'][t], last)
        prev, terminal = cells, data['terminals'][t]
    return 0.


def run(source, output, *, connected=True, fed=True, ticks=2048):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or ticks != 2048 or shutil.disk_usage(output.parent).free < 2*1024**3:
        raise ValueError('Need new output, a 2048-tick course and 2 GiB free')
    oldm = json.loads((source/'manifest.json').read_text())
    original = json.loads((source/'config.json').read_text())
    if oldm['mode'] != 'expectation' or oldm['torque'] != 0 or oldm['reset_prediction']:
        raise ValueError('Expected acquired unloaded prediction-feedback parent')
    if any(digest(p) != h for p, h in oldm['source_hashes'].items()):
        raise ValueError('Parent source changed')
    cfg, meta = append_energy_feedback(original, oldm['motor'], connected=connected)
    output.mkdir(exist_ok=False); path = output/'config.json'; path.write_text(encode(cfg)+'\n')
    setup_neuron_logger('CRITICAL')
    assembled, _, _, _ = fresh(path, 11, PredictiveReceptorNeuron)
    branch = load_checkpoint(source/'final.neural-checkpoint', trusted=True)
    install_energy_feedback(branch, assembled, original, cfg); del assembled
    body = BudgetRower()
    with np.load(source/'closed-loop.npz') as z:
        body.restore(z['physical_after'][-1])
    hashes = {str(source/f): digest(source/f) for f in ('manifest.json', 'config.json', 'closed-loop.npz', 'final.neural-checkpoint')}
    for obj in (run, append_energy_feedback, EnergyBudget, record_correction, normalized_afferents):
        path = Path(inspect.getfile(obj)).resolve(); hashes[str(path)] = digest(path)
    m = dict(oldm, source=str(source), source_hashes=hashes, energy_feedback=meta,
        energy_parameters=vars(body.organs.params), energy_fields=EnergyBudget.fields,
        timestep=body.model.opt.timestep, fed=fed, ticks=ticks,
        hypothesis='Continually adaptive predictive motor dynamics can reduce expenditure during scarcity and resume activity after digestion through neural feedback.',
        decision='One fixed circuit and budget; compare connected/disconnected under refed/unfed conditions. '
                 'Unmet energy is failure, and silence alone is not regulation. No tuning in this screen.',
        limits='Illustrative energy budget, not measured mammalian metabolism. No thermal or sleep mechanism. '
               'A nutrient bolus at tick 1024 is an external laboratory intervention, not autonomous feeding. '
               'No energy-dependent host motor brake; motion after unmet demand is not viable behavior.')
    (output/'manifest.json').write_text(encode(m)+'\n')
    save_checkpoint(branch, output/'initial.neural-checkpoint', sources=(__file__,))
    started = time.perf_counter()
    d = record_energy(BranchNetwork(branch), body, m, cfg, ticks, fed=fed)
    metrics = body_measures(d)
    raw = output/'closed-loop.npz'; np.savez_compressed(raw, **d)
    save_checkpoint(branch, output/'final.neural-checkpoint', sources=(__file__,))
    residuals = dict(energy=audit_energy(d, cfg, m), motor=audit_motor(d, cfg, m['motor']),
        history=audit_history(d, cfg, m['basis']), prediction=audit_record(d, cfg, m['bridge']),
        physics=audit_intervention(d, cfg, cut=False, torque=0., gain=m['gain'], start=256, stop=304))
    if any(digest(p) != h for p, h in hashes.items()):
        raise ValueError('Source changed during recording')
    result = dict(ticks=ticks, connected=connected, fed=fed, neurons=len(cfg['neurons']),
        metrics=metrics, residuals=residuals, final_organs=d['energy_after'][-1],
        raw_bytes=raw.stat().st_size, raw_sha256=digest(raw), seconds=time.perf_counter()-started)
    (output/'summary.json').write_text(encode(result)+'\n'); print(encode(result), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--disconnected', action='store_false', dest='connected')
    p.add_argument('--unfed', action='store_false', dest='fed')
    run(**vars(p.parse_args()))
