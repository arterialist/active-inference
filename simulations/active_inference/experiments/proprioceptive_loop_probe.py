"""Bounded motor-to-body-to-afferent experiment in the population brain.

The adapter only transduces actual joint position and muscle membrane state.
The existing neural recorder observes comparator and learning dynamics. No
decoded error, desired pose, stimulus class or host policy enters the network.
Physical integration state is recorded each tick, including solver warmstart.
"""
import argparse
import inspect
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np

from .association_route_probe import digest
from .composition_probe import encode, fingerprint
from .multimodal_pairing_probe import fresh
from .population_hierarchy import FIELDS
from .predictive_bridge_probe import record, audit_record
from .predictive_weight_transplant import install_weights
from ..components.body.research_rower import ResearchRower
from ..components.motor.proprioceptive_rower import append_proprioceptive_rower
from ..core.runtime_checkpoint import save_checkpoint
from neuron.extensions.experimental.predictive_receptor import PredictiveReceptorNeuron


class PhysicalLoop:
    """One sensed state -> one neural tick -> one physical integration step."""
    def __init__(self, net, body, motor, *, gain=8.):
        self.net, self.body, self.motor, self.gain = net, body, motor, gain
        self.rows = {key: [] for key in ('physical_before', 'physical_after',
            'joint_input', 'joint_position', 'muscle_state', 'actuator_ctrl', 'birth_kick')}

    @property
    def network(self):
        return self.net.network

    @property
    def current_tick(self):
        return self.net.current_tick

    def set_external_input(self, *args):
        return self.net.set_external_input(*args)

    def run_tick(self):
        before = self.body.state()
        position = self.body.data.qpos[self.body.positions].copy()
        sensory = self.body.sense()
        for nid, value in zip(self.motor['joint_position'], sensory, strict=True):
            self.net.set_external_input(nid, 0, float(value))
        kick = 5. if self.current_tick == 3 else 0.
        if kick:
            self.net.set_external_input(self.motor['cpg'][0], 0, kick)
        result = self.net.run_tick()
        muscles = np.array([self.network.neurons[n].S for n in self.motor['muscles']])
        self.body.step(muscles, gain=self.gain)
        values = (before, self.body.state(), sensory, position, muscles,
                  self.body.data.ctrl[self.body.actuators].copy(), kick)
        for key, value in zip(self.rows, values, strict=True):
            self.rows[key].append(value)
        return result


def record_loop(net, body, motor, bridge, ticks, *, gain=8.):
    loop = PhysicalLoop(net, body, motor, gain=gain)
    trial = dict(start=net.current_tick, stop=net.current_tick+ticks,
                 visual_clip=None, audio_clip=None)
    data = record(loop, [], {}, bridge, trial)
    data.update({k: np.asarray(v) for k, v in loop.rows.items()})
    data['neuron_ids'] = np.array(list(net.network.neurons))
    data['ticks'] = np.arange(trial['start'], trial['stop'])
    return data


def audit_physical(data, *, gain):
    """Re-integrate every recorded control, checking full state and transduction."""
    body = ResearchRower()
    body.restore(data['physical_before'][0])
    for t in range(len(data['ticks'])):
        np.testing.assert_array_equal(body.state(), data['physical_before'][t])
        np.testing.assert_array_equal(body.data.qpos[body.positions], data['joint_position'][t])
        np.testing.assert_array_equal(body.sense(), data['joint_input'][t])
        expected = gain*np.maximum(0., data['muscle_state'][t])
        np.testing.assert_array_equal(expected, data['actuator_ctrl'][t])
        body.step(data['muscle_state'][t], gain=gain)
        np.testing.assert_array_equal(body.state(), data['physical_after'][t])
    return 0.


def run(source, transplant, output, *, ticks=640, gain=8., seed=11):
    source, transplant, output = (Path(p).resolve() for p in (source, transplant, output))
    if not 64 <= ticks <= 1200 or not np.isfinite(gain) or gain < 0:
        raise ValueError('Expected bounded 64..1200 tick preflight and nonnegative gain')
    old = json.loads((source/'manifest.json').read_text())
    original = json.loads((source/'config.json').read_text())
    if len(original['neurons']) != 1761:
        raise ValueError('Expected retained 1761-cell audiovisual graph')
    for p, h in old['source_hashes'].items():
        if digest(p) != h:
            raise ValueError('Acquisition source changed: '+p)
    cfg, motor, bridge, selected = append_proprioceptive_rower(
        original, old['groups']['tactile_core'], seed=seed)
    if shutil.disk_usage(output.parent).free < 2*1024**3:
        raise OSError('Keep at least 2 GiB free before this recording')
    output.mkdir(exist_ok=False)
    path = output/'config.json'; path.write_text(encode(cfg)+'\n')
    net, _, members, _ = fresh(path, seed, PredictiveReceptorNeuron)
    with np.load(transplant) as z:
        install_weights(net, old['bridge'], z['start_weights'])
    body = ResearchRower()
    hashes = fingerprint()
    for obj in (run, append_proprioceptive_rower, ResearchRower, record, install_weights, fresh):
        p = Path(inspect.getfile(obj)).resolve(); hashes[str(p)] = digest(p)
    manifest = dict(source=str(source), source_config_sha256=digest(source/'config.json'),
        transplant=str(transplant), transplant_sha256=digest(transplant), source_hashes=hashes,
        neuron_ids=[n.id for n in members], fields=FIELDS, groups=old['groups'], motor=motor,
        bridge=bridge, selected=selected, gain=gain, seed=seed, ticks=ticks,
        mujoco_version=mujoco.__version__, state_signature=int(body.state_signature),
        joint_ranges_radians=body.model.jnt_range[body.joint_ids],
        timestep=body.model.opt.timestep,
        limits='Preflight, not autonomous action selection, mammalian cognition or consciousness. '
        'Only selected acquired audiovisual weights transfer; other cells start from config. '
        'No audiovisual playback in this proprioceptive experiment. Physical afferents enter '
        'the existing tactile core, not a separate disconnected graph. One neural tick per '
        'physics step differs from the historical reference default of six. No body policy. '
        'Full-cell display fields and selected predictor internals, not every intracellular field.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    save_checkpoint(net, output/'initial.neural-checkpoint', sources=(__file__,))
    started = time.perf_counter()
    data = record_loop(net, body, motor, bridge, ticks, gain=gain)
    # Persist before validating, including any unfavorable or failed trace.
    raw = output/'closed-loop.npz'; np.savez_compressed(raw, **data)
    neural_residual = audit_record(data, cfg, bridge)
    physical_residual = audit_physical(data, gain=gain)
    save_checkpoint(net, output/'final.neural-checkpoint', sources=(__file__,))
    scales = np.max(np.abs(body.model.jnt_range[body.joint_ids]), axis=1)
    exceed = np.abs(data['joint_position']) > scales
    index = {n:i for i,n in enumerate(data['neuron_ids'])}
    first = lambda a: int(np.flatnonzero(a)[0]) if np.any(a) else None
    result = dict(ticks=ticks, neurons=len(members), seconds=time.perf_counter()-started,
        raw_bytes=raw.stat().st_size, raw_sha256=digest(raw), neural_residual=neural_residual,
        physical_replay_residual=physical_residual,
        first_motor=first(np.any(data['actuator_ctrl'] != 0,axis=1)),
        first_joint_input=first(np.any(data['joint_input'] != 0,axis=1)),
        joint_limit_exceed_ticks=np.count_nonzero(exceed,axis=0),
        max_abs_joint_radians=np.max(np.abs(data['joint_position']),axis=0),
        first_tactile_core_release=first(np.any(data['cells'][:,[index[n] for n in old['groups']['tactile_core']],1]>0,axis=1)),
        max_weight_change=float(np.max(np.abs(data['weights']-data['start_weights']))))
    if any(digest(p)!=h for p,h in hashes.items()):
        raise ValueError('Runtime changed during recording')
    (output/'summary.json').write_text(encode(result)+'\n')
    print(encode(result),flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source','transplant','output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--ticks',type=int,default=640)
    p.add_argument('--gain',type=float,default=8.)
    p.add_argument('--seed',type=int,default=11)
    run(**vars(p.parse_args()))
