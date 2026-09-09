"""Preflight a continuous action-dependent learning and physical sweep loop.

Four matched birth-graph conditions: free/fused, resistive/fused,
resistive/sensory-only context, and resistive/fused with actuator transmission
cut. No gate score, correct direction, identity or host prediction is supplied
to neurons. Real media, delayed physical receptors and the neural birth kick
are the only external inputs. Every condition keeps positive adaptation.
"""
import argparse
from collections import deque
import inspect
import json
from pathlib import Path
import shutil

import numpy as np

from . import context_organization as base
from .temporal_verification import configure as original_configure, verify_learning
from .magnitude_feedback_probe import ReleaseObserver
from .magnitude_feedback_analysis import verify_returns
from ..components.body.loaded_hinge import LoadedHinge, afferents, XML, DT, MOTOR_GEAR
from ..components.motor.active_sweep import append_active_sweep
from neuron.extensions.experimental.magnitude_retrograde import MagnitudeRetrogradeNeuron


CONDITIONS = {'free_fused': (0., True, 1.), 'loaded_fused': (.8, True, 1.),
              'loaded_sensory': (.8, False, 1.), 'actuator_cut': (.8, True, 0.)}


def configure(seed=11, sensorimotor=True, width=192):
    cfg, g, selected = original_configure(seed, width=width)
    for n in cfg['neurons']:
        n['metadata']['retrograde_magnitude_error'] = True
    cfg, g = append_active_sweep(cfg, g, seed=seed, sensorimotor=sensorimotor)
    return cfg, g, selected


class PhysicalDelay:
    def __init__(self, initial=None):
        a = np.zeros((64, 6)) if initial is None else np.asarray(initial, dtype=float)
        if a.shape != (64, 6) or not np.isfinite(a).all():
            raise ValueError('Need sixty-four six-channel physical samples')
        self.queue = deque(row.copy() for row in a)

    def state(self):
        return np.asarray(self.queue)

    def step(self, value):
        a = np.asarray(value, dtype=float)
        if a.shape != (6,) or not np.isfinite(a).all():
            raise ValueError('Invalid physical sample')
        self.queue.append(a.copy())
        return self.queue.popleft()


def record(net, body, delay, features, groups, *, ticks=1024, coupling=1., observe=True):
    if coupling not in (0., 1.) or type(ticks) is not int or ticks < 1:
        raise ValueError('Invalid declared actuator intervention or duration')
    neurons = list(net.network.neurons.values())
    predictors = [net.network.neurons[n] for n in groups['prediction']]
    ext = sum((groups[r] for r in ('vision', 'audio', 'context', 'force', 'joint', 'velocity')), [])
    weights = lambda: np.array([[n.postsynaptic_points[s].u_i.info for s in n.prediction_ports] for n in predictors])
    initial = dict(body_initial=body.state(), delay_initial=delay.state(), weights_initial=weights(),
                   context_initial=np.array([n.prediction_context for n in predictors]),
                   error_initial=np.array([n.prediction_error for n in predictors]),
                   gate_initial=np.array([body.crossings, body.next_gate]))
    rows = {k: [] for k in ('cells','body','physical_states','raw_afferents','drive','weights','arrivals',
                           'errors','eta','gate','neural_command','birth_input')}
    def advance():
        for _ in range(ticks):
            t = net.current_tick
            # One actual recorded audiovisual episode loops continuously in this
            # physical preflight. Media identity never selects the resistance.
            values = np.r_[features['visual'][t % 300], features['auditory'][t % 300], 1., 0., np.zeros(6)]
            raw = afferents(body); values[194:200] = delay.step(raw)
            for nid, value in zip(ext, values):
                net.set_external_input(nid, 0, float(value))
            kick = 5. if t == 0 else 0.
            if kick:
                net.set_external_input(groups['cpg'][0], 0, kick)
            net.run_tick()
            command = net.network.neurons[groups['muscle'][0]].O-net.network.neurons[groups['muscle'][1]].O
            applied = coupling*command
            force, crossed = body.step(applied)
            values = dict(cells=base.cellular(neurons), body=[body.data.time,body.data.qpos[0],body.data.qvel[0],applied,force],
                physical_states=body.state(), raw_afferents=raw, drive=values, weights=weights(),
                arrivals=[n.prediction_arrivals.copy() for n in predictors],
                errors=[[n.prediction_error_used,n.prediction_error_arrival,n.prediction_error] for n in predictors],
                eta=[n.prediction_eta for n in predictors], gate=[body.crossings,body.next_gate,crossed],
                neural_command=command,birth_input=kick)
            for key, value in values.items():
                rows[key].append(value)
    if observe:
        with ReleaseObserver(net, groups['context'][0]) as observer:
            advance()
        returns = observer.arrays()
    else:
        advance(); returns = {}
    data = dict(**{k:np.asarray(v) for k,v in rows.items()}, **initial, **returns,
                neuron_ids=np.array([n.id for n in neurons]), delay_final=delay.state(),
                physical_parameters=np.array([body.drag,body.spring,body.gate,coupling]))
    if any(not np.isfinite(a).all() for a in data.values()):
        raise ValueError('Nonfinite active-sweep trace')
    return data


def verify(data, groups, features):
    drag,spring,gate,coupling = data['physical_parameters']
    if coupling not in (0., 1.):
        raise ValueError('Invalid actuator coupling')
    body = LoadedHinge(drag, spring, gate)
    body.restore(data['body_initial'], next_gate=int(data['gate_initial'][1]), crossings=int(data['gate_initial'][0]))
    delay = PhysicalDelay(data['delay_initial']); ids = list(data['neuron_ids'])
    muscles = data['cells'][:, [ids.index(n) for n in groups['muscle']], base.FIELDS.index('O')]
    if not np.array_equal(muscles[:,0]-muscles[:,1], data['neural_command']):
        raise ValueError('Command differs from actual neural outputs')
    if not np.array_equal(coupling*data['neural_command'], data['body'][:,3]):
        raise ValueError('Actuator differs from declared transmission')
    initial_tick = round(float(body.data.time)/DT)
    for t,row in enumerate(data['body']):
        q,v = float(body.data.qpos[0]),float(body.data.qvel[0])
        torque = -spring*q-drag*v
        raw = np.maximum([torque/MOTOR_GEAR,-torque/MOTOR_GEAR,q/.05,-q/.05,v/.2,-v/.2],0.)
        if not np.array_equal(raw,data['raw_afferents'][t]):
            raise ValueError('Environmental force or physical transducer differs')
        delivered = delay.step(raw)
        expected = np.r_[features['visual'][(initial_tick+t)%300],features['auditory'][(initial_tick+t)%300],1.,0.,delivered]
        if not np.array_equal(expected,data['drive'][t]):
            raise ValueError('Actual sensory delivery differs')
        if data['birth_input'][t] != (5. if initial_tick+t == 0 else 0.):
            raise ValueError('Unexpected external motor drive')
        force,crossed = body.step(float(row[3]))
        expected = [body.data.time,body.data.qpos[0],body.data.qvel[0],row[3],force]
        if not np.array_equal(expected,row) or not np.array_equal(body.state(),data['physical_states'][t]):
            raise ValueError('Physical integration differs')
        if not np.array_equal([body.crossings,body.next_gate,crossed],data['gate'][t]):
            raise ValueError('Physical sweep count differs')
    if not np.array_equal(delay.state(),data['delay_final']):
        raise ValueError('Physical delay history differs')
    verify_learning(data)
    if 'terminal_info' in data:
        verify_returns(data)
    return 0.


def run(output, seed=11, ticks=1024):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if type(ticks) is not int or not 256 <= ticks <= 4096:
        raise ValueError('Use a bounded 256..4096 tick structural preflight')
    if shutil.disk_usage(output.parent).free < 3*1024**3:
        raise OSError('Need 3 GiB reserve')
    source = Path(base.__file__).resolve().parents[3]/'.live/research/20260908_bounded_learning_paired_seed11/sensory-0.npz'
    with np.load(source) as z:
        features = {k:z[k] for k in ('visual','auditory')}
    hashes = base.fingerprint()
    for obj in (run, original_configure, append_active_sweep, LoadedHinge, MagnitudeRetrogradeNeuron, ReleaseObserver):
        p = str(Path(inspect.getfile(obj)).resolve()); hashes[p] = base.digest(p)
    # Include the original motor builder whose graph is reused.
    from .. import nmrower2
    hashes[str(Path(nmrower2.__file__).resolve())] = base.digest(nmrower2.__file__)
    output.mkdir(); rows = []; groups = {}; configs = {}
    for condition,(drag,fused,coupling) in CONDITIONS.items():
        if shutil.disk_usage(output).free < 3*1024**3:
            raise OSError('Storage reserve reached')
        cfg,g,selected = configure(seed, fused); groups[condition] = g
        config = output/f'{condition}.json'; config.write_text(base.encode(cfg)+'\n')
        configs[condition] = base.digest(config)
        net,_,_,_ = base.fresh(config, seed, MagnitudeRetrogradeNeuron)
        base.save_checkpoint(net, output/f'{condition}-initial.paula', sources=list(hashes))
        body = LoadedHinge(drag); delay = PhysicalDelay()
        data = record(net, body, delay, features, g, ticks=ticks, coupling=coupling)
        verify(data, g, features)
        path = output/f'{condition}.npz'; np.savez_compressed(path, **data)
        checkpoint = output/f'{condition}-final.paula'
        base.save_checkpoint(net, checkpoint, sources=list(hashes))
        physical = output/f'{condition}-final-body.npz'
        np.savez_compressed(physical, state=body.state(), delay=delay.state(), gate=[body.crossings,body.next_gate])
        rows.append(dict(condition=condition,file=path.name,sha256=base.digest(path),
                         checkpoint=checkpoint.name,checkpoint_sha256=base.digest(checkpoint),
                         physical=physical.name,physical_sha256=base.digest(physical),
                         ticks=ticks,gate_crossings=body.crossings))
        print(base.encode(dict(seed=seed,condition=condition,ticks=ticks,crossings=body.crossings)), flush=True)
    if any(base.digest(p) != h for p,h in hashes.items()):
        raise ValueError('Source changed during preflight')
    manifest = dict(seed=seed,groups=groups,source_hashes=hashes,physical_source=str(source),
                    physical_sha256=base.digest(source),config_hashes=configs,conditions=CONDITIONS,
                    xml=XML,fields=base.FIELDS,limits=__doc__)
    (output/'manifest.json').write_text(base.encode(manifest)+'\n')
    result = dict(rows=rows,executed_ticks=4*ticks)
    (output/'summary.json').write_text(base.encode(result)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path);p.add_argument('--seed',type=int,default=11);p.add_argument('--ticks',type=int,default=1024)
    a=p.parse_args();run(a.output,a.seed,a.ticks)
