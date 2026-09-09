"""Counterbalanced signed-residual learning with delayed embodied feedback.

The afferent delay is a physical-channel intervention, not a neural teaching
lesion. Force and joint information are both delayed, while audiovisual/context
inputs remain current. The delay line carries actual acquisition history,
which the current predictor treats as evidence, not as missing data. Learning
continues and may erase a useful expectation. Every supplied channel is recorded.
"""
import argparse
from collections import deque
import inspect
import json
from pathlib import Path
import shutil
import time

import numpy as np

from . import context_organization as base
from ..components.learning.opponent_prediction import couple_opponent_predictions


class AfferentDelay:
    def __init__(self, ticks=0, initial=None):
        if type(ticks) is not int or not 0 <= ticks <= 128:
            raise ValueError('Afferent delay must be 0..128 integer ticks')
        self.ticks = ticks
        a = np.zeros((ticks, 4)) if initial is None else np.asarray(initial, dtype=float)
        if a.shape != (ticks, 4) or not np.isfinite(a).all():
            raise ValueError('Invalid afferent history')
        self.queue = deque(row.copy() for row in a)

    def state(self):
        return np.asarray(list(self.queue)).reshape(self.ticks, 4)

    def step(self, value):
        value = np.asarray(value, dtype=float)
        if value.shape != (4,) or not np.isfinite(value).all():
            raise ValueError('Invalid physical afferent sample')
        if not self.ticks:
            return value.copy()
        self.queue.append(value.copy())
        return self.queue.popleft()


def course(net, arm, features, groups, selected, context, clip, ticks=364, delay=None):
    delay = AfferentDelay() if delay is None else delay
    neurons = list(net.network.neurons.values())
    ext = sum((groups[r] for r in ('vision','audio','context','force','joint')), [])
    predictors = [net.network.neurons[i] for i in groups['prediction']]
    q = lambda: [[n.postsynaptic_points[s].u_i.info for s in n.prediction_ports] for n in predictors]
    initial = dict(body_initial=arm.state(), weights_initial=np.array(q()),
        context_initial=np.array([n.prediction_context.copy() for n in predictors]),
        error_initial=np.array([n.prediction_error for n in predictors]),
        delay_initial=delay.state())
    records = {key: [] for key in ('cells','body','drive','weights','arrivals','errors',
                                 'eta','physical_states','raw_afferents')}
    for rel in range(ticks):
        v, force = base.physical_drive(features, context, clip, rel)
        angle = float(arm.data.qpos[0])
        raw = np.array([v[194],v[195],max(0.,angle),max(0.,-angle)])
        v[194:198] = delay.step(raw)
        for nid, value in zip(ext, v):
            net.set_external_input(nid, 0, float(value))
        net.run_tick()
        command = (net.network.neurons[groups['muscle'][0]].O -
                   net.network.neurons[groups['muscle'][1]].O)
        arm.step(command, force)
        values = dict(cells=base.cellular(neurons),
            body=[arm.data.time,arm.data.qpos[0],arm.data.qvel[0],command,force],
            drive=v,weights=q(),arrivals=[n.prediction_arrivals.copy() for n in predictors],
            errors=[[n.prediction_error_used,n.prediction_error_arrival,n.prediction_error] for n in predictors],
            eta=[n.prediction_eta for n in predictors],physical_states=arm.state(),raw_afferents=raw)
        for key, value in values.items():
            records[key].append(value)
    data = {key: np.asarray(value) for key,value in records.items()}
    data.update(initial, neuron_ids=np.array([n.id for n in neurons]),
        prediction_ids=np.array(groups['prediction']), delay_final=delay.state(),
        context_source_ids=np.array([[src for target,_,src in selected if target==nid]
                                    for nid in groups['prediction']]))
    if any(not np.isfinite(a).all() for a in data.values()):
        raise ValueError('Nonfinite record')
    return data


def verify_afferents(data):
    delay = AfferentDelay(len(data['delay_initial']), data['delay_initial'])
    arm = base.Arm(); arm.restore(data['body_initial'])
    for i, row in enumerate(data['body']):
        angle = float(arm.data.qpos[0]); force = float(row[4])
        raw = np.array([max(0.,force/base.FORCE),max(0.,-force/base.FORCE),
                        max(0.,angle),max(0.,-angle)])
        if not np.array_equal(raw, data['raw_afferents'][i]):
            raise ValueError('Raw physical afferent differs')
        if not np.array_equal(delay.step(raw), data['drive'][i,194:198]):
            raise ValueError('Delayed physical afferent differs')
        arm.step(float(row[3]), force)
    if not np.array_equal(delay.state(),data['delay_final']):
        raise ValueError('Final afferent history differs')
    return 0.


def run(output, seed=11, opponent=True, order=0):
    output = Path(output).resolve()
    if order not in (0, 1):
        raise ValueError('Order must be 0 or 1')
    if shutil.disk_usage(output.parent).free < 3*1024**3:
        raise OSError('Need 3 GiB reserve')
    source = Path(base.__file__).resolve().parents[3]/'.live/research/20260908_bounded_learning_paired_seed11'
    features = []; physical_sources = {}
    for clip in (0,1):
        p = source/f'sensory-{clip}.npz'; physical_sources[str(p)] = base.digest(p)
        with np.load(p) as z:
            features.append({k:z[k] for k in ('visual','auditory')})
    cfg, groups, selected = base.configure(seed)
    if opponent:
        cfg = couple_opponent_predictions(cfg,groups,groups['force'])
    output.mkdir(exist_ok=False)
    (output/'config.json').write_text(base.encode(cfg)+'\n')
    hashes = base.fingerprint()
    for obj in (run,base.run,base.append_predictive_bridge,base.cellular,base.fresh,couple_opponent_predictions):
        p = Path(inspect.getfile(obj)).resolve(); hashes[str(p)] = base.digest(p)
    m = dict(seed=seed,opponent=opponent,order=order,groups=groups,selected=selected,
        source_hashes=hashes,physical_sources=physical_sources,cell_fields=base.FIELDS,
        xml=base.XML,neural_tick_seconds=base.DT,
        protocol='16 acquisition episodes, then identical-state probes with intact/reset selected q and 0/64 tick somatic delay.',
        limitations='Externally cued contexts, two recordings, one joint. Not a learned latent context or semantic recognition. '
        'Opposing errors change effective learning gain as well as credit. Delaying both somatic channels tests a new '
        'sensor timing condition; its prehistory is the actual last 64 acquisition samples. No learning freeze or host policy.')
    (output/'manifest.json').write_text(base.encode(m)+'\n')
    net,_,_,_ = base.fresh(output/'config.json',seed,base.PredictiveReceptorNeuron)
    arm = base.Arm(); began = time.perf_counter(); rows = []
    def record(data, name):
        if shutil.disk_usage(output).free < 3*1024**3:
            raise OSError('Storage reserve reached')
        np.savez_compressed(output/name,**data)
        return dict(file=name,sha256=base.digest(output/name),
            physics_residual=base.verify_physics(data),learning_residual=base.verify_learning(data),
            afferent_residual=verify_afferents(data))
    base.save_checkpoint(net,output/'initial.paula',sources=[__file__])
    for context in (0,1):
        for repeat in range(4):
            sequence = (0,1) if (repeat+order)%2==0 else (1,0)
            for clip in sequence:
                data = course(net,arm,features,groups,selected,context,clip)
                rows.append(record(data,f'train-c{context}-r{repeat}-v{clip}.npz'))
            print(base.encode(dict(stage='train',context=context,repeat=repeat,
                                  seconds=time.perf_counter()-began)),flush=True)
    base.save_checkpoint(net,output/'final.paula',sources=[__file__])
    afferent_history=data['raw_afferents'][-64:].copy()
    np.savez_compressed(output/'final-body.npz',state=arm.state(),afferent_history=afferent_history)
    probes = []
    for delay_ticks in (0,64):
        for reset in (False,True):
            for context in (0,1):
                for clip in (0,1):
                    branch = base.load_checkpoint(output/'final.paula',trusted=True).network
                    body = base.Arm();body.restore(arm.state())
                    if reset:
                        for nid,sid,_ in selected:
                            branch.network.neurons[nid].postsynaptic_points[sid].u_i.info=0.
                    data=course(branch,body,features,groups,selected,context,clip,
                                ticks=192,delay=AfferentDelay(delay_ticks,
                                    afferent_history if delay_ticks else None))
                    probes.append(record(data,f'probe-d{delay_ticks}-r{int(reset)}-c{context}-v{clip}.npz'))
        print(base.encode(dict(stage='probe',delay=delay_ticks,seconds=time.perf_counter()-began)),flush=True)
    if any(base.digest(p)!=h for p,h in hashes.items()):
        raise ValueError('Source changed during course')
    summary=dict(training=rows,probes=probes,ticks=16*364+16*192,
                 seconds=time.perf_counter()-began,neurons=len(net.network.neurons))
    (output/'summary.json').write_text(base.encode(summary)+'\n')
    print(base.encode(dict(stage='complete',ticks=summary['ticks'],seconds=summary['seconds'])),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--order',type=int,choices=(0,1),default=0)
    p.add_argument('--baseline',action='store_true')
    a=p.parse_args();run(a.output,a.seed,not a.baseline,a.order)
