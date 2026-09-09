"""Paired embodied acquisition with matched or mismatched verification timing.

All bodily afferents have a continuous 64-tick physical delay during acquisition
and probing. The only paired architecture difference is the extra dendritic
delay on prediction-to-comparison paths and its birth attenuation compensation.
Both use quarter-strength mixed sensory drive, half-strength comparisons and a
64-tick exponential context trace. No host error or temporal teaching gate.
"""
import argparse
import inspect
import math
from pathlib import Path
import shutil
import time

import numpy as np

from . import context_organization as base
from .opponent_context import AfferentDelay,course,verify_afferents
from ..components.learning.opponent_prediction import couple_opponent_predictions
from ..components.learning.temporal_verification import configure_verification


def configure(seed=11,aligned=True,width=192):
    cfg,g,s=base.configure(seed,width=width)
    cfg=couple_opponent_predictions(cfg,g,g['force'])
    return configure_verification(cfg,g,prediction_lag=64 if aligned else 0),g,s


def verify_learning(data):
    q=data['weights_initial'].copy();x=data['context_initial'].copy();e=data['error_initial'].copy()
    dx=math.exp(-1/64);de=math.exp(-1/4);residual=0.
    for t,a in enumerate(data['arrivals']):
        x=dx*x+(1-dx)*a
        rate=1e-5*(1+499*abs(e)/(.01+abs(e)))
        q=np.minimum(1.,np.maximum(0.,q+rate[:,None]*e[:,None]*x))
        residual=max(residual,float(abs(q-data['weights'][t]).max()),
            float(abs(rate-data['eta'][t]).max()),float(abs(e-data['errors'][t,:,0]).max()))
        e=de*e+(1-de)*data['errors'][t,:,1]
        residual=max(residual,float(abs(e-data['errors'][t,:,2]).max()))
    if residual>2e-12 or np.any(data['eta']<=0):
        raise ValueError(f'Temporal learning audit failed: {residual}')
    return residual


def run(output,seed=11,aligned=True,order=0):
    output=Path(output).resolve()
    if order not in (0,1):raise ValueError('Order must be 0 or 1')
    if shutil.disk_usage(output.parent).free<3*1024**3:raise OSError('Need 3 GiB reserve')
    source=Path(base.__file__).resolve().parents[3]/'.live/research/20260908_bounded_learning_paired_seed11'
    features=[];physical_sources={}
    for clip in (0,1):
        path=source/f'sensory-{clip}.npz';physical_sources[str(path)]=base.digest(path)
        with np.load(path) as z:features.append({k:z[k] for k in ('visual','auditory')})
    cfg,groups,selected=configure(seed,aligned)
    output.mkdir(exist_ok=False);(output/'config.json').write_text(base.encode(cfg)+'\n')
    hashes=base.fingerprint()
    for obj in (run,configure_verification,couple_opponent_predictions,course,base.run,
                base.append_predictive_bridge,base.cellular,base.fresh):
        p=Path(inspect.getfile(obj)).resolve();hashes[str(p)]=base.digest(p)
    m=dict(seed=seed,aligned=aligned,order=order,groups=groups,selected=selected,
        source_hashes=hashes,physical_sources=physical_sources,cell_fields=base.FIELDS,
        xml=base.XML,neural_tick_seconds=base.DT,physical_delay=64,
        construction=cfg['metadata']['temporal_verification'],
        limitations='Two real clips, externally signalled context, one joint. No semantics, autonomous context '
        'inference or consciousness claim. Selected q reset retains all other acquired state and active learning. '
        'Matched timing is supplied anatomy. Long exponential credit is not exact delayed eligibility.')
    (output/'manifest.json').write_text(base.encode(m)+'\n')
    net,_,_,_=base.fresh(output/'config.json',seed,base.PredictiveReceptorNeuron)
    arm=base.Arm();delay=AfferentDelay(64);began=time.perf_counter();training=[];probes=[]
    def record(data,name):
        if shutil.disk_usage(output).free<3*1024**3:raise OSError('Storage reserve reached')
        np.savez_compressed(output/name,**data)
        return dict(file=name,sha256=base.digest(output/name),physics_residual=base.verify_physics(data),
                    learning_residual=verify_learning(data),afferent_residual=verify_afferents(data))
    base.save_checkpoint(net,output/'initial.paula',sources=[__file__])
    for context in (0,1):
        for repeat in range(4):
            for clip in ((0,1) if (repeat+order)%2==0 else (1,0)):
                data=course(net,arm,features,groups,selected,context,clip,delay=delay)
                training.append(record(data,f'train-c{context}-r{repeat}-v{clip}.npz'))
            print(base.encode(dict(stage='train',context=context,repeat=repeat,
                                  seconds=time.perf_counter()-began)),flush=True)
    base.save_checkpoint(net,output/'final.paula',sources=[__file__])
    np.savez_compressed(output/'final-body.npz',state=arm.state(),delay=delay.state())
    for reset in (False,True):
        for context in (0,1):
            for clip in (0,1):
                branch=base.load_checkpoint(output/'final.paula',trusted=True).network
                body=base.Arm();body.restore(arm.state())
                if reset:
                    for nid,sid,_ in selected:branch.network.neurons[nid].postsynaptic_points[sid].u_i.info=0.
                data=course(branch,body,features,groups,selected,context,clip,ticks=364,
                            delay=AfferentDelay(64,delay.state()))
                probes.append(record(data,f'probe-r{int(reset)}-c{context}-v{clip}.npz'))
        print(base.encode(dict(stage='probe',reset=reset,seconds=time.perf_counter()-began)),flush=True)
    if any(base.digest(p)!=h for p,h in hashes.items()):raise ValueError('Source changed during run')
    result=dict(training=training,probes=probes,ticks=24*364,seconds=time.perf_counter()-began,
                neurons=len(net.network.neurons))
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(stage='complete',ticks=result['ticks'],seconds=result['seconds'])),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=11)
    p.add_argument('--order',type=int,choices=(0,1),default=0);p.add_argument('--unmatched',action='store_true')
    a=p.parse_args();run(a.output,a.seed,not a.unmatched,a.order)
