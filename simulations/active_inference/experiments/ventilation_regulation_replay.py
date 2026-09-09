"""Executable initial-state replay of the complete neural/organ composition."""
import argparse
import json
from pathlib import Path
import random

import numpy as np

from . import context_organization as base
from .active_sweep_probe import PhysicalDelay
from .ventilation_regulation import CoupledHinge,OrganDelay,record
from .ventilation_regulation_audit import audit
from ..components.arbitration.ventilation_feedback import append_ventilation_feedback


def restore(neural,physical,groups):
    saved=base.load_checkpoint(neural,trusted=True)
    random.setstate(saved.python_rng);np.random.set_state(saved.numpy_rng)
    net=saved.network;body=CoupledHinge(net,groups)
    with np.load(physical) as z:
        body.restore(z['state'],crossings=int(z['gate'][0]),next_gate=int(z['gate'][1]))
        body.organs.restore(z['organ']);delay=PhysicalDelay(z['delay']);organ_delay=OrganDelay(z['organ_delay'])
    if abs(body.data.time-net.current_tick*.004)>1e-10: raise ValueError('Clock mismatch')
    return net,body,delay,organ_delay


def exact_prefix(full,prefix):
    if set(full)!=set(prefix): raise ValueError('Missing recorded fields')
    n=len(prefix['body'])
    if not 1<=n<=len(full['body']): raise ValueError('Empty or oversized replay')
    constants={'neuron_ids','terminal_ids','physical_parameters','reg_ids','cpg_ids','credit_mean','credit_decay','intervention'}
    for key,value in prefix.items():
        if key.endswith('_initial') or key in constants: expected=full[key]
        elif key=='delay_final':
            d=PhysicalDelay(full['delay_initial'])
            for raw in full['raw_afferents'][:n]:d.step(raw)
            expected=d.state()
        elif key=='organ_delay_final':
            d=OrganDelay(full['organ_delay_initial'])
            for raw in full['organ_raw'][:n]:d.step(raw)
            expected=d.state()
        elif key in ('retrograde_offsets','reg_return_offsets'):expected=full[key][:n+1]
        elif key=='retrograde_events':expected=full[key][:full['retrograde_offsets'][n]]
        elif key=='reg_returns':expected=full[key][:full['reg_return_offsets'][n]]
        else:expected=full[key][:n]
        if not np.array_equal(value,expected):raise ValueError('Executable replay differs: '+key)


def run(root,output,ticks=64):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    m=json.loads((root/'manifest.json').read_text());parent=Path(m['parent'])
    for p,h in m['sources'].items():
        if base.digest(p)!=h:raise ValueError('Source changed: '+p)
    pm=json.loads((parent/'manifest.json').read_text());original=json.loads(Path(pm['config']).read_text())
    cfg=json.loads((root/'config.json').read_text())
    expected,meta=append_ventilation_feedback(original,m['groups'],mode=m['mode'])
    if cfg!=expected or m['meta']!=meta:raise ValueError('Undeclared graph change')
    if base.digest(root/'ticks.npz')!=m['sha256']:raise ValueError('Evidence changed')
    with np.load(root/'ticks.npz') as f:full={k:f[k] for k in f.files}
    with np.load(pm['media']) as f:features={k:f[k] for k in ('visual','auditory')}
    net,body,delay,organ_delay=restore(root/'initial.paula',root/'initial-body.npz',m['groups'])
    data=record(net,body,delay,organ_delay,features,m['groups'],meta,ticks)
    exact_prefix(full,data);residual=audit(data,cfg,m['groups'],features)
    output.mkdir();np.savez_compressed(output/'ticks.npz',**data)
    result=dict(parent=str(root),seed=m['seed'],mode=m['mode'],ticks=ticks,residual=residual,
        exact_fields=sorted(data),sources={str(root/p):base.digest(root/p) for p in
        ('manifest.json','initial.paula','initial-body.npz','config.json','ticks.npz')},
        replay_source_sha256=base.digest(__file__),sha256=base.digest(output/'ticks.npz'))
    (output/'manifest.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(seed=m['seed'],mode=m['mode'],exact_ticks=ticks,residual=residual)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.root,a.output)
