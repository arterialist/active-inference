"""Same-graph oxygen-signal substitution in the acquired embodied brain.

Diagnostic only: after the real 64-tick sensory queue, hold the oxygen channel
at .25, .50 or .75, or leave it intact. Energy channels, real organ dynamics,
physical afferents, graph, native return paths and adaptation remain active.
The body is never clamped. Raw and delivered organ signals are both retained.
An intact continuation must exactly match the previous full 1024-tick record.

Checks cover actual sensory substitution, independent physical/resource and
predictive-learning recurrences and muscle-output identity. The unchanged
course also uses the full nine-cell added-path audit. Clamped courses do not
claim that latter auditor's complete algebraic reconstruction, since its
original protocol requires unmodified bodily signals.
"""
import argparse
import inspect
import json
from pathlib import Path
import shutil

import numpy as np

from . import context_organization as base
from .ventilation_regulation import OrganDelay,record
from .ventilation_regulation_replay import restore,exact_prefix
from .ventilation_regulation_audit import audit as audit_intact
from .ventilation_screen import audit as audit_body,observations
from .active_sweep_transfer import audit as audit_learning


CLAMPS={'intact':None,'held-low':.25,'held-initial':.5,'held-high':.75}


class ClampedOrganDelay(OrganDelay):
    def __init__(self,initial,oxygen):
        if oxygen not in (None,.25,.5,.75): raise ValueError('Undeclared sensory intervention')
        super().__init__(initial);self.oxygen=oxygen
    def step(self,value):
        delivered=super().step(value).copy()
        if self.oxygen is not None: delivered[2]=self.oxygen
        return delivered


def audit(z,cfg,g,features,oxygen):
    if oxygen not in CLAMPS.values(): raise ValueError('Undeclared clamp')
    audit_body(z);audit_learning(z,cfg,g,features,.8)
    n=len(z['body']);raw=z['organ_before'][:,[6,0]]/.3
    expected_raw=np.column_stack((raw[:,0],1-raw[:,0],raw[:,1]))
    np.testing.assert_array_equal(z['organ_before'][0],z['organ_initial'])
    np.testing.assert_array_equal(z['organ_before'][1:],z['organs'][:-1])
    np.testing.assert_array_equal(z['organ_raw'],expected_raw)
    np.testing.assert_array_equal(z['organ_delay_initial'],np.tile([.75,.25,.5],(64,1)))
    history=np.concatenate((z['organ_delay_initial'],expected_raw))
    delivered=history[:n].copy()
    if oxygen is not None: delivered[:,2]=oxygen
    np.testing.assert_array_equal(z['organ_drive'],delivered)
    np.testing.assert_array_equal(z['organ_delay_final'],history[n:])
    meta=cfg['metadata']['ventilation_feedback'];ri=list(z['reg_ids'])
    for j,nid in enumerate(meta['sensory_ids']):
        expected=np.zeros((n,4),dtype=np.float32);expected[:,0]=delivered[:,j]
        np.testing.assert_array_equal(z['reg_inputs'][:,ri.index(nid),0],expected)
    ids=list(z['neuron_ids']);oi=base.FIELDS.index('O')
    np.testing.assert_array_equal(z['muscles'],z['cells'][:,[ids.index(nid) for nid in g['muscle']],oi])
    if oxygen is None: return audit_intact(z,cfg,g,features)
    return None


def run(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free<3*1024**3: raise OSError('Need 3 GiB reserve')
    m=json.loads((root/'manifest.json').read_text());g=m['groups'];meta=m['meta']
    if m['mode']!='feedback' or m['ticks']!=1024 or m['start']!=6352:
        raise ValueError('Need corrected full-feedback course')
    sources=dict(m['sources'])
    for name in ('initial.paula','initial-body.npz','manifest.json','config.json','ticks.npz'):
        sources[str(root/name)]=base.digest(root/name)
    if sources[str(root/'ticks.npz')]!=m['sha256']: raise ValueError('Parent evidence changed')
    for obj in (run,restore,audit_intact,audit_body,audit_learning):
        p=str(Path(inspect.getfile(obj)).resolve());sources[p]=base.digest(p)
    for p,h in sources.items():
        if base.digest(p)!=h: raise ValueError('Changed source: '+p)
    pm=json.loads((Path(m['parent'])/'manifest.json').read_text())
    with np.load(pm['media']) as z: features={k:z[k] for k in ('visual','auditory')}
    cfg=json.loads((root/'config.json').read_text());output.mkdir()
    protocol=dict(parent=str(root),seed=m['seed'],start=6352,ticks=1024,clamps=CLAMPS,
                  groups=g,meta=meta,sources=sources,limits=__doc__)
    (output/'protocol.json').write_text(base.encode(protocol)+'\n');rows=[]
    for name,oxygen in CLAMPS.items():
        net,body,delay,od=restore(root/'initial.paula',root/'initial-body.npz',g)
        od=ClampedOrganDelay(od.state(),oxygen)
        z=record(net,body,delay,od,features,g,meta,1024)
        path=output/(name+'.npz');np.savez_compressed(path,**z)
        # Retain raw evidence even if a boundary or replay check fails.
        residual=audit(z,cfg,g,features,oxygen)
        if oxygen is None:
            with np.load(root/'ticks.npz') as f: exact_prefix({k:f[k] for k in f.files},z)
        base.save_checkpoint(net,output/(name+'.paula'),sources=list(sources))
        np.savez_compressed(output/(name+'-body.npz'),state=body.state(),delay=delay.state(),
            organ=body.organs.state(),organ_delay=od.state(),oxygen_clamp=[-1. if oxygen is None else oxygen],
            gate=[body.crossings,body.next_gate])
        item=dict(name=name,oxygen_clamp=oxygen,file=path.name,sha256=base.digest(path),
                  checkpoint=name+'.paula',physical=name+'-body.npz',
                  exact_previous_ticks=1024 if oxygen is None else 0,
                  full_added_path_residual=residual,observations=observations(z))
        for key in ('checkpoint','physical'): item[key+'_sha256']=base.digest(output/item[key])
        rows.append(item)
        print(base.encode(dict(seed=m['seed'],name=name,
            **{k:v for k,v in item['observations'].items() if k.endswith('_tick')})),flush=True)
    if any(base.digest(p)!=h for p,h in sources.items()): raise ValueError('Source changed during clamp course')
    result=dict(protocol,rows=rows,recorded_ticks=4096)
    (output/'manifest.json').write_text(base.encode(result)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('output')
    a=p.parse_args();run(a.root,a.output)
