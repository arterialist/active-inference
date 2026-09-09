"""Physical oxygen challenge and recovery in the continuing 603-cell brain.

Environmental oxygen fraction is .21 for local ticks 0:512, .105 for
512:1280 and .21 for 1280:2048. Nothing announces the change to the brain.
Compare real delayed organ feedback with the same-graph oxygen reading held
at .5. Actual oxygen, energy, mechanics, both sensory queues and adaptation
continue. The opening 512 ticks must reproduce each earlier fixed-world
record exactly. This is a predetermined challenge, not a tuned pass criterion.

The audit independently reconstructs resource equations, physical integration,
predictive learning and actual organ delivery. It does not rederive all cells
or claim the original fixed-input added-path audit covers the substituted case.
"""
import argparse
from dataclasses import replace
import inspect
import json
from pathlib import Path
import shutil

import numpy as np

from . import context_organization as base
from .active_sweep_transfer import audit as audit_learning
from .ventilation_input_clamp import ClampedOrganDelay
from .ventilation_regulation import record
from .ventilation_regulation_replay import restore, exact_prefix
from .ventilation_screen import observations
from ..components.body.ventilation import VentilationOrgans, VentilationParameters, research_energy_parameters
from ..components.body.loaded_hinge import DT, MOTOR_GEAR


TICKS = 2048
SCHEDULE = ((0, 512, .21), (512, 1280, .105), (1280, TICKS, .21))
CONDITIONS = {'intact': None, 'held-initial': .5}


class ChangingAirOrgans(VentilationOrgans):
    def __init__(self, state, world_tick=0):
        super().__init__()
        self.restore(state)
        if type(world_tick) is not int or not 0 <= world_tick <= TICKS:
            raise ValueError('Invalid world clock')
        self.world_tick = world_tick
        self.air = []

    def advance(self, *args, **kwargs):
        if self.world_tick >= TICKS:
            raise ValueError('World course exhausted')
        fraction = next(f for a,b,f in SCHEDULE if a <= self.world_tick < b)
        self.params = replace(self.params, oxygen_fraction=fraction)
        result = super().advance(*args, **kwargs)
        self.air.append(fraction)
        self.world_tick += 1
        return result


def audit_resources(z):
    """Independent resource ledger on separately audited physical positions."""
    p, ep = VentilationParameters(), research_energy_parameters()
    n = len(z['body'])
    if not 0 < n <= TICKS:
        raise ValueError('Invalid duration')
    expected_air = np.array([.105 if 512 <= t < 1280 else .21 for t in range(n)])
    np.testing.assert_array_equal(z['air_fraction'], expected_air)
    np.testing.assert_array_equal(z['intervention'], [1.,1.,1.])
    np.testing.assert_array_equal(z['organ_initial'], VentilationOrgans().state())
    for key,width in (('muscles',2),('organs',13),('exchange',9)):
        if z[key].shape != (n,width):
            raise ValueError('Missing resource field: '+key)
    state = z['organ_initial'].copy()
    # exchange contains both angles' chamber volumes. The initial angle is
    # checked through the first independently reconstructed physical step.
    from ..components.body.loaded_hinge import LoadedHinge
    b = LoadedHinge(.8)
    b.restore(z['body_initial'], crossings=int(z['gate_initial'][0]), next_gate=int(z['gate_initial'][1]))
    q0 = float(b.data.qpos[0])
    for t,m in enumerate(z['muscles']):
        q1 = float(z['body'][t,1])
        v = p.chamber_mid_ml + p.chamber_swing_ml*np.tanh(p.linkage_per_rad*np.array([q0,q1]))
        fresh = max(0., v[1]-v[0]); uptake = expected_air[t]*fresh
        demand = p.oxygen_demand_ml_s*DT
        available = state[0]+uptake; debt = max(0., demand-available)
        remaining = max(0., available-demand); spill = max(0., remaining-p.oxygen_capacity_ml)
        work = MOTOR_GEAR*(max(0.,m[0]*(q1-q0))+max(0.,m[1]*(q0-q1)))
        activation = DT*(m[0]**2+m[1]**2)
        np.testing.assert_allclose(z['exchange'][t], [*v,fresh,uptake,demand,debt,spill,work,activation], rtol=0,atol=2e-13)
        state[0] = min(p.oxygen_capacity_ml,remaining)
        state[1:6] += [fresh,uptake,demand,debt,spill]
        digested = min(state[7],ep.digestion_w*DT)
        cost = ep.basal_w*DT+work/ep.efficiency+ep.activation_w*activation
        available = state[6]+digested; debt = max(0.,cost-available)
        remaining = max(0.,available-cost); spill = max(0.,remaining-ep.capacity_j)
        state[6] = min(ep.capacity_j,remaining); state[7] -= digested
        state[8:12] += [digested,cost,debt,spill]
        np.testing.assert_allclose(z['organs'][t],state,rtol=0,atol=2e-12)
        q0 = q1


def audit(z,cfg,g,features,oxygen):
    if oxygen not in CONDITIONS.values():
        raise ValueError('Undeclared sensory control')
    audit_learning(z,cfg,g,features,.8)
    audit_resources(z)
    n = len(z['body'])
    np.testing.assert_array_equal(z['organ_before'][0],z['organ_initial'])
    np.testing.assert_array_equal(z['organ_before'][1:],z['organs'][:-1])
    raw = np.column_stack((z['organ_before'][:,6]/.3,1-z['organ_before'][:,6]/.3,z['organ_before'][:,0]/.3))
    np.testing.assert_array_equal(z['organ_raw'],raw)
    np.testing.assert_array_equal(z['organ_delay_initial'],np.tile([.75,.25,.5],(64,1)))
    history = np.concatenate((z['organ_delay_initial'],raw)); delivered = history[:n].copy()
    if oxygen is not None:
        delivered[:,2] = oxygen
    np.testing.assert_array_equal(z['organ_drive'],delivered)
    np.testing.assert_array_equal(z['organ_delay_final'],history[n:])
    ids = list(z['reg_ids'])
    for j,nid in enumerate(cfg['metadata']['ventilation_feedback']['sensory_ids']):
        expected = np.zeros((n,4),dtype=np.float32); expected[:,0] = delivered[:,j]
        np.testing.assert_array_equal(z['reg_inputs'][:,ids.index(nid),0],expected)
    ids = list(z['neuron_ids'])
    np.testing.assert_array_equal(z['muscles'],z['cells'][:,[ids.index(n) for n in g['muscle']],base.FIELDS.index('O')])


def run(root,output):
    root,output = Path(root).resolve(),Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free < 3*1024**3:
        raise OSError('Need 3 GiB reserve')
    parent = json.loads((root/'manifest.json').read_text())
    verified = Path(parent['parent'])
    m = json.loads((verified/'manifest.json').read_text())
    if parent['ticks'] != 1024 or m['mode'] != 'feedback' or m['start'] != 6352:
        raise ValueError('Need verified four-condition clamp parent')
    g,meta = m['groups'],m['meta']; sources = dict(parent['sources'])
    sources[str(root/'manifest.json')] = base.digest(root/'manifest.json')
    for obj in (run,record,restore,audit_learning,VentilationOrgans):
        path = str(Path(inspect.getfile(obj)).resolve()); sources[path] = base.digest(path)
    for name in CONDITIONS:
        row = next(r for r in parent['rows'] if r['name']==name)
        sources[str(root/row['file'])] = row['sha256']
    for p,h in sources.items():
        if base.digest(p) != h:
            raise ValueError('Changed source: '+p)
    cfg = json.loads((verified/'config.json').read_text())
    pm = json.loads((Path(m['parent'])/'manifest.json').read_text())
    with np.load(pm['media']) as f:
        features = {k:f[k] for k in ('visual','auditory')}
    output.mkdir()
    protocol = dict(parent=str(root),seed=m['seed'],groups=g,meta=meta,sources=sources,
                    start=6352,ticks=TICKS,schedule=SCHEDULE,conditions=CONDITIONS,limits=__doc__)
    (output/'protocol.json').write_text(base.encode(protocol)+'\n')
    rows = []
    for name,oxygen in CONDITIONS.items():
        net,body,delay,od = restore(verified/'initial.paula',verified/'initial-body.npz',g)
        body.organs = ChangingAirOrgans(body.organs.state())
        od = ClampedOrganDelay(od.state(),oxygen)
        z = record(net,body,delay,od,features,g,meta,TICKS)
        z['air_fraction'] = np.asarray(body.organs.air)
        path = output/(name+'.npz'); np.savez_compressed(path,**z)
        audit(z,cfg,g,features,oxygen)
        # Full opening prefix, including arrivals, returns and weights. Use
        # the reference slicing implementation rather than dropping end states.
        with np.load(root/(name+'.npz')) as f:
            old = {k:f[k] for k in f.files}
        exact_prefix(old,prefix(z,512))
        base.save_checkpoint(net,output/(name+'.paula'),sources=list(sources))
        np.savez_compressed(output/(name+'-body.npz'),state=body.state(),delay=delay.state(),
            organ=body.organs.state(),organ_delay=od.state(),world_tick=[body.organs.world_tick],
            oxygen_clamp=[-1. if oxygen is None else oxygen],gate=[body.crossings,body.next_gate])
        item = dict(name=name,file=path.name,sha256=base.digest(path),observations=observations(z),
                    exact_opening_ticks=512,checkpoint=name+'.paula',physical=name+'-body.npz')
        for k in ('checkpoint','physical'):
            item[k+'_sha256'] = base.digest(output/item[k])
        rows.append(item)
        print(base.encode(dict(seed=m['seed'],name=name,observations=item['observations'])),flush=True)
        del z
    if any(base.digest(p)!=h for p,h in sources.items()):
        raise ValueError('Source changed during course')
    result = dict(protocol,rows=rows,checked_ticks=2*TICKS)
    (output/'manifest.json').write_text(base.encode(result)+'\n')
    return result


def prefix(z,n):
    """Slice a recorded course including queues and variable-size return data."""
    from .active_sweep_probe import PhysicalDelay
    from .ventilation_regulation import OrganDelay
    constants = {'neuron_ids','terminal_ids','physical_parameters','reg_ids','cpg_ids','credit_mean','credit_decay','intervention'}
    result = {}
    for k,v in z.items():
        if k == 'air_fraction':
            continue
        if k.endswith('_initial') or k in constants:
            result[k] = v
        elif k in ('delay_final','organ_delay_final'):
            d = PhysicalDelay(z['delay_initial']) if k=='delay_final' else OrganDelay(z['organ_delay_initial'])
            raw = z['raw_afferents'] if k=='delay_final' else z['organ_raw']
            for sample in raw[:n]: d.step(sample)
            result[k] = d.state()
        elif k in ('retrograde_offsets','reg_return_offsets'):
            result[k] = v[:n+1]
        elif k in ('retrograde_events','reg_returns'):
            offsets = z['retrograde_offsets'] if k=='retrograde_events' else z['reg_return_offsets']
            result[k] = v[:offsets[n]]
        else:
            result[k] = v[:n]
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('root'); p.add_argument('output')
    a = p.parse_args(); run(a.root,a.output)
