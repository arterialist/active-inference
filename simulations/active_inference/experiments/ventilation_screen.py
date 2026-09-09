"""Physical-necessity screen using recorded, unmodified PAULA muscle histories.

This is an offline body replay, NOT neural self-regulation. The additional
organs receive movement but do not yet send afferents. Natural replay must
reproduce every saved MuJoCo integration state. Diagnostic silence, actuator
cut, exchange cut, and 2x/4x activation are explicit open-loop controls, never
organism policies or learned behavior. All cases use the complete 1024-tick
course. Resource equations are fixed before running, with no passing criterion
used to select their parameters. Full per-tick outputs precede summaries.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from ..components.body import ventilation as organ_module
from ..components.body import energy_budget as energy_module
from ..components.body import loaded_hinge as hinge_module
from ..components.body.ventilation import VentilatedHinge, VentilationOrgans, VentilationParameters
from ..components.body.loaded_hinge import LoadedHinge, DT, MOTOR_GEAR
from .context_organization import FIELDS


SEEDS = (11, 23, 44, 77)
CASES = (('transfer','released-intact'), ('transfer','released-reset'),
         ('transfer','loaded-intact'), ('transfer','loaded-reset'),
         ('return','current'), ('return','reset'))
CONTROLS = {'silent':(0.,1.,1.), 'actuator-cut':(1.,0.,1.),
            'exchange-cut':(1.,1.,0.), 'double':(2.,1.,1.), 'quadruple':(4.,1.,1.)}


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def replay(source, groups, *, scale=1., transmission=1., exchange=1.):
    drag, spring, gate, old_transmission = source['physical_parameters']
    if old_transmission != 1. or scale not in (0.,1.,2.,4.):
        raise ValueError('Unexpected input record or activation intervention')
    b = VentilatedHinge(drag, spring, gate)
    b.restore(source['body_initial'], crossings=int(source['gate_initial'][0]),
              next_gate=int(source['gate_initial'][1]))
    ids = list(source['neuron_ids'])
    original = source['cells'][:,[ids.index(n) for n in groups['muscle']],FIELDS.index('O')]
    if not np.array_equal(original[:,0]-original[:,1],source['body'][:,3]):
        raise ValueError('Recorded action differs from actual PAULA muscle outputs')
    muscles = scale*original
    rows = {k:[] for k in ('body','physical_states','organs','exchange','gate')}
    initial = b.organs.state()
    for t,m in enumerate(muscles):
        force, crossed = b.step_muscles(m,transmission=transmission,exchange=exchange)
        if scale == transmission == 1.:
            np.testing.assert_array_equal(b.state(),source['physical_states'][t])
        rows['body'].append([b.data.time,b.data.qpos[0],b.data.qvel[0],transmission*(m[0]-m[1]),force])
        rows['physical_states'].append(b.state()); rows['organs'].append(b.organs.state())
        rows['exchange'].append(b.last_exchange.copy()); rows['gate'].append([b.crossings,b.next_gate,crossed])
    result = {k:np.asarray(v) for k,v in rows.items()}
    result.update(muscles=muscles,organ_initial=initial,body_initial=source['body_initial'],
                  gate_initial=source['gate_initial'],physical_parameters=np.array([drag,spring,gate,transmission]),
                  intervention=np.array([scale,transmission,exchange]))
    audit(result)
    return result


def audit(z):
    """Independent recurrences; do not call the producer's organ advance()."""
    p, ep = VentilationParameters(), organ_module.research_energy_parameters()
    n = len(z['muscles'])
    if n < 1:
        raise ValueError('Empty physical evidence')
    for key,width in (('muscles',2),('body',5),('organs',13),('exchange',9),('gate',3)):
        if z[key].shape != (n,width):
            raise ValueError('Missing or misaligned tick fields: '+key)
    if z['physical_states'].shape[0] != n:
        raise ValueError('Missing physical integration states')
    np.testing.assert_array_equal(z['organ_initial'],VentilationOrgans().state())
    if (z['intervention'].shape != (3,) or z['intervention'][0] not in (0.,1.,2.,4.)
            or z['intervention'][1] not in (0.,1.) or z['intervention'][2] not in (0.,1.)
            or z['physical_parameters'].shape != (4,)
            or z['intervention'][1] != z['physical_parameters'][3]):
        raise ValueError('Undeclared physical intervention')
    drag,spring,gate,transmission = z['physical_parameters']
    b = LoadedHinge(drag,spring,gate)
    b.restore(z['body_initial'],next_gate=int(z['gate_initial'][1]),crossings=int(z['gate_initial'][0]))
    if any(not np.isfinite(a).all() for a in z.values()):
        raise ValueError('Nonfinite trace')
    state = z['organ_initial'].copy()
    for t,m in enumerate(z['muscles']):
        q0 = float(b.data.qpos[0]); command=float(transmission*(m[0]-m[1]))
        force,crossed=b.step(command); q1=float(b.data.qpos[0])
        np.testing.assert_array_equal(z['physical_states'][t], b.state())
        np.testing.assert_array_equal(z['body'][t],[b.data.time,q1,b.data.qvel[0],command,force])
        np.testing.assert_array_equal(z['gate'][t],[b.crossings,b.next_gate,crossed])
        volume=p.chamber_mid_ml+p.chamber_swing_ml*np.tanh(p.linkage_per_rad*np.array([q0,q1]))
        fresh=max(0.,volume[1]-volume[0]); uptake=z['intervention'][2]*p.oxygen_fraction*fresh
        demand=p.oxygen_demand_ml_s*DT; available=state[0]+uptake
        unmet=max(0.,demand-available); remaining=max(0.,available-demand)
        spill=max(0.,remaining-p.oxygen_capacity_ml)
        work=transmission*MOTOR_GEAR*(max(0.,m[0]*(q1-q0))+max(0.,m[1]*(q0-q1)))
        activation=DT*(m[0]**2+m[1]**2)
        expected=[*volume,fresh,uptake,demand,unmet,spill,work,activation]
        np.testing.assert_allclose(z['exchange'][t],expected,rtol=0,atol=2e-13)
        state[0]=min(p.oxygen_capacity_ml,remaining)
        state[1:6]+=[fresh,uptake,demand,unmet,spill]
        digested=min(state[7],ep.digestion_w*DT)
        cost=ep.basal_w*DT+work/ep.efficiency+ep.activation_w*activation
        fuel=state[6]+digested; debt=max(0.,cost-fuel); residual=max(0.,fuel-cost)
        overflow=max(0.,residual-ep.capacity_j)
        state[6]=min(ep.capacity_j,residual); state[7]-=digested
        state[8:12]+=[digested,cost,debt,overflow]
        np.testing.assert_allclose(z['organs'][t],state,rtol=0,atol=2e-12)
    return 0.


def observations(z):
    # Numerical tolerance only, not a tolerated physiological debt. Full raw
    # reserves and cumulative debts remain visible even below this threshold.
    failure={}
    for name,col in (('oxygen',4),('energy',10)):
        indices=np.flatnonzero(z['organs'][:,col]>1e-12)
        failure[name+'_first_debt_tick']=int(indices[0]) if len(indices) else None
    return dict(**failure,final=z['organs'][-1].tolist(),
                oxygen_empty_ticks=np.flatnonzero(z['organs'][:,0]<=1e-12).tolist(),
                energy_empty_ticks=np.flatnonzero(z['organs'][:,6]<=1e-12).tolist())


def run(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free < 3*1024**3: raise OSError('Need 3 GiB reserve')
    sources={str(Path(p).resolve()):digest(p) for p in
             (__file__,organ_module.__file__,energy_module.__file__,hinge_module.__file__)}
    protocol=dict(limits=__doc__,seeds=SEEDS,ticks=1024,cases=CASES,controls=CONTROLS,
                  organ_parameters=asdict(VentilationParameters()),
                  energy_parameters=asdict(organ_module.research_energy_parameters()),
                  organ_fields=VentilationOrgans.fields,exchange_fields=VentilationOrgans.exchange_fields,
                  body_fields=('time_s','angle_rad','velocity_rad_s','command','environmental_torque_Nm'),
                  debt_reporting_tolerance=1e-12,sources=sources)
    output.mkdir(); (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    cases=[]
    for seed in SEEDS:
        for family,name in CASES:
            parent=root/f'20260909_active_sweep_{family}_seed{seed}'
            mp=parent/'manifest.json'; manifest=json.loads(mp.read_text()); sources[str(mp)]=digest(mp)
            row=next(r for r in manifest['rows'] if r.get('kind',r.get('name'))==name)
            path=parent/row['file']
            if digest(path)!=row['sha256']: raise ValueError('Source recording changed: '+str(path))
            sources[str(path)]=row['sha256']
            with np.load(path) as f:
                z={k:f[k] for k in ('body','body_initial','gate_initial','physical_parameters',
                                   'physical_states','cells','neuron_ids')}
            if len(z['body'])!=1024: raise ValueError('Incomplete source course')
            variants={'natural':(1.,1.,1.)}
            if (family,name)==('return','current'): variants.update(CONTROLS)
            for label,(scale,transmission,exchange) in variants.items():
                data=replay(z,manifest['groups'],scale=scale,transmission=transmission,exchange=exchange)
                key=f's{seed}-{family}-{name}-{label}'; target=output/(key+'.npz')
                np.savez_compressed(target,**data)
                item=dict(key=key,seed=seed,source=str(path),file=target.name,sha256=digest(target),
                          observations=observations(data))
                cases.append(item)
                short={k:v for k,v in item['observations'].items() if k.endswith('_tick')}
                print(json.dumps(dict(key=key,**short)),flush=True)
    for p,h in sources.items():
        if digest(p)!=h: raise ValueError('Input or source changed during screen: '+p)
    protocol.update(sources=sources,cases=cases,checked_ticks=1024*len(cases))
    (output/'manifest.json').write_text(json.dumps(protocol,indent=2)+'\n')
    return protocol


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root',type=Path); p.add_argument('output',type=Path)
    a=p.parse_args(); run(a.root,a.output)
