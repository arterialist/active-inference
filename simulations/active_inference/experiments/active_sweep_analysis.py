"""Audit the action-dependent sweep and trace physical influence on learning.

All claims use complete tick sequences. Crossing counts locate events, not
proof of learned control. Acquisition contains no matched acquired-weight
intervention, so this analysis cannot certify retained functional memory.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from . import context_organization as base
from .active_sweep_probe import CONDITIONS, verify
from .body_state_memory_analysis import intervals
from .eligibility_reference_intervention_analysis import onset, changed_ticks
from ..components.body.loaded_hinge import LoadedHinge, MOTOR_GEAR, DT


def mechanics(z):
    arm = LoadedHinge(*z['physical_parameters'][:3]); arm.restore(z['body_initial'])
    mujoco.mj_forward(arm.model,arm.data)
    mass=float(arm.data.qM[0]);damping=float(arm.model.dof_damping[0])
    q,v,command,force=z['body'][:,1:].T
    q0=np.r_[arm.data.qpos[0],q[:-1]];v0=np.r_[arm.data.qvel[0],v[:-1]]
    torque=MOTOR_GEAR*command;passive=-damping*v
    residual=torque+force+passive-mass*(v-v0)/DT
    if np.max(abs(residual))>2e-12 or np.max(abs(q-q0-DT*v))>2e-12:
        raise ValueError('Mechanical step balance differs')
    motor_work=torque*(q-q0);load_work=force*(q-q0);damping_work=passive*(q-q0)
    kinetic=.5*mass*(v*v-v0*v0);implicit_loss=.5*mass*(v-v0)**2
    if np.max(abs(motor_work+load_work+damping_work-kinetic-implicit_loss))>2e-12:
        raise ValueError('Mechanical work balance differs')
    return np.column_stack((torque,force,passive,motor_work,load_work,damping_work,kinetic,implicit_loss))


def verify_predictive_arrivals(z,cfg,groups):
    nodes={n['id']:n for n in cfg['neurons']};ids=list(z['neuron_ids']);terms=list(map(tuple,z['terminal_ids']))
    edges={(c['target_neuron'],c['target_synapse']):(c['source_neuron'],c['source_terminal']) for c in cfg['connections']}
    for j,nid in enumerate(groups['prediction']):
        for column,port in enumerate(nodes[nid]['metadata']['prediction_ports']):
            source,terminal=edges[nid,port]
            out=z['cells'][:-1,ids.index(source),base.FIELDS.index('O')]
            info=z['terminal_info'][:-1,terms.index((source,terminal))]
            expected=(out*info).astype(np.float32).astype(float)
            if not np.array_equal(expected,z['arrivals'][1:,j,column]):
                raise ValueError('Selected arrivals do not match source-neuron release')


def analyze(roots,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    arrays={};cases=[];comparisons=[];seeds=set();sources={};checked=0
    for root in (Path(p).resolve() for p in roots):
        m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text());seed=m['seed']
        if seed in seeds:raise ValueError('Duplicate seed')
        seeds.add(seed)
        for p,h in m['source_hashes'].items():
            if base.digest(p)!=h:raise ValueError('Source changed: '+p)
        if base.digest(m['physical_source'])!=m['physical_sha256']:raise ValueError('Media changed')
        with np.load(m['physical_source']) as f:features={k:f[k] for k in ('visual','auditory')}
        rows=s['rows'];identities=[r['condition'] for r in rows]
        if len(identities)!=len(CONDITIONS) or set(identities)!=set(CONDITIONS):raise ValueError('Incomplete conditions')
        records={}
        for row in rows:
            c=row['condition'];g=m['groups'][c]
            for field in ('checkpoint','physical'):
                if base.digest(root/row[field])!=row[field+'_sha256']:raise ValueError('Missing/changed executable state')
            if base.digest(root/f'{c}.json')!=m['config_hashes'][c]:raise ValueError('Graph changed')
            cfg=json.loads((root/f'{c}.json').read_text())
            path=root/row['file']
            if base.digest(path)!=row['sha256']:raise ValueError('Record changed')
            with np.load(path) as f:z={k:f[k] for k in f.files}
            if len(z['body'])!=row['ticks'] or len(z['body'])<256:raise ValueError('Incorrect duration')
            if list(z['physical_parameters'][[0,3]])!=[CONDITIONS[c][0],CONDITIONS[c][2]]:
                raise ValueError('Wrong physical condition')
            if cfg['metadata']['active_sweep']['sensorimotor']!=CONDITIONS[c][1]:raise ValueError('Wrong neural condition')
            verify(z,g,features);verify_predictive_arrivals(z,cfg,g);mech=mechanics(z)
            records[c]=z;checked+=len(z['body']);ids=list(z['neuron_ids'])
            out=z['cells'][:,:,base.FIELDS.index('O')]
            prediction=out[:,ids.index(g['prediction'][0])]-out[:,ids.index(g['prediction'][1])]
            name=f's{seed}_{c}'
            arrays[name]=np.column_stack((z['body'],prediction,z['gate'],z['raw_afferents'],mech))
            arrays[name+'_learning']=np.concatenate((z['errors'].reshape(len(out),-1),z['eta']),axis=1)
            arrays[name+'_cpg']=out[:,[ids.index(n) for n in g['cpg']]]
            arrays[name+'_context']=out[:,[ids.index(n) for n in g['mixed_0']+g['mixed_1']]]
            cases.append(dict(seed=seed,condition=c,trace=name,gate_ticks=np.flatnonzero(z['gate'][:,2]).tolist(),
                total_crossings=int(z['gate'][-1,0]),first_selected_weight_change=onset(z['weights'],np.broadcast_to(z['weights_initial'],z['weights'].shape)),
                moving=intervals(z['body'][:,2]!=0),nonzero_environmental_force=intervals(z['body'][:,4]!=0),
                prediction_agrees_current_force=intervals(prediction*z['body'][:,4]>0),
                prediction_opposes_current_force=intervals(prediction*z['body'][:,4]<0),
                clock_spikes={str(n):np.flatnonzero(out[:,ids.index(n)]>0).tolist() for n in g['cpg']}))
        if s['executed_ticks']!=sum(r['ticks'] for r in rows):raise ValueError('Incorrect executed count')
        for other in ('actuator_cut','free_fused','loaded_sensory'):
            live=records['loaded_fused'];counter=records[other];g=m['groups']['loaded_fused']
            if any(not np.array_equal(live[f],counter[f]) for f in ('body_initial','delay_initial','weights_initial','context_initial','error_initial')):
                raise ValueError('Mismatched initial states')
            name=f's{seed}_loaded_vs_{other}';first={}
            for field in ('cells','drive','raw_afferents','weights','errors','eta','body','terminal_info'):
                first[field]=onset(live[field],counter[field]);arrays[name+'_'+field+'_changed']=changed_ticks(live[field],counter[field])
            ids=list(live['neuron_ids']);populations={}
            for role,members in g.items():
                ix=[ids.index(n) for n in members]
                populations[role]=onset(live['cells'][:,ix,base.FIELDS.index('O')],counter['cells'][:,ix,base.FIELDS.index('O')])
            arrays[name+'_prediction_error_effect']=live['errors']-counter['errors']
            arrays[name+'_body_effect']=live['body']-counter['body']
            comparisons.append(dict(seed=seed,other=other,first_difference=first,population_output=populations))
        for name in ('manifest.json','summary.json'):sources[str(root/name)]=base.digest(root/name)
    if len(seeds)<4:raise ValueError('Need four graph seeds')
    output.mkdir();np.savez_compressed(output/'per-tick.npz',**arrays)
    result=dict(cases=cases,comparisons=comparisons,seeds=sorted(seeds),checked_ticks=checked,sources=sources,
        producer_sha256=base.digest(__file__),columns=['time','angle','velocity','applied_command','environmental_torque',
        'signed_prediction','crossings','next_gate','crossed','force_positive','force_negative','position_positive',
        'position_negative','velocity_positive','velocity_negative','motor_torque','load_torque','passive_torque',
        'motor_work','load_work','damping_work','kinetic_change','implicit_loss'],
        limits=__doc__+' Selected neural arrivals checked from tick 1; tick-zero in-flight history is not inferred. '
        'Mechanical work is not metabolic cost. Tiny finite-precision divergences are not behavioral benefits.')
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(seeds=sorted(seeds),checked_ticks=checked,cases=len(cases))),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();analyze(a.roots,a.output)
