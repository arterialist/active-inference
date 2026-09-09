"""Audit physical-state dependence of memory use, without conflating objectives.

Keep prediction, pose, acceleration, actuator work and numerical dissipation
separate. Actuator work is physical work, not neural metabolic energy or ALERM
free energy. No composite success score or hidden preferred trajectory is used.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from . import context_organization as base
from .crossed_av_analysis import verify_stimuli
from .magnitude_feedback_analysis import verify_returns
from .opponent_context_analysis import read_record
from .temporal_verification import verify_learning


def mechanics(data):
    base.verify_physics(data)
    arm=base.Arm();arm.restore(data['body_initial']);mujoco.mj_forward(arm.model,arm.data)
    mass=float(arm.data.qM[0]);damping=float(arm.model.dof_damping[0])
    body=data['body'];q=body[:,1];v=body[:,2]
    q0=np.r_[arm.data.qpos[0],q[:-1]];v0=np.r_[arm.data.qvel[0],v[:-1]]
    motor=base.FORCE*body[:,3];load=body[:,4];passive=-damping*v
    total=motor+load+passive
    acceleration=(v-v0)/base.DT
    if np.max(abs(total-mass*acceleration))>2e-12:
        raise ValueError('Compiled mechanical force balance differs')
    if np.max(abs(q-q0-base.DT*v))>2e-12:raise ValueError('Position integration differs')
    delta=q-q0;wm=motor*delta;wl=load*delta;wd=passive*delta
    kinetic=.5*mass*(v*v-v0*v0);numerical=.5*mass*(v-v0)**2
    if np.max(abs(wm+wl+wd-kinetic-numerical))>2e-12:
        raise ValueError('Mechanical work balance differs')
    return np.column_stack((motor,load,passive,total,acceleration,wm,wl,wd,kinetic,numerical))


def intervals(mask):
    a=np.asarray(mask,dtype=bool)
    edges=np.flatnonzero(np.diff(np.r_[False,a,False]))
    return [[int(a),int(b)] for a,b in edges.reshape(-1,2)]


def analyze(roots,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    records={};cases=[];sources={};seen=set();checked=0
    for root in map(lambda p:Path(p).resolve(),roots):
        m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text())
        if m['seed'] in seen:raise ValueError('Duplicate seed')
        seen.add(m['seed']);conditions={}
        if s['executed_ticks']!=3072 or s['exact_reference_ticks']!=768:raise ValueError('Incomplete course')
        for p,h in {**m['source_hashes'],**m['physical_sources']}.items():
            if base.digest(p)!=h:raise ValueError('Source changed')
        features=[]
        for clip in (0,1):
            paths=[p for p in m['physical_sources'] if Path(p).name==f'sensory-{clip}.npz']
            if len(paths)!=1:raise ValueError('Ambiguous media')
            with np.load(paths[0]) as z:features.append({k:z[k] for k in ('visual','auditory')})
        for row in s['records']:
            key=(row['physical'],row['weights'],row['video'],row['audio'])
            if key in conditions:raise ValueError('Duplicate probe')
            z=read_record(root,row,m,learning_auditor=verify_learning);verify_returns(z)
            verify_stimuli(z,features,row,False);mech=mechanics(z)
            if len(z['body'])!=192:raise ValueError('Wrong probe duration')
            if row['physical']=='acquired' and not row['acquired_prefix_exact']:
                raise ValueError('Missing exact source replay')
            ids=list(z['neuron_ids']);o=z['cells'][:,:,base.FIELDS.index('O')]
            prediction=o[:,ids.index(m['groups']['prediction'][0])]-o[:,ids.index(m['groups']['prediction'][1])]
            tr=np.column_stack((z['body'],prediction,prediction*np.sign(z['body'][:,4]),mech))
            name=f's{m["seed"]}_{Path(row["file"]).stem}'
            records[name]=tr;conditions[key]=(z,tr);checked+=192
        expected={(p,w,v,a) for p in ('acquired','rest') for w in ('learned','reset') for v in (0,1) for a in (0,1)}
        if set(conditions)!=expected:raise ValueError('Incomplete factorial')
        for v,a in ((0,0),(0,1),(1,0),(1,1)):
            divergence={}
            for w in ('learned','reset'):
                acquired,rest=[conditions[p,w,v,a][0] for p in ('acquired','rest')]
                for field in ('weights_initial','delay_initial','context_initial','error_initial','terminal_initial'):
                    if not np.array_equal(acquired[field],rest[field]):raise ValueError('Unmatched neural/history state')
                for field in ('cells','weights','arrivals','errors','eta','terminal_info','drive'):
                    if not np.array_equal(acquired[field][:64],rest[field][:64]):
                        raise ValueError('Physical intervention reached neural system before physical delay')
                changed=np.flatnonzero(np.any(acquired['cells']!=rest['cells'],axis=(1,2)))
                divergence[w]=int(changed[0]) if len(changed) else None
            effects={}
            for p in ('acquired','rest'):
                (lz,l),(rz,r)=[conditions[p,w,v,a] for w in ('learned','reset')]
                if not np.array_equal(lz['body_initial'],rz['body_initial']):raise ValueError('Mismatched reset body')
                if np.any(rz['weights_initial']):raise ValueError('Incomplete selected reset')
                diff=np.column_stack((abs(l[:,1])-abs(r[:,1]),abs(l[:,10])-abs(r[:,10]),
                                      abs(l[:,7])-abs(r[:,7]),l[:,12]-r[:,12],l[:,6],l[:,3]-r[:,3]))
                key=f's{m["seed"]}_{p}_v{v}_a{a}_effect';records[key]=diff
                effects[p]=dict(trace=key,correct_prediction_worse_pose=intervals((diff[:,4]>0)&(diff[:,0]>0)),
                    correct_prediction_worse_total_torque=intervals((diff[:,4]>0)&(diff[:,1]>0)),
                    pose_delta_at63=float(diff[63,0]),pose_delta_at191=float(diff[191,0]))
            cases.append(dict(seed=m['seed'],video=v,audio=a,first_body_to_neural_divergence=divergence,effects=effects))
        sources[str(root/'manifest.json')]=base.digest(root/'manifest.json')
        sources[str(root/'summary.json')]=base.digest(root/'summary.json')
    if not cases:raise ValueError('No evidence')
    output.mkdir();np.savez_compressed(output/'per-tick.npz',**records)
    result=dict(cases=cases,checked_ticks=checked,sources=sources,producer_sha256=base.digest(__file__),
        columns=['time','angle','velocity','command','load','prediction','prediction_along_load',
          'motor_torque','load_torque','passive_torque','total_torque','acceleration','motor_work',
          'load_work','damping_work','kinetic_change','implicit_step_dissipation'],
        effect_columns=['delta_absolute_pose','delta_absolute_total_torque','delta_absolute_motor_torque',
                        'delta_motor_work','learned_prediction_along_load','delta_command'],
        limits='Retrospectively selected checkpoint, physical relocation and selected-weight reset. '
          'Body/history state is acquired, not a fresh cue-only memory test. All branches keep learning. '
          'No single objective is assumed optimal. Work is not metabolic energy; numerical loss is identified.')
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(cases=len(cases),checked_ticks=checked)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.roots,a.output)
