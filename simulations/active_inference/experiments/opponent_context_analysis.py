"""Inspect full embodied trajectories before interpreting window summaries."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import context_organization as base
from .opponent_context import verify_afferents


WINDOWS = ((0,32),(32,64),(64,128),(128,192))


def read_record(root, row, manifest, *, learning_auditor=base.verify_learning):
    path = root/row['file']
    if base.digest(path) != row['sha256']:
        raise ValueError('Recorded data changed')
    with np.load(path) as z:
        data = {k:z[k] for k in z.files}
    learning_auditor(data); base.verify_physics(data); verify_afferents(data)
    ids = list(data['neuron_ids']); groups=manifest['groups']
    muscles=data['cells'][:,[ids.index(n) for n in groups['muscle']],base.FIELDS.index('O')]
    if not np.array_equal(muscles[:,0]-muscles[:,1],data['body'][:,3]):
        raise ValueError('Physical command differs from recorded neural outputs')
    return data


def compare_feedback(roots, output):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    cases=[];traces={};provenance={}
    for root in map(lambda p:Path(p).resolve(),roots):
        m=json.loads((root/'manifest.json').read_text())
        summary=json.loads((root/'summary.json').read_text())
        if base.digest(Path(__file__).with_name('opponent_feedback_probe.py'))!=m['producer_sha256']:
            raise ValueError('Feedback producer changed')
        source=Path(m['source']); original=json.loads((source/'summary.json').read_text())
        sm=json.loads((source/'manifest.json').read_text())
        if base.digest(source/'manifest.json')!=m['source_manifest_sha256']:
            raise ValueError('Parent manifest changed')
        for p,h in {**sm['source_hashes'],**sm['physical_sources']}.items():
            if base.digest(p)!=h:raise ValueError('Parent source changed')
        provenance[str(root/'summary.json')]=base.digest(root/'summary.json')
        for row in summary['results']:
            delay,context,clip=(row[k] for k in ('delay','context','clip'))
            name=f'probe-d{delay}-r0-c{context}-v{clip}.npz'
            old=next(r for r in original['probes'] if r['file']==name)
            a=read_record(source,old,sm);b=read_record(root,row,m)
            if not np.array_equal(a['body_initial'],b['body_initial']):
                raise ValueError('Feedback control body mismatch')
            if not np.array_equal(a['weights_initial'],b['weights_initial']):
                raise ValueError('Feedback control changed selected memory')
            record=dict(seed=m['seed'],order=m['order'],delay=delay,context=context,clip=clip,
                        unchanged_replay_exact=row['unchanged_replay_exact'],conditions={})
            for label,z in (('original',a),('half',b)):
                torque=z['body'][:,4]+base.FORCE*z['body'][:,3]
                key=f's{m["seed"]}_o{m["order"]}_d{delay}_c{context}_v{clip}_{label}'
                traces[key]=np.column_stack((z['body'],torque,z['errors'][:,:,1],z['eta']))
                record['conditions'][label]=[dict(start=start,stop=stop,
                    mean_abs_net_torque=float(abs(torque[start:stop]).mean()),
                    peak_abs_angle=float(abs(z['body'][start:stop,1]).max()),
                    angle_end=float(z['body'][stop-1,1])) for start,stop in WINDOWS]
            cases.append(record)
    output.mkdir()
    np.savez_compressed(output/'per-tick.npz',**traces)
    result=dict(cases=cases,provenance=provenance,
        columns=['time_s','angle_rad','velocity_rad_s','command','load_Nm','net_torque_Nm',
                 'teaching_0','teaching_1','eta_0','eta_1'],
        scope='Acquired-state feedback intervention; it cannot repair or evaluate a new acquisition history.')
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(feedback_comparisons=len(cases))),flush=True)
    return result


def compare(roots, output):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    cases=[];traces={};provenance={};seen=set();checked_ticks=0
    for root in map(lambda p:Path(p).resolve(),roots):
        m=json.loads((root/'manifest.json').read_text())
        summary=json.loads((root/'summary.json').read_text())
        condition=(m['seed'],m['opponent'],m['order'])
        if condition in seen:
            raise ValueError('Duplicate condition')
        seen.add(condition)
        for p,h in {**m['source_hashes'],**m['physical_sources']}.items():
            if base.digest(p)!=h:
                raise ValueError(f'Source changed: {p}')
        provenance[str(root/'manifest.json')]=base.digest(root/'manifest.json')
        provenance[str(root/'summary.json')]=base.digest(root/'summary.json')
        prefix=f's{m["seed"]}_a{int(m["opponent"])}_o{m["order"]}'
        for row in summary['training']:
            z=read_record(root,row,m);checked_ticks+=len(z['body'])
            traces[f'{prefix}_{Path(row["file"]).stem}']=z['body']
        records={r['file']:r for r in summary['probes']}
        for delay in (0,64):
            for context in (0,1):
                for clip in (0,1):
                    pair=[]
                    for reset in (0,1):
                        name=f'probe-d{delay}-r{reset}-c{context}-v{clip}.npz'
                        z=read_record(root,records[name],m);checked_ticks+=len(z['body'])
                        pair.append(z)
                        ids=list(z['neuron_ids'])
                        p=z['cells'][:,[ids.index(i) for i in m['groups']['prediction']],base.FIELDS.index('O')]
                        torque=z['body'][:,4]+base.FORCE*z['body'][:,3]
                        # Exact per-tick values remain accessible. Common prediction
                        # amplitude is a neural statistic, NOT measured metabolic cost.
                        trace=np.column_stack((z['body'],torque,p[:,0],p[:,1],
                            z['drive'][:,194:198],z['errors'][:,:,1],
                            z['eta'],abs(z['weights']-z['weights_initial']).max(axis=(1,2))))
                        key=f'{prefix}_d{delay}_r{reset}_c{context}_v{clip}'
                        traces[key]=trace
                        cases.append(dict(key=key,seed=m['seed'],opponent=m['opponent'],
                            order=m['order'],delay=delay,reset=reset,context=context,clip=clip,
                            windows=[dict(start=a,stop=b,mean_abs_net_torque=float(abs(torque[a:b]).mean()),
                                peak_abs_angle=float(abs(z['body'][a:b,1]).max()),
                                command_end=float(z['body'][b-1,3]),angle_end=float(z['body'][b-1,1]),
                                max_weight_change=float(abs(z['weights'][a:b]-z['weights_initial']).max()))
                                for a,b in WINDOWS],min_eta=float(z['eta'].min())))
                    if not np.array_equal(pair[0]['body_initial'],pair[1]['body_initial']):
                        raise ValueError('Matched body states differ')
                    if not np.array_equal(pair[0]['delay_initial'],pair[1]['delay_initial']):
                        raise ValueError('Matched delay states differ')
                    if not np.array_equal(pair[0]['drive'][:,:196],pair[1]['drive'][:,:196]):
                        raise ValueError('Physical stimuli/load evidence differ across reset')
    output.mkdir()
    np.savez_compressed(output/'per-tick.npz',**traces)
    result=dict(cases=cases,provenance=provenance,checked_ticks=checked_ticks,
        probe_columns=['time_s','angle_rad','velocity_rad_s','command','load_Nm','net_torque_Nm',
            'prediction_0','prediction_1','supplied_force_0','supplied_force_1','supplied_joint_0',
            'supplied_joint_1','teaching_0','teaching_1','eta_0','eta_1','max_selected_q_change'],
        training_columns=['time_s','angle_rad','velocity_rad_s','command','load_Nm'],
        scope='Windows index full trajectories, not acceptance. Basal plasticity stays positive. '
              'No pure gain-matched comparison, semantic generalization or consciousness claim.')
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(cases=len(cases),checked_ticks=checked_ticks)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--feedback',action='store_true')
    p.add_argument('roots',type=Path,nargs='+')
    a=p.parse_args();(compare_feedback if a.feedback else compare)(a.roots,a.output)
