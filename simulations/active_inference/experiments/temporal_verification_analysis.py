"""Full-trajectory comparison of learned action under delayed verification."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import context_organization as base
from .opponent_context_analysis import read_record
from .temporal_verification import verify_learning


WINDOWS=((0,32),(32,64),(64,128),(128,256),(256,300),(300,364))


def compare_transplants(roots, output):
    """Audit the selected-weight intervention without inheriting body displacement."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    traces={};cases=[];provenance={};conditions=set();checked_ticks=0
    for root in map(lambda p:Path(p).resolve(),roots):
        m=json.loads((root/'manifest.json').read_text())
        summary=json.loads((root/'summary.json').read_text())
        identity=(m['seed'],m['order'],m['aligned'])
        if identity in conditions:raise ValueError('Duplicate condition')
        conditions.add(identity)
        if base.digest(Path(__file__).with_name('temporal_memory_transplant.py'))!=m['producer_sha256']:
            raise ValueError('Transplant producer changed')
        source=Path(m['source']);parent=json.loads((source/'manifest.json').read_text())
        for path,h in ((source/'manifest.json',m['source_manifest_sha256']),
                       (source/'initial.paula',m['initial_checkpoint_sha256']),
                       (source/'final.paula',m['learned_checkpoint_sha256'])):
            if base.digest(path)!=h:raise ValueError('Parent evidence changed')
        for p,h in {**parent['source_hashes'],**parent['physical_sources']}.items():
            if base.digest(p)!=h:raise ValueError(f'Source changed: {p}')
        for field in ('seed','order','aligned','groups','selected'):
            if m[field]!=parent[field]:raise ValueError('Parent configuration mismatch')
        if not summary['birth_replay_exact']:raise ValueError('No exact birth replay')
        provenance[str(root/'manifest.json')]=base.digest(root/'manifest.json')
        provenance[str(root/'summary.json')]=base.digest(root/'summary.json')
        rows={(r['weights'],r['context'],r['clip']):r for r in summary['results']}
        expected={(w,c,v) for w in ('birth','learned') for c in (0,1) for v in (0,1)}
        if set(rows)!=expected or len(summary['results'])!=8:raise ValueError('Incomplete/duplicate probes')
        for context in (0,1):
            for clip in (0,1):
                pair={}
                for weights in ('birth','learned'):
                    z=read_record(root,rows[weights,context,clip],m,learning_auditor=verify_learning)
                    if len(z['body'])!=96:raise ValueError('Wrong probe duration')
                    checked_ticks+=len(z['body']);pair[weights]=z
                    ids=list(z['neuron_ids'])
                    p=z['cells'][:,[ids.index(n) for n in m['groups']['prediction']],base.FIELDS.index('O')]
                    direction=np.sign(z['body'][:,4])
                    if not np.all(direction==(-1 if context^clip else 1)):
                        raise ValueError('Unexpected physical contingency')
                    torque=z['body'][:,4]+base.FORCE*z['body'][:,3]
                    prediction=(p[:,0]-p[:,1])*direction
                    dq=abs(z['weights']-z['weights_initial']).max(axis=(1,2))
                    key=f's{m["seed"]}_o{m["order"]}_a{int(m["aligned"])}_{weights}_c{context}_v{clip}'
                    traces[key]=np.column_stack((z['body'],torque,p,prediction,
                        z['body'][:,1]*direction,z['errors'][:,:,1],z['eta'],dq))
                    cases.append(dict(key=key,seed=m['seed'],order=m['order'],aligned=m['aligned'],
                        weights=weights,context=context,clip=clip,min_eta=float(z['eta'].min()),
                        wrong_prediction_ticks=np.flatnonzero(prediction<0).tolist(),
                        windows=[dict(start=a,stop=b,min_signed_prediction=float(prediction[a:b].min()),
                            max_signed_prediction=float(prediction[a:b].max()),
                            mean_abs_net_torque=float(abs(torque[a:b]).mean()),
                            peak_abs_angle=float(abs(z['body'][a:b,1]).max()),
                            angle_end=float(z['body'][b-1,1]),max_weight_change=float(dq[a:b].max()),
                            max_abs_teaching=float(abs(z['errors'][a:b,:,1]).max()))
                            for a,b in ((0,32),(32,64),(64,96))]))
                for field in ('body_initial','delay_initial','context_initial','error_initial','neuron_ids'):
                    if not np.array_equal(pair['birth'][field],pair['learned'][field]):
                        raise ValueError(f'Branch mismatch: {field}')
                if any(np.any(pair['birth'][field]) for field in ('delay_initial','context_initial','error_initial','weights_initial')):
                    raise ValueError('Birth history is not empty')
                if not np.array_equal(pair['birth']['body_initial'],base.Arm().state()):
                    raise ValueError('Body does not start at rest')
                if not np.array_equal(pair['birth']['drive'][:,:196],pair['learned']['drive'][:,:196]):
                    raise ValueError('External stimuli differ')
                if any(np.any(z['drive'][:64,194:198]) for z in pair.values()):
                    raise ValueError('Fresh bodily evidence arrived early')
    if not conditions:raise ValueError('No experimental conditions')
    for seed,order,aligned in conditions:
        if (seed,order,not aligned) not in conditions:raise ValueError('Missing paired architecture')
    output.mkdir();np.savez_compressed(output/'per-tick.npz',**traces)
    result=dict(cases=cases,provenance=provenance,checked_ticks=checked_ticks,
        columns=['time_s','angle_rad','velocity_rad_s','command','load_Nm','net_torque_Nm',
                 'prediction_0','prediction_1','prediction_along_load','angle_along_load',
                 'teaching_0','teaching_1','eta_0','eta_1','max_selected_q_change'],
        limits='Selected-weight transfer into birth brain and resting body; not full engram transfer. '
               '64 ticks without new somatic evidence, all plasticity active. Sign alone is not acceptance. '
               'Birth replay reported by producer; analysis reaudits all 768 probe ticks per source.')
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(cases=len(cases),checked_ticks=checked_ticks)),flush=True)
    return result


def compare(roots,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    traces={};cases=[];provenance={};conditions=set();checked_ticks=0;acquisition=[];retention=[]
    for root in map(lambda p:Path(p).resolve(),roots):
        m=json.loads((root/'manifest.json').read_text())
        summary=json.loads((root/'summary.json').read_text())
        identity=(m['seed'],m['order'],m['aligned'])
        if identity in conditions:raise ValueError('Duplicate condition')
        conditions.add(identity)
        for p,h in {**m['source_hashes'],**m['physical_sources']}.items():
            if base.digest(p)!=h:raise ValueError(f'Source changed: {p}')
        provenance[str(root/'manifest.json')]=base.digest(root/'manifest.json')
        provenance[str(root/'summary.json')]=base.digest(root/'summary.json')
        prefix=f's{m["seed"]}_o{m["order"]}_a{int(m["aligned"])}'
        previous_context=None
        for row in summary['training']:
            z=read_record(root,row,m,learning_auditor=verify_learning);checked_ticks+=len(z['body'])
            traces[f'{prefix}_{Path(row["file"]).stem}']=z['body']
            acquisition.append(dict(key=f'{prefix}_{row["file"]}',seed=m['seed'],aligned=m['aligned'],
                order=m['order'],max_selected_q=float(z['weights'].max()),
                min_eta=float(z['eta'].min()),max_eta=float(z['eta'].max()),
                max_abs_net_torque=float(abs(z['body'][:,4]+.2*z['body'][:,3]).max())))
            if previous_context is not None:
                mask=np.isin(z['context_source_ids'],m['groups']['mixed_0'])
                change=abs(z['weights']-previous_context)
                traces[f'{prefix}_retention_{Path(row["file"]).stem}']=np.column_stack((
                    change[:,mask].max(axis=1),change[:,~mask].max(axis=1),
                    z['arrivals'][:,mask].max(axis=1)))
                retention.append(dict(seed=m['seed'],order=m['order'],aligned=m['aligned'],file=row['file'],
                    old_bank_end_max_change=float(change[-1][mask].max()),
                    old_bank_incoming_end_max=float(z['arrivals'][-1][mask].max())))
            if row['file']==f'train-c0-r3-v{m["order"]}.npz':
                previous_context=z['weights'][-1].copy()
        records={r['file']:r for r in summary['probes']}
        for context in (0,1):
            for clip in (0,1):
                pair=[]
                for reset in (0,1):
                    row=records[f'probe-r{reset}-c{context}-v{clip}.npz']
                    z=read_record(root,row,m,learning_auditor=verify_learning);checked_ticks+=len(z['body'])
                    pair.append(z)
                    ids=list(z['neuron_ids']);pids=m['groups']['prediction']
                    prediction=z['cells'][:,[ids.index(n) for n in pids],base.FIELDS.index('O')]
                    net=z['body'][:,4]+.2*z['body'][:,3]
                    dq=abs(z['weights']-z['weights_initial']).max(axis=(1,2))
                    key=f'{prefix}_r{reset}_c{context}_v{clip}'
                    traces[key]=np.column_stack((z['body'],net,prediction,z['errors'][:,:,1],z['eta'],dq))
                    cases.append(dict(key=key,seed=m['seed'],order=m['order'],aligned=m['aligned'],
                        context=context,clip=clip,reset=reset,min_eta=float(z['eta'].min()),
                        windows=[dict(start=a,stop=b,mean_abs_net_torque=float(abs(net[a:b]).mean()),
                            peak_abs_angle=float(abs(z['body'][a:b,1]).max()),
                            angle_end=float(z['body'][b-1,1]),max_weight_change=float(dq[a:b].max()),
                            max_abs_teaching=float(abs(z['errors'][a:b,:,1]).max())) for a,b in WINDOWS]))
                for key in ('body_initial','delay_initial','context_initial','error_initial'):
                    if not np.array_equal(pair[0][key],pair[1][key]):raise ValueError(f'Branch mismatch: {key}')
                if not np.array_equal(pair[0]['drive'][:,:196],pair[1]['drive'][:,:196]):
                    raise ValueError('Matched physical stimuli differ')
    if not conditions:raise ValueError('No experimental conditions')
    for seed,order,aligned in conditions:
        if (seed,order,not aligned) not in conditions:raise ValueError('Missing paired architecture')
    output.mkdir();np.savez_compressed(output/'per-tick.npz',**traces)
    result=dict(cases=cases,acquisition=acquisition,retention=retention,provenance=provenance,checked_ticks=checked_ticks,
        probe_columns=['time_s','angle_rad','velocity_rad_s','command','load_Nm','net_torque_Nm',
                       'prediction_0','prediction_1','teaching_0','teaching_1','eta_0','eta_1','max_selected_q_change'],
        training_columns=['time_s','angle_rad','velocity_rad_s','command','load_Nm'],
        retention_columns=['old_bank_max_q_change','new_bank_max_q_change','old_bank_max_arrival'],
        limits='Within-seed paired timing comparison. Orders balanced across seeds, not factorial. '
               'No acceptance from summaries; full traces retained. Selected reset keeps learning active.')
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(cases=len(cases),checked_ticks=checked_ticks)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('roots',type=Path,nargs='+')
    p.add_argument('--transplants',action='store_true')
    a=p.parse_args();(compare_transplants if a.transplants else compare)(a.roots,a.output)
