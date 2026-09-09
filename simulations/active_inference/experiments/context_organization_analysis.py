"""Per-tick causal comparison of the context-bound embodied learning test.

No classifier is fitted. World-applied torque, actual muscle torque, physical
motion and local learned-state changes define the observations. All intervals
are retained; endpoint results do not decide acceptance.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .context_organization import digest, encode, FORCE, verify_learning, verify_physics


def first_difference(a,b):
    x=np.asarray(a)!=np.asarray(b)
    if x.ndim>1:x=x.reshape(len(x),-1).any(axis=1)
    where=np.flatnonzero(x)
    return int(where[0]) if len(where) else None


def load_record(root,row):
    p=root/row['file']
    if digest(p)!=row['sha256']:raise ValueError('Record changed')
    with np.load(p) as z:data={k:z[k] for k in z.files}
    verify_learning(data);verify_physics(data)
    return data


def analyze(root,output):
    root=Path(root).resolve();output=Path(output).resolve()
    m=json.loads((root/'manifest.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    for p,h in {**m['source_hashes'],**m['physical_sources']}.items():
        if digest(p)!=h:raise ValueError('Source changed')
    output.mkdir(exist_ok=False)
    rows={r['file']:r for r in summary['probes']}
    traces={};comparisons=[]
    for ctx in (0,1):
        for clip in (0,1):
            cases={k:load_record(root,rows[f'probe-{k}-c{ctx}-v{clip}.npz']) for k in ('intact','reset')}
            a,b=cases['intact'],cases['reset']
            if not np.array_equal(a['body_initial'],b['body_initial']):raise ValueError('Body branch differs')
            if not np.array_equal(a['drive'][:,:196],b['drive'][:,:196]):raise ValueError('World inputs differ')
            record=dict(context=ctx,clip=clip,
                first_command_difference=first_difference(a['body'][:,3],b['body'][:,3]),
                first_angle_difference=first_difference(a['body'][:,1],b['body'][:,1]),
                interventions={})
            for name,z in cases.items():
                net_torque=z['body'][:,4]+FORCE*z['body'][:,3]
                trace=np.column_stack((z['body'],net_torque,z['eta'].min(axis=1)))
                traces[f'c{ctx}_v{clip}_{name}']=trace
                windows=[]
                for start,stop in ((0,32),(32,64),(64,128),(128,256),(256,300),(300,364)):
                    segment=z['body'][start:stop]
                    windows.append(dict(start=start,stop=stop,
                        mean_abs_net_torque=float(np.mean(abs(net_torque[start:stop]))),
                        peak_abs_angle=float(np.max(abs(segment[:,1]))),
                        mean_command=float(segment[:,3].mean()),
                        angle_start=float(segment[0,1]),angle_end=float(segment[-1,1])))
                record['interventions'][name]=dict(windows=windows,
                    min_eta=float(z['eta'].min()),max_eta=float(z['eta'].max()),
                    max_probe_weight_change=float(np.max(abs(z['weights'][-1]-z['weights_initial']))))
            comparisons.append(record)
    # Acquisition record retained as actual per-tick output and body trajectory.
    acquisition=[];phase=[];bank_weights=[]
    for row in summary['training']:
        z=load_record(root,row)
        acquisition.append(z['body']);phase.extend([row['file']]*len(z['body']))
        bank_weights.append(z['weights'][-1])
    np.savez_compressed(output/'per-tick.npz',**traces,acquisition=np.concatenate(acquisition),
        acquisition_episode=np.array(phase),episode_final_weights=np.array(bank_weights))
    result=dict(source=str(root),manifest_sha256=digest(root/'manifest.json'),
        summary_sha256=digest(root/'summary.json'),seed=m['seed'],contextual=m['contextual'],
        columns=['time_s','angle_rad','velocity_rad_s','command','load_Nm','net_torque_Nm','min_eta'],
        comparisons=comparisons,
        interpretation='Learned-state intervention on an externally signalled two-context force task. '
        'Reset probes continue learning, so late convergence is not memory-free evidence. '
        'No automatic pass criterion, latent-context inference or semantic generalization claim.')
    (output/'summary.json').write_text(encode(result)+'\n')
    print(encode(dict(source=str(root),comparisons=len(comparisons))),flush=True)
    return result


def compare(base,output,seeds=(11,23,44,77)):
    """Compare contextual/blind acquisition and neural teaching-path lesions."""
    base=Path(base).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    rows=[];traces={};retention=[];provenance={};raw_bytes=0
    for seed in seeds:
        for architecture in ('intact','blind'):
            root=base/f'20260909_context_organization_{architecture}_seed{seed}'
            m=json.loads((root/'manifest.json').read_text())
            s=json.loads((root/'summary.json').read_text())
            if m['seed']!=seed or m['contextual']!=(architecture=='intact'):raise ValueError('Wrong condition')
            provenance[str(root/'summary.json')]=digest(root/'summary.json')
            for p,h in {**m['source_hashes'],**m['physical_sources']}.items():
                if digest(p)!=h:raise ValueError('Source changed')
            records={r['file']:r for r in s['training']+s['probes']}
            a=load_record(root,records['train-c0-r3-v0.npz'])
            b=load_record(root,records['train-c1-r3-v0.npz'])
            mask=np.isin(a['context_source_ids'],m['groups']['mixed_0'])
            difference=abs(b['weights'][-1]-a['weights'][-1])
            retention.append(dict(seed=seed,architecture=architecture,
                old_bank_max_change=float(difference[mask].max()),
                new_bank_max_change=float(difference[~mask].max()),
                old_bank_max_weight=float(a['weights'][-1][mask].max())))
            del a,b
            for context in (0,1):
                for clip in (0,1):
                    for state in ('intact','reset'):
                        r=records[f'probe-{state}-c{context}-v{clip}.npz']
                        data=load_record(root,r);raw_bytes+=(root/r['file']).stat().st_size
                        key=f'{architecture}_s{seed}_c{context}_v{clip}_{state}'
                        traces[key]=data['body']
                        rows.append(_case_row(data,seed,architecture,context,clip,state))
        root=base/f'20260909_context_organization_expression_seed{seed}'
        m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text())
        if digest(Path(__file__).with_name('context_organization_expression.py'))!=m['source_sha256']:
            raise ValueError('Expression producer changed')
        provenance[str(root/'summary.json')]=digest(root/'summary.json')
        for r in s['results']:
            data=load_record(root,r);raw_bytes+=(root/r['file']).stat().st_size
            if np.any(data['errors'][:,:,1]!=0):raise ValueError('Teaching lesion has nonzero arrivals')
            key=f'intact_s{seed}_c{r["context"]}_v{r["clip"]}_teaching_cut'
            traces[key]=data['body']
            rows.append(_case_row(data,seed,'intact',r['context'],r['clip'],'teaching_cut'))
    output.mkdir()
    np.savez_compressed(output/'all-probe-body-ticks.npz',**traces)
    result=dict(seeds=list(seeds),cases=rows,retention=retention,provenance=provenance,
        checked_probe_raw_bytes=raw_bytes,body_columns=['time_s','angle_rad','velocity_rad_s','command','load_Nm'],
        limits='Per-window values index retained complete trajectories; they are not an acceptance rule. '
        'Teaching-path lesions retain positive rates and decaying preexisting errors. '
        'Architecture comparison is not activity matched. No proof both senses are needed or model-based reasoning exists.')
    (output/'summary.json').write_text(encode(result)+'\n')
    return result


def _case_row(data,seed,architecture,context,clip,state):
    body=data['body'];net=body[:,4]+FORCE*body[:,3]
    windows=[]
    for start,stop in ((0,32),(32,64),(64,128),(128,256),(256,300),(300,364)):
        windows.append(dict(start=start,stop=stop,
            mean_abs_net_torque=float(abs(net[start:stop]).mean()),
            peak_abs_angle=float(abs(body[start:stop,1]).max()),
            angle_end=float(body[stop-1,1]),command_end=float(body[stop-1,3])))
    return dict(seed=seed,architecture=architecture,context=context,clip=clip,state=state,
        windows=windows,min_eta=float(data['eta'].min()),
        max_weight_change=float(abs(data['weights'][-1]-data['weights_initial']).max()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();analyze(a.root,a.output)
