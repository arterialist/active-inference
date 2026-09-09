"""Full-tick architecture and learned-weight comparison for motor history."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .association_route_probe import digest
from .associative_mismatch_audit import contextual_weight_mask
from .composition_probe import encode
from .temporal_body_probe import audit_history
from .predictive_bridge_probe import audit_record
from .proprioceptive_capacity_audit import envelope
from .proprioceptive_learning_audit import verify_physics,first_difference,intervals


def quantities(data,m):
    index={int(n):i for i,n in enumerate(data['neuron_ids'])}
    bridge=m['bridge']
    pos=data['cells'][:,[index[n] for n in bridge['error_positive']],1]
    neg=data['cells'][:,[index[n] for n in bridge['error_negative']],1]
    return dict(error=pos+neg,prediction=data['cells'][:,[index[n] for n in bridge['prediction']],1],
        joint_input=data['joint_input'],rate=data['eta'])


def describe(effect):
    # Preserve signs for each channel and the full trajectory, not just a mean.
    mean=effect.mean(axis=1)
    return dict(negative_mean_intervals=intervals(mean<0),negative_ticks_per_channel=(effect<0).sum(axis=0),
        minimum_mean=float(mean.min()),maximum_mean=float(mean.max()),
        last164_mean=float(mean[-164:].mean()),last164_per_channel=effect[-164:].mean(axis=0))


def run(short,multiscale,branches,output):
    roots=dict(short=Path(short).resolve(),multiscale=Path(multiscale).resolve())
    manifests={};configs={};data={};sources=[];arrays={};report={};max_residual=0.
    for mode,root in roots.items():
        m=json.loads((root/'manifest.json').read_text());manifests[mode]=m
        cfg=json.loads((root/'config.json').read_text());configs[mode]=cfg
        for p,h in m['source_hashes'].items():
            if digest(p)!=h:raise ValueError('Changed acquisition runtime: '+p)
        done=json.loads((root/'summary.json').read_text())
        if digest(root/'closed-loop.npz')!=done['raw_sha256']:raise ValueError('Changed acquisition raw data')
        with np.load(root/'closed-loop.npz') as z:d={k:z[k] for k in z.files}
        data[mode]=d
        max_residual=max(max_residual,audit_history(d,cfg,m['basis']),audit_record(d,cfg,m['bridge']),verify_physics(d,m['gain']))
        q=quantities(d,m)
        for key,value in q.items():arrays[mode+'_acquisition_'+key]=value
        ps=m['bridge']['prediction'];nodes={n['id']:n for n in cfg['neurons']}
        upper=envelope(d['arrivals'],np.array([nodes[n]['params']['lambda_param'] for n in ps]),
            np.array([nodes[n]['params']['delta_decay'] for n in ps]),
            np.array([nodes[n]['metadata']['prediction_cap'] for n in ps]))
        ix={int(n):i for i,n in enumerate(d['neuron_ids'])}
        target=d['cells'][:,[ix[n] for n in m['motor']['joint_position']],1]
        mask=target>upper+.01;mask[:64]=False
        if np.any(q['prediction']>upper):raise ValueError('Recorded output exceeds envelope')
        arrays[mode+'_capacity_upper']=upper;arrays[mode+'_capacity_exceed']=mask
        report[mode]=dict(capacity_exceed_ticks_per_channel=mask.sum(axis=0),
            context_sum_min_after256=float(d['arrivals'][256:].sum(axis=2).min()))
        sources.append(dict(path=str(root/'closed-loop.npz'),sha256=digest(root/'closed-loop.npz')))
    # The experimental structural contrast is lambda alone, not fan-in/gain.
    a,b=(deepcopy(configs[k]) for k in ('short','multiscale'))
    for cfg in (a,b):
        cfg['metadata']['temporal_basis'].pop('mode')
        for n in cfg['neurons']:
            if n['id'] in manifests['short']['basis']:n['params'].pop('lambda_param')
    if a!=b:raise ValueError('Architecture contrast includes undeclared changes')
    for key in data['short']:
        if key.startswith('start_'):np.testing.assert_array_equal(data['short'][key],data['multiscale'][key])
    body_difference=first_difference(data['short']['physical_after'],data['multiscale']['physical_after'])
    sensory_difference=first_difference(data['short']['joint_input'],data['multiscale']['joint_input'])
    effect=arrays['short_acquisition_error']-arrays['multiscale_acquisition_error']
    arrays['acquisition_multiscale_benefit']=effect
    report['architecture']=dict(benefit=describe(effect),first_physical_difference=body_difference,
        first_sensory_difference=sensory_difference)
    for mode in roots:
        reference=None;mode_manifest=manifests[mode];cfg=configs[mode]
        mask=contextual_weight_mask(cfg,mode_manifest['bridge'])
        for condition in ('learned','birth','shuffled'):
            root=Path(branches).resolve()/f'20260909_temporal_branch_{mode}_{condition}_seed{mode_manifest["seed"]}'
            m=json.loads((root/'manifest.json').read_text())
            if m['mode']!=mode or m['condition']!=condition or Path(m['source'])!=roots[mode]:
                raise ValueError('Unmatched branch')
            for p,h in m['source_hashes'].items():
                if digest(p)!=h:raise ValueError('Changed branch parent')
            done=json.loads((root/'summary.json').read_text())
            if digest(root/'closed-loop.npz')!=done['raw_sha256']:raise ValueError('Changed branch raw')
            with np.load(root/'closed-loop.npz') as z:d={k:z[k] for k in z.files}
            max_residual=max(max_residual,audit_history(d,cfg,m['basis']),audit_record(d,cfg,m['bridge']),verify_physics(d,m['gain']))
            if reference is None:reference=d
            for key in d:
                if key.startswith('start_') and key not in ('start_incoming_info','start_weights'):
                    np.testing.assert_array_equal(d[key],reference[key])
            np.testing.assert_array_equal(d['start_incoming_info'][~mask],reference['start_incoming_info'][~mask])
            np.testing.assert_array_equal(d['physical_before'][0],reference['physical_before'][0])
            q=quantities(d,m)
            for key,value in q.items():arrays[f'{mode}_{condition}_{key}']=value
            if condition!='learned':
                benefit=q['error']-arrays[f'{mode}_learned_error']
                arrays[f'{mode}_{condition}_minus_learned_error']=benefit
                report[mode][condition]=dict(benefit=describe(benefit),
                    first_error_difference=first_difference(q['error'],arrays[f'{mode}_learned_error']),
                    first_physical_difference=first_difference(d['physical_after'],reference['physical_after']),
                    first_sensory_difference=first_difference(d['joint_input'],reference['joint_input']))
            sources.append(dict(path=str(root/'closed-loop.npz'),sha256=done['raw_sha256']))
    output=Path(output).resolve();output.mkdir(exist_ok=False)
    np.savez_compressed(output/'effects-per-tick.npz',**arrays)
    result=dict(sources=sources,max_residual=max_residual,results=report,
        interpretation='Acquisition: short minus multiscale error. Branches: control minus learned error. '
        'Positive means less opponent error in the named candidate, not semantic accuracy. '
        'Retain all negative intervals and channel effects. Matched architecture and within-mode weight-only branches. '
        'No independent original graph seeds, body perturbation generalization or learned action-selection claim.')
    (output/'summary.json').write_text(encode(result)+'\n');print(encode(report),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('short','multiscale','branches','output'):p.add_argument('--'+key,type=Path,required=True)
    run(**vars(p.parse_args()))
