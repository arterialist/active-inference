"""Data-only verification of pre-spike observations and a release intervention."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from .eligibility_media_audit import check_branch_start,verify_ledger


def read_state(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def check_thresholds(rows,cells,nid,initial,trial):
    if len(rows)!=len(cells):raise ValueError('Missing threshold ticks')
    last=initial['neurons'][str(nid)]['t_last_fire'];previous=initial['neurons'][str(nid)]['S']
    margins=[]
    for t,row in enumerate(rows):
        numeric=('pre_S','old_S','I_t','dS','active_threshold','r','b','c','dt','lambda_param')
        if not all(np.isfinite(row[k]) for k in numeric) or row['dt']<=0 or row['lambda_param']<=0 or row['c']<0:
            raise ValueError('Nonfinite or invalid threshold observation')
        if row['S_dtype'] not in ('float32','float64') or row['I_dtype'] not in ('float32','float64'):
            raise ValueError('Unsupported scalar precision')
        tick=trial['start']+t;since=None if last is None else tick-last
        if row['tick']!=tick or row['neuron']!=nid or row['since_last']!=since or row['old_S']!=previous:
            raise ValueError('Threshold state continuity differs')
        active=row['b'] if since is not None and since<=row['c'] else row['r']
        if abs(row['pre_S'])<.005:active=row['r']
        if row['active_threshold']!=active:raise ValueError('Wrong active threshold')
        # Reproduce the recorded scalar precision, not an assumed float64 flow.
        old=np.asarray(row['old_S'],dtype=row['S_dtype'])[()]
        current=np.asarray(row['I_t'],dtype=row['I_dtype'])[()]
        # Native I_t starts as Python 0.0 on an empty-arrival tick. NumPy 2
        # treats it as a weak scalar. A strong np.float64 zero would promote
        # the float32 S update; the recorded float32 result rules that out.
        if row['S_dtype']=='float32' and row['I_dtype']=='float64' and row['I_t']==0.:
            current=0.
        increment=(row['dt']/row['lambda_param'])*(-old+current)
        integrated=float(np.clip(old+increment,-1000.,1000.))
        if abs(float(increment)-row['dS'])>1e-12 or abs(integrated-row['pre_S'])>1e-12:
            raise ValueError('Recorded membrane integration differs')
        fire=row['pre_S']>=active and (since is None or since>=row['c'])
        if row['will_fire']!=fire or bool(cells[t,nid-1,1]>0)!=fire or cells[t,nid-1,5]!=row['r']:
            raise ValueError('Firing rule or output differs')
        expected=0. if fire else row['pre_S']
        if cells[t,nid-1,0]!=expected:raise ValueError('Post-spike reset differs')
        if fire:last=tick
        previous=expected;margins.append(row['pre_S']-active)
    return np.array(margins)


def check_release(evidence,source,tick,block):
    before,after=evidence['before'],evidence['after']
    if evidence['source']!=source or evidence['tick']!=tick or evidence['block']!=block or before['tick']!=tick+1:
        raise ValueError('Release timing differs')
    found=[r for r in before['presynaptic_wheel'] if r[0]==tick+1 and r[1]=='tuple' and r[2][0]==source]
    if found!=evidence['found'] or evidence['removed']!=(found if block else []):
        raise ValueError('Release selection differs')
    expected=deepcopy(before)
    for row in evidence['removed']:expected['presynaptic_wheel'].remove(row)
    if expected!=after:raise ValueError('Undeclared release-state change')
    if block and (not found or before['neurons'][str(source)]['O']<=0):raise ValueError('No actual spike to block')


def audit(recording,output):
    root,output=Path(recording),Path(output)
    m=json.loads((root/'manifest.json').read_text());summary=json.loads((root/'summary.json').read_text())
    if not summary['training_exact'] or not summary['observer_controls_exact']:raise ValueError('Replay check failed')
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in m['source_files_sha256'].items()):
        raise ValueError('Source artifact changed')
    source=Path(m['source']);old=json.loads((source/'manifest.json').read_text())
    cfg=json.loads((source/'config.json').read_text());ports=old['selected_ports'];ns={n['id']:n for n in cfg['neurons']}
    if list(ns)!=list(range(1,len(ns)+1)) or any(n['params']['eta_post']<=0 or n['params']['eta_retro']<=0 for n in ns.values()):
        raise ValueError('Unexpected cell indexing or frozen adaptation')
    parent=read_state(source/'training-final-state.json.gz');donor=read_state(Path(m['opposite'])/'training-final-state.json.gz')
    order=[(nid,p['synapse_id']) for nid in ns for p in cfg['synaptic_points'] if p['type']=='postsynaptic' and p['neuron_id']==nid]
    lookup={v:i for i,v in enumerate(order)};take=[lookup[n,s] for n,s,_ in ports]
    if m['neuron'] in {src for _,_,src in ports}:raise ValueError('This ledger audit requires an unselected outgoing source')
    original={(p['neuron_id'],p['synapse_id']):p['u_i']['info'] for p in cfg['synaptic_points'] if p['type']=='postsynaptic'}
    conditions=('unchanged','opposite_mean','opposite_residual','opposite_selected','residual_block')
    if tuple(m['conditions'])!=conditions:raise ValueError('Wrong condition set')
    trial=summary['trial']
    if trial!=dict(start=parent['tick'],stop=parent['tick']+old['clip_ticks'],visual_clip=m['clip'],audio_clip=None):
        raise ValueError('Wrong stimulus protocol')
    data={};details={};curves={};max_error=0.
    for row in summary['branches']:
        c=row['condition']
        if c not in conditions or c in data:raise ValueError('Unexpected branch')
        for name,hash_key in (('file','sha256'),('detail','detail_sha256')):
            if hashlib.sha256((root/row[name]).read_bytes()).hexdigest()!=row[hash_key]:raise ValueError('Branch changed')
        detail=read_state(root/row['detail']);start=detail['start'];factor='opposite_residual' if c=='residual_block' else c
        check_branch_start(start,parent,donor,original,ports,factor)
        with np.load(root/row['file']) as z:
            if not np.array_equal(z['incoming_info_before'],[start['neurons'][str(n)]['synapses'][str(s)][0] for n,s in order]):
                raise ValueError('Wrong initial weights')
            q=np.array([start['neurons'][str(n)]['synapses'][str(s)][0] for n,s,_ in ports])
            pre=np.array([start['eligibility'][str(n)]['pre'][ns[n]['metadata']['eligibility_ports'].index(s)] for n,s,_ in ports])
            post=np.array([start['eligibility'][str(n)]['post'] for n,_,_ in ports])
            spikes=np.array([start['neurons'][str(n)]['O']>0 for n in ns])
            end,_,_,_,error,_=verify_ledger(z,cfg,ports,q,pre,post,spikes)
            if not np.array_equal(end,z['incoming_info_after'][take]):raise ValueError('Wrong final weights')
            max_error=max(max_error,error)
            if c!='residual_block':
                with np.load(Path(m['reference'])/f'{c}-{m["clip"]}.npz') as ref:
                    if set(ref.files)!=set(z.files) or any(not np.array_equal(z[k],ref[k]) for k in z.files):raise ValueError('Passive observation differs')
            data[c]=z['cells'];details[c]=detail
            curves['margin/'+c]=check_thresholds(detail['thresholds'],z['cells'],m['neuron'],start,trial)
        check_release(detail['release'],m['neuron'],trial['start']+m['probe_tick'],c=='residual_block')
    if set(data)!=set(conditions):raise ValueError('Missing branches')
    if details['residual_block']['release']['before']!=details['opposite_residual']['release']['before']:
        raise ValueError('Conditions differ before intervention')
    if not np.array_equal(data['residual_block'][:m['probe_tick']+1],data['opposite_residual'][:m['probe_tick']+1]):
        raise ValueError('Intervention has an anticipatory effect')
    effects={}
    for role,ids in {**old['groups'],'all_neurons':list(ns)}.items():
        effects[role]={};idx=np.array(ids)-1
        for c in ('opposite_residual','residual_block'):
            for reference in ('unchanged','opposite_residual'):
                a=data[c][:,idx,1]>0;b=data[reference][:,idx,1]>0;changed=a!=b;t=np.flatnonzero(changed.any(axis=1))
                key=c+'/'+reference
                effects[role][key]=dict(changed_cell_ticks=int(changed.sum()),first_tick=int(t[0]) if len(t) else None,
                    changed_cells_per_tick=changed.sum(axis=1).tolist())
                curves[role+'/'+key]=changed
    output.mkdir(parents=True,exist_ok=False);np.savez_compressed(output/'ticks.npz',**curves)
    result=dict(structurally_valid=True,recording=str(root),max_update_residual=max_error,effects=effects,
        thresholds_at_intervention={c:details[c]['thresholds'][m['probe_tick']] for c in conditions},
        removed_releases=details['residual_block']['release']['removed'],
        limits='One selected case. A transmission block tests causal influence, not memory content, recognition, useful reasoning or consciousness. Local spike history and retrograde events are intentionally retained.')
    (output/'summary.json').write_text(json.dumps(result,allow_nan=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.recording,a.output)
