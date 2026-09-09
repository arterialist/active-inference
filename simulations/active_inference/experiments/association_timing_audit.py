"""Data-only temporal audit separating prefix responses from completed cues."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from .association_balance_audit import checked,IntegrationAudit
from .association_continual_audit import verify_dynamics
from .association_cue_audit import classify_response


def temporal_response(consumer,schedule,coordinates):
    other=[i for i in range(32) if i not in coordinates];rows=[]
    for pulse in (0,8,16,24):
        active=np.flatnonzero(schedule[pulse:pulse+4].any(axis=1))
        if not len(active):continue
        # Five ticks from an external receptor pulse to this specific neural
        # consumer. This is a declared feedforward alignment, not a decision
        # deadline imposed on arbitrary brains or an invented ms conversion.
        begin=pulse+5;completion=pulse+int(active[-1])+5;end=pulse+13
        late=consumer[completion:end];right=int(late[:,coordinates].sum());wrong=int(late[:,other].sum())
        status='both' if right and wrong else 'expected_only' if right else 'other_only' if wrong else 'silent'
        rows.append(dict(pulse=pulse,first_input=pulse+int(active[0]),last_input=pulse+int(active[-1]),
            aligned_completion=completion,stop=end,late_status=status,
            late_expected_coverage=int(late[:,coordinates].any(axis=0).sum()),
            late_other_coverage=int(late[:,other].any(axis=0).sum()),
            prefix_expected_spikes=int(consumer[begin:completion,coordinates].sum()),
            prefix_other_spikes=int(consumer[begin:completion,other].sum())))
    return rows


def audit(root,output):
    root,output=Path(root),Path(output)
    m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text())
    source=Path(m['source']);old=json.loads((source/'manifest.json').read_text());cfg=json.loads((source/'config.json').read_text());g=old['groups']
    if not s['training_exact'] or not s['synchronous_controls_exact'] or old['sensitivity']!=.25:raise ValueError('Wrong acquisition/control')
    schedule_path=root/'schedules.npz'
    if hashlib.sha256(schedule_path.read_bytes()).hexdigest()!=s['schedule_sha256']:raise ValueError('Changed schedules')
    with np.load(schedule_path) as z:schedules={k:z[k] for k in z.files}
    starts={}
    for checkpoint in (32,128):
        with gzip.open(checked(root,s['starts'][str(checkpoint)]),'rt') as f:starts[checkpoint]=json.load(f)
        with gzip.open(source/f'{checkpoint}-trained-start.json.gz','rt') as f:
            if starts[checkpoint]!=json.load(f):raise ValueError('Changed checkpoint')
    cases={c['name']:c for c in m['cases']};timings=('synchronous','shared_delay','fixed_spread','resampled_spread','minority_first','majority_first')
    if m['cases']!=old['cases'] or tuple(m['timings'])!=timings or m['checkpoints']!=[32,128]:raise ValueError('Changed physical protocol')
    required={(k,c,t) for k in starts for c in cases for t in timings};seen=set();rows=[];evidence={};er=0.;ir=0.
    for item in s['probes']:
        key=item['checkpoint'],item['case'],item['timing']
        if key not in required or key in seen:raise ValueError('Unexpected/duplicate probe')
        seen.add(key);checkpoint,name,timing=key;case=cases[name];schedule=schedules[name+'/'+timing]
        if schedule.shape!=(64,32) or not np.isin(schedule,[0,1]).all():raise ValueError('Wrong schedule shape')
        selected=case['selected_receptors'];own=set(old['masks']['vision'][case['cue']]);expected=np.zeros(32,int);expected[[n-1 for n in selected]]=4
        if not np.array_equal(schedule.sum(axis=0),expected):raise ValueError('Receptor dose changed')
        for pulse in (0,8,16,24):
            expected_block=np.zeros(32,int);expected_block[[n-1 for n in selected]]=1
            if not np.array_equal(schedule[pulse:pulse+4].sum(axis=0),expected_block):raise ValueError('Volleys overlap or dose differs')
            for nid in selected:
                at=int(np.flatnonzero(schedule[pulse:pulse+4,nid-1])[0])
                if timing=='synchronous' and at!=0 or timing=='shared_delay' and at!=3:raise ValueError('Wrong common delay')
                if timing=='minority_first' and at!=(3 if nid in own else 0):raise ValueError('Wrong minority-first order')
                if timing=='majority_first' and at!=(0 if nid in own else 3):raise ValueError('Wrong majority-first order')
        if timing=='fixed_spread' and any(not np.array_equal(schedule[:4],schedule[t:t+4]) for t in (8,16,24)):raise ValueError('Fixed spread changed')
        start=starts[checkpoint]
        iq=np.array([[start['neurons'][str(n)]['synapses'][str(i)][0] for i in range(35,67)] for n in g['auditory']])
        last=np.array([start['neurons'][str(n)]['t_last_fire'] for n in g['auditory']])
        with np.load(checked(root,item)) as z:raw={k:z[k] for k in z.files}
        retina=np.concatenate((np.zeros((1,32),bool),schedule[:-1].astype(bool)))
        if not np.array_equal(retina,raw['states'][:,:32,1]>0) or raw['states'][:,32:64,1].any():raise ValueError('Sensory spike/dropout mismatch')
        _,_,dynamics,e,i=verify_dynamics(raw,start,IntegrationAudit(cfg,start,g),iq,last,cfg,g,.25);er=max(er,e);ir=max(ir,i)
        if timing=='synchronous':
            with np.load(source/f'{checkpoint}-trained-{name}.npz') as z:
                if any(not np.array_equal(raw[k],z[k]) for k in raw):raise ValueError('Scheduled control differs')
        consumer=raw['states'][:,96:128,1]>0;sound=case['cue'] if old['mapping']=='paired' else 1-case['cue'];coords=[n-33 for n in old['masks']['audio'][sound]]
        rows.append(dict(checkpoint=checkpoint,case=name,kind=case['kind'],timing=timing,
            **classify_response(consumer,coords,case['kind']),volleys=temporal_response(consumer,schedule,coords)))
        prefix=f'{checkpoint}/{name}/{timing}';evidence[prefix+'/consumer']=consumer;evidence[prefix+'/preactivation']=dynamics
    if seen!=required:raise ValueError('Missing probes')
    output.mkdir(parents=True,exist_ok=False);np.savez_compressed(output/'dynamics.npz',**evidence)
    result=dict(structurally_valid=True,seed=old['seed'],mapping=old['mapping'],selected_update_residual=er,
        inhibitory_update_residual=ir,probes=rows,
        limits='Strict whole-probe selectivity is descriptive, not sufficient to judge temporal behavior. Per-volley late responses follow the last input plus the verified five-tick feedforward path. No learned decision consumer or body is tested.')
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.recording,a.output)
