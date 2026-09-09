"""Data-only audit of sensory balance, learning and actual neural outputs.

Reconstructs auditory dendritic queues and every somatic threshold decision
from recorded incoming signals and pre-update weights. No simulator imports.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np

from .association_cue_audit import initial_snapshot, classify_response
from .eligibility_association_audit import verify_trace


class Record(dict):
    @property
    def files(self): return list(self)


class IntegrationAudit:
    def __init__(self, cfg, start, groups):
        self.groups=groups;self.tick=start['tick'];self.rows=[]
        self.params={n['id']:n['params'] for n in cfg['neurons']}
        self.distances={(p['neuron_id'],p['synapse_id']):p['distance_to_hillock']
                        for p in cfg['synaptic_points'] if p['type']=='postsynaptic'}
        self.soma={n:start['neurons'][str(n)]['S'] for n in groups['auditory']}
        self.last={n:start['neurons'][str(n)]['t_last_fire'] for n in groups['auditory']}
        self.queues={n:[tuple(x) for x in start['neurons'][str(n)]['dendritic_queue']] for n in groups['auditory']}
        if any(self.queues.values()):raise ValueError('This preparation begins after a quiet interval')

    def check(self, raw):
        # NPZ __getitem__ decompresses on every access. Materialize once per
        # bounded trial, never inside the neuron/synapse loops.
        raw={k:raw[k] for k in ('states','all_arrivals','all_input_weights','all_input_plast')}
        current=[]
        for t in range(len(raw['states'])):
            row=[]
            for j,n in enumerate(self.groups['auditory']):
                p=self.params[n];q=self.queues[n]
                if p['lambda_param']!=1. or any(p['w_r']) or any(p['w_b']):raise ValueError('Changed operating rule')
                for sid in np.flatnonzero(raw['all_arrivals'][t,j]>0):
                    v=np.float32(raw['all_arrivals'][t,j,sid])*float(raw['all_input_weights'][t,j,sid]+raw['all_input_plast'][t,j,sid])
                    heapq.heappush(q,(self.tick+self.distances[n,int(sid)],'hillock',v,int(sid)))
                total=0.;exc=0.;inh=0.
                while q and q[0][0]<=self.tick:
                    _,_,v,sid=heapq.heappop(q)
                    arriving=v*(p['delta_decay']**self.distances[n,sid]);total+=arriving
                    if arriving<0:inh+=float(arriving)
                    else:exc+=float(arriving)
                soma=self.soma[n];soma+=(1./p['lambda_param'])*(-soma+total)
                age=np.inf if self.last[n] is None else self.tick-self.last[n]
                threshold=p['b_base'] if age<=p['c'] else p['r_base']
                if abs(soma)<.005:threshold=p['r_base']
                fired=bool(soma>=threshold and age>=p['c'])
                row.append((float(soma),threshold,exc,inh))
                if fired:soma=0.;self.last[n]=self.tick
                self.soma[n]=soma
                actual=raw['states'][t,n-1]
                if fired!=(actual[1]>0) or float(soma)!=actual[0] or actual[5]!=p['r_base']:
                    raise ValueError(f'Somatic reconstruction differs tick={self.tick} neuron={n}: {soma}, {actual[0]}, fire={fired}/{actual[1]}')
            current.append(row);self.tick+=1
        return np.asarray(current)


def read_gzip(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def checked(root,item):
    path=root/item['file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Changed artifact')
    return path


def audit(recording,output):
    root,output=Path(recording),Path(output)
    m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text())
    cfg=json.loads((root/'config.json').read_text());g=m['groups']
    if len(cfg['neurons'])!=176 or not s['observer_controls_exact'] or m['mapping'] not in ('paired','swapped'):
        raise ValueError('Wrong preparation')
    if any(n['params']['eta_post']<=0 or n['params']['eta_retro']<=0 for n in cfg['neurons']):raise ValueError('Frozen learning')
    if {e['target_neuron'] for e in cfg['external_inputs']}!=set(g['vision']+g['audio']):raise ValueError('Hidden input')
    initial=initial_snapshot(cfg);trained=read_gzip(root/'trained-state.json.gz')
    residual=0.;starts={};configs={};rows=[];evidence={}
    required_states={'initial','trained','reset_selected','remove_inhibition','restore_threshold'}
    if m.get('source'):
        if not s.get('training_exact') or not s.get('unchanged_controls_exact'):raise ValueError('Rescue controls differ')
        required_states={'trained','reset_inhibition'}
    if set(s['starts'])!=required_states or len(m['cases'])!=24:raise ValueError('Missing conditions')
    for state in required_states:
        item=read_gzip(checked(root,s['starts'][state]));expected=deepcopy(initial if state=='initial' else trained)
        branch_cfg=deepcopy(cfg)
        for n in g['auditory']:
            if state in ('reset_selected','remove_inhibition','reset_inhibition'):
                ports=range(32) if state=='reset_selected' else range(35,67)
                for sid in ports:expected['neurons'][str(n)]['synapses'][str(sid)][0]=0. if state=='remove_inhibition' else initial['neurons'][str(n)]['synapses'][str(sid)][0]
        if state=='restore_threshold':
            for n in branch_cfg['neurons']:
                if n['id'] in g['auditory']:n['params'].update(r_base=.65,b_base=.90)
        params={str(n['id']):dict(r=n['params']['r_base'],b=n['params']['b_base']) for n in branch_cfg['neurons']}
        if item!={'state':expected,'threshold_parameters':params}:raise ValueError('Wrong state/parameter intervention')
        starts[state]=expected;configs[state]=branch_cfg

    def verify(raw,start,branch_cfg,integrator):
        nonlocal residual
        if raw['states'].shape[1:]!=(176,8) or any(not np.isfinite(raw[k]).all() for k in raw.files):raise ValueError('Invalid full trace')
        if any(raw[k].shape!=(len(raw['states']),32,67) for k in ('all_arrivals','all_input_weights','all_input_plast')):raise ValueError('Invalid input ledger')
        q=np.array([[start['neurons'][str(n)]['synapses'][str(i)][0] for i in range(32)] for n in g['auditory']])
        pre=np.array([start['eligibility'][str(n)]['pre'] for n in g['auditory']]);post=np.array([start['eligibility'][str(n)]['post'] for n in g['auditory']])
        prev=np.array([start['neurons'][str(n)]['O']>0 for n in g['vision']])
        selected=Record({k:raw[k] for k in ('states','weights','arrivals','eta','pre','post')})
        selected['states']=selected['states'][:,:144]
        *_,error=verify_trace(selected,q,pre,post,prev,branch_cfg,g,True);residual=max(residual,error)
        if not np.array_equal(raw['all_input_weights'][:,:,0:32],np.concatenate((q[None],raw['weights'][:-1]))):raise ValueError('Selected pre-update weights differ')
        if not np.array_equal(raw['all_arrivals'][:,:,:32],raw['arrivals']):raise ValueError('Input taps disagree')
        inh=raw['states'][:,144:176,1]>0
        before=np.array([start['neurons'][str(n)]['O']>0 for n in g['sensory_inhibition']])
        delivered=np.concatenate((before[None],inh[:-1]))
        if not np.array_equal(raw['all_arrivals'][:,:,35:]>0,np.broadcast_to(delivered[:,None,:],(len(inh),32,32))):raise ValueError('Inhibitory source/arrival mismatch')
        return integrator.check(raw)

    # Training continuity is checked in both the local learning and soma ledgers.
    start=deepcopy(initial);integrator=IntegrationAudit(cfg,start,g)
    if len(s['training'])!=32 or len(m['trials'])!=32:raise ValueError('Missing acquisition')
    for item,trial in zip(s['training'],m['trials']):
        with np.load(checked(root,item)) as raw:
            if len(raw['states'])!=trial['ticks']:raise ValueError('Wrong training duration')
            verify(raw,start,cfg,integrator)
            for j,n in enumerate(g['auditory']):
                for sid in range(32):start['neurons'][str(n)]['synapses'][str(sid)][0]=float(raw['weights'][-1,j,sid])
                start['eligibility'][str(n)].update(pre=raw['pre'][-1,j].tolist(),post=float(raw['post'][-1,j]))
            for n in g['vision']+g['sensory_inhibition']:start['neurons'][str(n)]['O']=float(raw['states'][-1,n-1,1])
    for n in g['auditory']:
        if start['eligibility'][str(n)]['pre']!=trained['eligibility'][str(n)]['pre'] or start['eligibility'][str(n)]['post']!=trained['eligibility'][str(n)]['post']:
            raise ValueError('Acquired traces differ')
        for sid in range(32):
            if start['neurons'][str(n)]['synapses'][str(sid)][0]!=trained['neurons'][str(n)]['synapses'][str(sid)][0]:raise ValueError('Acquired weights differ')
    specs=dict(clean=(16,0,1),omit25=(12,0,2),omit50=(8,0,2),omit75=(4,0,2),
               replace25=(12,4,2),balanced=(8,8,2),silence=(0,0,1))
    required_cases={(kind,cue,sample) for kind,(_,_,count) in specs.items() for cue in (0,1) for sample in range(count)}
    found=set();cases={}
    for case in m['cases']:
        key=case['kind'],case['cue'],case['sample']
        if key not in required_cases or key in found or case['name'] in cases:raise ValueError('Unexpected physical case')
        if (case['own_count'],case['foreign_count'])!=specs[case['kind']][:2]:raise ValueError('Wrong declared dose')
        if len(set(case['selected_receptors']))!=len(case['selected_receptors']):raise ValueError('Duplicate receptor')
        found.add(key);cases[case['name']]=case
    if found!=required_cases:raise ValueError('Missing physical cases')
    seen=set()
    for item in s['probes']:
        state,name=item['state'],item['case'];key=state,name
        if key in seen or state not in starts or name not in cases:raise ValueError('Unexpected/duplicate probe')
        seen.add(key);case=cases[name];start=starts[state];branch_cfg=configs[state]
        selected=set(case['selected_receptors']);cue=case['cue']
        if len(selected)!=case['own_count']+case['foreign_count'] or len(selected&set(m['masks']['vision'][cue]))!=case['own_count'] or len(selected&set(m['masks']['vision'][1-cue]))!=case['foreign_count']:
            raise ValueError('Wrong physical composition')
        with np.load(checked(root,item)) as raw:
            if len(raw['states'])!=64:raise ValueError('Wrong probe duration')
            native=verify(raw,start,branch_cfg,IntegrationAudit(branch_cfg,start,g))
            vision=np.zeros((64,32),bool)
            for t in (1,9,17,25):vision[t,[n-1 for n in selected]]=True
            if not np.array_equal(vision,raw['states'][:,:32,1]>0) or raw['states'][:,32:64,1].any():raise ValueError('Wrong sensory drive')
            consumer=raw['states'][:,96:128,1]>0
            sound=cue if m['mapping']=='paired' else 1-cue
            coordinates=[n-33 for n in m['masks']['audio'][sound]]
            rows.append(dict(state=state,case=name,kind=case['kind'],cue=cue,**classify_response(consumer,coordinates,case['kind'])))
            evidence[state+'/'+name+'/consumer']=consumer
            evidence[state+'/'+name+'/preactivation']=native
            evidence[state+'/'+name+'/inhibition']=raw['states'][:,144:176,1]>0
    if seen!={(state,name) for state in starts for name in cases}:raise ValueError('Missing probes')
    output.mkdir(parents=True,exist_ok=False);np.savez_compressed(output/'audited_dynamics.npz',**evidence)
    result=dict(structurally_valid=True,seed=m['seed'],mapping=m['mapping'],max_update_residual=residual,probes=rows,
        preactivation_fields=['pre_S','active_threshold','excitatory_current','inhibitory_current'],
        limits='Checks all auditory soma decisions, selected eligibility updates, inhibitory delivery, probe sensory drive and neural consumer timing. Does not independently reconstruct every inhibitory-cell/native plasticity update. Synthetic synchronous preparation, not embodied acceptance.')
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.recording,a.output)
