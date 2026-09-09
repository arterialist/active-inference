"""Data-only soma, eligibility and native-weight audit for the evidence bank."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import numpy as np

from .association_balance_audit import checked


class EvidenceAudit:
    def __init__(self, cfg, groups):
        self.ids=np.array(groups['evidence']); self.groups=groups
        ns={n['id']:n for n in cfg['neurons']}; self.ns=[ns[int(n)] for n in self.ids]
        points={(p['neuron_id'],p['synapse_id']):p for p in cfg['synaptic_points'] if p['type']=='postsynaptic'}
        self.q=np.array([[points[n,i]['u_i']['info'] for i in range(67)] for n in self.ids])
        self.birth=self.q.copy(); self.r=np.array([n['params']['r_base'] for n in self.ns])
        self.b=2*self.r; self.n=len(self.ids); self.tick=0
        self.soma=np.zeros(self.n,np.float32);self.last=np.full(self.n,-np.inf)
        self.pre=np.zeros((self.n,32));self.post=np.zeros(self.n);self.m=np.zeros(self.n);self.f=np.zeros(self.n)
        self.delayed=np.zeros((self.n,35),np.float32);self.previous=np.zeros(len(cfg['neurons']),bool)
        edges={(e['target_neuron'],e['target_synapse']):e for e in cfg['connections']}
        self.sources=np.array([[edges[n,i]['source_neuron']-1 for i in range(67)] for n in self.ids])
        for n in self.ns:
            p=n['params'];meta=n['metadata']
            if (p['lambda_param']!=4 or p['c']!=3 or p['b_base']!=2*p['r_base'] or p['delta_decay']!=.99 or
                p['beta_avg']!=.9 or p['gamma']!=[.99,.99] or any(p['w_r']) or any(p['w_b']) or any(p['w_tref']) or
                p['eta_post']!=1e-5 or meta['eligibility_ports']!=list(range(32)) or
                meta['eligibility_tau_pre']!=4 or meta['eligibility_tau_post']!=4 or
                meta['eligibility_cap']!=1 or meta.get('eligibility_alpha',1)!=1 or
                meta['plasticity_rate_boost']!=499 or meta['plasticity_rate_half_saturation']!=.01 or
                meta['native_port_modulation']!=[dict(port=i,sensitivity=.25) for i in range(35,67)]):
                raise ValueError('Unexpected evidence dynamics')
        for nid in self.ids:
            if [points[nid,i]['distance_to_hillock'] for i in range(67)] != [1]*35+[0]*32:
                raise ValueError('Changed dendritic delays')
            if any(points[nid,i]['u_i']['plast'] for i in range(67)):
                raise ValueError('Plastic throughput is not audited')
        self.max_error=0.

    def check(self, raw):
        cells=raw['states']; length=len(cells)
        if cells.shape!=(length,368,8) or not length or any(not np.isfinite(x).all() for x in raw.values()):
            raise ValueError('Invalid full trace')
        for name in ('input','before','after','plast'):
            if raw['evidence_'+name].shape!=(length,192,67):raise ValueError('Invalid synaptic ledger')
        records=[]
        for t in range(length):
            x=raw['evidence_input'][t].astype(np.float32); q=raw['evidence_before'][t]
            if raw['evidence_plast'][t].any():raise ValueError('Unexpected plastic throughput')
            if not np.allclose(q,self.q,atol=2e-12,rtol=0):raise ValueError('Weight continuity differs')
            expected=self.previous[self.sources].copy();expected[:,33:35]=False
            if not np.array_equal(x>0,expected):raise ValueError('Neural delivery differs')
            if not np.allclose(raw['evidence_pre'][t],self.pre,atol=1e-14,rtol=1e-14) or not np.allclose(raw['evidence_post'][t],self.post,atol=1e-14,rtol=1e-14):
                raise ValueError('Eligibility continuity differs')
            eta=1e-5*(1+499*self.m/(.01+self.m))
            if not np.allclose(eta,raw['evidence_eta'][t],atol=1e-16,rtol=1e-14):raise ValueError('Rate is not locally modulated')
            self.m=.99*self.m+.01*.25*self.previous[self.sources[:,33:35]].sum(axis=1)
            self.f=.9*self.f+.1*self.previous[self.ids-1]
            tref=np.clip(201-195*np.clip(3*self.f,0,1),6,201)
            # Native scalar arithmetic is float32 once a buffer input arrives.
            # PAULA's heap sorts equal-arrival events by local potential before
            # synapse index. Preserve that order, not a floating-point sum in
            # arbitrary port order or float64 precision.
            local=x*q.astype(np.float32)
            v=np.concatenate((local[:,35:],self.delayed),axis=1)
            order=np.argsort(v,axis=1)
            attenuation=np.broadcast_to(np.array([1.]*32+[.99]*35,np.float32),v.shape)
            arriving=np.take_along_axis(v,order,axis=1)*np.take_along_axis(attenuation,order,axis=1)
            current=np.cumsum(arriving,axis=1,dtype=np.float32)[:,-1]
            self.delayed=local[:,:35].copy()
            before=self.soma+.25*(-self.soma+current)
            age=self.tick-self.last
            threshold=np.where(age<=3,self.b,self.r)
            threshold=np.where(abs(before)<.005,self.r,threshold)
            fired=(before>=threshold.astype(np.float32)) & (age>=3)
            self.soma=np.where(fired,np.float32(0.),before)
            self.last[fired]=self.tick
            actual=cells[t,self.ids-1]
            if not np.array_equal(actual[:,1]>0,fired) or not np.array_equal(actual[:,0],self.soma.astype(float)):
                bad=np.flatnonzero((actual[:,1]>0)!=fired).tolist()
                raise ValueError(f'Somatic reconstruction differs tick={self.tick}, spikes={bad}, S error={abs(actual[:,0]-self.soma).max()}')
            if not np.allclose(actual[:,3],self.m,atol=1e-15,rtol=1e-14) or not np.allclose(actual[:,6],tref,atol=1e-13,rtol=1e-14):
                raise ValueError('Modulator or native timing window differs')
            pre=self.pre*np.exp(-.25);post=self.post*np.exp(-.25)
            active=x[:,:32]>0;plus=fired[:,None]*pre;minus=active*post[:,None];total=plus+minus
            target=np.divide(plus,total,out=np.zeros_like(plus),where=total>0)
            decay=np.exp(-eta[:,None]*total)
            predicted=q.copy();predicted[:,:32]=q[:,:32]*decay+target*(1-decay)
            # Every other incoming port retains bounded native plasticity.
            # Their incoming mod/plastic values are zero when informationally
            # active in this declared graph.
            native=q[:,32:]; info=x[:,32:]; active_native=info>0
            error=info-native.astype(np.float32)
            norm=np.linalg.norm(np.stack((error,np.zeros_like(error),np.zeros_like(error),np.zeros_like(error)),axis=-1),axis=-1).astype(float)
            rate=np.broadcast_to(eta[:,None],native.shape).copy()
            rate[:,3:]=1e-5*(1+.25*(eta[:,None]/1e-5-1))
            mag=abs(native); a=norm-.02; b=norm/10
            h=np.divide(np.expm1(a*rate),a,out=rate.copy(),where=a!=0)
            growing=mag*np.exp(a*rate)/(1+b*mag*h)
            shrinking=mag*np.exp(-(norm+.02)*rate)
            direction=(self.tick-self.last<=tref)[:,None]
            predicted[:,32:]=np.where(active_native,np.copysign(np.where(direction,growing,shrinking),native),native)
            error=float(abs(predicted-raw['evidence_after'][t]).max());self.max_error=max(self.max_error,error)
            if error>2e-12:raise ValueError(f'Local weight update differs: {error}')
            self.q=raw['evidence_after'][t].copy();self.pre=pre+active;self.post=post+fired
            records.append(np.stack((before,threshold,current),axis=-1))
            self.previous=cells[t,:,1]>0;self.tick+=1
        return np.asarray(records)


def audit(root, output):
    root,output=Path(root),Path(output)
    m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text());cfg=json.loads((root/'config.json').read_text())
    if len(s['training'])!=128 or len(s['probes'])!=112 or not s['baseline_exact']:raise ValueError('Incomplete run')
    checked(root,dict(file='config.json',sha256=s['config_sha256']))
    checked(root,dict(file='schedules.npz',sha256=s['schedules_sha256']))
    if any(n['params']['eta_post']<=0 or n['params']['eta_retro']<=0 for n in cfg['neurons']):raise ValueError('Frozen adaptation')
    groups=m['groups'];state=EvidenceAudit(cfg,groups);parents={};rows=[];recorded={};residual=0.
    for i,item in enumerate(s['training']):
        with np.load(checked(root,item)) as z:raw={k:z[k] for k in z.files}
        state.check(raw)
        if i+1 in (32,128):
            parents[i+1]=deepcopy(state)
            with gzip.open(checked(root,s['starts'][str(i+1)]),'rt') as f:snap=json.load(f)
            if snap['tick']!=state.tick:raise ValueError('Checkpoint clock differs')
            for j,nid in enumerate(groups['evidence']):
                node=snap['neurons'][str(nid)];e=snap['eligibility'][str(nid)]
                if node['S']!=float(state.soma[j]) or node['t_last_fire']!=state.last[j]:raise ValueError('Checkpoint soma differs')
                if not np.array_equal(state.q[j],[node['synapses'][str(k)][0] for k in range(67)]):raise ValueError('Checkpoint weights differ')
                if not np.allclose(state.pre[j],e['pre'],atol=1e-14,rtol=1e-14) or not np.isclose(state.post[j],e['post'],atol=1e-14,rtol=1e-14):raise ValueError('Checkpoint eligibility differs')
    residual=state.max_error
    with np.load(root/'schedules.npz') as schedules:
        seen=set()
        for p in s['probes']:
            cp,key,condition=p['checkpoint'],p['case'],p['state'];identity=(cp,key,condition)
            if identity in seen:raise ValueError('Duplicate probe')
            seen.add(identity);branch=deepcopy(parents[cp])
            if condition=='reset_evidence':branch.q[:,:32]=branch.birth[:,:32]
            elif condition!='trained':raise ValueError('Unexpected intervention')
            with np.load(checked(root,p)) as z:raw={k:z[k] for k in z.files}
            schedule=schedules[key]
            retinal=np.concatenate((np.zeros((1,32),bool),schedule[:-1]>0))
            if not np.array_equal(raw['states'][:,:32,1]>0,retinal) or raw['states'][:,32:64,1].any():raise ValueError('Wrong sensory probe')
            dynamics=branch.check(raw);residual=max(residual,branch.max_error)
            activity=raw['states'][:,np.array(groups['evidence'])-1,1]>0
            pattern=activity.reshape(len(activity),32,6)
            counts=[]
            for cue in (0,1):
                sound=cue if m['mapping']=='paired' else 1-cue
                coords=np.array(m['masks']['audio'][sound])-33
                counts.append(pattern[:,coords,:].sum(axis=1).tolist())
            rows.append(dict(checkpoint=cp,case=key,state=condition,spikes_by_tick_cue_level=counts,
                active_ticks=np.flatnonzero(activity.any(axis=1)).tolist()))
            name=f'{cp}/{condition}/{key}'
            recorded[name+'/spikes']=activity;recorded[name+'/dynamics']=dynamics
        required={(cp,k,'trained') for cp in (32,128) for k in schedules.files}
        required|={(cp,f'clean-cue{c}-sample0/synchronous','reset_evidence') for cp in (32,128) for c in (0,1)}
        if seen!=required:raise ValueError('Missing probe')
    output.mkdir(parents=True,exist_ok=False);np.savez_compressed(output/'dynamics.npz',**recorded)
    result=dict(seed=m['seed'],mapping=m['mapping'],architecture=m['architecture'],structurally_valid=True,
        local_update_residual=residual,probes=rows,
        limits='All added-cell soma decisions and incoming weight updates reconstructed. This is evidence encoding, not a learned decision, hierarchy or consciousness. Original baseline equality is checked by the experiment runner.')
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.recording,a.output)
