"""Independent arithmetic replay of the 65-cell graded/spiking consumer."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import numpy as np
from .association_balance_audit import checked


class ConsumerAudit:
    def __init__(self,cfg):
        ns=cfg['neurons']
        if [n['id'] for n in ns]!=list(range(1,66)):raise ValueError('Changed population')
        self.lam=np.array([n['params']['lambda_param'] for n in ns],np.float32)
        if not np.isin(self.lam[:32],[1,16]).all() or not (self.lam[32:]==1).all():raise ValueError('Unknown timescale')
        if any(n['params']['eta_post']!=1e-6 or n['params']['eta_retro']!=1e-7 or n['params']['delta_decay']!=1 for n in ns):raise ValueError('Changed native rule')
        self.counts=np.array([6]*32+[32]+[2]*32);self.offsets=np.r_[0,np.cumsum(self.counts)]
        points={(p['neuron_id'],p['synapse_id']):p for p in cfg['synaptic_points'] if p['type']=='postsynaptic'}
        self.q=np.array([points[n,i]['u_i']['info'] for n,count in enumerate(self.counts,1) for i in range(count)])
        self.soma=np.zeros(65,np.float32);self.o=np.zeros(65);self.f=np.zeros(65);self.last=np.full(65,-np.inf)
        self.info=np.ones(65,np.float32);self.error=np.zeros(288,np.float32);self.active=np.zeros(288,bool)
        self.direction=np.full(65,-1.);self.delayed=np.zeros(32,np.float32);self.tick=0;self.max_error=0.

    def check(self,raw):
        cells=raw['states'];length=len(cells)
        if not length or cells.shape!=(length,65,8) or raw['source'].shape!=(length,192) or raw['source'][-1].any():raise ValueError('Invalid boundary record')
        if any(raw[k].shape!=(length,288) for k in ('incoming','before','after')) or raw['terminal_info'].shape!=(length,65):raise ValueError('Invalid ledger')
        if any(not np.isfinite(x).all() for x in raw.values()):raise ValueError('Nonfinite trace')
        records=[]
        for t in range(length):
            previous_info=self.info.copy()
            # Retrograde events from the preceding tick. Pool -> each
            # integrator precedes contrast -> integrator; pool receives the
            # contrast-cell events in neuron order. Arithmetic is float32.
            delta=np.float32(1e-7)*(-self.error[192:224])
            self.info[:32]+=np.where(self.active[192:224],delta,np.float32(0))
            errors=self.error[224:].reshape(32,2);active=self.active[224:].reshape(32,2)
            directed=errors*self.direction[33:,None].astype(np.float32)
            delta=np.float32(1e-7)*directed
            self.info[:32]+=np.where(active[:,0],delta[:,0],np.float32(0))
            for j in range(32):
                if active[j,1]:self.info[32]+=delta[j,1]
            if not np.array_equal(raw['terminal_info'][t],self.info.astype(float)):raise ValueError(f'Terminal update differs at {self.tick}')
            expected=np.zeros(288,np.float32)
            if t:expected[:192]=raw['source'][t-1]
            release=(self.o*previous_info.astype(float)).astype(np.float32)
            expected[192:224]=release[:32]
            expected[224::2]=release[:32];expected[225::2]=release[32]
            x=raw['incoming'][t].astype(np.float32)
            if not np.array_equal(x,expected):raise ValueError(f'Delivery differs at {self.tick}')
            q=raw['before'][t]
            if not np.allclose(q,self.q,atol=2e-12,rtol=0):raise ValueError('Incoming weight continuity differs')
            local=x*q.astype(np.float32);current=np.zeros(65,np.float32)
            current[:32]=np.cumsum(np.sort(local[:192].reshape(32,6),axis=1),axis=1,dtype=np.float32)[:,-1]
            current[32]=np.cumsum(np.sort(local[192:224]),dtype=np.float32)[-1]
            pair=np.stack((self.delayed,local[225::2]),axis=1)
            current[33:]=np.cumsum(np.sort(pair,axis=1),axis=1,dtype=np.float32)[:,-1]
            self.delayed=local[224::2].copy()
            self.f=.9*self.f+.1*self.o
            tref=3*self.counts-(3*self.counts-6)*np.clip(3*self.f,0,1)
            before=self.soma+(np.float32(1)/self.lam)*(-self.soma+current)
            fired=np.zeros(65,bool);fired[33:]=(before[33:]>=np.float32(.01)) & (self.tick-self.last[33:]>=3)
            self.last[fired]=self.tick
            self.soma=np.where(fired,np.float32(0),before)
            self.o=np.r_[np.maximum(self.soma[:33].astype(float),0.),fired[33:].astype(float)]
            if not np.array_equal(cells[t,:,0],self.soma.astype(float)) or not np.array_equal(cells[t,:,1],self.o):raise ValueError(f'Soma/release differs at {self.tick}')
            thresholds=np.r_[np.full(33,1e12),np.full(32,.01)]
            if not np.array_equal(cells[t,:,5],thresholds) or not np.array_equal(cells[t,:,6],thresholds):raise ValueError('Threshold state differs')
            if cells[t,:,3:5].any() or not np.allclose(cells[t,:,2],self.f,rtol=1e-14,atol=1e-15) or not np.allclose(cells[t,:,7],tref,rtol=1e-14,atol=1e-13):raise ValueError('Regulatory state differs')
            self.direction=np.where(self.tick-self.last<=tref,1.,-1.)
            self.active=x>0;self.error=x-q.astype(np.float32)
            norm=np.linalg.norm(np.stack((self.error,np.zeros(288,np.float32),np.zeros(288,np.float32),np.zeros(288,np.float32)),axis=-1),axis=-1).astype(float)
            delta=1e-6*np.repeat(self.direction,self.counts)*norm*q
            delta=np.where(delta>0,delta*np.maximum(0,1-q/10),delta)
            predicted=np.where(self.active,q+delta-1e-6*.02*q,q)
            error=float(abs(predicted-raw['after'][t]).max());self.max_error=max(self.max_error,error)
            if error>2e-12:raise ValueError('Native postsynaptic update differs')
            self.q=raw['after'][t].copy();self.tick+=1
            records.append(np.stack((before,current),axis=-1))
        return np.asarray(records)


def audit(root,output):
    root,output=Path(root),Path(output);m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text())
    source_root=Path(m['source']);source_summary=json.loads((source_root/'summary.json').read_text())
    if len(s['training'])!=384 or len(s['probes'])!=336:raise ValueError('Incomplete experiment')
    conditions=('evidence_slow','evidence_fast','saturated_slow')
    wanted_training={(c,i) for c in conditions for i in range(128)}
    if {(p['condition'],p['index']) for p in s['training']}!=wanted_training:raise ValueError('Wrong acquisition grid')
    source_probes={(p['checkpoint'],p['state'],p['case']):p for p in source_summary['probes']}
    wanted_probes={(c,*key) for c in conditions for key in source_probes}
    if {(p['condition'],p['checkpoint'],p['state'],p['case']) for p in s['probes']}!=wanted_probes:raise ValueError('Wrong probe grid')
    states={c:ConsumerAudit(json.loads((root/f'{c}.json').read_text())) for c in conditions};parents={};rows=[];evidence={};residual=0.
    for item in s['training']:
        state=states[item['condition']]
        if state.tick!=96*item['index']:raise ValueError('Acquisition sequence differs')
        with np.load(checked(root,item)) as z:raw={k:z[k] for k in z.files}
        with np.load(checked(source_root,source_summary['training'][item['index']])) as z:
            cells=z['states'];wanted=np.repeat(cells[:,64:96,1]>0,6,axis=1) if item['condition']=='saturated_slow' else cells[:,176:,1]>0
        if not np.array_equal(raw['source'],wanted):raise ValueError('Acquisition does not replay the declared neural source')
        state.check(raw)
        if item['index']+1 in (32,128):
            key=f'{item["condition"]}/{item["index"]+1}';parents[key]=deepcopy(state)
            with gzip.open(checked(root,s['starts'][key]),'rt') as f:point=json.load(f)
            if point['tick']!=state.tick:raise ValueError('Checkpoint time differs')
            actual=np.array([point['neurons'][str(n)]['S'] for n in range(1,66)])
            if not np.array_equal(actual,state.soma.astype(float)):raise ValueError('Checkpoint soma differs')
    seen=set()
    for p in s['probes']:
        key=(p['condition'],p['checkpoint'],p['state'],p['case'])
        if key in seen:raise ValueError('Duplicate probe')
        if p['source_file']!=source_probes[key[1:]]:raise ValueError('Mislabeled source probe')
        seen.add(key);state=deepcopy(parents[f'{p["condition"]}/{p["checkpoint"]}'])
        with np.load(checked(root,p)) as z:raw={k:z[k] for k in z.files}
        with np.load(checked(Path(m['source']),p['source_file'])) as z:
            source=z['states'];wanted=np.repeat(source[:,64:96,1]>0,6,axis=1) if p['condition']=='saturated_slow' else source[:,176:,1]>0
        if not np.array_equal(wanted,raw['source']):raise ValueError('Wrong input history')
        dynamic=state.check(raw);residual=max(residual,state.max_error)
        output_spikes=raw['states'][:,33:,1]>0;counts=[]
        for cue in (0,1):
            sound=cue if m['mapping']=='paired' else 1-cue
            counts.append(output_spikes[:,np.array(m['masks']['audio'][sound])-33].sum(axis=1).tolist())
        rows.append(dict(condition=key[0],checkpoint=key[1],state=key[2],case=key[3],spikes_by_tick_cue=counts))
        name='/'.join(map(str,key));evidence[name+'/choice']=output_spikes;evidence[name+'/dynamics']=dynamic
    output.mkdir(parents=True,exist_ok=False);np.savez_compressed(output/'dynamics.npz',**evidence)
    result=dict(seed=m['seed'],mapping=m['mapping'],structurally_valid=True,local_update_residual=residual,probes=rows,
        limits='Isolated boundary replay only. Reconstructs delivery, soma/graded release, native incoming plasticity and native terminal updates for the consumer. Does not test back-action on the learned source or embodiment.')
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.recording,a.output)
