"""Reconstruct physical drive and sensory-neuron dynamics in real-media records.

No simulator or fitted decoder is used. This observer distinguishes sensory
availability, threshold conversion and downstream recruitment. It does not
equate dense activity with lost information or identify an animal vocalization.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np

from .association_balance_audit import checked
from .association_route_probe import digest


class ReceptorAudit:
    def __init__(self, cfg, groups, expected_threshold=.6, graded_gain=0.):
        self.ids = np.array(groups['vision']+groups['touch'])
        if not np.array_equal(self.ids,np.arange(1,193)): raise ValueError('Requires the declared first 192 receptors')
        ns = {n['id']:n for n in cfg['neurons']}
        points = {(p['neuron_id'],p['synapse_id']):p for p in cfg['synaptic_points'] if p['type']=='postsynaptic'}
        if any(e['target_neuron'] in self.ids for e in cfg['connections']):
            raise ValueError('Receptor has unmodeled neural inputs')
        for nid in self.ids:
            p, md = ns[nid]['params'], ns[nid]['metadata']
            if (p['lambda_param']!=1 or p['c']!=3 or p['delta_decay']!=.99 or p['beta_avg']!=.9
                    or p['r_base']!=expected_threshold or p['b_base']!=expected_threshold+.25 or p['num_inputs']!=2
                    or p['eta_post']!=1e-6 or any(p['w_r']) or any(p['w_b']) or any(p['w_tref'])
                    or not md['bounded_plasticity'] or md['plasticity_rate_boost']!=0):
                raise ValueError('Unmodeled receptor dynamics')
            for sid, weight in ((0,2.),(1,0.)):
                point = points[nid,sid]
                if point['distance_to_hillock']!=1 or point['u_i']['info']!=weight or point['u_i']['plast']!=0 or any(point['u_i']['adapt']):
                    raise ValueError('Unmodeled receptor synapse')
        self.q = np.full(192,2.); self.s = np.zeros(192,np.float32)
        self.r = expected_threshold; self.b = expected_threshold+.25
        if graded_gain not in (0.,.25):raise ValueError('Unmodeled graded gain')
        self.graded_gain=graded_gain
        self.delayed = np.zeros(192,np.float32); self.last = np.full(192,-np.inf)
        self.o = np.zeros(192); self.f = np.zeros(192); self.tick = 0

    def check(self, cells, values):
        if values.shape != (len(cells),192) or not np.isfinite(values).all() or (values<0).any():
            raise ValueError('Invalid physical receptor drive')
        n = len(cells); drives = np.zeros((n,192),np.float32); margin = np.zeros((n,192)); weights=np.zeros((n,192))
        for t in range(n):
            x = np.where(t%4==self.ids%4,2*values[t],0).astype(np.float32)
            drives[t] = x
            current = self.delayed*np.float32(.99)
            self.delayed = x*self.q.astype(np.float32)
            self.f = .9*self.f+(1-.9)*self.o
            before = self.s+(-self.s+current)
            age = self.tick-self.last
            threshold = np.where(age<=3,self.b,self.r)
            threshold = np.where(abs(before)<.005,self.r,threshold)
            fired = (before>=threshold.astype(np.float32)) & (age>=3)
            if self.graded_gain:fired[:]=False
            self.s = np.where(fired,np.float32(0),before); self.last[fired] = self.tick
            self.o = self.graded_gain*np.maximum(self.s.astype(float),0.) if self.graded_gain else fired.astype(float)
            margin[t] = before-threshold
            actual = cells[t,self.ids-1]
            if not np.array_equal(actual[:,0],self.s.astype(float)) or not np.array_equal(actual[:,1],self.o):
                raise ValueError(f'Receptor soma/output mismatch at {self.tick}')
            if actual[:,3:5].any() or not np.allclose(actual[:,2],self.f,atol=1e-14,rtol=1e-14):
                raise ValueError('Receptor regulatory state mismatch')
            if not (actual[:,5]==(1e12 if self.graded_gain else self.r)).all() or not (actual[:,6]==6).all():
                raise ValueError('Receptor thresholds/window mismatch')
            error = x-self.q.astype(np.float32)
            norm = np.linalg.norm(np.stack((error,np.zeros(192,np.float32),np.zeros(192,np.float32),np.zeros(192,np.float32)),axis=-1),axis=-1).astype(float)
            a = norm-.02; b = norm/10.; eta = np.full(192,1e-6)
            h = np.divide(np.expm1(a*eta),a,out=eta.copy(),where=a!=0)
            growing = self.q*np.exp(a*eta)/(1+b*self.q*h)
            shrinking = self.q*np.exp(-(norm+.02)*eta)
            self.q = np.where(x>0,np.where(self.tick-self.last<=6,growing,shrinking),self.q)
            weights[t]=self.q
            self.tick += 1
        return dict(drive=drives,margin=margin,receptor_spikes=np.zeros((n,192),bool) if self.graded_gain else cells[:,self.ids-1,1]>0,
                    physical_values=values,reconstructed_weights=weights)


def physical_values(features, trial):
    length = trial['stop']-trial['start']; values = np.zeros((length,192))
    for key, field, start in (('visual_clip','visual',0),('audio_clip','auditory',96)):
        if trial[key] is not None:
            src = features[trial[key]][field]
            values[:,start:start+96] = src[np.arange(length)%len(src)]
    return values


def audit(recording, output):
    root, output = Path(recording).resolve(), Path(output).resolve()
    m = json.loads((root/'manifest.json').read_text()); s = json.loads((root/'summary.json').read_text())
    cfg = json.loads((root/'config.json').read_text()); source = Path(m['source_recording'])
    features = []
    for clip in (0,1):
        path = source/f'sensory-{clip}.npz'
        if digest(path)!=m['source_files_sha256'][path.name]: raise ValueError('Changed sensory recording')
        with np.load(path) as z: features.append({k:z[k] for k in z.files})
    initial = ReceptorAudit(cfg,m['groups']); current = deepcopy(initial)
    output.mkdir(parents=True,exist_ok=False); rows = []
    def save(item, trial, state, label):
        with np.load(checked(root,item)) as z:
            cells = z['cells']; before = z['incoming_info_before']; after = z['incoming_info_after']
        if not np.allclose(before[:384:2],state.q,atol=2e-12,rtol=0) or before[1:384:2].any():
            raise ValueError('Receptor weight continuity mismatch')
        data = state.check(cells,physical_values(features,trial))
        if not np.allclose(after[:384:2],state.q,atol=2e-12,rtol=0) or after[1:384:2].any():
            raise ValueError('Receptor weight update mismatch')
        # All downstream populations remain available per tick, without a fitted readout.
        for role, ids in m['groups'].items(): data[role+'_spikes'] = cells[:,np.array(ids)-1,1]>0
        name = label+'.npz'; np.savez_compressed(output/name,**data)
        rows.append(dict(file=name,sha256=digest(output/name),source=item['file'],trial=trial))
    for item, trial in zip(s['episodes'],m['trials'],strict=True):
        if current.tick!=trial['start']: raise ValueError('Broken acquisition continuity')
        save(item,trial,current,f'experience-{item["episode"]:03d}')
    for p in s['probes']:
        state = deepcopy(initial if p['condition']=='initial' else current)
        if state.tick!=p['start_tick']: raise ValueError('Probe start mismatch')
        t = state.tick
        trial = dict(start=t,stop=t+m['clip_ticks'],visual_clip=p['clip'] if p['sense']=='visual' else None,
                     audio_clip=p['clip'] if p['sense']=='audio' else None)
        save(p,trial,state,p['file'][:-4])
    result = dict(seed=m['seed'],mapping=m['mapping'],structurally_valid=True,records=rows,
                  source=str(root),receptor_ids=initial.ids.tolist(),
                  limits='Reconstructs receptor somata and informational output flags using physical drive and local incoming learning. Does not reconstruct outgoing source-terminal release or all downstream neurons. Dense activity and unequal stimulus energy alone are not proof of lost content.')
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a = p.parse_args(); audit(a.recording,a.output)
