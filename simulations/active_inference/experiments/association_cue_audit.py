"""Data-only cue-completion audit with neural consumer and full-tick controls."""
from copy import deepcopy
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from .eligibility_association_audit import verify_trace


def initial_snapshot(cfg):
    value=dict(tick=0,neurons={},presynaptic_wheel=[],retrograde_wheel=[],eligibility={})
    for n in cfg['neurons']:
        nid=n['id'];p=n['params'];points=[x for x in cfg['synaptic_points'] if x['neuron_id']==nid]
        value['neurons'][str(nid)]=dict(S=0.,O=0.,r=p['r_base'],b=p['b_base'],t_ref=p['c']*p['num_inputs'],
            F_avg=0.,M=[0.]*p['num_neuromodulators'],t_last_fire=None,dendritic_queue=[],
            synapses={str(x['synapse_id']):[x['u_i']['info'],x['u_i']['plast'],x['u_i']['adapt'],x['potential']]
                      for x in points if x['type']=='postsynaptic'},
            terminals={str(x['terminal_id']):[x['u_o']['info'],x['u_o']['mod'],x['u_i_retro']]
                       for x in points if x['type']=='presynaptic'})
        value['eligibility'][str(nid)]=dict(pre=[0.]*len(n['metadata']['eligibility_ports']),post=0.,last_tick=None,updates=0)
    return value


def classify_response(consumer,expected_coordinates,kind):
    seen=consumer.any(axis=0);correct=np.array(expected_coordinates,dtype=int)
    other=np.array([i for i in range(consumer.shape[1]) if i not in set(correct)])
    right=int(seen[correct].sum());wrong=int(seen[other].sum())
    if kind=='balanced':status='balanced_both' if right and wrong else 'balanced_one' if right or wrong else 'balanced_silent'
    elif kind=='silence':status='spontaneous' if seen.any() else 'quiet'
    elif wrong:status='mixed' if right else 'wrong_only'
    else:status='complete_selective' if right==len(correct) else 'partial_selective' if right else 'silent'
    return dict(status=status,expected_coverage=right,other_coverage=wrong,
        expected_spikes=int(consumer[:,correct].sum()),other_spikes=int(consumer[:,other].sum()),
        expected_spikes_per_tick=consumer[:,correct].sum(axis=1).tolist(),
        other_spikes_per_tick=consumer[:,other].sum(axis=1).tolist(),
        windows=[dict(start=t,stop=t+8,expected_coverage=int(consumer[t:t+8,correct].any(axis=0).sum()),
                      other_coverage=int(consumer[t:t+8,other].any(axis=0).sum())) for t in range(0,len(consumer),8)])


def audit(recording,output):
    root,output=Path(recording),Path(output);m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text())
    if not s['training_exact'] or not s['clean_controls_exact']:raise ValueError('Replay not exact')
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in m['source_files_sha256'].items()):raise ValueError('Source changed')
    source=Path(m['source']);old=json.loads((source/'manifest.json').read_text());cfg=json.loads((source/'config.json').read_text());g=old['groups']
    if m['seed']!=old['seed'] or m['mapping']!=old['mapping'] or old['mapping'] not in ('paired','swapped'):raise ValueError('Wrong assignment')
    if any(n['params']['eta_post']<=0 or n['params']['eta_retro']<=0 for n in cfg['neurons']):raise ValueError('Frozen adaptation')
    with gzip.open(source/'trained-state.json.gz','rt') as f:trained=json.load(f)
    initial=initial_snapshot(cfg);reset=deepcopy(trained)
    for nid in g['auditory']:
        for sid in range(32):reset['neurons'][str(nid)]['synapses'][str(sid)][0]=initial['neurons'][str(nid)]['synapses'][str(sid)][0]
    starts={}
    for state,expected in (('initial',initial),('trained',trained),('reset_selected',reset)):
        meta=s['starts'][state];path=root/meta['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('State artifact changed')
        with gzip.open(path,'rt') as f:starts[state]=json.load(f)
        if starts[state]!=expected:raise ValueError('Undeclared starting-state change')
    specs=dict(clean=(16,0,1),omit25=(12,0,2),omit50=(8,0,2),omit75=(4,0,2),replace25=(12,4,2),balanced=(8,8,2),silence=(0,0,1))
    required_cases={(kind,cue,sample) for kind,(_,_,n) in specs.items() for cue in (0,1) for sample in range(n)}
    cases={};present=set()
    for c in m['cases']:
        key=c['kind'],c['cue'],c['sample'];actual=set(c['selected_receptors'])
        if key not in required_cases or key in present or c['name'] in cases:raise ValueError('Unexpected/duplicate physical case')
        own,foreign,_=specs[c['kind']]
        if (c['own_count'],c['foreign_count'])!=(own,foreign) or len(actual)!=len(c['selected_receptors']) or len(actual)!=own+foreign:
            raise ValueError('Wrong physical dose')
        if len(actual&set(old['masks']['vision'][c['cue']]))!=own or len(actual&set(old['masks']['vision'][1-c['cue']]))!=foreign:
            raise ValueError('Incorrect cue composition')
        cases[c['name']]=c;present.add(key)
    if present!=required_cases:raise ValueError('Missing physical cases')
    required={(state,name) for state in starts for name in cases};seen=set();rows=[];residual=0.;rasters={}
    for item in s['probes']:
        state,name=item['state'],item['case'];key=state,name
        if key not in required or key in seen:raise ValueError('Unexpected probe')
        seen.add(key);case=cases[name];start=starts[state];p=root/item['file']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Probe changed')
        q=np.array([[start['neurons'][str(n)]['synapses'][str(i)][0] for i in range(32)] for n in g['auditory']])
        pre=np.array([start['eligibility'][str(n)]['pre'] for n in g['auditory']]);post=np.array([start['eligibility'][str(n)]['post'] for n in g['auditory']])
        previous=np.array([start['neurons'][str(n)]['O']>0 for n in g['vision']])
        with np.load(p) as raw:
            if len(raw['states'])!=64:raise ValueError('Wrong probe duration')
            *_,error=verify_trace(raw,q,pre,post,previous,cfg,g,True);residual=max(residual,error)
            cells=raw['states'];vision=np.zeros((64,len(g['vision'])),bool)
            idx=[g['vision'].index(n) for n in case['selected_receptors']]
            for tick in (1,9,17,25):vision[tick,idx]=True
            if not np.array_equal(vision,cells[:,np.array(g['vision'])-1,1]>0):raise ValueError('Physical receptor spikes differ')
            if np.any(cells[:,np.array(g['audio'])-1,1]):raise ValueError('Sound delivered during recall')
            if case['kind']=='clean' and state in ('initial','trained'):
                with np.load(source/f'probe-{state}-{case["cue"]}.npz') as old_data:
                    if any(not np.array_equal(raw[k],old_data[k]) for k in raw.files):raise ValueError('Clean control differs')
            consumer=cells[:,np.array(g['consumer'])-1,1]>0
        sound=case['cue'] if m['mapping']=='paired' else 1-case['cue']
        coordinates=[g['audio'].index(n) for n in old['masks']['audio'][sound]]
        rows.append(dict(state=state,case=name,kind=case['kind'],cue=case['cue'],sample=case['sample'],
            expected_sound=None if case['kind'] in ('balanced','silence') else sound,
            **classify_response(consumer,coordinates,case['kind'])))
        rasters[state+'/'+name]=consumer
    if seen!=required:raise ValueError('Missing probes')
    output.mkdir(parents=True,exist_ok=False);np.savez_compressed(output/'consumer_rasters.npz',**rasters)
    result=dict(structurally_valid=True,seed=m['seed'],mapping=m['mapping'],max_update_residual=residual,probes=rows,
        criterion='Complete selective means all expected consumer coordinates fire at least once and no opposite coordinates fire. Per-tick and window records reveal latency, intermittency and partial expression. Balanced mixtures have no correct class.',
        limits='Synthetic receptor masks and a coordinate-preserving neural relay, not a learned semantic consumer. No real-media generalization, long retention, sequential interference or embodiment. Two physical subsets per corruption level per graph seed.')
    (output/'summary.json').write_text(json.dumps(result,allow_nan=False,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.recording,a.output)
