"""Independent full-tick audit of continued association and inhibitory learning."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import numpy as np
from .association_balance_audit import IntegrationAudit,Record,checked
from .association_cue_audit import initial_snapshot,classify_response
from .eligibility_association_audit import verify_trace


def inhibitory_flow(raw, initial, last, tick, cfg, groups, sensitivity):
    q=initial.copy();last=last.copy();residual=0.
    ns={n['id']:n for n in cfg['neurons']};ids=np.array(groups['auditory'])-1
    basal=np.array([ns[n]['params']['eta_post'] for n in groups['auditory']])
    for t in range(len(raw['states'])):
        before=raw['all_input_weights'][t,:,35:]
        residual=max(residual,float(np.max(abs(q-before))))
        incoming=raw['all_arrivals'][t,:,35:];active=incoming>0
        if raw['all_input_plast'][t,:,35:].any():raise ValueError('Unexpected inhibitory plastic throughput')
        # All sensory and inhibitory terminals have zero modulatory output and
        # receive only zero-modulation retrograde errors in this graph.
        error=(incoming.astype(np.float32)-before.astype(np.float32))
        norm=np.linalg.norm(np.stack((error,np.zeros_like(error),np.zeros_like(error),np.zeros_like(error)),axis=-1),axis=-1).astype(float)
        eta=(basal*(1.+sensitivity*(raw['eta'][t]/basal-1.)))[:,None]
        if not np.all(eta>0):raise ValueError('Frozen inhibitory learning')
        firing=raw['states'][t,ids,1]>0;last[firing]=tick+t
        positive=(tick+t-last<=raw['states'][t,ids,6])[:,None]
        mag=abs(before);a=norm-.02;b=norm/10.
        # Here active inhibitory ports have norm > 1 and a > 0.
        z=np.exp(-a*eta)
        growth=mag/(z+(b*mag/a)*(-np.expm1(-a*eta)))
        decay=mag*np.exp(-(norm+.02)*eta)
        q=np.where(active,-np.where(positive,growth,decay),before)
    if residual>2e-12:raise ValueError(f'Inhibitory local update differs: {residual}')
    return q,last,residual


def verify_dynamics(raw,start,integrator,iq,last,cfg,g,sens):
    """Reusable data-only verification for one bounded 176-cell trace."""
    if raw['states'].shape[1:]!=(176,8) or any(not np.isfinite(v).all() for v in raw.values()):raise ValueError('Invalid trace')
    q=np.array([[start['neurons'][str(n)]['synapses'][str(i)][0] for i in range(32)] for n in g['auditory']])
    pre=np.array([start['eligibility'][str(n)]['pre'] for n in g['auditory']]);post=np.array([start['eligibility'][str(n)]['post'] for n in g['auditory']])
    prev=np.array([start['neurons'][str(n)]['O']>0 for n in g['vision']])
    old=Record({k:raw[k] for k in ('states','weights','pre','post','arrivals','eta')});old['states']=old['states'][:,:144]
    *_,error=verify_trace(old,q,pre,post,prev,cfg,g,True)
    if not np.array_equal(raw['all_input_weights'][:,:,:32],np.concatenate((q[None],raw['weights'][:-1]))):raise ValueError('Pre-update association mismatch')
    iq,last,ih_error=inhibitory_flow(raw,iq,last,integrator.tick,cfg,g,sens)
    before=np.array([start['neurons'][str(n)]['O']>0 for n in g['sensory_inhibition']])
    expected=np.concatenate((before[None],raw['states'][:-1,144:176,1]>0))
    if not np.array_equal(raw['all_arrivals'][:,:,35:]>0,np.broadcast_to(expected[:,None,:],(len(expected),32,32))):raise ValueError('Inhibitory input mismatch')
    dynamics=integrator.check(raw)
    return iq,last,dynamics,error,ih_error


def audit(root,output):
    root,output=Path(root),Path(output)
    m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text())
    cfg=json.loads((root/'config.json').read_text());g=m['groups'];sens=m['sensitivity']
    if sens not in (.25,1.) or m['checkpoints']!=[32,128] or len(m['trials'])!=128 or len(s['training'])!=128:raise ValueError('Wrong protocol')
    if sens==1. and not s['control_exact']:raise ValueError('Failed inherited control')
    if any(n['params']['eta_post']<=0 or n['params']['eta_retro']<=0 for n in cfg['neurons']):raise ValueError('Frozen learning')
    if {e['target_neuron'] for e in cfg['external_inputs']}!=set(g['vision']+g['audio']):raise ValueError('Unexpected external input')
    for p in cfg['synaptic_points']:
        if p['type']=='presynaptic' and p['neuron_id'] in g['vision']+g['sensory_inhibition'] and any(p['u_o']['mod']):
            raise ValueError('Inhibitory-flow audit assumes zero-modulation sensory/inhibitory terminals')
    for n in cfg['neurons']:
        expected=[dict(port=s,sensitivity=sens) for s in range(35,67)] if n['id'] in g['auditory'] else []
        if n['metadata'].get('native_port_modulation',[])!=expected:raise ValueError('Undeclared sensitivity')
    initial=initial_snapshot(cfg);start=deepcopy(initial);integrator=IntegrationAudit(cfg,initial,g)
    last=np.full(32,-np.inf);iq=np.full((32,32),-.08);residual=0.;ih_residual=0.;parents={};rows=[];rasters={}

    def verify(raw,start,integrator,iq,last):
        nonlocal residual,ih_residual
        iq,last,dynamics,error,ih_error=verify_dynamics(raw,start,integrator,iq,last,cfg,g,sens)
        residual=max(residual,error);ih_residual=max(ih_residual,ih_error)
        return iq,last,dynamics

    def sensory(raw,cue,sound,physical=None):
        expected=np.zeros((len(raw['states']),64),bool)
        for role,which in (('vision',cue),('audio',sound)):
            if which is None:continue
            ids=physical if role=='vision' and physical is not None else m['masks'][role][which]
            for t in (1,9,17,25):expected[t,[n-1 for n in ids]]=True
        if not np.array_equal(expected,raw['states'][:,:64,1]>0):raise ValueError('Wrong physical drive')

    for index,item in enumerate(s['training']):
        with np.load(checked(root,item)) as z:raw={k:z[k] for k in z.files}
        if len(raw['states'])!=96:raise ValueError('Training duration differs')
        trial=m['trials'][index];sensory(raw,trial['cue'],trial['sound'])
        iq,last,_=verify(raw,start,integrator,iq,last)
        for j,n in enumerate(g['auditory']):
            for sid in range(32):start['neurons'][str(n)]['synapses'][str(sid)][0]=float(raw['weights'][-1,j,sid])
            start['eligibility'][str(n)].update(pre=raw['pre'][-1,j].tolist(),post=float(raw['post'][-1,j]))
        for n in g['vision']+g['sensory_inhibition']:start['neurons'][str(n)]['O']=float(raw['states'][-1,n-1,1])
        checkpoint=index+1
        if checkpoint in m['checkpoints']:
            with gzip.open(checked(root,s['starts'][f'{checkpoint}-trained']),'rt') as f:parent=json.load(f)
            for j,n in enumerate(g['auditory']):
                node=parent['neurons'][str(n)]
                if node['S']!=integrator.soma[n] or node['t_last_fire']!=integrator.last[n]:raise ValueError('Wrong checkpoint soma')
                np.testing.assert_allclose([node['synapses'][str(i)][0] for i in range(35,67)],iq[j],rtol=0,atol=2e-12)
                for i in range(32):
                    if node['synapses'][str(i)][0]!=start['neurons'][str(n)]['synapses'][str(i)][0]:raise ValueError('Wrong checkpoint association')
                for field in ('pre','post'):
                    if parent['eligibility'][str(n)][field]!=start['eligibility'][str(n)][field]:raise ValueError('Wrong checkpoint eligibility trace')
            parents[checkpoint]=parent
    cases={c['name']:c for c in m['cases']};seen=set()
    specs=dict(clean=(16,0,1),omit25=(12,0,2),omit50=(8,0,2),omit75=(4,0,2),replace25=(12,4,2),balanced=(8,8,2),silence=(0,0,1))
    required={(kind,cue,sample) for kind,(_,_,count) in specs.items() for cue in (0,1) for sample in range(count)}
    if len(cases)!=24 or len(m['cases'])!=24 or {(c['kind'],c['cue'],c['sample']) for c in cases.values()}!=required:raise ValueError('Missing physical cases')
    for c in cases.values():
        selected=set(c['selected_receptors']);own,other,_=specs[c['kind']]
        if len(selected)!=len(c['selected_receptors']) or len(selected)!=own+other or (c['own_count'],c['foreign_count'])!=(own,other):raise ValueError('Wrong cue dose')
        if len(selected&set(m['masks']['vision'][c['cue']]))!=own or len(selected&set(m['masks']['vision'][1-c['cue']]))!=other:raise ValueError('Wrong cue composition')
    for item in s['probes']:
        checkpoint,state,name=item['checkpoint'],item['state'],item['case'];key=checkpoint,state,name
        if key in seen or checkpoint not in parents or state not in ('trained','reset_selected') or name not in cases:raise ValueError('Unexpected probe')
        seen.add(key);case=cases[name];expected=deepcopy(parents[checkpoint])
        if state=='reset_selected':
            for n in g['auditory']:
                for i in range(32):expected['neurons'][str(n)]['synapses'][str(i)][0]=initial['neurons'][str(n)]['synapses'][str(i)][0]
        with gzip.open(checked(root,s['starts'][f'{checkpoint}-{state}']),'rt') as f:start=json.load(f)
        if start!=expected:raise ValueError('Undeclared probe intervention')
        iq=np.array([[start['neurons'][str(n)]['synapses'][str(i)][0] for i in range(35,67)] for n in g['auditory']])
        last=np.array([start['neurons'][str(n)]['t_last_fire'] if start['neurons'][str(n)]['t_last_fire'] is not None else -np.inf for n in g['auditory']])
        with np.load(checked(root,item)) as z:raw={k:z[k] for k in z.files}
        if len(raw['states'])!=64:raise ValueError('Probe duration differs')
        sensory(raw,case['cue'],None,case['selected_receptors'])
        _,_,dynamics=verify(raw,start,IntegrationAudit(cfg,start,g),iq,last)
        sound=case['cue'] if m['mapping']=='paired' else 1-case['cue'];coords=[n-33 for n in m['masks']['audio'][sound]]
        consumer=raw['states'][:,96:128,1]>0
        rows.append(dict(checkpoint=checkpoint,state=state,case=name,kind=case['kind'],**classify_response(consumer,coords,case['kind'])))
        prefix=f'{checkpoint}/{state}/{name}';rasters[prefix+'/consumer']=consumer;rasters[prefix+'/preactivation']=dynamics
    if seen!={(k,s,n) for k in parents for s in ('trained','reset_selected') for n in cases}:raise ValueError('Missing probe')
    output.mkdir(parents=True,exist_ok=False);np.savez_compressed(output/'dynamics.npz',**rasters)
    result=dict(structurally_valid=True,seed=m['seed'],mapping=m['mapping'],sensitivity=sens,
        selected_update_residual=residual,inhibitory_update_residual=ih_residual,probes=rows,
        limits='Reconstructs selected excitatory and all auditory inhibitory input updates and auditory somatic decisions. Does not reconstruct every other neuron. Synthetic synchronous inputs; finite continued-learning horizon.')
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.recording,a.output)
