"""Independent sensory-state and selected-association audit of graded media runs.

The recorder does not contain every source-terminal amplitude or every
downstream input. This audit does not claim to reconstruct those quantities.
"""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import numpy as np

from .association_balance_audit import checked
from .association_route_probe import digest
from .eligibility_media_audit import verify_ledger
from .media_drive_audit import ReceptorAudit,physical_values
from .multimodal_pairing_audit import readout,contrast
from .population_state_audit import KINDS,tick_projection


def check_config(cfg,original,m):
    restored=deepcopy(cfg);gain=.25 if m['condition']=='graded' else 0.
    if m['condition'] not in ('graded','spiking'):raise ValueError('Unknown condition')
    if restored['metadata'].pop('sensory_release',None)!=m['condition']:raise ValueError('Undeclared release')
    sensory=set(m['groups']['vision']+m['groups']['touch'])
    for n in restored['neurons']:
        if n['id'] in sensory:
            for key,value in (('graded_gain',gain),('graded_S0',0.),('graded_max',0.)):
                if n['metadata'].pop(key,None)!=value:raise ValueError('Unexpected release metadata')
            if n['metadata'].get('eligibility_ports',[]):raise ValueError('Undefined graded spike eligibility')
    if restored!=original:raise ValueError('Undeclared graph or model change')
    return gain


def verify_rates(cells,previous_m,cfg,ports):
    ns={n['id']:n for n in cfg['neurons']};targets=np.array([n-1 for n,_,_ in ports])
    prior=np.concatenate((previous_m[None,:],cells[:-1,:,3]),axis=0)[:,targets]
    prior=np.maximum(prior,0.)
    boost=np.array([ns[n]['metadata'].get('plasticity_rate_boost',0.) for n,_,_ in ports])
    half=np.array([ns[n]['metadata'].get('plasticity_rate_half_saturation',.1) for n,_,_ in ports])
    expected=1+boost*prior/(half+prior)
    if not np.allclose(cells[:,targets,7],expected,atol=1e-12,rtol=1e-14):raise ValueError('Local gain does not follow prior modulator')


def audit(recording,output):
    root,output=Path(recording).resolve(),Path(output).resolve()
    m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text())
    cfg=json.loads((root/'config.json').read_text());source=Path(m['source'])
    old=json.loads((source/'manifest.json').read_text());original=json.loads((source/'config.json').read_text())
    gain=check_config(cfg,original,m)
    if any(m[k]!=old[k] for k in ('seed','mapping','groups','selected_ports','trials')):raise ValueError('Changed protocol')
    features=[]
    for clip in (0,1):
        path=Path(old['source_recording'])/f'sensory-{clip}.npz'
        if digest(path)!=old['source_files_sha256'][path.name]:raise ValueError('Changed sensory source')
        with np.load(path) as z:features.append({k:z[k] for k in z.files})
    ports=list(map(tuple,m['selected_ports']));points={(p['neuron_id'],p['synapse_id']):p for p in cfg['synaptic_points'] if p['type']=='postsynaptic'}
    order=[(n['id'],p['synapse_id']) for n in cfg['neurons'] for p in cfg['synaptic_points'] if p['type']=='postsynaptic' and p['neuron_id']==n['id']]
    lookup={key:i for i,key in enumerate(order)};take=np.array([lookup[n,sid] for n,sid,_ in ports])
    initial=np.array([points[key]['u_i']['info'] for key in order])
    q=initial[take].copy();pre=np.zeros(len(ports));post=pre.copy();previous=np.zeros(len(cfg['neurons']),bool);previous_m=np.zeros(len(cfg['neurons']))
    receptor=ReceptorAudit(cfg,m['groups'],graded_gain=gain);initial_receptor=deepcopy(receptor)
    residual=0.;training=[];samples={};rows=[];full_weights=initial.copy()
    def verify(item,trial,values,r,start,expected_full):
        with np.load(checked(root,item)) as z:data={k:z[k] for k in z.files}
        if not np.array_equal(data['incoming_info_before'],expected_full):raise ValueError('Full incoming-weight boundary mismatch')
        if len(data['cells'])!=trial['stop']-trial['start']:raise ValueError('Changed recording duration')
        r.check(data['cells'],values)
        if not np.allclose(data['incoming_info_after'][:384:2],r.q,atol=2e-12,rtol=0):raise ValueError('Sensory incoming update mismatch')
        verify_rates(data['cells'],start[4],cfg,ports)
        result=verify_ledger(data,cfg,ports,*start[:4],enabled=True)
        if not np.array_equal(result[0],data['incoming_info_after'][take]):raise ValueError('Association endpoint differs')
        return data,result
    if len(s['training'])!=len(m['trials']):raise ValueError('Incomplete acquisition')
    for i,(item,trial) in enumerate(zip(s['training'],m['trials'],strict=True)):
        if item['episode']!=i or receptor.tick!=trial['start']:raise ValueError('Broken acquisition sequence')
        data,result=verify(item,trial,physical_values(features,trial),receptor,(q,pre,post,previous,previous_m),full_weights)
        q,pre,post,previous,error,change=result;previous_m=data['cells'][-1,:,3].copy()
        full_weights=data['incoming_info_after'].copy();residual=max(residual,error)
        training.append(dict(episode=i,selected_update_l1_by_tick=change[:,3].tolist()))
    with gzip.open(root/'trained-state.json.gz','rt') as f:snapshot=json.load(f)
    for i,(nid,sid,_) in enumerate(ports):
        node=snapshot['neurons'][str(nid)];e=snapshot['eligibility'][str(nid)]
        selected=next(n for n in cfg['neurons'] if n['id']==nid)['metadata']['eligibility_ports']
        if node['synapses'][str(sid)][0]!=q[i] or e['pre'][selected.index(sid)]!=pre[i] or e['post']!=post[i]:raise ValueError('Learned snapshot mismatch')
    required={(state,sense,clip,g) for state in ('initial','continuation') for sense in ('visual','audio') for clip in (0,1) for g in (.5,1.,2.)}
    if len(s['probes'])!=24:raise ValueError('Incomplete probes')
    for p in s['probes']:
        key=p['state'],p['sense'],p['clip'],p['gain']
        if key not in required or key in samples:raise ValueError('Wrong or duplicate probe')
        is_initial=p['state']=='initial';r=deepcopy(initial_receptor if is_initial else receptor)
        if r.tick!=p['start_tick']:raise ValueError('Probe starts at wrong neural time')
        feature=features[p['clip']];length=len(feature['visual']);values=np.zeros((length,192))
        if p['sense']=='visual':
            values[:,:96]=1-np.clip(p['gain']*(1-feature['visual']),0,1);values[:3,:96]=0
        else:
            db=feature['band_db']+20*np.log10(p['gain'])
            values[:,96:]=np.clip((db[:,:,None]-np.array([-65.,-45.,-25.]))/20,0,1).reshape(-1,96)
        start=(initial[take],np.zeros(len(ports)),np.zeros(len(ports)),np.zeros(len(cfg['neurons']),bool),np.zeros(len(cfg['neurons']))) if is_initial else (q,pre,post,previous,previous_m)
        data,result=verify(p,dict(start=r.tick,stop=r.tick+length),values,r,start,initial if is_initial else full_weights)
        residual=max(residual,result[4]);samples[key]=data['cells']
        rows.append(dict(**p,spikes_by_tick={role:(data['cells'][:,np.array(m['groups'][role])-1,1]>0).sum(axis=1).tolist() for role in ('visual_core','tactile_core','upper_core','mismatch_candidate')}))
    readouts={};curves={}
    for role in ('tactile_core','upper_core'):
        ids=m['groups'][role];readouts[role]={}
        for g in (.5,1.,2.):
            for kind in KINDS:
                value=lambda state,sense:np.stack([readout(samples[state,sense,clip,g],ids,kind) for clip in (0,1)])
                ref=value('initial','audio');now=value('continuation','audio')
                readouts[role][f'{g}/{kind}']=dict(before=contrast(value('initial','visual'),ref),
                    after_original_reference=contrast(value('continuation','visual'),ref),
                    after_current_reference=contrast(value('continuation','visual'),now))
                if kind=='population_centered_rate':
                    for state in ('initial','continuation'):
                        curve=tick_projection([samples[state,'visual',clip,g] for clip in (0,1)],ids,ref)
                        if curve is not None:curves[f'{role}/{g}/{state}']=curve
    output.mkdir(parents=True,exist_ok=False);np.savez_compressed(output/'observer-traces.npz',**curves)
    result=dict(structurally_valid=True,seed=m['seed'],mapping=m['mapping'],condition=m['condition'],
        max_selected_update_residual=residual,training=training,probes=rows,readouts=readouts,
        limits='Sensory somata/output and incoming learning plus selected associative updates reconstructed. Receptor output O is before source-terminal scaling, whose full tick history was not recorded. Other downstream dynamics are observed, not independently reconstructed. No semantic or embodied acceptance claim.')
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


def compare_recall_factors(recording, output):
    """Check declared branch starts and retain sound-reference projections by tick.

    Projections compare response patterns, not semantic categories or accuracy.
    Sound references are fixed graded-parent references, not branch-specific
    audio trials. This comparison does not validate a memory-content claim.
    """
    root, output = Path(recording).resolve(), Path(output).resolve()
    m = json.loads((root/'manifest.json').read_text())
    s = json.loads((root/'summary.json').read_text())
    source, donor = Path(m['source']), Path(m['donor'])
    ss = json.loads((source/'summary.json').read_text())
    states = {}
    for name, path in (('graded', source/'trained-state.json.gz'),
                       ('spiking', donor/'trained-state.json.gz'),
                       ('initial', source/'initial-state.json.gz')):
        with gzip.open(path, 'rt') as f:
            states[name] = json.load(f)
    required = {(w, e, c) for w in states for e in ('graded', 'spiking') for c in (0, 1)}
    if len(s['probes']) != len(required):
        raise ValueError('Incomplete factorial')
    cells = {}; trajectories = {}; rows = []
    for p in s['probes']:
        key = p['weights'], p['expression'], p['clip']
        if key not in required or key in cells:
            raise ValueError('Duplicate or unexpected factor')
        expected = deepcopy(states['graded'])
        for n, sid, _ in m['selected_ports']:
            expected['neurons'][str(n)]['synapses'][str(sid)][0] = states[p['weights']]['neurons'][str(n)]['synapses'][str(sid)][0]
        with gzip.open(root/(Path(p['file']).stem+'-start.json.gz'), 'rt') as f:
            if json.load(f) != expected:
                raise ValueError('Undeclared recorded-state intervention')
        with np.load(checked(root, p)) as z:
            c = z['cells'].copy()
        if not np.isfinite(c).all():
            raise ValueError('Nonfinite factor record')
        cells[key] = c
        a = c[:, np.array(m['groups']['tactile_core'])-1]
        spikes = (a[:, :, 1] > 0).sum(axis=1)
        if not np.array_equal(spikes, p['auditory_spikes_by_tick']):
            raise ValueError('Summary spike trace differs')
        name = '/'.join(map(str, key)); trajectories[name+'/spikes'] = spikes
        trajectories[name+'/s_minus_r'] = a[:, :, 0]-a[:, :, 5]
        rows.append(dict(weights=key[0], expression=key[1], clip=key[2],
            total_spikes=int(spikes.sum()), first_spike=int(np.flatnonzero(spikes)[0]) if spikes.any() else None,
            spike_ticks=np.flatnonzero(spikes).tolist()))
    projections = {}
    for role in ('tactile_core', 'upper_core'):
        ids = m['groups'][role]
        for reference_state in ('initial', 'continuation'):
            refs = []
            for clip in (0, 1):
                p = next(p for p in ss['probes'] if p['state']==reference_state
                         and p['sense']=='audio' and p['clip']==clip and p['gain']==1.)
                with np.load(checked(source, p)) as z:
                    refs.append(readout(z['cells'], ids, 'population_centered_rate'))
            for w in states:
                for e in ('graded', 'spiking'):
                    curve = tick_projection([cells[w, e, c] for c in (0, 1)], ids, np.array(refs))
                    name = f'{role}/{reference_state}/{w}/{e}'
                    if curve is not None:
                        trajectories[name+'/projection'] = curve
                        projections[name] = dict(post32_mean=float(curve[32:].mean()),
                            windows=[dict(start=t, stop=min(t+32, len(curve)), mean=float(curve[t:t+32].mean()))
                                     for t in range(0, len(curve), 32)])
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'trajectories.npz', **trajectories)
    result = dict(recorded_starts_valid=True, recording=str(root), probes=rows, projections=projections,
        limits='Snapshots cover declared recorded state, not all hidden implementation fields. '
               'S-r is not the pre-reset firing margin on spike ticks. '
               'Auditory reference alignment is an observer, not learned content, a neural consumer or semantic accuracy.')
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.recording,a.output)
