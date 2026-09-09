"""Data-only factorial audit of experience-dependent audiovisual interactions.

Compare the same four physical inputs across initial, trained and selected-reset
states, under both experience assignments. No activity template or classifier
defines a prediction. A factorial interaction can reveal learned sensitivity to
the relation; it is not by itself a prediction-error computation or awareness.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .eligibility_media_audit import verify_ledger, check_branch_start
from .multimodal_pairing_audit import sensory_marginals

CONDITIONS = ('initial', 'unchanged', 'reset_selected')


def weight_components(delta):
    """Orthogonal bookkeeping, not a claim about where information resides."""
    delta=np.asarray(delta,dtype=float)
    if delta.ndim!=2 or not delta.size or not np.isfinite(delta).all(): raise ValueError('Need finite target-by-input differences')
    common=np.full_like(delta,delta.mean())
    target=np.broadcast_to(delta.mean(axis=1,keepdims=True),delta.shape)-common
    within=delta-common-target
    parts=dict(common_shift=common,target_profile=target,within_target=within)
    squared=float(np.sum(delta*delta))
    return parts,{k:float(np.sum(v*v))/squared if squared>0 else None for k,v in parts.items()}


def interaction(samples):
    """Original cross-pair minus original same-pair, with equal physical dose.

Positive means off-diagonal combinations produce the greater response.
It is not labelled 'error'. Separable video/audio effects cancel algebraically.
All inputs must describe the same ticks, neurons and measurement fields.
"""
    if set(samples) != {(0,0),(0,1),(1,0),(1,1)}: raise ValueError('Need all four physical combinations')
    values = list(samples.values())
    if any(v.shape != values[0].shape or not np.isfinite(v).all() for v in values): raise ValueError('Incompatible samples')
    return .5*(samples[0,1]+samples[1,0]-samples[0,0]-samples[1,1])


def read_state(path):
    with gzip.open(path, 'rt') as f: return json.load(f)


def load_record(root):
    meta = json.loads((root/'manifest.json').read_text()); summary = json.loads((root/'summary.json').read_text())
    if meta.get('mode') != 'congruence' or meta['conditions'] != list(CONDITIONS): raise ValueError('Wrong protocol')
    source = Path(meta['source_recording']); m = json.loads((source/'manifest.json').read_text())
    raw_source = Path(m['source_recording'])
    if any(hashlib.sha256((raw_source/name).read_bytes()).hexdigest()!=h for name,h in m['source_files_sha256'].items()):
        raise ValueError('Original media or source configuration changed')
    if not summary['full_training_replay_exact'] or summary['seed'] != m['seed'] or summary['mapping'] != m['mapping']:
        raise ValueError('Training replay provenance differs')
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p,h in meta['source_files_sha256'].items()):
        raise ValueError('Source artifact changed')
    cfg = json.loads((source/'config.json').read_text()); ports = m['selected_ports']
    ns = {n['id']: n for n in cfg['neurons']}
    if list(ns) != list(range(1,len(ns)+1)): raise ValueError('Cell indexing differs')
    if any(n['params']['eta_post'] <= 0 or n['params']['eta_retro'] <= 0 for n in ns.values()): raise ValueError('Frozen adaptation')
    points = {(p['neuron_id'],p['synapse_id']):p for p in cfg['synaptic_points'] if p['type']=='postsynaptic'}
    order = [(n,p['synapse_id']) for n in ns for p in cfg['synaptic_points'] if p['type']=='postsynaptic' and p['neuron_id']==n]
    lookup = {p:i for i,p in enumerate(order)}; take = [lookup[n,s] for n,s,_ in ports]
    original = {p:value['u_i']['info'] for p,value in points.items()}
    initial = read_state(source/'initial-state.json.gz'); trained = read_state(source/'training-final-state.json.gz')
    required = {(c,v,a) for c in CONDITIONS for v in (0,1) for a in (0,1)}
    declared = {(s['condition'],s['trial']['visual_clip'],s['trial']['audio_clip']) for s in meta['probe_schedule']}
    if declared != required or len(meta['probe_schedule']) != len(required): raise ValueError('Schedule incomplete')
    cells = {}; residual = 0.
    for b in summary['branches']:
        trial = b['trial']; key = b['condition'],trial['visual_clip'],trial['audio_clip']
        if key not in required or key in cells: raise ValueError('Missing/duplicate physical condition')
        if any(hashlib.sha256((root/b[field]).read_bytes()).hexdigest()!=b[digest]
               for field,digest in (('file','sha256'),('start_file','start_sha256'))): raise ValueError('Probe artifact changed')
        parent = initial if key[0]=='initial' else trained
        start = read_state(root/b['start_file'])
        check_branch_start(start,parent,parent,original,ports,'reset_selected' if key[0]=='reset_selected' else 'unchanged')
        if trial != dict(start=parent['tick'],stop=parent['tick']+m['clip_ticks'],visual_clip=key[1],audio_clip=key[2]):
            raise ValueError('Physical probe timing differs')
        q = np.array([start['neurons'][str(n)]['synapses'][str(s)][0] for n,s,_ in ports])
        pre = np.array([start['eligibility'][str(n)]['pre'][ns[n]['metadata']['eligibility_ports'].index(s)] for n,s,_ in ports])
        post = np.array([start['eligibility'][str(n)]['post'] for n,_,_ in ports])
        previous = np.array([start['neurons'][str(n)]['O']>0 for n in ns])
        with np.load(root/b['file']) as z:
            if z['cells'].shape != (m['clip_ticks'],len(ns),8) or not np.array_equal(z['incoming_info_before'],
                  [start['neurons'][str(n)]['synapses'][str(s)][0] for n,s in order]): raise ValueError('Initial weights or cells differ')
            end,_,_,_,error,_ = verify_ledger(z,cfg,ports,q,pre,post,previous)
            if not np.array_equal(end,z['incoming_info_after'][take]): raise ValueError('Selected final weights differ')
            residual = max(residual,error); cells[key] = z['cells']
    if set(cells)!=required: raise ValueError('Incomplete probe set')
    return m,cfg,cells,residual


def audit(paired, swapped, output):
    roots = [Path(paired).resolve(),Path(swapped).resolve()]; output=Path(output)
    recordings = [load_record(p) for p in roots]
    a,b=recordings
    if [x[0]['mapping'] for x in recordings]!=['paired','swapped'] or a[0]['seed']!=b[0]['seed'] or a[1]!=b[1]:
        raise ValueError('Unmatched assignments')
    if sensory_marginals(Path(a[0]['source_recording']),a[0])!=sensory_marginals(Path(b[0]['source_recording']),b[0]):
        raise ValueError('Different sensory marginals')
    for v in (0,1):
        for sound in (0,1):
            if not np.array_equal(a[2]['initial',v,sound],b[2]['initial',v,sound]): raise ValueError('Initial probes differ')
    # Equal physical inputs need not produce equal receptor spikes: these
    # neurons also learn. Report upstream differences instead of freezing them
    # or attributing every downstream difference to associative cortical inputs.
    sensory = np.array(a[0]['groups']['vision']+a[0]['groups']['touch'])-1
    receptor_differences = []
    for condition in CONDITIONS:
        for v in (0,1):
            for sound in (0,1):
                changed = (a[2][condition,v,sound][:,sensory,1]>0)!=(b[2][condition,v,sound][:,sensory,1]>0)
                times=np.flatnonzero(changed.any(axis=1))
                receptor_differences.append(dict(condition=condition,visual=v,audio=sound,
                    different_spike_entries=int(changed.sum()),first_tick=int(times[0]) if len(times) else None))
    fields = a[0]['fields']; curves={}; regions={}; interaction_by_mapping={}
    grouped={}
    for n,s,_ in a[0]['selected_ports']: grouped.setdefault(n,[]).append(s)
    if len({len(sids) for sids in grouped.values()})!=1: raise ValueError('Unequal row fan-in for parameter decomposition')
    parameters=[]
    for root in roots:
        source=Path(json.loads((root/'manifest.json').read_text())['source_recording'])
        final=read_state(source/'training-final-state.json.gz')
        parameters.append(np.array([[final['neurons'][str(n)]['synapses'][str(s)][0] for s in sids] for n,sids in grouped.items()]))
    components,fractions=weight_components(parameters[0]-parameters[1])
    curves.update({f'parameter_delta/{key}': value for key,value in components.items()})
    for m,cfg,cells,_ in recordings:
        for condition in CONDITIONS:
            values={(v,s):cells[condition,v,s].copy() for v in (0,1) for s in (0,1)}
            for value in values.values(): value[:,:,1] = value[:,:,1]>0
            interaction_by_mapping[m['mapping'],condition]=interaction(values)
    for role,ids in a[0]['groups'].items():
        take=np.array(ids)-1; regions[role]={}
        for condition in CONDITIONS:
            jp,js=[interaction_by_mapping[mapping,condition][:,take] for mapping in ('paired','swapped')]
            # Both compare the same physical combinations. This removes the
            # initial physical interaction and isolates assignment dependence.
            difference=jp-js
            for name,value in (('paired',jp),('swapped',js),('assignment_difference',difference)):
                curves[f'{role}/{condition}/{name}']=value
                regions[role][f'{condition}/{name}']={}
                for j,field in enumerate(fields):
                    temporal=value[:,:,j].mean(axis=1)
                    regions[role][f'{condition}/{name}'][field]=dict(
                        all_ticks=float(temporal.mean()),onset_0_32=float(temporal[:32].mean()),after_32=float(temporal[32:].mean()),
                        windows=[{'start':t,'stop':min(t+32,len(temporal)),'mean':float(temporal[t:t+32].mean())}
                                 for t in range(0,len(temporal),32)])
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'interactions.npz',**curves)
    result=dict(seed=a[0]['seed'],recordings=list(map(str,roots)),structurally_valid=True,
                max_update_residual=max(x[3] for x in recordings),fields=fields,regions=regions,receptor_differences=receptor_differences,
                weight_difference_squared_norm_fractions=fractions,
                definition='J=(response01+response10-response00-response11)/2. Assignment difference is Jpaired-Jswapped. Positive means relative enhancement for experienced-inconsistent combinations, not proven error coding.',
                limits='Two graph seeds, two clips, no delayed outcome, omission or behavioral consumer. Post-learning branches still adapt. A response interaction is not a uniquely identified prediction mechanism or a consciousness test.')
    (output/'summary.json').write_text(json.dumps(result,allow_nan=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paired',type=Path,required=True);p.add_argument('--swapped',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.paired,a.swapped,a.output)
