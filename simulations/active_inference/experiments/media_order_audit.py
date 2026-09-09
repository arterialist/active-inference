"""Data-only order-controlled audiovisual comparison, with full-tick contrasts.

No classification score is used as an acceptance criterion. Fixed sound-only
references are observers. They do not feed back into neural activity.
"""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import json
from pathlib import Path

import numpy as np

from .association_balance_audit import checked
from .association_route_probe import digest
from .eligibility_media_audit import verify_ledger
from .graded_media_audit import check_config, verify_rates
from .media_drive_audit import ReceptorAudit, physical_values


def crossed_effect(paired0, swapped0, paired1, swapped1):
    arrays = [np.asarray(a, dtype=float) for a in (paired0, swapped0, paired1, swapped1)]
    if any(a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Incompatible crossed responses')
    a0, a1 = (arrays[0]-arrays[1])/2, (arrays[2]-arrays[3])/2
    return (a0+a1)/2, (a0-a1)/2


def reference_projections(diff, reference):
    """Separate total recruitment from nonuniform ensemble differences.

    Discarding common-mode activity would exclude a legitimate intensity code.
    Keeping only common mode would mistake intensity for a distinct ensemble.
    """
    diff=np.asarray(diff,dtype=float);reference=np.asarray(reference,dtype=float)
    if reference.shape!=(2,diff.shape[-1]) or not np.isfinite(reference).all() or not np.isfinite(diff).all():
        raise ValueError('Invalid reference projection')
    raw=reference[0]-reference[1]
    axes=dict(raw=raw,spatial=raw-raw.mean(),common=np.full_like(raw,raw.mean()))
    return {k:diff@axis/float(axis@axis) if float(axis@axis)>1e-16 else None for k,axis in axes.items()}


def temporal_summary(v):
    v=np.asarray(v,dtype=float)
    if v.ndim!=1 or not len(v) or not np.isfinite(v).all():raise ValueError('Invalid temporal trace')
    return dict(full_mean=float(v.mean()),onset_mean=float(v[:32].mean()),
        post32_mean=float(v[32:].mean()) if len(v)>32 else None,
        positive_ticks=int((v>0).sum()),negative_ticks=int((v<0).sum()),zero_ticks=int((v==0).sum()),
        windows=[dict(start=t,stop=min(t+32,len(v)),value=float(v[t:t+32].mean())) for t in range(0,len(v),32)])


def verify_protocol(manifests):
    if set(manifests) != {(m, o) for m in ('paired', 'swapped') for o in (0, 1)}:
        raise ValueError('Need both assignments and both orders')
    for (mapping, order), m in manifests.items():
        if m['mapping'] != mapping or m['order'] != order:raise ValueError('Wrong history key')
        if m['repeats'] < 1:raise ValueError('Need positive exposure count')
        trials = m['trials']; seq = []
        for i, trial in enumerate(trials):
            if trial['start'] != (0 if i==0 else trials[i-1]['stop']):raise ValueError('Broken time sequence')
            if trial['stop'] <= trial['start']:raise ValueError('Nonpositive exposure')
            if i % 2:
                if trial['phase']!='withdrawal' or trial['visual_clip'] is not None or trial['audio_clip'] is not None or trial['stop']-trial['start']!=96:
                    raise ValueError('Changed withdrawal')
            else:
                if trial['phase']!='experience':raise ValueError('Changed exposure')
                v = trial['visual_clip']; a = trial['audio_clip']
                if v not in (0,1) or a != (v if mapping=='paired' else 1-v):raise ValueError('Wrong association')
                seq.append(v)
        if seq != [order, 1-order]*m['repeats'] or len(trials)!=4*m['repeats']:
            raise ValueError('Wrong exposure order')
    for field in ('visual_clip', 'audio_clip'):
        sequences = {m:Counter(tuple((t[field], t['stop']-t['start']) for t in manifests[m,o]['trials']) for o in (0,1))
                     for m in ('paired','swapped')}
        if sequences['paired'] != sequences['swapped']:raise ValueError('Unbalanced sensory sequence marginals')


def load_state(path):
    with gzip.open(path, 'rt') as f:return json.load(f)


def compare_backchannel(recording, output):
    """Locate the first effects of a synaptic-weight intervention on each route."""
    root, output=Path(recording).resolve(),Path(output).resolve()
    m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text())
    samples={}
    for p in s['branches']:
        key=p['reset'],p['cut']
        if key in samples:raise ValueError('Duplicate backchannel branch')
        with np.load(checked(root,p)) as z:samples[key]={k:z[k] for k in z.files}
    if set(samples)!={(r,c) for r in (False,True) for c in (False,True)}:raise ValueError('Missing backchannel branch')
    if not all(s[k] for k in ('exact_acquisition_replay','intact_prefixes_exact','parent_unchanged')):
        raise ValueError('Replay or observer control failed')
    selected={(n,sid):src for n,sid,src in m['selected_ports']}
    for (reset,cut),d in samples.items():
        if cut and len(d['retro_events']):raise ValueError('Selected cut events were delivered')
        if not cut and len(d['removed_retro']):raise ValueError('Intact events were removed')
        ev=d['removed_retro'] if cut else d['retro_events']
        if not len(ev) or any(selected.get((int(row[1]),int(row[2])))!=int(row[3]) for row in ev):
            raise ValueError('Wrong feedback route')
    traces={};rows={}
    def save(name,a,b):
        diff=b-a;mask=diff.reshape(len(diff),-1)!=0;where=np.flatnonzero(mask.any(axis=1))
        traces[name]=diff
        rows[name]=dict(first_tick=int(where[0]) if len(where) else None,
                        different_values_by_tick=mask.sum(axis=1).tolist(),max_difference=float(abs(diff).max()))
    for cut in (False,True):
        a,b=samples[False,cut],samples[True,cut];condition='cut' if cut else 'intact'
        for role,ids in m['groups'].items():
            for field,index in (('S',0),('O',1),('M0',3)):
                save(f'{condition}/{role}/{field}',a['cells'][:,np.array(ids)-1,index],b['cells'][:,np.array(ids)-1,index])
        save(condition+'/selected_arrivals',a['arrivals'],b['arrivals'])
        for role in ('visual_core','tactile_core','upper_core'):
            take=[i for i,(n,_) in enumerate(m['terminal_order']) if n in m['groups'][role]]
            save(condition+'/'+role+'/terminal_info',a['terminals'][:,take,0],b['terminals'][:,take,0])
    output.mkdir(parents=True,exist_ok=False);np.savez_compressed(output/'trajectories.npz',**traces)
    result=dict(routes_valid=True,recording=str(root),comparisons=rows,
        limits='First divergences are consequences of the matched intervention, not universal conduction constants. '
               'Full input/soma/terminal differences retained. Return events are observed, not independently reconstructed here. '
               'One cue, one history, short prefix; no conclusion that feedback should be removed from the brain.')
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


def compare(paths, output):
    roots = [Path(p).resolve() for p in paths]; output = Path(output).resolve()
    manifests = {}; locations = {}
    for root in roots:
        m = json.loads((root/'manifest.json').read_text()); key = m['mapping'], m['order']
        if key in manifests:raise ValueError('Duplicate history')
        manifests[key] = m;locations[key] = root
    verify_protocol(manifests)
    base = manifests['paired',0]; source_cfg = json.loads((Path(base['source'])/'config.json').read_text())
    birth = load_state(locations['paired',0]/'initial-state.json.gz')
    reference_config = (locations['paired',0]/'config.json').read_bytes()
    features = []
    for path, sha in base['physical_sources'].items():
        if digest(path)!=sha:raise ValueError('Changed physical source')
    for clip in (0,1):
        path = next(Path(p) for p in base['physical_sources'] if Path(p).name==f'sensory-{clip}.npz')
        with np.load(path) as z:features.append({k:z[k] for k in z.files})
    cells = {}; residual = 0.; histories = []
    for key, m in manifests.items():
        root = locations[key]; summary = json.loads((root/'summary.json').read_text())
        if any(m[k]!=base[k] for k in ('seed','groups','selected_ports','physical_sources','repeats','source')):
            raise ValueError('Unmatched graph or media')
        if (root/'config.json').read_bytes()!=reference_config or load_state(root/'initial-state.json.gz')!=birth:
            raise ValueError('Different initial neural state')
        cfg = json.loads(reference_config); check_config(cfg, source_cfg, dict(m, condition='graded'))
        ns = {n['id']:n for n in cfg['neurons']}; ports = m['selected_ports']
        order = [(n['id'],p['synapse_id']) for n in cfg['neurons'] for p in cfg['synaptic_points']
                 if p['type']=='postsynaptic' and p['neuron_id']==n['id']]
        def weight_vector(state):return np.array([state['neurons'][str(n)]['synapses'][str(sid)][0] for n,sid in order])
        def start_values(state):
            return (np.array([state['neurons'][str(n)]['synapses'][str(sid)][0] for n,sid,_ in ports]),
                np.array([state['eligibility'][str(n)]['pre'][ns[n]['metadata']['eligibility_ports'].index(sid)] for n,sid,_ in ports]),
                np.array([state['eligibility'][str(n)]['post'] for n,_,_ in ports]),
                np.array([state['neurons'][str(n)]['O']>0 for n in ns]),
                np.array([state['neurons'][str(n)]['M'][0] for n in ns]))
        start = start_values(birth); weights = weight_vector(birth)
        receptor = ReceptorAudit(cfg, m['groups'], graded_gain=.25)
        if len(summary['training'])!=len(m['trials']):raise ValueError('Missing training')
        for item, trial in zip(summary['training'], m['trials'], strict=True):
            if item['trial']!=trial:raise ValueError('Changed acquisition declaration')
            with np.load(checked(root,item)) as z:data={k:z[k] for k in z.files}
            if not np.array_equal(data['incoming_info_before'],weights):raise ValueError('Broken weight continuity')
            verify_rates(data['cells'], start[4], cfg, ports)
            result = verify_ledger(data, cfg, ports, *start[:4]);residual=max(residual,result[4])
            receptor.check(data['cells'], physical_values(features,trial))
            if not np.allclose(data['incoming_info_after'][:384:2],receptor.q,atol=2e-12,rtol=0):raise ValueError('Sensory learning differs')
            weights=data['incoming_info_after'];start=(*result[:4],data['cells'][-1,:,3])
        parent=load_state(root/'trained-state.json.gz')
        if not np.array_equal(weight_vector(parent),weights):raise ValueError('Trained weight state differs')
        for actual,expected in zip(start_values(parent),start,strict=True):
            if not np.array_equal(actual,expected):raise ValueError('Trained traces differ')
        required={(state,expr,sense,clip) for state,expr,sense in (
            ('initial','spiking','visual'),('initial','graded','audio'),('trained','graded','visual'),
            ('trained','spiking','visual'),('reset_selected','spiking','visual'),('trained','graded','audio')) for clip in (0,1)}
        seen=set()
        for p in summary['probes']:
            case=p['state'],p['expression'],p['sense'],p['clip']
            if case not in required or case in seen:raise ValueError('Unexpected probe')
            seen.add(case);expected=deepcopy(birth if p['state']=='initial' else parent)
            if p['state']=='reset_selected':
                for n,sid,_ in ports:expected['neurons'][str(n)]['synapses'][str(sid)][0]=birth['neurons'][str(n)]['synapses'][str(sid)][0]
            if digest(root/p['start_file'])!=p['start_sha256'] or load_state(root/p['start_file'])!=expected:
                raise ValueError('Undeclared recorded-state change')
            start=start_values(expected)
            trial=p['trial'];clip=p['clip'];sense=p['sense']
            if (trial['start']!=expected['tick'] or trial['stop']-trial['start']!=len(features[clip]['visual'])
                    or trial['visual_clip']!=(clip if sense=='visual' else None)
                    or trial['audio_clip']!=(clip if sense=='audio' else None)):
                raise ValueError('Changed probe input or time')
            with np.load(checked(root,p)) as z:data={k:z[k] for k in z.files}
            if not np.array_equal(data['incoming_info_before'],weight_vector(expected)):raise ValueError('Probe weights differ')
            result=verify_ledger(data,cfg,ports,*start[:4]);residual=max(residual,result[4]);verify_rates(data['cells'],start[4],cfg,ports)
            if p['expression']=='graded':
                r=ReceptorAudit(cfg,m['groups'],graded_gain=.25) if p['state']=='initial' else deepcopy(receptor)
                r.check(data['cells'],physical_values(features,p['trial']))
            silent=m['groups']['touch'] if p['sense']=='visual' else m['groups']['vision']
            if data['cells'][:,np.array(silent)-1,1].any():raise ValueError('Silent receptors emitted')
            # All cells' output histories are kept, not only a selected decoder.
            cells[key+case]=data['cells'][:,:,1].copy()
        if seen!=required:raise ValueError('Missing probe')
        histories.append(dict(mapping=key[0],order=key[1],ticks=summary['ticks']))
    for state,expr,sense,clip in required:
        if state=='initial':
            v=cells['paired',0,state,expr,sense,clip]
            if any(not np.array_equal(v,cells[m,o,state,expr,sense,clip]) for m,o in manifests):
                raise ValueError('Initial output control differs')
    traces={};rows={}
    for role in ('tactile_core','upper_core','mismatch_candidate'):
        ids=np.array(base['groups'][role])-1
        ref=[(cells['paired',0,'initial','graded','audio',c][32:,ids]>0).mean(axis=0) for c in (0,1)]
        axis=ref[0]-ref[1];axis-=axis.mean();norm=float(axis@axis)
        effects={}
        for state,expr in (('trained','graded'),('trained','spiking'),('reset_selected','spiking')):
            contrasts={}
            for mapping,order in manifests:
                a=(cells[mapping,order,state,expr,'visual',0][:,ids]>0).astype(float)
                b=(cells[mapping,order,state,expr,'visual',1][:,ids]>0).astype(float)
                diff=a-b;contrasts[mapping,order]=diff
                name=f'{role}/{state}/{expr}/{mapping}/{order}'
                traces[name+'/cue_difference']=diff
                channel_rows={}
                for channel,v in reference_projections(diff,ref).items():
                    if v is not None:
                        traces[name+'/'+channel+'_reference_projection']=v
                        channel_rows[channel]=temporal_summary(v)
                if norm>1e-16:
                    v=(diff-diff.mean(axis=1,keepdims=True))@axis/norm
                    traces[name+'/sound_projection']=v
                    rows[name]=dict(**temporal_summary(v),channels=channel_rows)
                else:rows[name]=dict(post32_mean=None,windows=[],channels=channel_rows)
            main,interaction=crossed_effect(contrasts['paired',0],contrasts['swapped',0],contrasts['paired',1],contrasts['swapped',1])
            name=f'{role}/{state}/{expr}';traces[name+'/assignment_effect']=main
            traces[name+'/assignment_by_order']=interaction;effects[state,expr]=main
            if norm>1e-16:
                v=main@axis/norm;traces[name+'/assignment_projection']=v
                rows[name]=temporal_summary(v)
        traces[role+'/selected_weight_contribution']=effects['trained','spiking']-effects['reset_selected','spiking']
    output.mkdir(parents=True,exist_ok=False);np.savez_compressed(output/'trajectories.npz',**traces)
    result=dict(protocol_valid=True,initial_outputs_exact=True,max_selected_update_residual=residual,
        histories=histories,readouts=rows,
        limits='One graph seed, two clips, two crossed orders. Sequence marginals are controlled across orders; '
               'nonlinear joint-history effects remain. Projections are fixed initial sound-reference observers, '
               'not accuracy, animal-category recognition or evidence of a neural consumer. '
               'Spiking-cue sensory somata and all downstream currents are observed, not independently reconstructed.')
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recording',type=Path,nargs=4,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();compare(a.recording,a.output)
