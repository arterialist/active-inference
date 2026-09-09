"""Independent audit of birth-placement and mean-growth interventions."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np

from .association_balance_audit import checked
from .association_route_probe import digest
from .eligibility_media_audit import verify_ledger
from .graded_media_audit import verify_rates
from .media_drive_audit import physical_values
from .media_order_audit import load_state,verify_protocol,reference_projections,temporal_summary,crossed_effect
from .media_order_control import ledger_start
from .media_weight_identity_audit import quiet_receptor_state,verify_local_current,first_difference


def acquired_contrasts(samples):
    names=('intact','initial','cycle1','cycle2','cycle3','birth_cycle1','birth_cycle2','birth_cycle3','mean_growth')
    if set(samples)!={(name,cue) for name in names for cue in (0,1)}:
        raise ValueError('Missing matched cue/weight condition')
    shape=samples['intact',0].shape
    if any(x.shape!=shape or not np.isfinite(x).all() for x in samples.values()):
        raise ValueError('Incompatible trajectories')
    d={name:samples[name,0]-samples[name,1] for name in names}
    learned_placement=d['intact']-sum(d[f'cycle{i}'] for i in (1,2,3))/3
    birth_placement=d['initial']-sum(d[f'birth_cycle{i}'] for i in (1,2,3))/3
    return dict(birth_placement=birth_placement, acquired_placement=learned_placement-birth_placement,
                input_specific_growth=d['intact']-d['mean_growth'])


def compare(recordings,output):
    roots=[Path(p).resolve() for p in recordings]
    manifests=[json.loads((p/'manifest.json').read_text()) for p in roots]
    courses=[json.loads((Path(m['course'])/'manifest.json').read_text()) for m in manifests]
    verify_protocol({(m['mapping'],m['order']):m for m in courses})
    if len(roots)!=4:raise ValueError('Need four distinct histories')
    base=courses[0]; config_bytes=(Path(manifests[0]['course'])/'config.json').read_bytes()
    cfg=json.loads(config_bytes); ports=base['selected_ports']
    initial_state=load_state(Path(manifests[0]['course'])/'initial-state.json.gz')
    q0=np.array([initial_state['neurons'][str(n)]['synapses'][str(sid)][0] for n,sid,_ in ports])
    rows,traces,effects,comparisons={},{},{},{}; residual=0.
    for root,m,cm in zip(roots,manifests,courses,strict=True):
        course=Path(m['course']); source=Path(m['source']); s=json.loads((root/'summary.json').read_text())
        old=json.loads((source/'summary.json').read_text())
        if not s['intact_control_exact'] or s['acquisition_ticks']!=0:raise ValueError('Cached control failed')
        if any(m[k]!=cm[k] for k in ('mapping','order','seed','groups','selected_ports')):raise ValueError('Wrong history')
        if (course/'config.json').read_bytes()!=config_bytes or load_state(course/'initial-state.json.gz')!=initial_state:
            raise ValueError('Birth or configuration differs')
        if any(digest(p)!=h for p,h in {**m['source_hashes'],**m.get('continuation_sources',{})}.items()):raise ValueError('Source changed')
        parent=load_state(course/'checkpoint-16-state.json.gz')
        q1=np.array([parent['neurons'][str(n)]['synapses'][str(sid)][0] for n,sid,_ in ports])
        expected={f'birth_cycle{i}':q0.copy() for i in (1,2,3)};expected['mean_growth']=q0.copy()
        for nid in sorted({n for n,_,_ in ports}):
            indices=[i for i,(n,_,_) in enumerate(ports) if n==nid]
            if len(indices)!=4:raise ValueError('Wrong number of selected inputs')
            for shift in (1,2,3):
                for j,i in enumerate(indices):expected[f'birth_cycle{shift}'][i]=q0[indices[(j-shift)%4]]
            expected['mean_growth'][indices]+=(q1[indices]-q0[indices]).sum()/4
        with np.load(root/'weights.npz') as z:
            if not np.array_equal(z['initial'],q0) or not np.array_equal(z['learned'],q1):raise ValueError('Wrong source weights')
            for condition,q in expected.items():
                if not np.array_equal(z[condition],q):raise ValueError('Wrong factor weights')
        sensory_parent=quiet_receptor_state(cfg,m['groups'],parent)
        features=[]
        for clip in (0,1):
            p=next(Path(p) for p in cm['physical_sources'] if Path(p).name==f'sensory-{clip}.npz')
            with np.load(p) as z:features.append({k:z[k] for k in z.files})
        order=[(n['id'],p['synapse_id']) for n in cfg['neurons'] for p in cfg['synaptic_points'] if p['type']=='postsynaptic' and p['neuron_id']==n['id']]
        take=np.array([order.index((n,sid)) for n,sid,_ in ports])
        full=np.array([parent['neurons'][str(n)]['synapses'][str(sid)][0] for n,sid in order])
        samples={};seen=set();history=f"{m['mapping']}/{m['order']}"
        for p in old['probes']:
            with np.load(checked(source,p)) as z:samples[p['condition'],p['cue']]=z['cells'][:,:,1].copy()
        for p in s['probes']:
            condition,cue=p['condition'],p['cue'];key=condition,cue
            if condition not in expected or cue not in (None,0,1) or key in seen:raise ValueError('Invalid branch')
            seen.add(key)
            if p['trial']!=dict(start=parent['tick'],stop=parent['tick']+300,visual_clip=cue,audio_clip=None):raise ValueError('Wrong physical trial')
            with np.load(checked(root,p)) as z:d={k:z[k] for k in z.files}
            weights=full.copy();weights[take]=expected[condition]
            if not np.array_equal(d['incoming_info_before'],weights):raise ValueError('Unexpected initial weights')
            state=deepcopy(parent)
            for (n,sid,_),q in zip(ports,expected[condition],strict=True):state['neurons'][str(n)]['synapses'][str(sid)][0]=float(q)
            start=ledger_start(state,cfg,ports); result=verify_ledger(d,cfg,ports,*start[:4]);residual=max(residual,result[4])
            verify_rates(d['cells'],start[4],cfg,ports)
            verify_local_current(d['arrivals'],d['weights'],expected[condition],d['selected_local_current'])
            sensory=deepcopy(sensory_parent);sensory.check(d['cells'],physical_values(features,p['trial']))
            if not np.allclose(d['incoming_info_after'][:384:2],sensory.q,atol=2e-12,rtol=0):raise ValueError('Wrong sensory endpoint')
            if not np.array_equal(d['incoming_info_after'][take],result[0]):raise ValueError('Wrong selected endpoint')
            samples[key]=d['cells'][:,:,1].copy()
            if condition=='mean_growth':
                with np.load(source/f'intact-cue-{cue}.npz') as z:
                    for field in ('selected_local_current','arrivals','terminals'):
                        comparisons[f'{history}/{cue}/{field}']=first_difference(z[field],d[field])
                    for role in ('tactile_core','upper_core','visual_core'):
                        ids=np.array(m['groups'][role])-1
                        for field,column in (('S',0),('O',1)):
                            comparisons[f'{history}/{cue}/{role}/{field}']=first_difference(z['cells'][:,ids,column],d['cells'][:,ids,column])
        if seen!={(c,v) for c in expected for v in (None,0,1)}:raise ValueError('Missing factor branch')
        for role in ('tactile_core','upper_core','mismatch_candidate'):
            ids=np.array(m['groups'][role])-1
            for (condition,cue),value in samples.items():
                x=value[:,ids];spikes=(x>0).sum(axis=1);times=np.flatnonzero(spikes)
                name=f'{history}/{condition}/{cue}/{role}'
                traces[name]=x
                rows[name]=dict(events=int(spikes.sum()),late=int(spikes[32:].sum()),first=int(times[0]) if len(times) else None,last=int(times[-1]) if len(times) else None)
            value=acquired_contrasts({(c,v):a[:,ids] for (c,v),a in samples.items() if v is not None})
            effects[m['mapping'],m['order'],role]=value
    projections={}
    for role in ('tactile_core','upper_core','mismatch_candidate'):
        ids=np.array(base['groups'][role])-1;refs=[]
        for clip in (0,1):
            with np.load(Path(base['source'])/f'initial-graded-audio-{clip}.npz') as z:refs.append((z['cells'][32:,ids,1]>0).mean(axis=0))
        for mapping,order in (('paired',0),('swapped',0),('paired',1),('swapped',1)):
            course=Path(next(m['course'] for m in manifests if m['mapping']==mapping and m['order']==order))
            current=[]
            for clip in (0,1):
                with np.load(course/f'checkpoint-16-context-vNone-a{clip}.npz') as z:current.append((z['cells'][96:,ids,1]>0).mean(axis=0))
            for kind,values in effects[mapping,order,role].items():
                traces[f'{mapping}/{order}/{role}/{kind}']=values
                for channel,p in reference_projections(values,refs).items():
                    if p is not None:
                        key=f'{mapping}/{order}/{role}/{kind}/{channel}';traces[key]=p;projections[key]=temporal_summary(p)
            diff=traces[f'{mapping}/{order}/mean_growth/0/{role}']-traces[f'{mapping}/{order}/mean_growth/1/{role}']
            for basis,ref in (('original',refs),('current',current)):
                for channel,p in reference_projections(diff,ref).items():
                    if p is not None:
                        key=f'{mapping}/{order}/{role}/mean_growth/{basis}/{channel}';traces[key]=p;projections[key]=temporal_summary(p)
        for kind in ('birth_placement','acquired_placement','input_specific_growth'):
            main,interaction=crossed_effect(*(effects[m,o,role][kind] for m,o in (('paired',0),('swapped',0),('paired',1),('swapped',1))))
            for label,values in (('assignment',main),('assignment_by_order',interaction)):
                key=f'{role}/{kind}/{label}';traces[key]=values
                for channel,p in reference_projections(values,refs).items():
                    if p is not None:traces[key+'/'+channel]=p;projections[key+'/'+channel]=temporal_summary(p)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'trajectories.npz',**traces)
    result=dict(factors_valid=True,selected_potentials_exact=True,graded_sensory_valid=True,max_selected_update_residual=residual,
        rows=rows,projections=projections,first_effects=comparisons,recordings=[str(p) for p in roots],audit_source_sha256=digest(__file__),
        limits='One graph seed. Conditional effects from learned fast states, not complete memory erasure or neural-use tests. '
               'Mean growth preserves target sums only at intervention onset; subsequent plasticity and return paths may diverge. '
               'Independent selected and sensory reconstructions do not cover all hidden state.')
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result
