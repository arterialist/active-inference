"""Retain per-tick clamp contrasts and first changes along the embodied loop."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import context_organization as base
from .ventilation_input_clamp import CLAMPS,audit


def first(a,b):
    if a.shape!=b.shape: raise ValueError('Mismatched compared arrays')
    hit=np.flatnonzero(np.any(a.reshape(len(a),-1)!=b.reshape(len(b),-1),axis=1))
    return int(hit[0]) if len(hit) else None


def analyze(roots,output,conflicts=()):
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    cases=[];arrays={};sources={};seen=set();checked=0
    conflict_roots={}
    for p in map(lambda p:Path(p).resolve(),conflicts):
        m=json.loads((p/'manifest.json').read_text())
        if m['seed'] in conflict_roots:raise ValueError('Duplicate conflict seed')
        conflict_roots[m['seed']]=(p,m)
    if conflicts and set(conflict_roots)!={11,23,44,77}:raise ValueError('Need all conflict seeds')
    for root in map(lambda p:Path(p).resolve(),roots):
        m=json.loads((root/'manifest.json').read_text());seed=m['seed'];parent=Path(m['parent'])
        if seed in seen: raise ValueError('Duplicate seed')
        seen.add(seed)
        if len(m['rows'])!=4 or {r['name'] for r in m['rows']}!=set(CLAMPS):
            raise ValueError('Incomplete clamp course')
        for p,h in m['sources'].items():
            if base.digest(p)!=h: raise ValueError('Source changed: '+p)
            sources[p]=h
        cfg=json.loads((parent/'config.json').read_text())
        old=json.loads((Path(json.loads((parent/'manifest.json').read_text())['parent'])/'manifest.json').read_text())
        with np.load(old['media']) as f:features={k:f[k] for k in ('visual','auditory')}
        records={}
        for row in m['rows']:
            p=root/row['file'];sources[str(p)]=row['sha256']
            if base.digest(p)!=row['sha256']:raise ValueError('Changed recording')
            with np.load(p) as f:z={k:f[k] for k in f.files}
            if len(z['body'])!=1024 or row['oxygen_clamp']!=CLAMPS[row['name']]:
                raise ValueError('Undeclared course')
            audit(z,cfg,m['groups'],features,row['oxygen_clamp']);checked+=len(z['body'])
            records[row['name']]=z
        ref=records['intact'];ids=list(ref['neuron_ids']);oi=base.FIELDS.index('O');meta=m['meta'];g=m['groups']
        roles={'oxygen_afferent':[meta['oxygen']],'oxygen_comparator':[meta['deficit']],
               'phase_relays':meta['relays'],'energy_alarm':[meta['energy']['alarm']],
               'rhythm':g['cpg'],'muscle':g['muscle'],'prediction':g['prediction']}
        outcomes=[]
        for name,z in records.items():
            onset={key:first(ref[key],z[key]) for key in ('organ_raw','organ_drive','reg_inputs','reg_q_after',
                'cells','muscles','body','drive','weights','terminal_info')}
            onset.update({role+'_output':first(ref['cells'][:,[ids.index(n) for n in ns],oi],
                z['cells'][:,[ids.index(n) for n in ns],oi]) for role,ns in roles.items()})
            for key in ref:
                if key.endswith('_initial'):np.testing.assert_array_equal(ref[key],z[key],err_msg=key)
            key=f's{seed}_{name}'
            arrays[key+'_body']=z['body'];arrays[key+'_organs']=z['organs']
            arrays[key+'_organ_raw']=z['organ_raw'];arrays[key+'_organ_drive']=z['organ_drive']
            columns=sum(roles.values(),[]);arrays[key+'_outputs']=z['cells'][:,[ids.index(n) for n in columns],oi]
            arrays[key+'_neuron_ids']=np.array(columns)
            arrays[key+'_body_difference']=z['body']-ref['body']
            activation=np.cumsum(.1*.004*z['muscles']**2,axis=0)
            arrays[key+'_energy_ledger']=np.column_stack((np.cumsum(z['exchange'][:,7]),activation))
            phase_events=[];ri=list(z['reg_ids'])
            for phase,relay in zip((0,2),meta['relays']):
                for t in np.flatnonzero(z['cells'][:,ids.index(g['cpg'][phase]),oi]>0):
                    if t+2>=len(z['body']):continue
                    phase_events.append(dict(source_tick=int(t),phase=phase,relay_output_tick=int(t+2),
                        actual_oxygen_fraction=float(z['organ_raw'][t,2]),
                        delivered_oxygen_fraction=float(z['organ_drive'][t,2]),
                        deficit_arrival=float(z['reg_inputs'][t+1,ri.index(relay),1,0]),
                        relay_output=float(z['cells'][t+2,ids.index(relay),oi])))
            alarm=z['cells'][:,ids.index(meta['energy']['alarm']),oi]
            condition=dict(name=name,first_difference=onset,
                first_debt={label:(int(ix[0]) if len(ix) else None) for label,col in (('oxygen',4),('energy',10))
                    for ix in [np.flatnonzero(z['organs'][:,col]>1e-12)]},
                alarm_active_ticks=np.flatnonzero(alarm>0).tolist(),
                phase_events=sorted(phase_events,key=lambda event:event['source_tick']),
                muscle_activation_cost_j=activation[-1],
                oxygen_min=float(z['organs'][:,0].min()),energy_min=float(z['organs'][:,6].min()),
                final_organs=z['organs'][-1])
            outcomes.append(condition)
        conflict=None
        if conflicts:
            cp,cm=conflict_roots[seed]
            if Path(cm['parent'])!=parent or cm['oxygen_clamp']!=.25 or cm['ticks']!=1024:
                raise ValueError('Unmatched conflict course')
            for p,h in cm['sources'].items():
                if base.digest(p)!=h:raise ValueError('Conflict source changed: '+p)
                sources[p]=h
            if base.digest(cp/'ticks.npz')!=cm['sha256']:raise ValueError('Conflict recording changed')
            sources[str(cp/'ticks.npz')]=cm['sha256'];sources[str(cp/'manifest.json')]=base.digest(cp/'manifest.json')
            cut_cfg=json.loads((cp/'config.json').read_text());expected=json.loads((parent/'config.json').read_text())
            expected_changes=[]
            for _,nid,sid in meta['energy']['ports']:
                p=next(p for p in expected['synaptic_points'] if p['type']=='postsynaptic' and
                       (p['neuron_id'],p['synapse_id'])==(nid,sid))
                expected_changes.append([nid,sid,p['u_i']['info'],0.]);p['u_i']['info']=0.
            if cut_cfg!=expected or cm['changed_weights']!=expected_changes:raise ValueError('Undeclared graph intervention')
            with np.load(cp/'ticks.npz') as f:cut={k:f[k] for k in f.files}
            audit(cut,cut_cfg,g,features,.25);checked+=len(cut['body'])
            intact=records['held-low']
            for k in intact:
                if k.endswith('_initial'):np.testing.assert_array_equal(intact[k],cut[k],err_msg=k)
            active=np.flatnonzero(intact['cells'][:,ids.index(meta['energy']['alarm']),oi]>0)
            if not len(active):raise ValueError('Energy pathway was not recruited')
            onset={k:first(intact[k],cut[k]) for k in ('cells','body','muscles','organ_raw','organ_drive',
                                                      'terminal_info','weights','reg_q_after')}
            # No downstream state divergence before the first alarm release.
            for k in ('cells','body','muscles','organ_raw','organ_drive','weights'):
                np.testing.assert_array_equal(intact[k][:active[0]],cut[k][:active[0]],err_msg=k)
            key=f's{seed}_energy-cut';arrays[key+'_body']=cut['body'];arrays[key+'_organs']=cut['organs']
            columns=sum(roles.values(),[]);arrays[key+'_outputs']=cut['cells'][:,[ids.index(n) for n in columns],oi]
            arrays[key+'_neuron_ids']=np.array(columns)
            arrays[key+'_difference']=cut['organs']-intact['organs']
            conflict=dict(first_alarm_tick=int(active[0]),first_difference=onset,
                final_intact=intact['organs'][-1],final_cut=cut['organs'][-1],
                minimum_energy_intact=float(intact['organs'][:,6].min()),minimum_energy_cut=float(cut['organs'][:,6].min()))
        cases.append(dict(seed=seed,conditions=outcomes,roles=roles,energy_conflict=conflict))
        sources[str(root/'manifest.json')]=base.digest(root/'manifest.json')
    if seen!={11,23,44,77}: raise ValueError('Need all four seeds')
    output.mkdir();np.savez_compressed(output/'per-tick.npz',**arrays)
    result=dict(cases=cases,sources=sources,checked_ticks=checked,
        first_difference_units='zero-based local neural tick index; not a universal conduction constant',
        limits='First differences trace intervention propagation, not mediation proof for each path. '
               'Exact differences can precede behaviorally important amplitudes. Full raw recordings remain primary.',
        source_sha256=base.digest(__file__),sha256=base.digest(output/'per-tick.npz'))
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(checked_ticks=checked,cases=[dict(seed=c['seed'],conditions=[
        {k:v for k,v in row.items() if k in ('name','first_difference','first_debt','oxygen_min','energy_min')}
        for row in c['conditions']]) for c in cases])),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',nargs='+');p.add_argument('--output',required=True)
    p.add_argument('--conflicts',nargs='*',default=[])
    a=p.parse_args();analyze(a.roots,a.output,a.conflicts)
