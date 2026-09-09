"""Matched phase-authorization and predictor-output interventions in changing air.

All four conditions append the same five existing-form PAULA cells and two
muscle inputs to the verified 603-cell checkpoint. Legacy conditions retain
old relay muscle weights and zero the new route. Authorized conditions reverse
those two route weights. Predictor-cut conditions additionally zero only the
two predictor-to-muscle weights; prediction, teaching, adaptation and native
returns remain. These cuts are diagnostics, not a proposed autonomous policy.

Every old cell and pending signal survives construction. The extra layer adds
two ticks; functional effects cannot all be attributed to cancellation alone.
The same-graph legacy control detects effects of adding the new plastic return
paths, which can differ from the earlier 603-cell experiment even at zero
forward motor weights. The actual .21/.105/.21 air course is unchanged.
"""
import argparse
from copy import deepcopy
import inspect
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np

from . import context_organization as base
from .crossed_av_continuation import isolated_rng
from .ventilation_regulation import record
from .ventilation_regulation_replay import restore
from .ventilation_changing_air import ChangingAirOrgans,TICKS,SCHEDULE,audit as audit_body
from .ventilation_screen import observations
from ..components.arbitration.phase_authorization import append_phase_authorization
from ..components.motor.sensory_correction import install_on_runtime
from ..core.external_input_state import synchronize_quiescent_external_inputs
from neuron.extensions.experimental.cascade_eligibility import CascadeEligibilityNeuron


CONDITIONS = ('legacy','authorized','legacy-predictor-cut','authorized-predictor-cut')


def condition_config(original,g,condition):
    if condition not in CONDITIONS: raise ValueError('Undeclared intervention')
    old = original['metadata']['ventilation_feedback']
    cfg,phase = append_phase_authorization(original,drive=old['deficit'],
        phases=[g['cpg'][0],g['cpg'][2]],muscles=g['muscle'])
    points = {(p['neuron_id'],p['synapse_id']):p for p in cfg['synaptic_points'] if p['type']=='postsynaptic'}
    edges = {(e['target_neuron'],e['target_synapse']):e['source_neuron'] for e in cfg['connections']}
    selected = []
    for family,ports in (('legacy',old['ports']),('authorized',phase['ports'])):
        weight = 8. if (condition.startswith('authorized')) == (family=='authorized') else 0.
        for src,nid,sid in ports:
            if edges[nid,sid]!=src: raise ValueError('Unexpected relay source')
            points[nid,sid]['u_i']['info'] = weight
            selected.append([nid,sid,weight])
    if condition.endswith('predictor-cut'):
        predictor_ports = [(nid,sid) for (nid,sid),src in edges.items()
                           if nid in g['muscle'] and src in g['prediction']]
        if len(predictor_ports)!=2: raise ValueError('Expected two predictor muscle projections')
        for nid,sid in predictor_ports:
            points[nid,sid]['u_i']['info'] = 0.; selected.append([nid,sid,0.])
    meta = deepcopy(old); meta['neurons'] += phase['neurons']; meta['phase_authorization'] = phase
    return cfg,meta,selected


def install(net,fresh,original,cfg,selected):
    old = dict(net.network.neurons)
    synchronize_quiescent_external_inputs(net.network)
    install_on_runtime(SimpleNamespace(network=net),fresh,original,cfg)
    changed = []
    for nid,sid,value in selected:
        point = net.network.neurons[nid].postsynaptic_points[sid]
        changed.append([nid,sid,point.u_i.info,value]); point.u_i.info = value
    if any(net.network.neurons[nid] is not n for nid,n in old.items()):
        raise ValueError('Old cell was replaced')
    return changed


def audit_gate(z,cfg,meta):
    """Actual new-path releases and integration, with recorded initial queues."""
    import heapq
    phase = meta['phase_authorization']; ids = list(z['neuron_ids']); reg = list(z['reg_ids'])
    nodes = {n['id']:n for n in cfg['neurons']}; terms = list(map(tuple,z['terminal_ids']))
    points = {(p['neuron_id'],p['synapse_id']):p for p in cfg['synaptic_points'] if p['type']=='postsynaptic'}
    edges = {(e['target_neuron'],e['target_synapse']):(e['source_neuron'],e['source_terminal']) for e in cfg['connections']}
    old_cells = z['reg_cells_initial']; old_terms = z['terminal_initial']
    queue = {nid:[] for nid in phase['neurons']}; start = round(z['body'][0,0]/.004)-1
    for nid,t,s,v in z['reg_queues_initial']:
        if nid in queue: heapq.heappush(queue[int(nid)],(int(t),'hillock',np.float32(v),int(s)))
    residual = 0.
    for t in range(len(z['body'])):
        for nid in phase['neurons']:
            j = reg.index(nid); node = nodes[nid]; inputs = np.zeros_like(z['reg_inputs'][t,j])
            if t==0: inputs[:] = z['reg_arriving_initial'][j]
            else:
                for s in range(2):
                    if (nid,s) in edges:
                        src,term = edges[nid,s]
                        inputs[s,0] = old_cells[ids.index(src),base.FIELDS.index('O')]*old_terms[terms.index((src,term))]
            np.testing.assert_array_equal(z['reg_inputs'][t,j],inputs)
            for s in range(2):
                if inputs[s,0]>0:
                    v = inputs[s,0]*float(z['reg_q_before'][t,j,s])
                    np.testing.assert_array_equal(z['reg_scheduled'][t,j,s],v)
                    heapq.heappush(queue[nid],(start+t+points[nid,s]['distance_to_hillock'],'hillock',v,s))
            current = 0.
            while queue[nid] and queue[nid][0][0]<=start+t:
                _,_,v,s = heapq.heappop(queue[nid])
                current += v*node['params']['delta_decay']**points[nid,s]['distance_to_hillock']
            prior = np.float32(old_cells[ids.index(nid),base.FIELDS.index('S')])
            s = np.clip(prior+(-prior+current)/node['params']['lambda_param'],-1000.,1000.)
            out = node['metadata']['graded_gain']*max(0.,float(s))
            actual = z['cells'][t,ids.index(nid),[base.FIELDS.index('S'),base.FIELDS.index('O')]]
            residual = max(residual,float(np.max(abs(actual-[s,out]))))
            np.testing.assert_allclose(actual,[s,out],rtol=0,atol=3e-6)
        old_cells = z['cells'][t]; old_terms = z['terminal_info'][t]
    return residual


def run(root,output):
    root,output = Path(root).resolve(),Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free<3*1024**3: raise OSError('Need 3 GiB reserve')
    parent = json.loads((root/'manifest.json').read_text())
    clamp = Path(parent['parent']); cm = json.loads((clamp/'manifest.json').read_text())
    verified = Path(cm['parent']); vm = json.loads((verified/'manifest.json').read_text())
    if parent['ticks']!=TICKS or parent['schedule']!=[list(s) for s in SCHEDULE] or vm['mode']!='feedback':
        raise ValueError('Need completed changing-air parent')
    sources = dict(parent['sources']); sources[str(root/'manifest.json')] = base.digest(root/'manifest.json')
    for obj in (run,append_phase_authorization,install_on_runtime,synchronize_quiescent_external_inputs,record,audit_body):
        p = str(Path(inspect.getfile(obj)).resolve()); sources[p] = base.digest(p)
    for p,h in sources.items():
        if base.digest(p)!=h: raise ValueError('Changed source: '+p)
    original = json.loads((verified/'config.json').read_text()); g = vm['groups']
    pm = json.loads((Path(vm['parent'])/'manifest.json').read_text())
    with np.load(pm['media']) as f: features = {k:f[k] for k in ('visual','auditory')}
    output.mkdir(); rows = []
    protocol = dict(parent=str(root),seed=parent['seed'],groups=g,start=6352,ticks=TICKS,
                    schedule=SCHEDULE,conditions=CONDITIONS,sources=sources,limits=__doc__)
    (output/'protocol.json').write_text(base.encode(protocol)+'\n')
    for condition in CONDITIONS:
        folder = output/condition; folder.mkdir()
        cfg,meta,selected = condition_config(original,g,condition)
        (folder/'config.json').write_text(base.encode(cfg)+'\n')
        net,body,delay,od = restore(verified/'initial.paula',verified/'initial-body.npz',g)
        with isolated_rng(): fresh = base.fresh(folder/'config.json',parent['seed'],CascadeEligibilityNeuron)[0]
        changed = install(net,fresh,original,cfg,selected)
        body.organs = ChangingAirOrgans(body.organs.state())
        base.save_checkpoint(net,folder/'initial.paula',sources=list(sources))
        np.savez_compressed(folder/'initial-body.npz',state=body.state(),delay=delay.state(),
            organ=body.organs.state(),organ_delay=od.state(),world_tick=[0],gate=[body.crossings,body.next_gate])
        z = record(net,body,delay,od,features,g,meta,TICKS); z['air_fraction'] = np.asarray(body.organs.air)
        np.savez_compressed(folder/'ticks.npz',**z)
        audit_body(z,cfg,g,features,None); residual = audit_gate(z,cfg,meta)
        base.save_checkpoint(net,folder/'final.paula',sources=list(sources))
        np.savez_compressed(folder/'final-body.npz',state=body.state(),delay=delay.state(),
            organ=body.organs.state(),organ_delay=od.state(),world_tick=[TICKS],gate=[body.crossings,body.next_gate])
        item = dict(condition=condition,meta=meta,changed_weights=changed,observations=observations(z),
                    residual=residual,files={f:base.digest(folder/f) for f in
                        ('config.json','ticks.npz','initial.paula','initial-body.npz','final.paula','final-body.npz')})
        (folder/'manifest.json').write_text(base.encode(dict(protocol,**item))+'\n'); rows.append(item)
        print(base.encode(dict(seed=parent['seed'],condition=condition,residual=residual,
            first_debt={k:v for k,v in item['observations'].items() if k.endswith('_tick')})),flush=True)
        del z,net,body,delay,od,fresh
    if any(base.digest(p)!=h for p,h in sources.items()): raise ValueError('Source changed during course')
    result = dict(protocol,rows=rows,checked_ticks=len(CONDITIONS)*TICKS)
    (output/'manifest.json').write_text(base.encode(result)+'\n'); return result


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('root'); p.add_argument('output')
    a = p.parse_args(); run(a.root,a.output)
