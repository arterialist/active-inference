"""Full-population, real-media screen of the new local predictive bridge.

No simulation body or live server. Every completed cellular tick, predictive
weight, context/error trace and information terminal is retained. Two physical
recordings cannot establish category learning. Short acquisition is a screen,
not acceptance. Positive adaptation continues during all probes.
"""
import argparse
from copy import deepcopy
import inspect
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .composition_probe import encode, fingerprint, k
from .association_route_probe import digest
from .multimodal_pairing_probe import fresh, inputs
from .population_hierarchy import cellular, weight_values, FIELDS
from .media_order_control import protocol
from ..components.learning.predictive_bridge import append_predictive_bridge
from ..core.runtime_checkpoint import save_checkpoint, load_checkpoint
from neuron.neuron import setup_neuron_logger
from neuron.extensions.experimental.predictive_receptor import PredictiveReceptorNeuron


def record(net, features, groups, bridge, trial):
    members = list(net.network.neurons.values())
    predictors = [net.network.neurons[i] for i in bridge['prediction']]
    comparators = [net.network.neurons[i] for role in ('error_positive', 'error_negative') for i in bridge[role]]
    points = [p for n in members for p in n.postsynaptic_points.values()]
    q = lambda: np.array([[n.postsynaptic_points[s].u_i.info for s in n.prediction_ports] for n in predictors])
    terminals = lambda: np.array([n.presynaptic_points[k.TERM].u_o.info for n in members])
    start = dict(weights=q(), context=np.array([n.prediction_context for n in predictors]),
        error=np.array([n.prediction_error for n in predictors]), cells=cellular(members),
        terminals=terminals(), incoming_info=weight_values(points),
        comparator_weights=np.array([[p.u_i.info for p in n.postsynaptic_points.values()] for n in comparators]),
        comparator_due=np.array([sum(v*n.params.delta_decay**n.distances[s] for t, _, v, s in n.propagation_queue
                                    if t <= net.current_tick) for n in comparators]))
    names = ('cells', 'terminals', 'weights', 'context', 'arrivals', 'error', 'error_used',
             'error_arrival', 'eta', 'comparator_weights', 'comparator_arrivals', 'comparator_scheduled')
    rows = {key: [] for key in names}
    comparator_set = {id(n) for n in comparators}
    original_tick = PredictiveReceptorNeuron.tick
    observed = {}
    def observed_tick(n, external_inputs, current_tick, dt=1.):
        if id(n) not in comparator_set:
            return original_tick(n, external_inputs, current_tick, dt)
        arriving = n.input_buffer[:, 0].copy()
        result = original_tick(n, external_inputs, current_tick, dt)
        # Base PAULA clears the delivery buffer before returning. Capture it
        # before tick; read the resulting local potential only where driven.
        observed[n.id] = (arriving, [p.potential if arriving[s] > 0 else 0.
                                     for s,p in n.postsynaptic_points.items()])
        return result
    for t in range(trial['start'], trial['stop']):
        if t != net.current_tick: raise ValueError('Nonconsecutive neural time')
        for nid, value in inputs(features, groups, trial, t): net.set_external_input(nid, 0, value)
        PredictiveReceptorNeuron.tick = observed_tick
        try:
            net.run_tick()
        finally:
            PredictiveReceptorNeuron.tick = original_tick
        rows['cells'].append(cellular(members)); rows['terminals'].append(terminals())
        rows['weights'].append(q())
        for key, attr in (('context','prediction_context'), ('arrivals','prediction_arrivals'),
                          ('error','prediction_error'), ('error_used','prediction_error_used'),
                          ('error_arrival','prediction_error_arrival'), ('eta','prediction_eta')):
            rows[key].append(np.array([getattr(n, attr) for n in predictors]))
        rows['comparator_weights'].append(np.array([[p.u_i.info for p in n.postsynaptic_points.values()] for n in comparators]))
        rows['comparator_arrivals'].append(np.array([observed[n.id][0] for n in comparators]))
        rows['comparator_scheduled'].append(np.array([observed[n.id][1] for n in comparators]))
    data = {key: np.asarray(value) for key, value in rows.items()}
    data.update({f'start_{key}': value for key, value in start.items()})
    data['end_incoming_info'] = weight_values(points)
    if any(not np.isfinite(a).all() for a in data.values()):
        raise FloatingPointError('Nonfinite recorded state')
    return data


def audit_record(data, config, bridge):
    """Independent algebra and transmission checks, no learning-helper calls."""
    ids = [n['id'] for n in config['neurons']]; index = {n:i for i,n in enumerate(ids)}
    nodes = {n['id']: n for n in config['neurons']}
    connections = {(c['target_neuron'],c['target_synapse']):c['source_neuron'] for c in config['connections']}
    ps = bridge['prediction']; cs = bridge['error_positive'] + bridge['error_negative']
    src = np.array([[index[connections[n,s]] for s in nodes[n]['metadata']['prediction_ports']] for n in ps])
    error_sources = np.array([[index[connections[n,s]] for s,_ in nodes[n]['metadata']['prediction_error_ports']] for n in ps])
    polarities = np.array([[sign for _,sign in nodes[n]['metadata']['prediction_error_ports']] for n in ps])
    cmp_sources = np.array([[index[connections[n,s]] for s in (0,1)] for n in cs])
    md = [nodes[n]['metadata'] for n in ps]
    dx = np.exp(-1/np.array([m['prediction_tau_context'] for m in md]))[:,None]
    de = np.exp(-1/np.array([m['prediction_tau_error'] for m in md]))
    caps = np.array([m['prediction_cap'] for m in md])[:,None]
    basal = np.array([nodes[n]['params']['eta_post'] for n in ps])
    boost = np.array([m['prediction_boost'] for m in md]); half = np.array([m['prediction_half'] for m in md])
    lam = np.array([nodes[n]['params']['lambda_param'] for n in cs])
    decay = np.array([nodes[n]['params']['delta_decay'] for n in cs])
    q, x, e = (data['start_'+key].copy() for key in ('weights','context','error'))
    prev, terminal = data['start_cells'], data['start_terminals']
    cq, due = data['start_comparator_weights'], data['start_comparator_due']
    residual = 0.

    def same(actual, expected, label):
        nonlocal residual
        d = float(np.max(np.abs(actual-expected))); residual = max(residual,d)
        if not np.allclose(actual, expected, rtol=0, atol=2e-12):
            raise ValueError(f'{label}: residual {d}')

    for t, cells in enumerate(data['cells']):
        # Source output is multiplied by its actual terminal at release and
        # rounded by the native float32 delivery buffer one tick later.
        released = prev[:,1] * terminal
        arrivals = released[src].astype(np.float32).astype(float)
        same(data['arrivals'][t], arrivals, 'context delivery')
        observed_error = (released[error_sources].astype(np.float32).astype(float)*polarities).sum(axis=1)
        same(data['error_arrival'][t], observed_error, 'neural error delivery')
        ca = released[cmp_sources].astype(np.float32).astype(float)
        same(data['comparator_arrivals'][t], ca, 'matched comparison delivery')
        same(data['comparator_scheduled'][t], ca.astype(np.float32)*cq.astype(np.float32), 'comparison synaptic current')
        # Native information and membrane arithmetic retains numpy float32
        # scalars. Mirror operation order, including each dendrite's decay.
        previous_s = prev[[index[n] for n in cs],0].astype(np.float32)
        expected_s = previous_s + (1/lam).astype(np.float32)*(-previous_s+due.astype(np.float32))
        same(cells[[index[n] for n in cs],0], expected_s, 'comparison integration')
        same(cells[[index[n] for n in cs],1], np.maximum(0,expected_s), 'opponent release')
        same(data['error_used'][t], e, 'one-tick local receptor causality')
        x = dx*x+(1-dx)*arrivals
        eta = basal*(1+boost*np.abs(e)/(half+np.abs(e)))
        q = np.clip(q+eta[:,None]*e[:,None]*x, 0, caps)
        e = de*e+(1-de)*observed_error
        same(data['context'][t],x,'context receptor equation')
        same(data['eta'][t],eta,'positive local learning rate')
        same(data['weights'][t],q,'selected weight equation')
        same(data['error'][t],e,'error receptor equation')
        due = (data['comparator_scheduled'][t].astype(np.float32)*decay[:,None].astype(np.float32)).sum(axis=1,dtype=np.float32)
        cq = data['comparator_weights'][t]; prev = cells; terminal = data['terminals'][t]
    return residual


def run(source, media, output, *, mapping='paired', order=0, seed=11, repeats=2):
    source, media, output = map(lambda p:Path(p).resolve(), (source,media,output))
    old = json.loads((source/'manifest.json').read_text())
    if any(digest(p)!=h for p,h in old['source_hashes'].items()):
        raise ValueError('Prior runtime changed')
    original = json.loads((source/'config.json').read_text())
    cfg, bridge, edges, selected = append_predictive_bridge(original, old['groups']['vision'],
        old['groups']['touch'], seed=seed)
    features = []
    for clip in (0,1):
        if digest(media/f'sensory-{clip}.npz') != old['physical_sources'][str(media/f'sensory-{clip}.npz')]:
            raise ValueError('Physical stimulus differs from the prior recording')
        with np.load(media/f'sensory-{clip}.npz') as z: features.append({k:z[k] for k in z.files})
    length = int(features[0]['ticks'])
    if length != 300 or int(features[1]['ticks']) != length: raise ValueError('Expected original 300-tick recordings')
    if shutil.disk_usage(output.parent).free < 750*1024**2: raise OSError('Less than 750 MiB available')
    output.mkdir(exist_ok=False); cfg_path=output/'config.json'; cfg_path.write_text(encode(cfg)+'\n')
    hashes=fingerprint()
    for obj in (run, append_predictive_bridge, fresh, protocol, inputs):
        p=Path(inspect.getfile(obj)).resolve(); hashes[str(p)]=digest(p)
    trials=protocol(length,repeats,mapping,order)
    manifest=dict(source=str(source),source_config_sha256=digest(source/'config.json'),
        base_graph_seed=original['metadata'].get('seed'),bridge_seed=seed,mapping=mapping,order=order,
        repeats=repeats,groups=old['groups'],bridge=bridge,edges=edges,selected=selected,
        fields=FIELDS,trials=trials,source_hashes=hashes,
        physical_sources={str(media/f'sensory-{i}.npz'):digest(media/f'sensory-{i}.npz') for i in (0,1)},
        limitations='Screen, not acceptance. Two clips; existing 1152-cell birth graph plus 320 cells. '
                    'Bridge predicts physical auditory receptor channels from visual receptors. '
                    'No hierarchical completion, semantic understanding, action or embodiment claim. '
                    'Native learning and retrograde coupling remain active; no teaching/recall flag. '
                    'The original unaligned mismatch circuit is retained but not called a prediction error.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    setup_neuron_logger('CRITICAL')
    net,_,members,_=fresh(cfg_path,seed,PredictiveReceptorNeuron)
    assert len(members)==1472
    save_checkpoint(net,output/'initial.neural-checkpoint',sources=(__file__,))
    entries=[]; started=time.perf_counter(); used=0

    def perform(network,trial,name,**tags):
        nonlocal used
        data=record(network,features,old['groups'],bridge,trial)
        residual=audit_record(data,cfg,bridge)
        path=output/(name+'.npz'); np.savez_compressed(path,**data)
        used += path.stat().st_size
        entries.append(dict(file=path.name,sha256=digest(path),trial=trial,audit_residual=residual,**tags))
        (output/'progress.json').write_text(encode(dict(entries=entries,bytes=used))+'\n')
        print(encode(dict(file=path.name,tick=network.current_tick,seconds=time.perf_counter()-started,
                          bytes=used,residual=residual)),flush=True)
        if used>180*1024**2 or shutil.disk_usage(output).free<600*1024**2:
            raise OSError('Research recording disk budget reached; partial records retained')

    for i,trial in enumerate(trials): perform(net,trial,f'experience-{i:02d}',phase=trial['phase'])
    save_checkpoint(net,output/'trained.neural-checkpoint',sources=(__file__,))
    for state in ('initial','trained','reset_selected'):
        parent='initial' if state=='initial' else 'trained'
        for clip in (0,1,None):
            if state=='reset_selected' and clip is None: continue
            branch=load_checkpoint(output/f'{parent}.neural-checkpoint',trusted=True)
            probe=branch.network
            if state=='reset_selected':
                birth={(p['neuron_id'],p['synapse_id']):p['u_i']['info'] for p in cfg['synaptic_points'] if p['type']=='postsynaptic'}
                for nid,sid,_ in selected: probe.network.neurons[nid].postsynaptic_points[sid].u_i.info=birth[nid,sid]
            t=probe.current_tick
            trial=dict(start=t,stop=t+length,visual_clip=clip,audio_clip=None)
            perform(probe,trial,f'probe-{state}-{clip}',phase='probe',state=state,clip=clip)
    if any(digest(p)!=h for p,h in hashes.items()): raise ValueError('Runtime changed during experiment')
    result=dict(entries=entries,seconds=time.perf_counter()-started,bytes=used,
                ticks=sum(e['trial']['stop']-e['trial']['start'] for e in entries))
    (output/'summary.json').write_text(encode(result)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','media','output'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--mapping',choices=('paired','swapped'),default='paired')
    p.add_argument('--order',type=int,choices=(0,1),default=0)
    p.add_argument('--seed',type=int,default=11); p.add_argument('--repeats',type=int,default=2)
    a=p.parse_args();run(**vars(a))
