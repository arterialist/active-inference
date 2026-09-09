"""Learned parallel evidence population, before categorical output saturation.

Uses existing PAULA somatic dynamics and local learning extensions. Six cells
per auditory coordinate vary threshold, not category preference. All acquire
their own visual synapses from birth. No trained weight copying or host readout
controls the network. This tests an interface, not a completed decision circuit.
"""
import argparse
from copy import deepcopy
import gzip
import inspect
import json
from pathlib import Path
import time

import numpy as np

from .association_balance_probe import balanced_config, record_trial
from .association_route_probe import digest
from .composition_probe import encode, fingerprint, k
from .eligibility_association_probe import dynamic_snapshot, protocol
from .multimodal_pairing_probe import fresh
from .sensory_schedule import receptor_schedule, ScheduledReceptors
from .association_cue_probe import cue_cases
from neuron.extensions.experimental.port_modulation import PortModulationNeuron

THRESHOLDS = (.04, .08, .16, .32, .64, 1.28)


def evidence_config(seed, architecture='diverse'):
    if architecture not in ('diverse', 'homogeneous'):
        raise ValueError(architecture)
    cfg, groups = balanced_config(seed)
    for n in cfg['neurons']:
        if n['id'] in groups['auditory']:
            n['metadata']['native_port_modulation'] = [dict(port=i, sensitivity=.25) for i in range(35, 67)]
    original_neurons = {n['id']: deepcopy(n) for n in cfg['neurons']}
    original_points = deepcopy(cfg['synaptic_points'])
    original_edges = deepcopy(cfg['connections'])
    # A new terminal per source isolates this projection's native retrograde
    # adaptation. It does not disable retrograde learning on either projection.
    sources = sorted({e['source_neuron'] for e in original_edges if e['target_neuron'] in groups['auditory']})
    terminals = {}
    for src in sources:
        existing = [p for p in original_points if p['type']=='presynaptic' and p['neuron_id']==src]
        terminal = deepcopy(existing[0])
        terminal['terminal_id'] = max(p['terminal_id'] for p in existing)+1
        terminals[src] = terminal['terminal_id']; cfg['synaptic_points'].append(terminal)
    cursor = max(original_neurons)+1
    groups['evidence'] = []; groups['evidence_coordinates'] = []
    for coordinate, old_id in enumerate(groups['auditory']):
        ids = []
        for level, threshold in enumerate(THRESHOLDS):
            nid = cursor; cursor += 1; ids.append(nid)
            n = deepcopy(original_neurons[old_id]); n['id'] = nid
            r = threshold if architecture=='diverse' else float(np.sqrt(THRESHOLDS[0]*THRESHOLDS[-1]))
            n['params'].update(lambda_param=4., r_base=r, b_base=2*r)
            n['metadata'].update(role='evidence', auditory_coordinate=coordinate, threshold_level=level)
            cfg['neurons'].append(n)
            for p in original_points:
                if p['neuron_id'] != old_id: continue
                new = deepcopy(p); new['neuron_id'] = nid
                if new['type']=='postsynaptic' and new['synapse_id']==32:
                    # With lambda=4, audio-alone drive 8*.99/4=1.98 crosses
                    # every threshold during acquisition. This is a declared
                    # stronger afferent projection, not conductance-neutral.
                    new['u_i']['info'] = 8.
                cfg['synaptic_points'].append(new)
            for edge in original_edges:
                if edge['target_neuron'] != old_id: continue
                new = deepcopy(edge); new['target_neuron'] = nid
                new['source_terminal'] = terminals[new['source_neuron']]
                cfg['connections'].append(new)
        groups['evidence_coordinates'].append(ids); groups['evidence'].extend(ids)
    cfg['metadata'].update(preparation='parallel-association-evidence-v0', architecture=architecture,
        scale='Six independently plastic cells per audio coordinate; input conductance per visual/inhibitory cell unchanged, total conductance increased sixfold. Audio teaching projection increased to 8 for slower integration. No normalisation claim.')
    return cfg, groups


def record(net, groups, masks, trial):
    original = PortModulationNeuron.tick
    ids = groups['evidence']; captured = []; order = []
    def observed(n, ext, tick, dt=1.):
        if n.id not in ids or net.network.neurons[n.id] is not n:
            return original(n, ext, tick, dt)
        ports = [n.postsynaptic_points[i] for i in range(67)]
        before = np.array([p.u_i.info for p in ports])
        incoming = n.input_buffer.copy()
        pre = n.eligibility_pre.copy(); post = n.eligibility_post
        eta = n.params.eta_post*n.rate_multiplier()
        events = original(n, ext, tick, dt)
        captured.append((incoming[:, 0], before, [p.u_i.info for p in ports],
                         [p.u_i.plast for p in ports], pre, post, eta))
        order.append(n.id)
        return events
    PortModulationNeuron.tick = observed
    try:
        data = record_trial(net, groups, masks, trial)
    finally:
        PortModulationNeuron.tick = original
    if order != ids*trial['ticks']: raise AssertionError('Wrong evidence observation order')
    for j, name in enumerate(('input','before','after','plast','pre','post','eta')):
        values = np.asarray([v[j] for v in captured])
        data['evidence_'+name] = values.reshape((trial['ticks'], len(ids))+values.shape[1:])
    return data


def test_schedules(masks, seed):
    schedules = {}
    for case in cue_cases(masks, seed):
        if case['kind'] not in ('clean', 'omit75', 'replace25', 'silence'): continue
        for timing in ('synchronous','resampled_spread','minority_first','majority_first'):
            schedules[case['name']+'/'+timing] = receptor_schedule(case, masks, seed, timing)
    # Actual event transitions, with no hidden boundary/reset sent to neurons.
    # Same cue total durations, both orders; 0/8/24 empty ticks between events.
    for first in (0, 1):
        for gap in (0, 8, 24):
            a = np.zeros((96, 32), np.uint8)
            for cue, start in ((first, 0), (1-first, 32+gap)):
                for t in range(start, start+32, 8):
                    a[t, np.array(masks['vision'][cue])-1] = 1
            schedules[f'transition-first{first}-gap{gap}'] = a
    return schedules


def run(source, output, architecture='diverse'):
    source, output = Path(source).resolve(), Path(output).resolve()
    old = json.loads((source/'manifest.json').read_text())
    if old['sensitivity'] != .25: raise ValueError('Expected quarter-sensitivity baseline')
    if any(digest(p)!=h for p,h in old['source_hashes'].items()): raise ValueError('Baseline runtime changed')
    cfg, groups = evidence_config(old['seed'], architecture)
    hashes = fingerprint()
    for fn in (run, balanced_config, record_trial, protocol, fresh, receptor_schedule, cue_cases):
        p = Path(inspect.getfile(fn)).resolve(); hashes[str(p)] = digest(p)
    schedules = test_schedules(old['masks'], old['seed'])
    output.mkdir(parents=True, exist_ok=False)
    path = output/'config.json'; path.write_text(encode(cfg)+'\n')
    manifest = dict(source=str(source), seed=old['seed'], mapping=old['mapping'],
        architecture=architecture, groups=groups, masks=old['masks'], trials=old['trials'],
        checkpoints=[32,128], source_hashes=hashes,
        limits='Parallel independently learned evidence encoding. No upper decision circuit, event segmentation, new neuron model or embodiment. Threshold diversity is compared to equal-size homogeneous cells. Audio projection is stronger to preserve acquisition access at lambda=4.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    np.savez_compressed(output/'schedules.npz', **schedules)
    net, *_ = fresh(path, old['seed'], PortModulationNeuron)
    initial = json.loads(dynamic_snapshot(net)); training=[]; probes=[]; starts={}; started=time.perf_counter()
    for i, trial in enumerate(old['trials']):
        data = record(net, groups, old['masks'], trial)
        with np.load(source/f'train-{i:03d}.npz') as z:
            for key in z.files:
                value = data[key][:,:176] if key=='states' else data[key]
                if not np.array_equal(value,z[key]): raise AssertionError(f'Original brain changed in acquisition: {i}/{key}')
        name=f'train-{i:03d}.npz'; np.savez_compressed(output/name,**data)
        training.append(dict(file=name,sha256=digest(output/name)))
        cp=i+1
        if cp not in (32,128): continue
        parent=dynamic_snapshot(net)
        name=f'{cp}-start.json.gz'
        with gzip.open(output/name,'wt') as f:f.write(parent+'\n')
        starts[str(cp)]=dict(file=name,sha256=digest(output/name))
        for key, schedule in schedules.items():
            branch=deepcopy(net)
            data=record(ScheduledReceptors(branch,schedule),groups,old['masks'],dict(cue=None,sound=None,ticks=len(schedule)))
            if key.endswith('/synchronous'):
                with np.load(source/f'{cp}-trained-{key.split("/")[0]}.npz') as z:
                    if any(not np.array_equal(data[k][:,:176] if k=='states' else data[k],z[k]) for k in z.files):
                        raise AssertionError('Original recall changed')
            name=f'{cp}-{key.replace("/","--")}.npz';np.savez_compressed(output/name,**data)
            probes.append(dict(checkpoint=cp,case=key,state='trained',file=name,sha256=digest(output/name)))
        # Intervention is confined to diagnostic clones. All bank visual weights
        # return to birth values, while old learned recall remains available.
        for cue in (0,1):
            key=f'clean-cue{cue}-sample0/synchronous';branch=deepcopy(net)
            for nid in groups['evidence']:
                for sid in range(32):
                    branch.network.neurons[nid].postsynaptic_points[sid].u_i.info=initial['neurons'][str(nid)]['synapses'][str(sid)][0]
            schedule=schedules[key]
            data=record(ScheduledReceptors(branch,schedule),groups,old['masks'],dict(cue=None,sound=None,ticks=len(schedule)))
            name=f'{cp}-reset-evidence-cue{cue}.npz';np.savez_compressed(output/name,**data)
            probes.append(dict(checkpoint=cp,case=key,state='reset_evidence',file=name,sha256=digest(output/name)))
        if dynamic_snapshot(net)!=parent: raise AssertionError('Probe mutated acquisition')
        print(encode(dict(checkpoint=cp,seconds=time.perf_counter()-started)),flush=True)
    if any(digest(p)!=h for p,h in hashes.items()): raise ValueError('Runtime changed')
    result=dict(training=training,probes=probes,starts=starts,baseline_exact=True,
        config_sha256=digest(path),schedules_sha256=digest(output/'schedules.npz'),
        ticks=12288+sum(len(schedules[p['case']]) for p in probes),seconds=time.perf_counter()-started)
    (output/'summary.json').write_text(encode(result)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--architecture',choices=('diverse','homogeneous'),default='diverse')
    a=p.parse_args();run(a.source,a.output,a.architecture)
