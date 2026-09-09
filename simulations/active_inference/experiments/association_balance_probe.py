"""Sensory-driven inhibition before associative spike saturation.

This 176-cell preparation adds a label-blind threshold-distributed inhibitory
population to the 144-cell learned association. It is a circuit hypothesis,
not a reproduction of an organism. Motif motivation: Assisi et al. 2007,
doi:10.1038/nn1947. Their temporal-window model is not implemented here.
All existing local learning remains active, including on inhibitory ports.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import inspect
import json
from pathlib import Path
import time

import numpy as np

from .association_cue_probe import cue_cases
from .association_route_probe import digest
from .composition_probe import encode, fingerprint, k, network_module
from .eligibility_association_probe import config, protocol, run_trial, dynamic_snapshot
from .multimodal_pairing_probe import fresh
from neuron.extensions.experimental.eligibility_trace import EligibilityTraceNeuron

STATES = ('initial', 'trained', 'reset_selected', 'remove_inhibition', 'restore_threshold')


def balanced_config(seed):
    cfg, groups = config(seed)
    groups['sensory_inhibition'] = list(range(145, 177))
    cfg['metadata'].update(preparation='eligibility-sensory-balance-v0',
        prediction='Lower threshold retains incomplete cues; input-dependent inhibition suppresses weaker conflicting evidence.')
    for n in cfg['neurons']:
        if n['id'] in groups['auditory']:
            n['params'].update(r_base=.25, b_base=.50, num_inputs=67)
    for j, nid in enumerate(groups['sensory_inhibition']):
        n = k.neuron(nid, r=j+.5, b=j+.75, c=3, lam=1., eta_post=1e-6,
            eta_retro=1e-7, w_r=[0., 0.], w_b=[0., 0.], w_tref=[0., 0.], delta_decay=.99,
            meta={'role':'sensory_inhibition', 'bounded_plasticity':True,
                  'eligibility_ports':[], 'plasticity_rate_boost':0.})
        n['params']['num_inputs'] = 32
        cfg['neurons'].append(n)
        cfg['synaptic_points'].append(k.term(nid, mod=[0., 0.]))
        for sid, src in enumerate(groups['vision']):
            cfg['synaptic_points'].append(k.syn(nid, sid, 1., 0, adapt=[0., 0.]))
            cfg['connections'].append(k.conn(src, nid, sid))
        for target in groups['auditory']:
            cfg['synaptic_points'].append(k.syn(target, 35+j, -.08, 0, adapt=[0., 0.]))
            cfg['connections'].append(k.conn(nid, target, 35+j))
    return cfg, groups


def record_trial(net, groups, masks, trial):
    """Read every core input before the existing transparent trace wrapper."""
    original = EligibilityTraceNeuron.tick
    arrivals, weights, plast, ids = [], [], [], []
    def observed(n, ext, tick, dt=1.):
        if n.id in groups['auditory'] and net.network.neurons[n.id] is n:
            ids.append(n.id)
            arrivals.append(n.input_buffer[:, 0].copy())
            weights.append([n.postsynaptic_points[i].u_i.info for i in range(67)])
            plast.append([n.postsynaptic_points[i].u_i.plast for i in range(67)])
        return original(n, ext, tick, dt)
    EligibilityTraceNeuron.tick = observed
    try:
        data = run_trial(net, groups, masks, trial)
    finally:
        EligibilityTraceNeuron.tick = original
    if ids != groups['auditory']*trial['ticks']:
        raise AssertionError('Unexpected sampling order')
    for key, values in (('all_arrivals', arrivals), ('all_input_weights', weights), ('all_input_plast', plast)):
        data[key] = np.asarray(values).reshape(trial['ticks'], 32, 67)
    return data


def run(output, seed=11, mapping='paired'):
    if mapping not in ('paired', 'swapped'):
        raise ValueError('Need opposite paired assignments')
    if network_module.MIN_CONNECTION_SIGNAL_TRAVEL_TICKS != 1 or network_module.MAX_CONNECTION_SIGNAL_TRAVEL_TICKS != 1:
        raise ValueError('Requires one-tick cleft delays')
    output = Path(output).resolve()
    cfg, groups = balanced_config(seed)
    masks, trials = protocol(groups, seed, mapping, 16)
    cases = cue_cases(masks, seed)
    output.mkdir(parents=True, exist_ok=False)
    path = output/'config.json'; path.write_text(encode(cfg)+'\n')
    hashes = fingerprint()
    for fn in (run, config, protocol, run_trial, cue_cases, fresh):
        p = Path(inspect.getfile(fn)).resolve(); hashes[str(p)] = digest(p)
    manifest = dict(seed=seed, mapping=mapping, groups=groups, masks=masks,
        trials=trials, cases=cases, states=STATES, source_hashes=hashes,
        design='32 label-blind inhibitory cells with thresholds .5..31.5 and unit sensory inputs; each projects -.08 to every auditory cell. Auditory r=.25, b=.50. No stimulus-specific inhibitory wiring.',
        calibration='Fixed from prior learned q~.177, unpaired q~.02. For four pure receptors expected net current .388; 12+4 minority -.33, majority .92; compared to r=.25. Not blind parameter selection.',
        limits='Synthetic synchronous cue test, not real media, recurrent memory, learned semantic consumer, hierarchy or embodiment. Removal branches are diagnostic interventions, not autonomous control.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    net, _, _, _ = fresh(path, seed, EligibilityTraceNeuron)
    initial = deepcopy(net); started = time.perf_counter(); training = []
    for index, trial in enumerate(trials):
        data = record_trial(net, groups, masks, trial)
        name = f'train-{index:03d}.npz'; np.savez_compressed(output/name, **data)
        training.append(dict(file=name, sha256=digest(output/name)))
    parent = dynamic_snapshot(net)
    with gzip.open(output/'trained-state.json.gz', 'wt') as f: f.write(parent+'\n')
    print(encode(dict(stage='trained', seconds=time.perf_counter()-started)), flush=True)
    points = {(p['neuron_id'],p['synapse_id']):p for p in cfg['synaptic_points'] if p['type']=='postsynaptic'}
    rows, starts = [], {}
    for state in STATES:
        base = deepcopy(initial if state=='initial' else net)
        expected = json.loads(dynamic_snapshot(base))
        for nid in groups['auditory']:
            n = base.network.neurons[nid]
            if state=='reset_selected':
                for sid in range(32):
                    q=points[nid,sid]['u_i']['info']; n.postsynaptic_points[sid].u_i.info=q
                    expected['neurons'][str(nid)]['synapses'][str(sid)][0]=q
            elif state=='remove_inhibition':
                for sid in range(35,67):
                    n.postsynaptic_points[sid].u_i.info=0.
                    expected['neurons'][str(nid)]['synapses'][str(sid)][0]=0.
            elif state=='restore_threshold':
                n.params.r_base=.65; n.params.b_base=.90
        if json.loads(dynamic_snapshot(base))!=expected:
            raise AssertionError('Undeclared state change')
        # Threshold parameter changes take effect on the next ordinary tick.
        params={str(n.id):dict(r=n.params.r_base,b=n.params.b_base) for n in base.network.neurons.values()}
        name=state+'-start.json.gz'
        with gzip.open(output/name,'wt') as f:f.write(encode(dict(state=expected,threshold_parameters=params))+'\n')
        starts[state]=dict(file=name,sha256=digest(output/name))
        for case in cases:
            branch=deepcopy(base); physical=deepcopy(masks)
            physical['vision'][case['cue']]=case['selected_receptors']
            trial=dict(cue=case['cue'],sound=None,ticks=64)
            data=record_trial(branch,groups,physical,trial)
            # A full-trace unobserved replay proves the added input tap is passive.
            if case['kind']=='clean':
                control=deepcopy(base); plain=run_trial(control,groups,physical,trial)
                if any(not np.array_equal(data[k],plain[k]) for k in plain) or dynamic_snapshot(control)!=dynamic_snapshot(branch):
                    raise AssertionError('Input observation altered dynamics')
            name=f'{state}-{case["name"]}.npz'; np.savez_compressed(output/name,**data)
            rows.append(dict(state=state,case=case['name'],file=name,sha256=digest(output/name)))
        if dynamic_snapshot(net)!=parent:raise AssertionError('Parent changed')
        print(encode(dict(stage=state, seconds=time.perf_counter()-started)),flush=True)
    if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('Runtime changed')
    result=dict(training=training,starts=starts,probes=rows,observer_controls_exact=True,
        ticks=3072+64*(len(rows)+10),seconds=time.perf_counter()-started)
    (output/'summary.json').write_text(encode(result)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=11)
    p.add_argument('--mapping',choices=('paired','swapped'),default='paired')
    a=p.parse_args();run(a.output,a.seed,a.mapping)
