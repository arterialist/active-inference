"""Controlled PAULA association preparation with a downstream neural consumer.

144 neurons: two 32-receptor sheets, a 32-cell auditory population, 32 neural
consumers and 16 activity-driven modulatory neurons. Input masks represent
receptor stimulation, not real media or categories. Visual-to-auditory wiring
contains every receptor equally, with identity-independent heterogeneity.
Auditory-to-consumer wiring preserves receptor coordinates. Assignment enters
only through which physical stimuli occur together, never through weights.

This isolates acquisition and cross-sensory reinstatement before scaling the
new local rule into the existing 1152-cell audiovisual architecture. It is not
a substitute for that architecture, generalization, hierarchy or embodiment.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import time

import numpy as np

from .composition_probe import encode, fingerprint, k, snapshot, network_module
from .multimodal_pairing_probe import fresh
from .population_hierarchy import cellular, weight_values
from neuron.extensions.experimental.eligibility_trace import EligibilityTraceNeuron


def config(seed=11, mode="eligibility"):
    if mode not in ("eligibility", "native"):
        raise ValueError(mode)
    rng = np.random.default_rng(seed)
    groups = {"vision": list(range(1, 33)), "audio": list(range(33, 65)),
              "auditory": list(range(65, 97)), "consumer": list(range(97, 129)), "modulator": list(range(129, 145))}
    neurons, points, connections, external = [], [], [], []
    for role, ids in groups.items():
        for j, nid in enumerate(ids):
            core = role == "auditory"
            n = k.neuron(nid, r=.65, b=.9, c=3, lam=1., eta_post=1e-5 if core else 1e-6,
                         eta_retro=1e-7, w_r=[0., 0.], w_b=[0., 0.], w_tref=[0., 0.], delta_decay=.99,
                         meta={"role": role, "bounded_plasticity": True,
                               "eligibility_ports": list(range(32)) if core and mode == "eligibility" else [],
                               "eligibility_cap": 1., "eligibility_tau_pre": 4., "eligibility_tau_post": 4.,
                               "plasticity_rate_boost": 499. if core else 0.,
                               "plasticity_rate_half_saturation": .01})
            n['params']['gamma'] = [.99, .99]
            neurons.append(n)
            term = k.term(nid, mod=[.25, 0.] if role == "modulator" else [0., 0.])
            if role == "modulator": term['u_o']['info'] = 0.
            points.append(term)
            if role in ("vision", "audio"):
                points += [k.syn(nid, 0, 2., adapt=[0., 0.]), k.syn(nid, 1, 0., adapt=[0., 0.])]
                external.append(k.ext(nid, 0))
            elif core:
                for sid, src in enumerate(groups['vision']):
                    points.append(k.syn(nid, sid, .02*float(rng.uniform(.85, 1.15)), adapt=[0., 0.]))
                    connections.append(k.conn(src, nid, sid))
                points.append(k.syn(nid, 32, 1.5, adapt=[0., 0.]))
                connections.append(k.conn(groups['audio'][j], nid, 32))
                for sid, src in enumerate((groups['modulator'][j % 16], groups['modulator'][(j+5) % 16]), 33):
                    points.append(k.syn(nid, sid, 0., adapt=[1., 0.]))
                    connections.append(k.conn(src, nid, sid))
            elif role == 'consumer':
                points += [k.syn(nid, 0, 1., adapt=[0., 0.]), k.syn(nid, 1, 0., adapt=[0., 0.])]
                connections.append(k.conn(groups['auditory'][j], nid, 0))
            else:
                sources = list(rng.choice(groups['vision'], 4, replace=False))+list(rng.choice(groups['audio'], 4, replace=False))
                for sid, src in enumerate(sources):
                    points.append(k.syn(nid, sid, .8, adapt=[0., 0.]))
                    connections.append(k.conn(int(src), nid, sid))
    for n in neurons:
        n['params']['num_inputs'] = sum(p['type']=='postsynaptic' and p['neuron_id']==n['id'] for p in points)
    return {"metadata": {"preparation": "eligibility-association-v0", "seed": seed, "mode": mode},
            "global_params": {"num_inputs": 1, "num_neuromodulators": 2}, "simulation_params": {"max_history": 1},
            "neurons": neurons, "synaptic_points": points, "connections": connections, "external_inputs": external}, groups


def protocol(groups, seed, mapping, repeats):
    if mapping not in ('paired', 'swapped', 'separated') or repeats < 1:
        raise ValueError('Invalid experience protocol')
    rng = np.random.default_rng(seed+811)
    masks = {role: [list(map(int, x)) for x in np.array_split(rng.permutation(groups[role]), 2)] for role in ('vision', 'audio')}
    trials = []
    for _ in range(repeats):
        for cue in rng.permutation(2):
            cue = int(cue)
            trials.append({'cue': cue, 'sound': 1-cue if mapping == 'swapped' else cue,
                           'audio_offset': 32 if mapping == 'separated' else 0, 'ticks': 96})
    return masks, trials


def dynamic_snapshot(net):
    value = snapshot(net)
    value['eligibility'] = {str(n.id): {'pre': n.eligibility_pre, 'post': n.eligibility_post,
        'last_tick': n.eligibility_last_tick, 'updates': n.eligibility_updates} for n in net.network.neurons.values()}
    return encode(value)


def run_trial(net, groups, masks, trial):
    neurons = list(net.network.neurons.values())
    cores = [net.network.neurons[n] for n in groups['auditory']]
    states, weights, pre, post, arrivals, eta = [], [], [], [], [], []
    # The input ledger is a read-only tap on the selected receptor arrivals.
    # Monkeypatch this experimental class only, restored even on failure.
    original = EligibilityTraceNeuron.tick
    sampled = {}
    def observed(n, ext, t, dt=1.):
        if n.id in groups['auditory'] and net.network.neurons[n.id] is n:
            sampled[n.id] = (n.input_buffer[:32, 0].copy(), n.params.eta_post*n.rate_multiplier())
        return original(n, ext, t, dt)
    EligibilityTraceNeuron.tick = observed
    try:
        for rel in range(trial['ticks']):
            for role, key, offset in (('vision', 'cue', 0), ('audio', 'sound', trial.get('audio_offset', 0))):
                which = trial.get(key)
                phase = rel-offset
                if which is not None and 0 <= phase < 32 and phase % 8 == 0:
                    for nid in masks[role][which]: net.set_external_input(nid, 0, 1.)
            net.run_tick()
            states.append(cellular(neurons))
            weights.append([[n.postsynaptic_points[i].u_i.info for i in range(32)] for n in cores])
            pre.append([n.eligibility_pre.copy() for n in cores])
            post.append([n.eligibility_post for n in cores])
            arrivals.append([sampled[n.id][0] for n in cores])
            eta.append([sampled[n.id][1] for n in cores])
    finally:
        EligibilityTraceNeuron.tick = original
    result = {k: np.array(v) for k, v in locals().items() if k in ('states', 'weights', 'pre', 'post', 'arrivals', 'eta')}
    if any(not np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('Nonfinite neural record')
    return result


def run(output, seed=11, mapping='paired', mode='eligibility', repeats=16):
    if network_module.MIN_CONNECTION_SIGNAL_TRAVEL_TICKS != 1 or network_module.MAX_CONNECTION_SIGNAL_TRAVEL_TICKS != 1:
        raise ValueError('Branch comparison requires deterministic cleft delays')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cfg, groups = config(seed, mode)
    masks, trials = protocol(groups, seed, mapping, repeats)
    path = output/'config.json'
    path.write_text(encode(cfg)+'\n')
    hashes = fingerprint()
    import hashlib
    hashes[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {'seed': seed, 'mapping': mapping, 'mode': mode, 'repeats': repeats, 'groups': groups,
                'masks': masks, 'trials': trials, 'source_hashes': hashes,
                'limits': 'Synthetic receptor patterns, not real image/audio recognition or embodiment. Consumer wiring copies auditory coordinates without learning. Only the cross-sensory association can be learned. Native output-terminal learning remains active.'}
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    net, _, neurons, syns = fresh(path, seed, EligibilityTraceNeuron)
    initial = deepcopy(net)
    start = time.perf_counter()
    for i, trial in enumerate(trials):
        data = run_trial(net, groups, masks, trial)
        np.savez_compressed(output/f'train-{i:03d}.npz', **data)
        if i % 8 == 7: print(encode({'training': i+1, 'seconds': time.perf_counter()-start}), flush=True)
    parent = dynamic_snapshot(net)
    with gzip.open(output/'trained-state.json.gz', 'wt') as f: f.write(parent+'\n')
    results = []
    for state, source in (('initial', initial), ('trained', net)):
        for cue in (0, 1):
            branch = deepcopy(source)
            if dynamic_snapshot(branch) != dynamic_snapshot(source): raise AssertionError('Branch differs')
            data = run_trial(branch, groups, masks, {'cue': cue, 'sound': None, 'ticks': 64})
            np.savez_compressed(output/f'probe-{state}-{cue}.npz', **data)
            consumer = data['states'][:, np.array(groups['consumer'])-1, 1] > 0
            counts = [int(consumer[:, [n-33 for n in masks['audio'][sound]]].sum()) for sound in (0, 1)]
            results.append({'state': state, 'cue': cue, 'consumer_sound_coordinates': counts})
            if dynamic_snapshot(net) != parent: raise AssertionError('Probe changed parent')
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):
        raise ValueError('Runtime changed')
    result = {'seed': seed, 'mapping': mapping, 'mode': mode, 'probes': results, 'seconds': time.perf_counter()-start,
              'ticks': 96*len(trials)+256, 'claim': 'Recorded diagnostic only. Requires independent audit and replication.'}
    (output/'summary.json').write_text(encode(result)+'\n')
    print(encode(result), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--mapping', choices=('paired', 'swapped', 'separated'), default='paired')
    p.add_argument('--mode', choices=('eligibility', 'native'), default='eligibility')
    p.add_argument('--repeats', type=int, default=16)
    a = p.parse_args()
    run(a.output, a.seed, a.mapping, a.mode, a.repeats)
