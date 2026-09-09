"""Causal necessity/sufficiency branches for recorded eligibility association.

Replay every training field exactly. Reset only learned cross-sensory weights
in the complete trained state, or transplant only those weights into a fresh
network. Both probes stay plastic. Neither manipulation is part of the brain.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .eligibility_association_probe import run_trial, dynamic_snapshot
from .multimodal_pairing_probe import fresh
from .composition_probe import encode
from neuron.extensions.experimental.eligibility_trace import EligibilityTraceNeuron


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    m = json.loads((source/'manifest.json').read_text())
    if m['mode'] != 'eligibility': raise ValueError('Requires eligibility record')
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p,h in m['source_hashes'].items()):
        raise ValueError('Recorded runtime changed')
    net, _, _, _ = fresh(source/'config.json', m['seed'], EligibilityTraceNeuron)
    initial = deepcopy(net)
    for i, tr in enumerate(m['trials']):
        data = run_trial(net, m['groups'], m['masks'], tr)
        with np.load(source/f'train-{i:03d}.npz') as raw:
            if set(raw.files) != set(data) or any(not np.array_equal(raw[k], data[k]) for k in data):
                raise AssertionError(f'Training replay differs at {i}')
    parent = dynamic_snapshot(net)
    with gzip.open(source/'trained-state.json.gz', 'rt') as f:
        if parent != encode(json.load(f)): raise AssertionError('Final dynamical state differs')
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for condition in ('unchanged', 'reset_selected', 'selected_only'):
        for cue in (0, 1):
            branch = deepcopy(initial if condition == 'selected_only' else net)
            donor = initial if condition == 'reset_selected' else net
            if condition != 'unchanged':
                for nid in m['groups']['auditory']:
                    for sid in branch.network.neurons[nid].eligibility_ports:
                        branch.network.neurons[nid].postsynaptic_points[sid].u_i.info = donor.network.neurons[nid].postsynaptic_points[sid].u_i.info
            # Verify the complete declared intervention, not only an endpoint.
            before = json.loads(dynamic_snapshot(branch))
            expected = json.loads(dynamic_snapshot(initial if condition == 'selected_only' else net))
            if condition != 'unchanged':
                donor_state = json.loads(dynamic_snapshot(donor))
                for nid in m['groups']['auditory']:
                    for sid in range(32):
                        expected['neurons'][str(nid)]['synapses'][str(sid)][0] = donor_state['neurons'][str(nid)]['synapses'][str(sid)][0]
            if before != expected: raise AssertionError('Undeclared state change')
            data = run_trial(branch, m['groups'], m['masks'], {'cue': cue, 'sound': None, 'ticks': 64})
            if condition == 'unchanged':
                with np.load(source/f'probe-trained-{cue}.npz') as raw:
                    if any(not np.array_equal(raw[k], data[k]) for k in data): raise AssertionError('Unchanged probe differs')
            name = f'{condition}-{cue}.npz'
            np.savez_compressed(output/name, **data)
            with gzip.open(output/f'{condition}-{cue}-start.json.gz', 'wt') as f: f.write(encode(before)+'\n')
            o = data['states'][:, np.array(m['groups']['consumer'])-1, 1] > 0
            counts = [int(o[:, [n-33 for n in m['masks']['audio'][a]]].sum()) for a in (0, 1)]
            rows.append({'condition': condition, 'cue': cue, 'counts': counts,
                         'event_ticks': [np.flatnonzero(o[:, [n-33 for n in m['masks']['audio'][a]]].any(axis=1)).tolist() for a in (0, 1)],
                         'file': name, 'sha256': hashlib.sha256((output/name).read_bytes()).hexdigest()})
            if dynamic_snapshot(net) != parent: raise AssertionError('Parent mutated')
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p,h in m['source_hashes'].items()):
        raise ValueError('Runtime changed during replay')
    result = {'source_recording': str(source), 'seed': m['seed'], 'mapping': m['mapping'],
              'full_training_replay_exact': True, 'full_declared_start_states_verified': True,
              'probes': rows, 'ticks': sum(t['ticks'] for t in m['trials'])+384,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limits': 'Selected incoming weights only. Necessity/sufficiency is conditional on this constructed topology and receptor-pattern task, not all memory or embodied cognition.'}
    (output/'summary.json').write_text(encode(result)+'\n')
    print(encode(result), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.source, a.output)
