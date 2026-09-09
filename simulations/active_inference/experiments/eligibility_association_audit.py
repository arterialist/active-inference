"""Independent data-only audit of the synthetic association preparation.

Checks every selected eligibility update against explicit exponentially weighted
pre/post histories, receptor arrival masks against upstream spikes, absence of
audio during probes, and delayed neural-consumer firing. Reports all seeds and
controls separately. No learned decoder and no consciousness inference.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def verify_trace(raw, initial_weights, initial_pre, initial_post, previous_vision, cfg, groups, enabled):
    cells, weights, arrivals, eta = (raw[k] for k in ('states', 'weights', 'arrivals', 'eta'))
    length = len(cells)
    if (cells.shape != (length, 144, 8) or weights.shape != (length, 32, 32) or
            arrivals.shape != weights.shape or eta.shape != (length, 32) or
            any(not np.isfinite(raw[k]).all() for k in raw.files)):
        raise ValueError('Bad trace shapes or values')
    ns = {n['id']: n for n in cfg['neurons']}
    core_ids = np.array(groups['auditory'])-1
    v = cells[:, np.array(groups['vision'])-1, 1] > 0
    expected_arrivals = np.concatenate((previous_vision[None], v[:-1]))
    if not np.array_equal(arrivals > 0, np.broadcast_to(expected_arrivals[:, None, :], arrivals.shape)):
        raise ValueError('Local arrival masks do not match presynaptic spikes')
    core, consumer = cells[:, core_ids, 1] > 0, cells[:, np.array(groups['consumer'])-1, 1] > 0
    if not np.array_equal(consumer[2:], core[:-2]):
        raise ValueError('Consumer does not match delayed neural input')
    pre, post, q = initial_pre.copy(), initial_post.copy(), initial_weights.copy()
    residual = 0.
    if enabled:
        ps = [ns[n] for n in groups['auditory']]
        tau_pre = np.array([n['metadata']['eligibility_tau_pre'] for n in ps])[:, None]
        tau_post = np.array([n['metadata']['eligibility_tau_post'] for n in ps])
        cap = np.array([n['metadata']['eligibility_cap'] for n in ps])[:, None]
        alpha = np.array([n['metadata'].get('eligibility_alpha', 1.) for n in ps])[:, None]
        basal = np.array([n['params']['eta_post'] for n in ps])
        if not np.allclose(eta, basal*cells[:, core_ids, 7], rtol=1e-14, atol=0):
            raise ValueError('Effective rate differs from local receptor')
        for t in range(length):
            pre *= np.exp(-1/tau_pre)
            post *= np.exp(-1/tau_post)
            a = (arrivals[t] > 0).astype(float)
            plus, minus = core[t, :, None]*pre, alpha*a*post[:, None]
            total = plus+minus
            # Independently evaluate the affine ODE solution.
            decay = np.exp(-eta[t, :, None]*total)
            target = np.divide(cap*plus, total, out=np.zeros_like(q), where=total > 0)
            q = q*decay+target*(1-decay)
            residual = max(residual, float(abs(q-weights[t]).max()))
            pre += a
            post += core[t]
            if not np.allclose(pre, raw['pre'][t], atol=1e-14, rtol=1e-14) or not np.allclose(post, raw['post'][t], atol=1e-14, rtol=1e-14):
                raise ValueError('Eligibility trace recurrence differs')
        if residual > 2e-12:
            raise ValueError('Weights do not follow the local eligibility equation')
    return weights[-1], raw['pre'][-1], raw['post'][-1], v[-1], residual


def audit(recording, output):
    root, output = Path(recording).resolve(), Path(output).resolve()
    m = json.loads((root/'manifest.json').read_text()); cfg = json.loads((root/'config.json').read_text())
    s = json.loads((root/'summary.json').read_text()); groups = m['groups']
    enabled = m['mode'] == 'eligibility'
    if m['mode'] not in ('eligibility', 'native') or len(cfg['neurons']) != 144:
        raise ValueError('Unknown preparation')
    if any(n['params']['eta_post'] <= 0 or n['params']['eta_retro'] <= 0 for n in cfg['neurons']):
        raise ValueError('Frozen plasticity')
    if set(e['target_neuron'] for e in cfg['external_inputs']) != set(groups['vision']+groups['audio']):
        raise ValueError('Unexpected external input route')
    points = {(p['neuron_id'], p['synapse_id']): p for p in cfg['synaptic_points'] if p['type'] == 'postsynaptic'}
    weights = np.array([[points[n, i]['u_i']['info'] for i in range(32)] for n in groups['auditory']])
    initial = weights.copy(); pre = np.zeros((32, 32 if enabled else 0)); post = np.zeros(32); previous = np.zeros(32, dtype=bool)
    residual = 0.
    for i, trial in enumerate(m['trials']):
        with np.load(root/f'train-{i:03d}.npz') as raw:
            if len(raw['states']) != trial['ticks']:
                raise ValueError('Missing training ticks')
            weights, pre, post, previous, error = verify_trace(raw, weights, pre, post, previous, cfg, groups, enabled)
            residual = max(residual, error)
    rows = []
    for state in ('initial', 'trained'):
        for cue in (0, 1):
            with np.load(root/f'probe-{state}-{cue}.npz') as raw:
                if len(raw['states']) != 64:
                    raise ValueError('Missing probe ticks')
                _, _, _, _, error = verify_trace(raw, initial if state == 'initial' else weights,
                    np.zeros_like(pre) if state == 'initial' else pre,
                    np.zeros_like(post) if state == 'initial' else post,
                    np.zeros_like(previous) if state == 'initial' else previous, cfg, groups, enabled)
                residual = max(residual, error)
                cells = raw['states']
                if np.any(cells[:, np.array(groups['audio'])-1, 1]):
                    raise ValueError('Auditory receptors fired in a silent probe')
                consumers = cells[:, np.array(groups['consumer'])-1, 1] > 0
                counts = [int(consumers[:, [n-33 for n in m['masks']['audio'][a]]].sum()) for a in (0, 1)]
                events = [np.flatnonzero(consumers[:, [n-33 for n in m['masks']['audio'][a]]].any(axis=1)).tolist() for a in (0, 1)]
                rows.append({'state': state, 'cue': cue, 'consumer_counts': counts, 'consumer_event_ticks': events})
    if [r['consumer_counts'] for r in rows] != [r['consumer_sound_coordinates'] for r in s['probes']]:
        raise ValueError('Summary misrepresents neural consumer events')
    output.mkdir(parents=True, exist_ok=False)
    result = {'recording': str(root), 'seed': m['seed'], 'mapping': m['mapping'], 'mode': m['mode'],
              'structurally_valid': True, 'max_update_residual': residual, 'probes': rows,
              'limits': 'Local eligibility equation independently checked only for enabled mode. Native weights are a behavioral control, not independently reconstructed here. Probe sensory silence means no receptor spikes; experimental inputs are defined in the saved protocol. Not real-media, category, hierarchical or embodied acceptance.'}
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); audit(a.recording, a.output)
