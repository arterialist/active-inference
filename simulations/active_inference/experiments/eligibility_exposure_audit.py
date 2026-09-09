"""Conditional learning exposure from actual per-tick eligibility records.

For the recorded pre/post activity and local rates, the selected incoming rule
has q(t) = A(t) q(0) + B(t). A is the remaining initial-weight coefficient, not
a fraction of information retained. This conditional affine decomposition is
not the closed-loop derivative: changing a weight can change future activity,
native return signals and rates. No learning rate is frozen or replaced here.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def affine_step(initial_coefficient, experience_term, plus, minus, eta, cap=1.):
    arrays = [np.asarray(v, dtype=float) for v in (initial_coefficient, experience_term, plus, minus, eta)]
    if any(v.shape != arrays[0].shape or not np.isfinite(v).all() for v in arrays):
        raise ValueError('Incompatible affine arrays')
    a, b, plus, minus, eta = arrays
    if not np.isfinite(cap) or cap <= 0 or np.any(a < 0) or np.any(a > 1) or np.any(b < 0) or np.any(plus < 0) or np.any(minus < 0) or np.any(eta <= 0):
        raise ValueError('Invalid affine state or learning coefficients')
    total = plus+minus
    # expm1 keeps small positive learning changes numerically visible.
    amount = -np.expm1(-eta*total)
    equilibrium = np.divide(cap*plus, total, out=np.zeros_like(total), where=total > 0)
    return a*(1-amount), b*(1-amount)+equilibrium*amount


def audit(recording, output):
    root, output = Path(recording).resolve(), Path(output).resolve()
    m = json.loads((root/'manifest.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    cfg = json.loads((root/'config.json').read_text())
    with gzip.open(root/'initial-state.json.gz', 'rt') as f:
        state = json.load(f)
    ports = m['selected_ports']
    indices = {n['id']: i for i, n in enumerate(cfg['neurons'])}
    targets = np.array([indices[n] for n, _, _ in ports])
    q0 = np.array([state['neurons'][str(n)]['synapses'][str(sid)][0] for n, sid, _ in ports])
    metadata = {n['id']: n['metadata'] for n in cfg['neurons']}
    tau_pre = np.array([metadata[n]['eligibility_tau_pre'] for n, _, _ in ports])
    tau_post = np.array([metadata[n]['eligibility_tau_post'] for n, _, _ in ports])
    alpha = np.array([metadata[n]['eligibility_alpha'] for n, _, _ in ports])
    caps = {metadata[n]['eligibility_cap'] for n, _, _ in ports}
    if len(caps) != 1:
        raise ValueError('Current audit requires a common positive cap')
    cap = caps.pop()
    pre = np.array([state['eligibility'][str(n)]['pre'][metadata[n]['eligibility_ports'].index(sid)] for n, sid, _ in ports])
    post = np.array([state['eligibility'][str(n)]['post'] for n, _, _ in ports])
    a, b = np.ones(len(ports)), np.zeros(len(ports))
    records = []; rows = []; residual = 0.; tick = state['tick']
    output.mkdir(parents=True, exist_ok=False)
    for item in summary['training']:
        path = root/item['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Changed training record')
        with np.load(path) as raw:
            d = {k: raw[k] for k in ('cells', 'arrivals', 'weights', 'eta', 'pre', 'post')}
        trial = item['trial']
        if trial['start'] != tick or trial['stop']-trial['start'] != len(d['cells']):
            raise ValueError('Changed training sequence')
        a_trace, b_trace = np.empty_like(d['weights']), np.empty_like(d['weights'])
        for t in range(len(d['cells'])):
            plus = (d['cells'][t, targets, 1] > 0)*pre*np.exp(-1/tau_pre)
            minus = (d['arrivals'][t] > 0)*post*np.exp(-1/tau_post)*alpha
            a, b = affine_step(a, b, plus, minus, d['eta'][t], cap)
            residual = max(residual, float(np.max(np.abs(a*q0+b-d['weights'][t]))))
            if residual > 2e-12:
                raise ValueError('Conditional decomposition disagrees with recorded weights')
            a_trace[t], b_trace[t] = a, b
            pre, post = d['pre'][t], d['post'][t]
        tick = trial['stop']
        name = item['file']
        np.savez_compressed(output/name, initial_coefficient=a_trace, experience_term=b_trace)
        records.append(dict(file=name, trial=trial, source_sha256=item['sha256']))
        rows.append(dict(stop=tick, visual_clip=trial['visual_clip'], audio_clip=trial['audio_clip'],
                         coefficient_quantiles=np.quantile(a, [0, .1, .5, .9, 1]).tolist()))
    result = dict(recording=str(root), selected_ports=ports, records=records, boundaries=rows,
                  max_reconstruction_residual=residual,
                  final_initial_coefficient_quantiles=np.quantile(a, [0, .1, .5, .9, 1]).tolist(),
                  limits='Conditional on observed spike/arrival/rate histories. Not a closed-loop perturbation derivative, '
                         'fraction of memory, semantic content or proof that more training would repair recall. '
                         'A prior full ledger audit is required to validate the recorded traces themselves.')
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    audit(a.recording, a.output)
