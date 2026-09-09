"""Trace where acquired selected-pathway weight changes came from.

Data-only analysis of the existing linear eligibility rule. For its recorded
activity, q_next = f*q + (1-f)*q_star. Partition q-q_birth into signed terms
tagged by the physical trial during which each update happened. Every term
decays by f; the current trial receives (1-f)*(q_star-q_birth). Their sum must
reconstruct every recorded weight at every tick.

These terms are conditional arithmetic, not causal credit or memory fractions.
Changing one trial changes subsequent activity, rates and return signals. The
withdrawal tag includes residual eligibility and recurrent activity. Labels are
used only here, never delivered to a neuron. Source traces are referenced, not
duplicated. The output retains full-tick diagnostics and final per-port terms.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .association_balance_audit import checked
from .association_route_probe import digest
from .eligibility_media_audit import verify_ledger
from .graded_media_audit import verify_rates
from .media_order_audit import load_state
from .media_order_control import ledger_start


def attributed_step(terms, birth, plus, minus, eta, tag):
    terms, birth, plus, minus, eta = [np.asarray(x, dtype=float)
                                    for x in (terms, birth, plus, minus, eta)]
    if (terms.ndim != 2 or birth.ndim != 1 or terms.shape[1:] != birth.shape
            or any(x.shape != birth.shape for x in (plus, minus, eta))
            or any(not np.isfinite(x).all() for x in (terms, birth, plus, minus, eta))
            or type(tag) is not int or not 0 <= tag < len(terms)
            or np.any(birth < 0) or np.any(birth > 1)
            or np.any(plus < 0) or np.any(minus < 0) or np.any(eta <= 0)):
        raise ValueError('Invalid conditional attribution inputs')
    total = plus + minus
    amount = -np.expm1(-eta * total)
    target = np.divide(plus, total, out=np.zeros_like(birth), where=total > 0)
    after = terms * (1 - amount)
    after[tag] += amount * (target - birth)
    return after


def growth_modes(change, targets):
    """Orthogonal constant, between-target and within-target components.

Equal fan-in is required so target means have equal weight. These are weight
coordinates, not separable dynamical functions of the recurrent network.
"""
    x = np.asarray(change, dtype=float)
    targets = np.asarray(targets)
    if x.ndim != 1 or targets.shape != x.shape or not len(x) or not np.isfinite(x).all():
        raise ValueError('Invalid growth vector')
    _, inverse, counts = np.unique(targets, return_inverse=True, return_counts=True)
    if not np.all(counts == counts[0]):
        raise ValueError('Unequal target fan-in')
    means = np.bincount(inverse, weights=x) / counts
    common = np.full_like(x, x.mean())
    between = means[inverse] - common
    within = x - means[inverse]
    return common, between, within


FIELDS = ('tick', 'trial', 'tag', 'potentiation_exposure', 'depression_exposure',
          'signed_update', 'absolute_update', 'growth_energy', 'global_energy',
          'between_target_energy', 'within_target_energy', 'max_weight_residual')


def audit(recording, output):
    root, output = Path(recording).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    m = json.loads((root / 'manifest.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    cfg = json.loads((root / 'config.json').read_text())
    source_hashes = {str(root / name): digest(root / name) for name in
                     ('manifest.json', 'summary.json', 'config.json', 'initial-state.json.gz')}
    state = load_state(root / 'initial-state.json.gz')
    ports = m['selected_ports']
    ns = {n['id']: n for n in cfg['neurons']}
    target_ids = np.array([n for n, _, _ in ports])
    cell_index = {nid: i for i, nid in enumerate(ns)}
    targets = np.array([cell_index[n] for n in target_ids])
    for n in np.unique(target_ids):
        meta = ns[n]['metadata']
        if any(meta[k] != v for k, v in dict(eligibility_tau_pre=4., eligibility_tau_post=4.,
                                             eligibility_alpha=1., eligibility_cap=1.).items()):
            raise ValueError('Audit is specific to the recorded unit-cap linear rule')
    current = ledger_start(state, cfg, ports)
    birth = current[0].copy()
    terms = np.zeros((3, len(ports)))
    traces, trials = [], []
    max_residual = 0.
    cursor = state['tick']
    if [t['trial'] for t in summary['training']] != m['trials']:
        raise ValueError('Incomplete or changed acquisition protocol')
    for i, item in enumerate(summary['training']):
        trial = item['trial']
        if trial['start'] != cursor:
            raise ValueError('Nonconsecutive acquisition')
        tag = 2 if trial['visual_clip'] is None else trial['visual_clip']
        if tag not in (0, 1, 2):
            raise ValueError('Unknown physical trial')
        path = checked(root, item)
        with np.load(path) as z:
            data = {k: z[k] for k in ('cells', 'weights', 'pre', 'post', 'arrivals', 'eta')}
        if len(data['cells']) != trial['stop'] - cursor:
            raise ValueError('Wrong trial length')
        result = verify_ledger(data, cfg, ports, *current[:4])
        verify_rates(data['cells'], current[4], cfg, ports)
        q, pre, post = (v.copy() for v in current[:3])
        rows = np.empty((len(data['cells']), len(FIELDS)))
        for t in range(len(rows)):
            plus = (data['cells'][t, targets, 1] > 0) * pre * np.exp(-.25)
            minus = (data['arrivals'][t] > 0) * post * np.exp(-.25)
            terms = attributed_step(terms, birth, plus, minus, data['eta'][t], tag)
            expected = birth + terms.sum(axis=0)
            residual = float(np.max(abs(expected - data['weights'][t])))
            max_residual = max(max_residual, residual, result[4])
            if residual > 2e-12:
                raise ValueError('Conditional terms do not reconstruct recorded weights')
            next_q = data['weights'][t]
            delta, growth = next_q - q, next_q - birth
            modes = growth_modes(growth, target_ids)
            rows[t] = (cursor + t, i, tag, np.sum(data['eta'][t] * plus),
                       np.sum(data['eta'][t] * minus), delta.sum(), np.abs(delta).sum(),
                       growth @ growth, *(v @ v for v in modes), residual)
            q, pre, post = next_q, data['pre'][t], data['post'][t]
        traces.append(rows)
        trials.append(dict(trial=trial, source_sha256=item['sha256'],
                           plus_exposure=float(rows[:, 3].sum()),
                           minus_exposure=float(rows[:, 4].sum()),
                           signed_update=float(rows[:, 5].sum())))
        cursor = trial['stop']
        current = (*result[:4], data['cells'][-1, :, 3].copy())
    growth = current[0] - birth
    modes = growth_modes(growth, target_ids)
    if any(digest(p) != h for p, h in source_hashes.items()):
        raise ValueError('Source metadata changed during analysis')
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'trajectory.npz', fields=np.array(FIELDS),
                        ticks=np.concatenate(traces), ports=np.array(ports),
                        birth=birth, learned=current[0], conditional_terms=terms,
                        growth_modes=np.array(modes))
    report = dict(recording=str(root), mapping=m['mapping'], order=m['order'], seed=m['seed'],
                  source_hashes=source_hashes, audit_source_sha256=digest(__file__),
                  trajectory_sha256=digest(output / 'trajectory.npz'),
                  tags=['visual_clip_0_present', 'visual_clip_1_present', 'withdrawal'],
                  max_weight_residual=max_residual, ticks=cursor-state['tick'],
                  final_mode_energies=[float(v @ v) for v in modes],
                  final_conditional_term_norms=np.linalg.norm(terms, axis=1).tolist(),
                  trials=trials,
                  limits='One recorded graph history. Conditional equation decomposition, not causal credit, '
                         'memory fractions or a counterfactual neural run. Target-level changes can carry '
                         'content. Weight coordinates are not independent closed-loop dynamical functions.')
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    audit(args.recording, args.output)
