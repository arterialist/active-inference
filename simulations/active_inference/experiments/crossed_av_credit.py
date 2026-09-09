"""Trace acquisition interference on a declared, fixed audiovisual interface.

No fitted readout or brain intervention. Each recorded weight vector is applied
to the same four factual pre-feedback input histories. These conditional outputs
show what each update does to that interface, not what a counterfactual coupled
brain would do. Native return signals can change subsequent sensory releases.
The saved factors reconstruct every training-tick by probe-tick comparison;
window extrema are indexes into those factors, not a behavioral acceptance rule.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import context_organization as base
from .crossed_av_capacity import filtered_context, native_predictor
from .opponent_context_analysis import read_record
from .temporal_verification import verify_learning
from .crossed_av_continuation_analysis import FACTORIAL


def credit_components(data):
    """Exact local accounting, separating pre-episode and new context traces.

    The error trace is factual in all terms. 'Old' names the origin of context
    activity only, not an error's origin or a semantic memory. Newly arriving
    releases may themselves contain earlier neural/queue state. Clipping is kept
    as its own correction; it cannot be assigned uniquely to either input age.
    """
    verify_learning(data)
    decay = math.exp(-1 / 64)
    x = data['context_initial'].copy()
    old = x.copy()
    q = data['weights_initial'].copy()
    rows = {k: [] for k in ('old_context', 'new_context', 'bound_correction')}
    for t, arrivals in enumerate(data['arrivals']):
        old *= decay
        x = decay * x + (1 - decay) * arrivals
        coefficient = data['eta'][t, :, None] * data['errors'][t, :, 0, None]
        prior = coefficient * old
        new = coefficient * (x - old)
        # Match the producer's floating-point expression before accounting for
        # clipping and the tiny difference introduced by splitting the sum.
        proposed = coefficient * x
        next_q = data['weights'][t]
        correction = next_q - q - proposed
        if not np.allclose(prior + new + correction, next_q - q, rtol=0, atol=2e-15):
            raise ValueError('Credit decomposition differs from recorded update')
        rows['old_context'].append(prior)
        rows['new_context'].append(new)
        rows['bound_correction'].append(correction)
        q = next_q
    return {k: np.asarray(v) for k, v in rows.items()}


def differential(values, source_ids):
    ids = [list(a) for a in source_ids]
    if len(ids) != 2 or len(set(ids[0])) != len(ids[0]) or set(ids[0]) != set(ids[1]):
        raise ValueError('Opponent source identities differ or repeat')
    order = [ids[1].index(i) for i in ids[0]]
    return values[..., 0, :] - values[..., 1, order]


def projected_bounds(weights, basis, start=16, stop=64):
    """Exact float64 conditional projection, with full factors retained by caller."""
    if weights.ndim != 2 or basis.ndim != 3 or weights.shape[1] != basis.shape[2]:
        raise ValueError('Expected acquisition-by-port and pair-by-probe-tick-by-port')
    if not 0 <= start < stop <= basis.shape[1]:
        raise ValueError('Invalid probe interval')
    response = np.einsum('tn,pkn->tpk', weights, basis[:, start:stop], optimize=True)
    return np.stack((response.min(axis=2), response.max(axis=2)), axis=2)


def analyze(analysis_root, output):
    analysis_root, output = Path(analysis_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    summary_path = analysis_root / 'summary.json'
    audited = json.loads(summary_path.read_text())
    arrays = {}; episodes = []; references = []; seen = set()
    for source in audited['sources']:
        root = Path(source['root'])
        if (base.digest(root / 'manifest.json') != source['manifest_sha256'] or
                base.digest(root / source['completed_record']) != source['completed_sha256']):
            raise ValueError('Audited source changed')
        m = json.loads((root / 'manifest.json').read_text())
        if (m['seed'], m['reverse']) in seen:
            raise ValueError('Repeated acquisition condition')
        seen.add((m['seed'], m['reverse']))
        for path, digest in {**m['source_hashes'], **m['physical_sources']}.items():
            if base.digest(path) != digest:
                raise ValueError('Executable or media source changed')
        parent = Path(m['parent'])
        for name, digest in m['parent_evidence'].items():
            if base.digest(parent / name) != digest:
                raise ValueError('Parent evidence changed')
        pm = json.loads((parent / 'manifest.json').read_text())
        ps = json.loads((parent / 'summary.json').read_text())
        progress = json.loads((root / source['completed_record']).read_text())
        prefix = f's{m["seed"]}_r{int(m["reverse"])}'
        basis = []; ids = None; residual = 0.
        for video, audio in ((0, 0), (0, 1), (1, 0), (1, 1)):
            row = next(r for r in ps['probes'] if
                (r['kind'], r['weights'], r['presentation'], r['video'], r['audio']) ==
                ('resting', 'learned', 'both', video, audio))
            z = read_record(parent, row, pm, learning_auditor=verify_learning)
            if np.any(z['weights'][:64] != z['weights_initial']) or np.any(z['drive'][:64, 194:198]):
                raise ValueError('Reference is not an unchanged pre-feedback prefix')
            if ids is None:
                ids = z['context_source_ids'].copy()
            if not np.array_equal(ids, z['context_source_ids']):
                raise ValueError('Reference input identity changed')
            order = [list(ids[1]).index(i) for i in ids[0]]
            if not np.array_equal(z['arrivals'][:64, 0], z['arrivals'][:64, 1][:, order]):
                raise ValueError('Reference opponents received different histories')
            for j, nid in enumerate(m['groups']['prediction']):
                actual = z['cells'][:64, list(z['neuron_ids']).index(nid), base.FIELDS.index('O')]
                native = native_predictor(z['arrivals'][:64, j], z['weights_initial'][j])
                residual = max(residual, float(abs(actual - native).max()))
            sign = 1 if video ^ audio ^ int(m['reverse']) == 0 else -1
            basis.append(sign * filtered_context(z['arrivals'][:64, 0]))
        if residual > 2e-12:
            raise ValueError('Native reference reconstruction failed')
        basis = np.stack(basis)
        arrays[prefix + '_signed_basis'] = basis
        arrays[prefix + '_source_ids'] = ids
        # Remove the declared task signs before measuring sensory overlap.
        signs = np.array([1., -1., -1., 1.]) * (-1 if m['reverse'] else 1)
        raw_basis = basis * signs[:, None, None]
        mode_basis = np.einsum('ij,jkn->ikn', FACTORIAL, raw_basis)
        arrays[prefix + '_raw_gram'] = np.einsum('pkn,qkn->kpq', raw_basis, raw_basis)
        arrays[prefix + '_mode_gram'] = np.einsum('pkn,qkn->kpq', mode_basis, mode_basis)
        references.append(dict(seed=m['seed'], reverse=m['reverse'], key=prefix,
                               native_residual=residual, parent=str(parent),
                               mode_norms_16_63=np.linalg.norm(mode_basis[:,16:64],axis=(1,2)).tolist()))
        for row in progress['training']:
            z = read_record(root, row, m, learning_auditor=verify_learning)
            if not np.array_equal(ids, z['context_source_ids']):
                raise ValueError('Acquisition input identity changed')
            key = prefix + '_' + Path(row['file']).stem
            components = credit_components(z)
            weights = differential(z['weights'], ids)
            arrays[key + '_q'] = weights
            arrays[key + '_q_initial'] = differential(z['weights_initial'], ids)
            bounds = projected_bounds(weights, basis)
            arrays[key + '_bounds'] = bounds
            for label, value in components.items():
                arrays[key + '_' + label] = differential(value, ids)
            # Retain both opponent magnitudes: subtracting them can conceal
            # large but mutually cancelling plastic updates.
            arrays[key + '_credit_l1'] = np.stack([
                abs(components[label]).sum(axis=2) for label in components], axis=1)
            episodes.append(dict(key=key, seed=m['seed'], reverse=m['reverse'],
                block=row['block'], video=row['video'], audio=row['audio'],
                source_sha256=row['sha256'], ticks=len(weights),
                first_bounds=bounds[0].tolist(), last_bounds=bounds[-1].tolist()))
    if not episodes:
        raise ValueError('No acquisition evidence')
    output.mkdir()
    np.savez_compressed(output / 'credit-factors.npz', **arrays)
    result = dict(episodes=episodes, references=references,
        audit_summary=str(summary_path), audit_sha256=base.digest(summary_path),
        analysis_source_sha256=base.digest(__file__),
        acquisition_ticks=sum(r['ticks'] for r in episodes),
        reconstruction='For each acquisition tick t and probe tick k: signed_basis[p,k,:] @ q[t,:]. '
                       'q is the positive-minus-negative selected weight, aligned by recorded source IDs. '
                       'Subtract consecutive q or use the three saved credit components for update effects.',
        pair_order=[[0,0],[0,1],[1,0],[1,1]], bounds_order=['min','max'], probe_interval=[16,64],
        mode_order=['common','visual','auditory','joint'],
        gram_axes=['probe_tick','pair_or_mode','pair_or_mode'],
        credit_l1_axes=['acquisition_tick','old/new/bound','opponent_channel'],
        limits='Fixed factual four-exposure interface, float64 linear response without native rounding. '
               'No counterfactual neural replay, fitted weights, acceptance, or semantic generalization. '
               'Credit age splits context activity, not teaching-error origin or residual neural release. '
               'Reference Gram matrices are interface geometry, not a closed-loop learning Jacobian. '
               'Bound corrections include subtraction-roundoff; no unique clipped credit attribution.')
    (output / 'summary.json').write_text(base.encode(result) + '\n')
    print(base.encode(dict(episodes=len(episodes), acquisition_ticks=result['acquisition_ticks'])), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('analysis_root', type=Path); p.add_argument('output', type=Path)
    a = p.parse_args(); analyze(a.analysis_root, a.output)
