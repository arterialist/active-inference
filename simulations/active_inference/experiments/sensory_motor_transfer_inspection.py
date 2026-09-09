"""Inspect transfer trajectories beyond a short-course resource pass.

This reads independently audited recordings; it never runs or changes a brain.
Cycles are bounded by actual phase-zero neural releases, not a fitted clock.
Incomplete opening/closing cycles remain explicitly identified. Every adverse
resource-contrast interval is retained, with end-exclusive local tick indices.
No finite trace is classified as a stable attractor or long-term survival.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import context_organization as base
from .body_state_memory_analysis import intervals
from .sensory_motor_transfer import BACKGROUNDS, MEMORIES, TICKS


def cycles(z, phase):
    """Describe each complete observed cycle without dropping partial ranges."""
    phase = np.asarray(phase)
    if (phase.shape != (len(z['body']),) or not np.isfinite(phase).all()
            or np.any(phase < 0)):
        raise ValueError('Expected one finite nonnegative phase signal per tick')
    # A sustained release is one onset, not repeated complete cycles.
    starts = np.flatnonzero((phase > 0) & ~np.r_[False, phase[:-1] > 0])
    rows = []
    for a, b in zip(starts[:-1], starts[1:]):
        angle = z['body'][a:b, 1]
        before = z['body'][a-1, 1] if a else z['body_initial'][1]
        travel = float(np.maximum(np.diff(np.r_[before, angle]), 0).sum())
        ex = z['exchange'][a:b]
        old = z['organs'][a-1] if a else z['organ_initial']
        rows.append(dict(start=int(a), end=int(b),
            angle_min=float(angle.min()), angle_max=float(angle.max()),
            angle_excursion=float(np.ptp(angle)), angle_midrange=float((angle.min()+angle.max())/2),
            oxygen_min=float(z['organs'][a:b, 0].min()), oxygen_max=float(z['organs'][a:b, 0].max()),
            reserve_change=z['organs'][b-1, [0, 6]]-old[[0, 6]],
            inspired_ml=float(ex[:, 2].sum()), uptake_ml=float(ex[:, 3].sum()),
            oxygen_demand_ml=float(ex[:, 4].sum()), positive_angular_travel_rad=travel,
            exchange_per_positive_rad=float(ex[:, 2].sum()/travel) if travel else None))
    partial = ([[0, int(starts[0])]] if len(starts) and starts[0] else [])
    partial += [[int(starts[-1]), len(phase)]] if len(starts) else [[0, len(phase)]]
    return dict(cycles=rows, partial_intervals=partial)


def resource_contrast(a, b, atol=1e-12):
    """Retain adverse intervals as well as favorable ones, never an average score."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite([a, b]).all():
        raise ValueError('Need matched finite resource trajectories')
    result = []
    for d in (a-b).T:
        result.append(dict(positive=intervals(d > atol), negative=intervals(d < -atol),
            indistinguishable=intervals(abs(d) <= atol), minimum=float(d.min()), maximum=float(d.max())))
    return result


def inspect(analysis, output):
    analysis, output = Path(analysis).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    m = json.loads((analysis/'summary.json').read_text())
    if m['checked_ticks'] != 32768 or {c['seed'] for c in m['cases']} != {11, 23, 44, 77}:
        raise ValueError('Need the complete four-seed audited comparison')
    sources = dict(m['sources'])
    for p, h in sources.items():
        if base.digest(p) != h:
            raise ValueError('Changed audited evidence: '+p)
    sources[str(analysis/'summary.json')] = base.digest(analysis/'summary.json')
    for module in (__file__, base.__file__, __import__(intervals.__module__, fromlist=['x']).__file__):
        sources[str(Path(module).resolve())] = base.digest(module)
    rows = []
    for c in m['cases']:
        seed = c['seed']; groups = c['groups']
        paths = [Path(p).parent for p in m['sources'] if Path(p).name == 'manifest.json'
                 and Path(p).parent.name == f'20260910_sensory_motor_transfer_seed{seed}']
        if len(paths) != 1:
            raise ValueError('Ambiguous audited record root')
        root = paths[0]; initial = {}; reference_phase = None
        for background in BACKGROUNDS:
            pair = {}
            for memory in MEMORIES:
                with np.load(root/f'{background}-{memory}.npz') as f:
                    z = {k:f[k] for k in f.files}
                if len(z['body']) != TICKS:
                    raise ValueError('Wrong course duration')
                now = {k:v for k,v in z.items() if k.endswith('_initial')}
                if background == 'familiar':
                    initial[memory] = now
                if set(now) != set(initial[memory]):
                    raise ValueError('Initial-state evidence fields differ')
                for k,v in now.items():
                    np.testing.assert_array_equal(v,initial[memory][k],err_msg=k)
                ids = list(z['neuron_ids']); oi = base.FIELDS.index('O')
                phase = z['cells'][:,[ids.index(n) for n in groups['cpg']],oi]
                if reference_phase is None:
                    reference_phase = phase.copy()
                np.testing.assert_array_equal(phase,reference_phase)
                pair[memory] = z['organs'][:,[0,6]]
                rows.append(dict(seed=seed,background=background,memory=memory,
                    **cycles(z,phase[:,0])))
                del z
            rows.append(dict(seed=seed,background=background,comparison='retained-minus-reset',
                resource_fields=['oxygen_ml','energy_j'],
                contrasts=resource_contrast(pair['retained'],pair['reset'])))
    result = dict(rows=rows,sources=sources,limits=__doc__,checked_ticks=32768,
        same_recorded_initials_across_backgrounds=True,same_all_cpg_release_trains=True)
    output.write_text(base.encode(result)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('analysis'); p.add_argument('output')
    args = p.parse_args(); inspect(args.analysis,args.output)
