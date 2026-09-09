"""Compare complete learning courses without selecting a favorable checkpoint.

Consumes independently audited trajectories. This observer never supplies a
readout, learning signal or action to the neural network. Every probe tick and
all physical-state/weight controls remain separate.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import context_organization as base
from .body_state_memory_analysis import intervals
from .crossed_av_continuation_analysis import PAIR_ORDER, factorial_modes


def contrast(reference, candidate):
    """Positive prediction delta is more load-aligned, not necessarily better control."""
    a, b = (np.asarray(x, dtype=float) for x in (reference, candidate))
    if a.shape != (96, 15) or b.shape != a.shape or not np.isfinite([a, b]).all():
        raise ValueError('Need two finite 96-tick audited probe trajectories')
    if not np.array_equal(a[:, 4], b[:, 4]):
        raise ValueError('Physical load differs')
    return np.column_stack((b[:, 8]-a[:, 8], abs(b[:, 1])-abs(a[:, 1]),
                            b[:, 3]-a[:, 3], a[:, 8], b[:, 8]))


def pair_events(predictions):
    x = np.asarray(predictions, dtype=float)
    if x.shape != (96, 4) or not np.isfinite(x).all():
        raise ValueError('Need all four finite pair trajectories')
    signed = x * np.array([1, -1, -1, 1])
    return dict(all_pairs_correct=intervals(np.all(signed > 0, axis=1)),
                positive_bias=intervals(np.all(x > 0, axis=1)),
                negative_bias=intervals(np.all(x < 0, axis=1)),
                any_pair_silent=intervals(np.any(x == 0, axis=1)))


def load_audit(root):
    root = Path(root).resolve()
    summary = json.loads((root/'summary.json').read_text())
    if summary['blocks'] != 16:
        raise ValueError('Need the full declared sixteen-block course')
    schedules = {}
    for source in summary['sources']:
        run = Path(source['root'])
        for name, field in [('manifest.json', 'manifest_sha256'),
                            ('completed-block-16.json', 'progress_sha256')]:
            if base.digest(run/name) != source[field]:
                raise ValueError('Audited source changed')
        manifest = json.loads((run/'manifest.json').read_text())
        if manifest['seed'] in schedules:
            raise ValueError('Duplicate graph seed')
        schedules[manifest['seed']] = manifest['schedule']
    if len(schedules) < 4:
        raise ValueError('Need at least four graph seeds')
    cases = {}
    for row in summary['cases']:
        key = tuple(row[k] for k in ('seed', 'blocks', 'kind', 'weights', 'video', 'audio'))
        if key in cases:
            raise ValueError('Duplicate probe')
        cases[key] = row
    expected = {(s,b,'resting','learned',v,a) for s in schedules for b in range(1,17) for v,a in PAIR_ORDER}
    expected |= {(s,b,'acquired',w,v,a) for s in schedules for b in (4,8,12,16)
                 for w in ('learned','reset') for v,a in PAIR_ORDER}
    if set(cases) != expected:
        raise ValueError('Incomplete probe family')
    return schedules, cases


def run(reference, candidate, output):
    reference, candidate, output = map(lambda p: Path(p).resolve(), (reference, candidate, output))
    if output.exists():
        raise FileExistsError(output)
    sa, ca = load_audit(reference); sb, cb = load_audit(candidate)
    if sa != sb or set(ca) != set(cb):
        raise ValueError('Courses differ in seed, schedule or probe identity')
    arrays = {}; cases = []; pairs = {}; groups = []
    with np.load(reference/'per-tick.npz') as za, np.load(candidate/'per-tick.npz') as zb:
        for key in sorted(ca):
            a = za[ca[key]['trace']]; b = zb[cb[key]['trace']]
            name = '_'.join(map(str, key)); diff = contrast(a, b)
            arrays[name] = diff
            cases.append(dict(identity=key, trace=name,
                prediction_improved=intervals(diff[:,0]>0), prediction_worsened=intervals(diff[:,0]<0),
                absolute_pose_reduced=intervals(diff[:,1]<0), absolute_pose_increased=intervals(diff[:,1]>0)))
            group = key[:4]; pair = key[4:]
            pairs.setdefault(group, {})[pair] = (a[:,6]-a[:,7], b[:,6]-b[:,7])
        for group, data in sorted(pairs.items()):
            entries = {}
            for j, label in enumerate(('reference','competition')):
                x = np.column_stack([data[p][j] for p in PAIR_ORDER])
                name = '_'.join(map(str, group))+'_'+label
                arrays[name+'_pairs'] = x; arrays[name+'_modes'] = factorial_modes(x)
                entries[label] = dict(trace=name, **pair_events(x))
            groups.append(dict(identity=group, **entries))
    output.mkdir(); np.savez_compressed(output/'per-tick.npz', **arrays)
    result = dict(cases=cases, groups=groups, matched_probe_ticks=len(cases)*96,
        sources={str(p/name):base.digest(p/name) for p in (reference,candidate)
                 for name in ('summary.json','per-tick.npz')},
        producer_sha256=base.digest(__file__), pair_order=PAIR_ORDER,
        difference_columns=['signed_prediction_delta','absolute_pose_delta','command_delta',
                            'reference_signed_prediction','competition_signed_prediction'],
        mode_columns=['common','visual','audio','joint'], intervals='Zero-based [start, stop).',
        limits='Observer of independent audits, not a new raw-equation audit. Resting ticks 16..63 '
               'precede new body feedback; acquired probes retain history. Comparisons across '
               'architectures include all accumulated state changes. Pose is not prediction, '
               'work or metabolic energy. No checkpoint selection or agent-level acceptance.')
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(matched_probe_ticks=result['matched_probe_ticks'], groups=len(groups))), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'candidate', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args(); run(args.reference, args.candidate, args.output)
