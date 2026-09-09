"""Find exact temporal aliases at the learned association's consumer interface.

This is a data-only test of a particular feedforward spike interface, not an
impossibility claim about PAULA, feedback, or access to other populations.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .association_balance_audit import checked


def channel_comparison(left, right):
    """Compare every recorded coordinate at every tick, without binning."""
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (176, 8):
        raise ValueError('Expected matching full population traces')
    if not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Empty or nonfinite channel evidence')
    spikes_a, spikes_b = a[:, 96:128, 1], b[:, 96:128, 1]
    return dict(
        consumer_output_exact=bool(np.array_equal(spikes_a, spikes_b)),
        consumer_recorded_state_exact=bool(np.array_equal(a[:, 96:128], b[:, 96:128])),
        auditory_output_exact=bool(np.array_equal(a[:, 64:96, 1], b[:, 64:96, 1])),
        auditory_recorded_state_exact=bool(np.array_equal(a[:, 64:96], b[:, 64:96])),
        differing_consumer_output_ticks=np.flatnonzero(np.any(spikes_a != spikes_b, axis=1)).tolist(),
        left_active_ticks=np.flatnonzero(np.any(spikes_a > 0, axis=1)).tolist(),
        right_active_ticks=np.flatnonzero(np.any(spikes_b > 0, axis=1)).tolist(),
        # These eight recorded fields are not a full dynamic snapshot. Weights,
        # terminals, queues and eligibility traces must not be inferred equal.
        auditory_field_max_difference=np.max(abs(a[:, 64:96]-b[:, 64:96]), axis=(0, 1)).tolist(),
    )


def audit(recording, timing_audit, output):
    root, prior, output = map(Path, (recording, timing_audit, output))
    m = json.loads((root/'manifest.json').read_text())
    s = json.loads((root/'summary.json').read_text())
    parent = json.loads((Path(m['source'])/'manifest.json').read_text())
    verified = json.loads((prior/'summary.json').read_text())
    if not verified['structurally_valid'] or (verified['seed'], verified['mapping']) != (parent['seed'], parent['mapping']):
        raise ValueError('Missing matching dynamics audit')
    items = {(p['checkpoint'], p['case'], p['timing']): p for p in s['probes']}
    if len(items) != 288 or not s['training_exact'] or not s['synchronous_controls_exact']:
        raise ValueError('Invalid timing experiment')
    schedule_path = root/'schedules.npz'
    if hashlib.sha256(schedule_path.read_bytes()).hexdigest() != s['schedule_sha256']:
        raise ValueError('Changed schedules')
    records, arrays = [], {}
    with np.load(schedule_path) as schedules, np.load(prior/'dynamics.npz') as dynamics:
        for checkpoint in (32, 128):
            checked(root, s['starts'][str(checkpoint)])
            for sample in (0, 1):
                for first_cue in (0, 1):
                    # Same physical category order, opposite majority. A has
                    # 12 first-cue receptors then 4 second-cue receptors; B
                    # has 4 first-cue receptors then 12 second-cue receptors.
                    keys = [(checkpoint, f'replace25-cue{first_cue}-sample{sample}', 'majority_first'),
                            (checkpoint, f'replace25-cue{1-first_cue}-sample{sample}', 'minority_first')]
                    raw, drives, names = [], [], []
                    for side, key in enumerate(keys):
                        cp, case, timing = key
                        name = f'{cp}/{case}/{timing}'
                        with np.load(checked(root, items[key])) as z:
                            raw.append(z['states'])
                        drive = schedules[case+'/'+timing]
                        if drive.shape != (64, 32) or not np.isin(drive, [0, 1]).all():
                            raise ValueError('Invalid physical drive')
                        drives.append(drive); names.append(name)
                        if not np.array_equal(dynamics[name+'/consumer'], raw[-1][:, 96:128, 1] > 0):
                            raise ValueError('Audited consumer and raw trace differ')
                        first_ids = np.array(parent['masks']['vision'][first_cue])-1
                        second_ids = np.array(parent['masks']['vision'][1-first_cue])-1
                        expected_first, expected_second = ((12, 4) if side == 0 else (4, 12))
                        for pulse in (0, 8, 16, 24):
                            if (int(drive[pulse, first_ids].sum()), int(drive[pulse+3, second_ids].sum())) != (expected_first, expected_second):
                                raise ValueError('Wrong physical majority/order')
                        if int(drive.sum()) != 64:
                            raise ValueError('Unexpected extra input')
                    if np.array_equal(drives[0], drives[1]):
                        raise ValueError('Identical inputs cannot establish aliasing')
                    name = f'{checkpoint}/sample{sample}/first{first_cue}'
                    result = channel_comparison(*raw)
                    records.append(dict(checkpoint=checkpoint, sample=sample, first_cue=first_cue,
                        traces=names, raw_files=[items[k] for k in keys], **result))
                    for side in range(2):
                        arrays[f'{name}/{side}/schedule'] = drives[side]
                        arrays[f'{name}/{side}/consumer'] = raw[side][:, 96:128]
                        arrays[f'{name}/{side}/auditory'] = raw[side][:, 64:96]
                        arrays[f'{name}/{side}/preactivation'] = dynamics[names[side]+'/preactivation']
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'witnesses.npz', **arrays)
    result = dict(seed=parent['seed'], mapping=parent['mapping'], witnesses=records,
        provenance={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (root/'manifest.json', root/'summary.json', prior/'summary.json', prior/'dynamics.npz')},
        limits='Equal consumer output histories cannot convey opposite cue majorities to an otherwise identical downstream observer of that channel alone. This does not equate full internal states, exclude other neural pathways, or predict feedback-coupled behavior. No upper population was simulated.')
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', type=Path, required=True)
    p.add_argument('--timing-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    audit(a.recording, a.timing_audit, a.output)
