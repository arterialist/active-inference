"""Recheck temporal-weight interventions and retain every movement contrast."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .association_route_probe import digest
from .associative_mismatch_audit import contextual_weight_mask
from .composition_probe import encode
from .hierarchical_body_audit import describe
from .proprioceptive_learning_audit import intervals
from .sensory_motor_placement import audit, temporal_permutation
from .sensory_motor_probe import body_measures


def run(roots, output):
    output = Path(output).resolve()
    if len(roots) != 8 or output.exists():
        raise ValueError('Need eight runs and a new output')
    records, prefixes, metrics, raw_hashes, reports, traces = {}, {}, {}, {}, {}, {}
    reference = {}
    for root in map(Path, roots):
        m = json.loads((root/'manifest.json').read_text())
        key = m['placement_seed'], m['torque']
        if key in records or key not in {(s, f) for s in (23, 44, 77, 101) for f in (0., .5)}:
            raise ValueError('Wrong factorial branch')
        cfg = json.loads((root/'config.json').read_text())
        s = json.loads((root/'summary.json').read_text())
        if any(digest(p) != h for p, h in m['source_hashes'].items()):
            raise ValueError('Source changed')
        raw = root/'closed-loop.npz'
        if digest(raw) != s['raw_sha256']:
            raise ValueError('Recording changed')
        with np.load(raw) as z:
            d = {k: z[k] for k in z.files}
        residuals = audit(d, cfg, m)
        calculated = body_measures(d)
        for field, v in calculated.items():
            if isinstance(v, str):
                assert v == s['metrics'][field]
            else:
                np.testing.assert_array_equal(v, s['metrics'][field])
        parent = Path(m['source'])
        with np.load(parent/'closed-loop.npz') as z:
            q, mapping = temporal_permutation(cfg, m['bridge'], z['start_weights'], key[0])
            np.testing.assert_array_equal(q, d['start_weights'])
            mask = contextual_weight_mask(cfg, m['bridge'])
            for field in z.files:
                if not field.startswith('start_') or field == 'start_weights':
                    continue
                if field == 'start_incoming_info':
                    np.testing.assert_array_equal(z[field][~mask], d[field][~mask])
                else:
                    np.testing.assert_array_equal(z[field], d[field])
            with np.load(root/'intervention.npz') as intervention:
                np.testing.assert_array_equal(intervention['before'], z['start_weights'])
                np.testing.assert_array_equal(intervention['after'], q)
                np.testing.assert_array_equal(intervention['mapping'], mapping)
            index = {int(n): i for i, n in enumerate(d['neuron_ids'])}
            ids = [index[n] for n in m['bridge']['prediction']]
            for field in ('actuator_ctrl', 'joint_position', 'forward_progress', 'error_arrival'):
                delta = z[field]-d[field]
                name = f'learned_minus_shuffle/{key}/{field}'
                traces[name] = delta; reports[name] = describe(delta)
            name = f'learned_minus_shuffle/{key}/prediction_S'
            delta = z['cells'][:, ids, 0]-d['cells'][:, ids, 0]
            traces[name] = delta; reports[name] = describe(delta)
            if key[1] not in reference:
                reference[key[1]] = {k: z[k] for k in ('joint_position', 'forward_progress')}
        traces[f'{key}/cumulative_positive_work'] = np.cumsum(np.maximum(d['sampled_actuator_power'], 0).sum(axis=1))*.004
        prefixes[key] = {k: hashlib.sha256(d[k][:256].tobytes()).hexdigest()
                         for k in ('cells', 'weights', 'physical_after', 'actuator_ctrl')}
        metrics[str(key)] = dict(calculated,
            last164_forward_change=float(d['forward_progress'][-1]-d['forward_progress'][-165]),
            minimum_learning_rate=float(d['eta'].min()), residuals=residuals)
        records[key] = {k: d[k] for k in ('joint_position', 'forward_progress', 'actuator_ctrl')}
        raw_hashes[str(raw.resolve())] = digest(raw)
        del d
    for seed in (23, 44, 77, 101):
        if prefixes[seed, 0.] != prefixes[seed, .5]:
            raise ValueError('Sham and load differ before force onset')
        sham, pulse = records[seed, 0.], records[seed, .5]
        distance = np.linalg.norm(pulse['joint_position']-sham['joint_position'], axis=1)
        native_distance = np.linalg.norm(reference[.5]['joint_position']-reference[0.]['joint_position'], axis=1)
        benefit = distance-native_distance
        name = f'learned_load_distance_benefit/permutation{seed}'
        traces[name] = benefit
        reports[name] = dict(describe(benefit), unfavorable_intervals=intervals(benefit < 0),
            last164_mean=float(benefit[-164:].mean()),
            shuffled_last164_load_distance=float(distance[-164:].mean()))
    output.mkdir(exist_ok=False)
    np.savez_compressed(output/'effects-per-tick.npz', **traces)
    result = dict(source_hash=digest(__file__), raw_hashes=raw_hashes, metrics=metrics,
        trajectories=reports, ticks=8*640, unchanged_replay_ticks=8*32,
        limits='Four within-group weight permutations of one acquired graph, not independent graph replication. '
               'This tests delayed weight placement, not all temporal representation or all learned prediction. '
               'Weight distribution preservation does not match effective time-varying current. '
               'No semantic, hierarchical or consciousness conclusion follows.')
    (output/'summary.json').write_text(encode(result)+'\n')
    print(encode(dict(metrics=metrics, load_comparisons={k:v for k,v in reports.items() if k.startswith('learned_load')})), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots', type=Path, nargs=8, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(**vars(p.parse_args()))
