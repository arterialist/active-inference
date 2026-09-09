"""Analyze load rejection, continued movement and acquired-weight action effects."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .association_route_probe import digest
from .associative_mismatch_audit import contextual_weight_mask
from .composition_probe import encode
from .hierarchical_body_audit import describe
from .hierarchical_body_perturbation import audit_intervention
from .sensory_motor_probe import audit_motor, body_measures
from .temporal_body_probe import audit_history
from .predictive_bridge_probe import audit_record
from .proprioceptive_learning_audit import intervals


def run(roots, output):
    output = Path(output).resolve()
    if len(roots) != 8 or output.exists():
        raise ValueError('Need eight declared runs and a new output')
    records, manifests, metrics, sources, prefix = {}, {}, {}, {}, {}
    cfg_by_mode = {}
    for root in map(lambda p: Path(p).resolve(), roots):
        m = json.loads((root/'manifest.json').read_text())
        label = m['mode']+('_reset' if m['reset_prediction'] else '')
        key = label, m['torque']
        if key in records or key not in {(mode, force) for mode in ('none', 'position', 'expectation', 'expectation_reset') for force in (0., .5)}:
            raise ValueError('Unexpected or duplicated condition')
        if any(digest(p) != h for p, h in m['source_hashes'].items()):
            raise ValueError('Source changed')
        cfg = json.loads((root/'config.json').read_text())
        s = json.loads((root/'summary.json').read_text())
        path = root/'closed-loop.npz'
        if digest(path) != s['raw_sha256']:
            raise ValueError('Raw record changed')
        with np.load(path) as z:
            d = {k: z[k] for k in z.files}
        audit_motor(d, cfg, m['motor']); audit_record(d, cfg, m['bridge'])
        audit_history(d, cfg, m['basis'])
        audit_intervention(d, cfg, cut=False, torque=m['torque'], gain=m['gain'],
                           start=m['force_start'], stop=m['force_stop'])
        before = {k: d[k].copy() for k in ('torso_pose', 'joint_velocity', 'sampled_actuator_power', 'forward_progress')}
        metric = body_measures(d)
        for k, a in before.items():
            np.testing.assert_array_equal(a, d[k])
        metric['last164_forward_change'] = float(d['forward_progress'][-1]-d['forward_progress'][-165])
        prefix[key] = {k: hashlib.sha256(d[k][:m['force_start']].tobytes()).hexdigest()
                       for k in ('cells', 'weights', 'physical_after', 'history_arrivals')}
        records[key] = {k: a.copy() for k, a in d.items() if k.startswith('start_') or k in
                       ('ticks', 'neuron_ids', 'joint_position', 'forward_progress', 'actuator_ctrl', 'weights', 'eta')}
        manifests[key], metrics[str(key)], sources[str(path)] = m, metric, digest(path)
        cfg_by_mode[label] = cfg
        del d
    base = manifests['none', 0.]
    for m in manifests.values():
        for k in ('source', 'ticks', 'force_start', 'force_stop', 'motor', 'bridge', 'basis', 'gain'):
            if m[k] != base[k]:
                raise ValueError('Mismatched protocol '+k)
    # All three architectures match exactly apart from the declared new
    # incoming weights and their metadata. Reset leaves the graph unchanged.
    for cfg in cfg_by_mode.values():
        for k in ('neurons', 'connections', 'external_inputs'):
            if cfg[k] != cfg_by_mode['none'][k]:
                raise ValueError('Unmatched architecture '+k)
    for mode in ('none', 'position', 'expectation', 'expectation_reset'):
        a, b = records[mode, 0.], records[mode, .5]
        if prefix[mode, 0.] != prefix[mode, .5]:
            raise ValueError('Conditions differ before the load')
        for k in a:
            if k.startswith('start_') or k in ('ticks', 'neuron_ids'):
                np.testing.assert_array_equal(a[k], b[k])
    mask = contextual_weight_mask(cfg_by_mode['expectation'], base['bridge'])
    for force in (0., .5):
        a, b = records['expectation', force], records['expectation_reset', force]
        for k in a:
            if k.startswith('start_') and k not in ('start_weights', 'start_incoming_info'):
                np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_array_equal(a['start_incoming_info'][~mask], b['start_incoming_info'][~mask])
    effects, report = dict(ticks=records['none', 0.]['ticks']), {}
    for mode in ('none', 'position', 'expectation', 'expectation_reset'):
        a, b = records[mode, 0.], records[mode, .5]
        deviation = b['joint_position']-a['joint_position']
        effects[mode+'/load_joint_difference'] = deviation
        effects[mode+'/load_joint_distance'] = np.linalg.norm(deviation, axis=1)
        for key in ('actuator_ctrl', 'forward_progress'):
            effects[mode+'/load_'+key] = b[key]-a[key]
        report[mode] = dict(joint_difference=describe(deviation),
            actuator_difference=describe(b['actuator_ctrl']-a['actuator_ctrl']),
            last164_mean_joint_distance=float(effects[mode+'/load_joint_distance'][-164:].mean()),
            endpoint_forward_load_penalty=float(a['forward_progress'][-1]-b['forward_progress'][-1]))
    for mode in ('position', 'expectation', 'expectation_reset'):
        benefit = effects['none/load_joint_distance']-effects[mode+'/load_joint_distance']
        effects[mode+'/joint_distance_reduction_vs_none'] = benefit
        report[mode]['reduction_vs_none'] = dict(full_trajectory=describe(benefit),
            unfavorable_intervals=intervals(benefit < 0), last164_mean=float(benefit[-164:].mean()))
    learned_effect = {}
    for force in (0., .5):
        a, b = records['expectation', force], records['expectation_reset', force]
        for key in ('actuator_ctrl', 'joint_position', 'forward_progress', 'weights'):
            name = f'learned_minus_reset/force{force}/{key}'
            effects[name] = a[key]-b[key]; learned_effect[name] = describe(effects[name])
    output.mkdir(exist_ok=False)
    np.savez_compressed(output/'effects-per-tick.npz', **effects)
    result = dict(raw_hashes=sources, audit_sha256=digest(__file__), metrics=metrics,
        comparisons=report, learned_effect=learned_effect,
        limits='One acquired graph. Corrective wiring is fixed; learned-weight effects do not establish hierarchy. '
               'Matched load/sham trajectories are not identical after the load; their difference is the measured response. '
               'Mechanical work is approximate; no metabolism or task-reward claim.')
    (output/'summary.json').write_text(encode(result)+'\n')
    print(encode(dict(comparisons=report, learned_effect=learned_effect)), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots', nargs=8, type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(**vars(p.parse_args()))
