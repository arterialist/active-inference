"""Full-trajectory causal comparison of physical disturbance and feedback cut.

No decoder is fitted, and a neural difference is not called useful regulation.
The factorial separates a cut's baseline effect from its effect on disturbance
response. Physical trajectories may differ, so neither contrast assumes matched
future sensation. All cellwise effects remain in the output, including zeros.
"""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path

import numpy as np

from .association_route_probe import digest
from .composition_probe import encode
from .hierarchical_body_perturbation import audit_intervention
from .predictive_bridge_probe import audit_record
from .proprioceptive_learning_audit import first_difference, intervals
from .temporal_body_probe import audit_history


def describe(x):
    x = np.asarray(x, dtype=float)
    flat = x.reshape(len(x), -1)
    if not np.isfinite(flat).all():
        raise ValueError('Nonfinite causal trajectory')
    maximum = np.max(np.abs(flat), axis=1)
    return dict(first_nonzero=first_difference(flat, np.zeros_like(flat)),
        nonzero_intervals=intervals(maximum > 0), maximum_absolute=float(maximum.max()),
        last64_maximum_absolute=float(maximum[-64:].max()),
        positive_cell_ticks=int(np.sum(flat > 0)), negative_cell_ticks=int(np.sum(flat < 0)))


def pathway_summary(cfg, bridge):
    roles = {n['id']: n['metadata'].get('role', 'unassigned') for n in cfg['neurons']}
    graph = defaultdict(list)
    pairs = Counter()
    for e in cfg['connections']:
        a, b = e['source_neuron'], e['target_neuron']
        graph[a].append(b); pairs[roles[a], roles[b]] += 1

    def reaches(sources, targets):
        seen = set(sources); queue = deque(sources)
        while queue:
            for n in graph[queue.popleft()]:
                if n not in seen:
                    seen.add(n); queue.append(n)
        return sorted(seen & set(targets))

    upper = [n for n in roles if roles[n] == 'upper_core']
    motor = [n for n in roles if roles[n] in ('body_cpg', 'body_muscle')]
    return dict(role_projection_counts={a+' -> '+b: count for (a, b), count in sorted(pairs.items())},
        motor_neurons_forward_reachable_from_upper=reaches(upper, motor),
        upper_neurons_forward_reachable_from_new_consumers=reaches(bridge['prediction_consumer'], upper),
        limitation='Configured forward projection graph only. Retrograde plastic signals and body-mediated paths are excluded. '
                   'Reachability neither proves transmission nor useful control.')


def observability(a, b):
    """Compare actual context, without mistaking it for the complete neural state."""
    result = {}
    for field in ('history_arrivals', 'history_weights', 'history_scheduled', 'arrivals',
                  'weights', 'eta', 'error_arrival', 'joint_input'):
        result[field] = describe(b[field].astype(float)-a[field].astype(float))
    context_equal = np.all(a['arrivals'] == b['arrivals'], axis=(1, 2))
    sensory_differs = np.any(a['joint_input'] != b['joint_input'], axis=1)
    # Equality is per tick, not equality of the complete preceding history.
    prefix_end = first_difference(a['arrivals'], b['arrivals'])
    if prefix_end is None:
        prefix_end = len(context_equal)
    prefix = np.arange(len(context_equal)) < prefix_end
    result['different_sensation_with_identical_context_prefix'] = intervals(prefix & sensory_differs)
    result['equal_context_and_different_sensation_ticks'] = int(np.sum(context_equal & sensory_differs))
    result['limit'] = ('Selected input arrival histories only. Error-driven weights, receptor traces, '
                       'and other populations can already differ and carry physical-state information.')
    return result


def run(roots, output):
    roots, output = [Path(p).resolve() for p in roots], Path(output).resolve()
    if len(roots) != 4 or len(set(roots)) != 4 or output.exists():
        raise ValueError('Need four unique factorial recordings and a new output')
    data, manifests, raw_hashes = {}, {}, {}
    cfg = None
    for root in roots:
        m = json.loads((root/'manifest.json').read_text())
        key = bool(m['cut']), float(m['torque'])
        if key in data or key not in {(cut, force) for cut in (False, True) for force in (0., .5)}:
            raise ValueError('Invalid or duplicated factorial condition')
        if any(digest(p) != h for p, h in m['source_hashes'].items()):
            raise ValueError('Experiment source changed')
        s = json.loads((root/'summary.json').read_text())
        raw = root/'closed-loop.npz'
        if digest(raw) != s['raw_sha256']:
            raise ValueError('Raw record changed')
        current_cfg = json.loads((Path(m['source'])/'config.json').read_text())
        if cfg is not None and cfg != current_cfg:
            raise ValueError('Different birth graphs')
        cfg = current_cfg
        with np.load(raw) as z:
            d = {k: z[k] for k in z.files}
        if len(d['ticks']) != m['ticks'] or not np.array_equal(d['neuron_ids'], m['neuron_ids']):
            raise ValueError('Tick count or column identity mismatch')
        np.testing.assert_array_equal(np.diff(d['ticks']), np.ones(m['ticks']-1))
        audit_intervention(d, cfg, cut=m['cut'], torque=m['torque'], gain=m['gain'],
                           start=m['force_start'], stop=m['force_stop'])
        audit_history(d, cfg, m['basis']); audit_record(d, cfg, m['bridge'])
        data[key], manifests[key] = d, m; raw_hashes[str(raw)] = digest(raw)
    base, bm = data[False, 0.], manifests[False, 0.]
    for key, d in data.items():
        m = manifests[key]
        for field in ('source', 'ticks', 'gain', 'force_start', 'force_stop', 'motor', 'bridge', 'basis'):
            if m[field] != bm[field]:
                raise ValueError('Unmatched branch '+field)
        for field in base:
            if field.startswith('start_') or field in ('neuron_ids', 'ticks'):
                np.testing.assert_array_equal(base[field], d[field])
        np.testing.assert_array_equal(base['physical_before'][0], d['physical_before'][0])
    for cut in (False, True):
        for field in ('cells', 'weights', 'physical_before', 'physical_after', 'feedback_before'):
            np.testing.assert_array_equal(data[cut, 0.][field][:64], data[cut, .5][field][:64])
    index = {int(n): i for i, n in enumerate(base['neuron_ids'])}
    groups = defaultdict(list)
    for n in cfg['neurons']:
        groups[n['metadata'].get('role', 'unassigned')].append(index[n['id']])
    trajectories, summaries = dict(ticks=base['ticks'], neuron_ids=base['neuron_ids']), {}
    for field in ('joint_input', 'joint_position', 'muscle_state', 'actuator_ctrl', 'weights', 'eta', 'cells'):
        arrays = {key: d[field].astype(float) for key, d in data.items()}
        contrasts = {'pulse_intact': arrays[False, .5]-arrays[False, 0.],
                     'pulse_cut': arrays[True, .5]-arrays[True, 0.],
                     'cut_sham': arrays[True, 0.]-arrays[False, 0.],
                     'cut_pulse': arrays[True, .5]-arrays[False, .5]}
        contrasts['interaction'] = contrasts['pulse_cut']-contrasts['pulse_intact']
        for name, x in contrasts.items():
            label = field+'/'+name
            trajectories[label] = x; summaries[label] = describe(x)
            if field == 'cells':
                for role, positions in groups.items():
                    for f, j in (('S', 0), ('O', 1), ('M1', 4), ('rate', 7)):
                        summaries[label+'/'+role+'/'+f] = describe(x[:, positions, j])
    activity = {}
    for (cut, force), d in data.items():
        label = f'cut={cut}/torque={force}'
        active_upper = np.any(d['cells'][:, groups['upper_core'], 1] > 0, axis=1)
        removed = d['feedback_before'][:, :, 0]-d['feedback_after'][:, :, 0]
        activity[label] = dict(upper_active_intervals=intervals(active_upper),
            nonzero_removed_arrivals=int(np.count_nonzero(removed)),
            maximum_removed_amplitude=float(np.max(removed)),
            minimum_local_prediction_rate=float(d['eta'].min()),
            maximum_prediction_weight_change=float(np.max(np.abs(d['weights']-d['start_weights']))))
    result = dict(audit_source_sha256=digest(__file__), raw_hashes=raw_hashes, summaries=summaries,
        observability=observability(data[False, 0.], data[False, .5]),
        activity=activity, pathways=pathway_summary(cfg, bm['bridge']), ticks=4*len(base['ticks']),
        limitation='A single-state causal screen, not replicated regulation or learned action. '
                   'No content task or memory criterion is present; suppression is not success. '
                   'Difference-of-differences measures interaction, not proof of compensation. '
                   'Physical actuator differences are analyzed exactly and may be numerically tiny.')
    output.mkdir(exist_ok=False)
    np.savez_compressed(output/'effects-per-tick.npz', **trajectories)
    (output/'summary.json').write_text(encode(result)+'\n')
    print(encode(dict(ticks=result['ticks'], activity=activity, pathways=result['pathways'])), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots', type=Path, nargs=4, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(**vars(p.parse_args()))
