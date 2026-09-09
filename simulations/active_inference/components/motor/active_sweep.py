"""Compose the existing four-phase PAULA rhythm with an embodied predictor.

The reference CPG's neuron parameters and delayed ring are copied from its
builder, not replaced by a host clock. Existing antagonist muscles receive
CPG phases 0 and 2. Mixed populations optionally receive the actual muscle
outputs and delayed position/velocity receptors. All conditions retain those
edges and their return paths; the sensory-only control sets added context
throughputs to zero. No learned classifier, target trajectory or action policy.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from ... import nmrower2 as reference

k = reference.k


def append_active_sweep(original, groups, *, seed=11, sensorimotor=True):
    cfg = deepcopy(original); g = deepcopy(groups)
    nodes = {n['id']: n for n in cfg['neurons']}
    if len(nodes) != len(cfg['neurons']) or any(len(g[r]) != 2 for r in ('muscle', 'joint', 'prediction')):
        raise ValueError('Need distinct original cells and opponent channels')
    scratch = Path(reference.build())
    try:
        motor = json.loads(scratch.read_text())
    finally:
        scratch.unlink()
    remap = {n: max(nodes)+i+1 for i, n in enumerate(reference.P)}
    g['cpg'] = list(remap.values())
    for node in motor['neurons']:
        if node['id'] not in remap:
            continue
        node['id'] = remap[node['id']]
        node['params'].update(num_inputs=2, eta_post=1e-7, eta_retro=1e-7)
        node['metadata'].update(role='cpg', bounded_plasticity=True, graded_gain=0.,
                                plasticity_rate_boost=0., retrograde_magnitude_error=False)
        cfg['neurons'].append(node); nodes[node['id']] = node
        cfg['synaptic_points'].append(k.syn(node['id'], 1, 0., adapt=[0., 0.]))
    for point in motor['synaptic_points']:
        if point['neuron_id'] not in remap:
            continue
        point['neuron_id'] = remap[point['neuron_id']]
        if point['type'] == 'presynaptic':
            point['terminal_id'] = k.TERM
        cfg['synaptic_points'].append(point)
    for edge in motor['connections']:
        if edge['source_neuron'] in remap and edge['target_neuron'] in remap:
            edge['source_neuron'] = remap[edge['source_neuron']]
            edge['target_neuron'] = remap[edge['target_neuron']]
            edge['source_terminal'] = k.TERM
            cfg['connections'].append(edge)
    cfg['external_inputs'].append(k.ext(g['cpg'][0], 0))
    g['velocity'] = [max(nodes)+1, max(nodes)+2]
    for nid in g['velocity']:
        node = k.neuron(nid, lam=2, c=3, eta_post=1e-7, eta_retro=1e-7, delta_decay=.99,
                        meta=dict(role='velocity', graded_gain=1., bounded_plasticity=True,
                                  plasticity_rate_boost=0., retrograde_magnitude_error=True))
        node['params']['num_inputs'] = 2
        cfg['neurons'].append(node); nodes[nid] = node
        cfg['synaptic_points'] += [k.term(nid), k.syn(nid, 0, 1., adapt=[0., 0.]),
                                   k.syn(nid, 1, 0., adapt=[0., 0.])]
        cfg['external_inputs'].append(k.ext(nid, 0))

    additions = []
    sources = {(e['target_neuron'], e['target_synapse']):e['source_neuron'] for e in cfg['connections']}
    for point in cfg['synaptic_points']:
        if (point['type'] == 'postsynaptic' and point['neuron_id'] in g['muscle']
                and sources.get((point['neuron_id'], point['synapse_id'])) in g['joint']):
            # Receptors now express q/.05 rather than q radians. Preserve the
            # original reflex's physical gain; do not accidentally amplify it 20x.
            point['u_i']['info'] *= .05
    def wire(source, target, weight, role):
        sid = nodes[target]['params']['num_inputs']
        if sid >= k.TERM:
            raise ValueError('Port budget exceeded')
        nodes[target]['params']['num_inputs'] += 1
        cfg['synaptic_points'].append(k.syn(target, sid, float(weight), adapt=[0., 0.]))
        cfg['connections'].append(k.conn(source, target, sid))
        additions.append([source, target, sid, float(weight), role])

    for phase, target in zip((g['cpg'][0], g['cpg'][2]), g['muscle']):
        wire(phase, target, 8., 'rhythmic_drive')
    rng = np.random.default_rng(seed+7811)
    for left, right in zip(g['mixed_0'], g['mixed_1']):
        for family in ('muscle', 'joint', 'velocity'):
            # Matched anatomical heterogeneity in the two contextual banks.
            signs = rng.choice([-1., 1.], 2)
            for source, sign in zip(g[family], signs):
                for target in (left, right):
                    wire(source, target, .125*sign if sensorimotor else 0., 'context_'+family)
    cfg['metadata']['active_sweep'] = dict(sensorimotor=sensorimotor, seed=seed, groups={r:g[r] for r in ('cpg','velocity')},
        additions=additions, birth_kick=5., joint_reflex_unit_compensation=.05,
        limits='Existing reference rhythm, additional motor conductance, and heterogeneous signed sensorimotor context. '
               'Added context conductance is intentional, not matched to sensory-only current. No learned rhythm or action selector.')
    return cfg, g
