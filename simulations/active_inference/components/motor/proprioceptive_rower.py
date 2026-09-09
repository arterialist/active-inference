"""Append a reference PAULA motor plant and physical joint-sensing pathway.

The existing rower builder and MuJoCo geometry remain unchanged. This research
composition remaps IDs, supplies positive basal plasticity, gives its existing
nonspiking muscles graded neural release, and adds joint-position afferents.
No action selector or outcome predictor runs in Python.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from ... import nmrower2 as reference
from ..learning.predictive_bridge import append_predictive_bridge

k = reference.k


def append_proprioceptive_rower(original, ascending_targets, *, seed=11):
    cfg = deepcopy(original)
    existing = {n['id'] for n in cfg['neurons']}
    if len(existing) != len(cfg['neurons']) or not set(ascending_targets) <= existing:
        raise ValueError('Invalid original population or ascending targets')
    temporary = Path(reference.build())
    try:
        motor = json.loads(temporary.read_text())
    finally:
        temporary.unlink()  # Only the exact scratch config just created above.
    remap = {n['id']:max(existing)+i+1 for i,n in enumerate(motor['neurons'])}
    muscle_ids = (reference.MLp,reference.MLr,reference.MRp,reference.MRr)
    for node in motor['neurons']:
        old = node['id']; node['id'] = remap[old]
        node['params'].update(eta_post=1e-7,eta_retro=1e-7)
        node['metadata'].update(role='body_muscle' if old in muscle_ids else 'body_cpg',
            bounded_plasticity=True,plasticity_rate_boost=0.)
        if old in muscle_ids:
            node['metadata']['graded_gain'] = 1.
        else:
            # The one-port historical CPG has inverted t_ref bounds. A reserved
            # zero-current port restores lower<=upper without driving the cell.
            motor['synaptic_points'].append(k.syn(old,1,0.,1,adapt=[0.,0.]))
            node['params']['num_inputs'] = 2
        cfg['neurons'].append(node)
    for point in motor['synaptic_points']:
        point['neuron_id'] = remap[point['neuron_id']]
        if point['type'] == 'presynaptic':
            point['terminal_id'] = k.TERM
        cfg['synaptic_points'].append(point)
    for connection in motor['connections']:
        connection['source_neuron'] = remap[connection['source_neuron']]
        connection['target_neuron'] = remap[connection['target_neuron']]
        connection['source_terminal'] = k.TERM
        cfg['connections'].append(connection)
    # No host steering channel is installed. Only the one-time birth kick.
    cfg['external_inputs'].append(k.ext(remap[reference.P[0]],0))
    groups = dict(cpg=[remap[n] for n in reference.P],muscles=[remap[n] for n in muscle_ids])
    groups['joint_position'] = list(range(max(remap.values())+1,max(remap.values())+5))
    for nid in groups['joint_position']:
        n = k.neuron(nid,lam=1,c=2,eta_post=1e-7,eta_retro=1e-7,delta_decay=.99,
            meta=dict(role='joint_position',graded_gain=1.,bounded_plasticity=True,plasticity_rate_boost=0.))
        n['params']['num_inputs'] = 2; cfg['neurons'].append(n)
        cfg['synaptic_points'] += [k.term(nid),k.syn(nid,0,1.,1,adapt=[0.,0.]),k.syn(nid,1,0.,1,adapt=[0.,0.])]
        cfg['external_inputs'].append(k.ext(nid,0))
    rng = np.random.default_rng(seed+5167)
    nodes = {n['id']:n for n in cfg['neurons']}
    ascending = []
    for target in ascending_targets:
        for source in rng.choice(groups['joint_position'],2,replace=False):
            sid = nodes[target]['params']['num_inputs']
            nodes[target]['params']['num_inputs'] += 1
            if sid >= k.TERM:
                raise ValueError('Ascending port collides with information terminal')
            cfg['synaptic_points'].append(k.syn(target,sid,1.5,1,adapt=[0.,0.]))
            cfg['connections'].append(k.conn(int(source),target,sid))
            ascending.append([int(source),target,sid])
    previous_bridge = deepcopy(cfg['metadata'].get('predictive_bridge'))
    cfg,bridge,edges,selected = append_predictive_bridge(cfg,groups['cpg']+groups['muscles'],
        groups['joint_position'],seed=seed,fanin=8,consumers=8)
    cfg['metadata']['proprioceptive_bridge'] = cfg['metadata'].pop('predictive_bridge')
    if previous_bridge is not None:
        cfg['metadata']['predictive_bridge'] = previous_bridge
    cfg['metadata']['proprioceptive_rower'] = dict(seed=seed,groups=groups,ascending=ascending,
        neuron_ticks_per_physics_step=1,muscle_gain=reference.GGAIN,
        modifications='Positive motor plasticity; standard terminal IDs; reserved CPG ports; graded muscle release. '
                      'CPG timing and body parameters inherited. Joint afferents ascend into the existing tactile core.')
    return cfg,groups,bridge,selected
