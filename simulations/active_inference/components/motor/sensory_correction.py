"""Optional opponent sensory currents into the existing antagonistic muscles.

Position mode is a fixed centering reflex. Expectation mode subtracts learned
predicted opponent position using existing neural error cells. Neither route
is a learned hierarchy or a biologically reconstructed spinal circuit.
All variants have the same cells/edges and positive basal adaptation. Only
the new muscle weights differ. Zero-current edges still carry native plastic
return signals. Four observation filters match the comparators' delay and
membrane constant; they do not guarantee matched complete signal dynamics.
"""
from copy import deepcopy

import numpy as np

from ..learning.predictive_bridge import k


def append_sensory_correction(original, motor, bridge, *, mode='position', gain=1.):
    if mode not in ('none', 'position', 'expectation') or not np.isfinite(gain) or gain <= 0:
        raise ValueError('Invalid correction mode or gain')
    cfg = deepcopy(original)
    nodes = {n['id']: n for n in cfg['neurons']}
    required = motor['joint_position']+motor['muscles']+bridge['error_positive']+bridge['error_negative']
    if any(n not in nodes for n in required) or any(len(ids) != 4 for ids in
            (motor['joint_position'], motor['muscles'], bridge['error_positive'], bridge['error_negative'])):
        raise ValueError('Expected four ordered opponent joint and muscle channels')
    cursor = max(nodes)+1; filters = list(range(cursor, cursor+4))
    for nid, source in zip(filters, motor['joint_position']):
        n = k.neuron(nid, lam=8, c=3, eta_post=1e-7, eta_retro=1e-7, delta_decay=.99,
            meta=dict(role='position_feedback_filter', graded_gain=1., bounded_plasticity=True,
                      plasticity_rate_boost=0.))
        n['params']['num_inputs'] = 2
        cfg['neurons'].append(n)
        cfg['synaptic_points'] += [k.syn(nid, 0, 1., 1, adapt=[0., 0.]),
                                  k.syn(nid, 1, 0., 1, adapt=[0., 0.]), k.term(nid)]
        cfg['connections'].append(k.conn(source, nid, 0))
    ports = []
    for joint in (0, 1):
        plus, minus = 2*joint, 2*joint+1
        # Positive coordinate deviation excites retraction and inhibits
        # protraction. Opponent channels reverse this for negative deviation.
        sources = [filters[plus], filters[minus],
                   bridge['error_positive'][plus], bridge['error_negative'][plus],
                   bridge['error_positive'][minus], bridge['error_negative'][minus]]
        deviation_signs = [1., -1., 1., -1., -1., 1.]
        for muscle, correction_sign in ((motor['muscles'][plus], -1.), (motor['muscles'][minus], 1.)):
            for j, (source, sign) in enumerate(zip(sources, deviation_signs)):
                sid = nodes[muscle]['params']['num_inputs']
                if sid >= k.TERM:
                    raise ValueError('Feedback port collides with terminal')
                nodes[muscle]['params']['num_inputs'] += 1
                selected = (mode == 'position' and j < 2) or (mode == 'expectation' and j >= 2)
                weight = gain*correction_sign*sign if selected else 0.
                cfg['synaptic_points'].append(k.syn(muscle, sid, weight, 1, adapt=[0., 0.]))
                cfg['connections'].append(k.conn(source, muscle, sid))
                ports.append([source, muscle, sid, weight])
    meta = dict(mode=mode, gain=gain, filters=filters, ports=ports,
        rule='Muscle current is minus signed observed position, or minus signed observed-minus-predicted position.',
        limits='Fixed anatomical signs and gain. No Python desired angle, online policy or error. '
               'New incoming conductance intentionally adds motor authority; cell count and wiring match modes.')
    cfg['metadata']['sensory_correction'] = meta
    return cfg, meta


def install_on_runtime(branch, fresh, original, cfg):
    """Install this declared additive graph delta without resetting old state.

    A fresh configured network supplies only new ports/cells. Existing neurons,
    learned weights, queues, terminals and extension traces remain the acquired
    objects. Expanded muscles keep their old membrane and queue, but their
    input count and t_ref upper bound necessarily grow. Both cache families
    are rebuilt against the new buffers. This is construction, never per-tick
    neural control. Caller must save the resulting executable checkpoint.
    """
    net = branch.network; topo = net.network
    old = {n['id']: n for n in original['neurons']}
    new = {n['id']: n for n in cfg['neurons']}
    if set(topo.neurons) != set(old) or cfg['connections'][:len(original['connections'])] != original['connections']:
        raise ValueError('Only additive graph installation is supported')
    vec = getattr(topo, '_ext_vec', None)
    if vec is not None and any(np.any(vec[k]) for k in ('info', 'plast', 'mod')):
        raise ValueError('Cannot discard pending external input')
    if cfg['external_inputs'] != original['external_inputs']:
        raise ValueError('Installation must preserve external inputs')
    for nid, node in new.items():
        candidate = fresh.network.neurons[nid]
        if nid not in old:
            topo.neurons[nid] = candidate
            continue
        cell = topo.neurons[nid]; count = node['params']['num_inputs']
        if count == cell.params.num_inputs:
            continue
        old_count = cell.params.num_inputs
        if count < old_count:
            raise ValueError('Not an additive port change')
        cell.params = deepcopy(cell.params); cell.params.num_inputs = count
        cell.upper_t_ref_bound = cell.params.c*count
        buffer = np.zeros((count, cell.input_buffer.shape[1]), dtype=cell.input_buffer.dtype)
        buffer[:old_count] = cell.input_buffer; cell.input_buffer = buffer
        for sid in range(old_count, count):
            cell.postsynaptic_points[sid] = candidate.postsynaptic_points[sid]
            cell.distances[sid] = candidate.distances[sid]
            cell.synapse_sources[sid] = candidate.synapse_sources[sid]
    topo.num_neurons = len(topo.neurons)
    topo.connections = [(e['source_neuron'], e['source_terminal'], e['target_neuron'], e['target_synapse'])
                        for e in cfg['connections']]
    topo.connection_cache.clear(); topo.fast_connection_cache.clear()
    for a, terminal, b, port in topo.connections:
        topo.connection_cache[a, terminal].append((b, port))
    topo.optimize_runtime_connections()
    topo._ext_vec = None
    from ...core.runtime_checkpoint import check_buffer_aliases
    check_buffer_aliases(net)
