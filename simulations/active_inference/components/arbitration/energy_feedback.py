"""A PAULA energy-deficit comparator inhibits the existing muscles.

Uses ordinary signed dendrites, graded rectification and weak bounded native
plasticity. The half-reserve reference is an explicit anatomical setting,
not a learned preference or reconstructed hypothalamic cell type. This is
homeostatic activity feedback, not proof of sleep or deliberative arbitration.
"""
from copy import deepcopy

from ..learning.predictive_bridge import k
from ..motor.sensory_correction import install_on_runtime


def append_energy_feedback(original, motor, *, connected=True):
    cfg = deepcopy(original)
    nodes = {n['id']: n for n in cfg['neurons']}
    energy, deficit, alarm = range(max(nodes)+1, max(nodes)+4)
    ids = [energy, deficit, alarm]
    for nid, role, lam in zip(ids, ('energy_afferent', 'energy_deficit_afferent', 'energy_activity_regulator'), (4., 4., 16.)):
        n = k.neuron(nid, lam=lam, c=3, eta_post=1e-7, eta_retro=1e-7, delta_decay=.99,
            meta=dict(role=role, graded_gain=1., bounded_plasticity=True, plasticity_rate_boost=0.))
        n['params']['num_inputs'] = 2
        cfg['neurons'].append(n)
        weights = (1., -1.) if nid == alarm else (1., 0.)
        cfg['synaptic_points'] += [k.syn(nid, s, w, 1, adapt=[0., 0.]) for s, w in enumerate(weights)]
        cfg['synaptic_points'].append(k.term(nid))
    cfg['external_inputs'] += [k.ext(energy, 0), k.ext(deficit, 0)]
    cfg['connections'] += [k.conn(deficit, alarm, 0), k.conn(energy, alarm, 1)]
    ports = []
    for nid in motor['muscles']:
        sid = nodes[nid]['params']['num_inputs']
        if sid >= k.TERM:
            raise ValueError('Energy-feedback port collides with terminal')
        nodes[nid]['params']['num_inputs'] += 1
        cfg['synaptic_points'].append(k.syn(nid, sid, -2. if connected else 0., 1, adapt=[0., 0.]))
        cfg['connections'].append(k.conn(alarm, nid, sid))
        ports.append([alarm, nid, sid])
    meta = dict(energy=energy, deficit=deficit, alarm=alarm, neurons=ids, ports=ports,
        connected=bool(connected),
        interpretation='Fixed half-reserve comparator and inhibitory muscle feedback. Positive local adaptation remains.',
        control='Disconnected condition keeps all cells, edges and sensory inputs; only new muscle q is zero. '
                'Native plastic return messages remain active, so this is not a forward-only lesion.')
    cfg['metadata']['energy_feedback'] = meta
    return cfg, meta


def install_energy_feedback(branch, fresh, original, cfg):
    """Reuse additive state-preserving installation, then add new sensory ports."""
    old_ids = {n['id'] for n in original['neurons']}
    old_ext = original['external_inputs']
    if cfg['external_inputs'][:len(old_ext)] != old_ext:
        raise ValueError('Existing external ports must remain unchanged')
    extra = set(fresh.network.external_inputs)-set(branch.network.network.external_inputs)
    if len(extra) != 2 or any(nid in old_ids for nid, _ in extra):
        raise ValueError('Expected exactly two external ports on new afferents')
    # First install the neural graph with the established contract. New
    # external inputs are a separate, declared construction operation.
    neural_cfg = dict(cfg, external_inputs=old_ext)
    install_on_runtime(branch, fresh, original, neural_cfg)
    topo = branch.network.network
    for key in sorted(extra):
        topo.external_inputs[key] = deepcopy(fresh.network.external_inputs[key])
        topo.free_synapses.append(key)
    topo._ext_vec = None
