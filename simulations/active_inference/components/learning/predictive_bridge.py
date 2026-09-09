"""Append a channel-aligned neural predictor without replacing the old brain.

Each target channel gets a graded predictor and two opponent comparison cells.
Context reaches the predictor only through plastic neural projections. The
observed and predicted information releases reach the same comparison cell with
equal initial magnitude and delay, with opposite signs. Comparator outputs
reach local predictor error receptors, never a host-computed loss.

Context/target ID lists specify anatomical channels, not stimulus identities.
The default research screen predicts auditory receptor activity from visual
receptor activity. Passing a core population is a separate, testable choice,
not evidence that the existing upper population has learned a hierarchy.
"""
from copy import deepcopy

import numpy as np

from simulations.paula_loader import ensure_paula_available

ensure_paula_available()
from paula_agent import ckit as k


def append_predictive_bridge(original, context_ids, target_ids, *, seed=11,
                             fanin=32, consumers=32, error_wiring='matched'):
    if error_wiring not in ('matched', 'rotated'):
        raise ValueError('Unknown error wiring')
    cfg = deepcopy(original)
    existing = {n['id'] for n in cfg['neurons']}
    if (not context_ids or not target_ids or len(set(context_ids)) != len(context_ids)
            or len(set(target_ids)) != len(target_ids) or
            not set(context_ids + target_ids) <= existing or
            not 1 <= fanin <= len(context_ids) or consumers < 1):
        raise ValueError('Invalid bridge channels or population sizes')
    terminals = {(p['neuron_id'], p['terminal_id']) for p in cfg['synaptic_points']
                 if p['type'] == 'presynaptic'}
    if any((i, k.TERM) not in terminals for i in context_ids + target_ids):
        raise ValueError('Source channel lacks the declared information terminal')
    rng = np.random.default_rng(seed + 31991)
    cursor = max(existing) + 1
    groups = {}
    for role, count in (('prediction', len(target_ids)), ('error_positive', len(target_ids)),
                        ('error_negative', len(target_ids)), ('prediction_consumer', consumers)):
        groups[role] = list(range(cursor, cursor + count)); cursor += count
    ports = {nid: 0 for ids in groups.values() for nid in ids}
    node_configs = {}
    for role, ids in groups.items():
        for nid in ids:
            predictor = role == 'prediction'
            n = k.neuron(nid, lam=4 if predictor else 16 if role == 'prediction_consumer' else 8,
                         c=3, r=.6, b=.85, eta_post=1e-5 if predictor else 1e-7,
                         eta_retro=1e-7, delta_decay=.99,
                         meta={'role': role, 'bounded_plasticity': True,
                               'plasticity_magnitude_decay': .02, 'graded_gain': 1.,
                               'graded_max': 0., 'plasticity_rate_boost': 0.})
            cfg['neurons'].append(n); node_configs[nid] = n
            cfg['synaptic_points'].append(k.term(nid))
    edges, selected = [], []

    def wire(src, tgt, weight, family):
        sid = ports[tgt]; ports[tgt] += 1
        cfg['synaptic_points'].append(k.syn(tgt, sid, weight, 1, adapt=[0., 0.]))
        cfg['connections'].append(k.conn(src, tgt, sid))
        edges.append([src, tgt, sid, family, True])
        return sid

    for channel, (target, predictor, pos, neg) in enumerate(zip(target_ids,
            groups['prediction'], groups['error_positive'], groups['error_negative'])):
        incoming = []
        for src in rng.choice(context_ids, fanin, replace=False):
            src = int(src)
            sid = wire(src, predictor, 1. / fanin, 'predictive_context')
            incoming.append(sid); selected.append([predictor, sid, src])
        wire(target, pos, 1., 'observed_channel'); wire(predictor, pos, -1., 'predicted_channel')
        wire(target, neg, -1., 'observed_channel'); wire(predictor, neg, 1., 'predicted_channel')
        # Rotating comparison feedback preserves degree and amplitudes but
        # teaches the wrong target channel. It is a declared causal control.
        error_channel = channel if error_wiring == 'matched' else (channel + 1) % len(target_ids)
        p = wire(groups['error_positive'][error_channel], predictor, 0., 'positive_teaching')
        m = wire(groups['error_negative'][error_channel], predictor, 0., 'negative_teaching')
        node_configs[predictor]['metadata'].update(prediction_ports=incoming,
            prediction_error_ports=[[p, 1], [m, -1]], prediction_tau_context=8.,
            prediction_tau_error=4., prediction_cap=1., prediction_boost=499., prediction_half=.01)
    for consumer in groups['prediction_consumer']:
        for src in rng.choice(groups['prediction'], min(12, len(target_ids)), replace=False):
            wire(int(src), consumer, 1. / min(12, len(target_ids)), 'predicted_readout')
    for nid, n in node_configs.items():
        n['params']['num_inputs'] = ports[nid]
        if ports[nid] < 2 or ports[nid] >= k.TERM:
            raise ValueError('Invalid input or terminal ID budget')
    cfg['metadata']['predictive_bridge'] = dict(seed=seed, fanin=fanin,
        context_ids=list(context_ids), target_ids=list(target_ids), error_wiring=error_wiring,
        interpretation='Experimental homologous-channel prediction, not accepted multimodal recall')
    return cfg, groups, edges, selected
