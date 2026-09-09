"""Couple two existing predictors through their signed residual.

This changes wiring, not a neuron equation. Each existing positive prediction
branch retains its contextual plasticity ports. Four ordinary comparison cells
now compare the signed target difference with the signed prediction difference.
Their rectified releases teach the existing local error receptors. Both branches
may adapt to one residual, so this is not a learning-gain-matched comparison.
It does not reproduce a biological excitatory/inhibitory plasticity rule.
"""
from copy import deepcopy

from simulations.paula_loader import ensure_paula_available

ensure_paula_available()
from paula_agent import ckit as k


def couple_opponent_predictions(original, groups, target_ids):
    cfg = deepcopy(original)
    roles = ('prediction', 'error_positive', 'error_negative')
    if len(target_ids) != 2 or any(len(groups[r]) != 2 for r in roles):
        raise ValueError('Opponent coupling requires two target/prediction channels')
    nodes = {n['id']: n for n in cfg['neurons']}
    if len(set(target_ids)) != 2 or not set(target_ids) <= nodes.keys():
        raise ValueError('Invalid opponent target channels')
    for channel in (0, 1):
        for role, sign in (('error_positive', 1.), ('error_negative', -1.)):
            target = groups[role][channel]
            node = nodes[target]
            for source, weight in ((target_ids[1-channel], -sign),
                                   (groups['prediction'][1-channel], sign)):
                sid = node['params']['num_inputs']
                if sid >= k.TERM:
                    raise ValueError('Opponent coupling exceeds port budget')
                node['params']['num_inputs'] += 1
                cfg['synaptic_points'].append(k.syn(target, sid, weight, 1, adapt=[0., 0.]))
                cfg['connections'].append(k.conn(source, target, sid))
    cfg['metadata']['opponent_prediction'] = {
        'target_ids': list(target_ids),
        'residual': '(target_0-target_1)-(prediction_0-prediction_1)',
        'scope': 'Ordinary signed comparison wiring; existing local learning unchanged',
    }
    return cfg
