"""Population-coded history using existing graded PAULA cells and dendrites.

Motivation: Kennedy et al. 2014, doi:10.1038/nn.3650, measured delayed,
temporally diverse corollary-discharge responses. This simple filter bank is
NOT their granule/UBC model, does not implement rebound, and uses no new neuron
equation. Native propagation, membrane integration and plasticity remain.
"""
from copy import deepcopy
import math

from simulations.paula_loader import ensure_paula_available
ensure_paula_available()
from paula_agent import ckit as k


def append_temporal_basis(original, sources, *, mode='multiscale',
                          timescales=(2.,8.,32.,128.), delays=(1,8,24,48)):
    if mode not in ('multiscale','short'):
        raise ValueError('Unknown temporal basis mode')
    if (not sources or len(set(sources))!=len(sources) or not timescales or not delays
            or any(not math.isfinite(v) or v<1 for v in timescales)
            or any(type(d) is not int or d<1 for d in delays)):
        raise ValueError('Invalid source IDs or temporal constants')
    cfg=deepcopy(original)
    existing={n['id'] for n in cfg['neurons']}
    terminals={(p['neuron_id'],p['terminal_id']) for p in cfg['synaptic_points'] if p['type']=='presynaptic'}
    if not set(sources)<=existing or any((n,k.TERM) not in terminals for n in sources):
        raise ValueError('Source lacks a declared information terminal')
    basis=[];cursor=max(existing)+1
    for source in sources:
        for band,tau in enumerate(timescales):
            for delay in delays:
                lam=float(tau if mode=='multiscale' else timescales[0])
                node=k.neuron(cursor,lam=lam,c=2,eta_post=1e-7,eta_retro=1e-7,
                    delta_decay=.99,meta=dict(role='motor_history',graded_gain=1.,
                    bounded_plasticity=True,plasticity_rate_boost=0.,history_source=source,
                    history_band=band,history_delay=delay))
                node['params']['num_inputs']=2
                cfg['neurons'].append(node)
                cfg['synaptic_points'] += [k.syn(cursor,0,1.,delay,adapt=[0.,0.]),
                    k.syn(cursor,1,0.,1,adapt=[0.,0.]),k.term(cursor)]
                cfg['connections'].append(k.conn(source,cursor,0))
                basis.append(cursor);cursor+=1
    cfg['metadata']['temporal_basis']=dict(mode=mode,sources=list(sources),neurons=basis,
        timescales=list(timescales),delays=list(delays),
        normalization='One unit incoming weight per basis cell. No compensating gain for delay attenuation. '
        'Short and multiscale variants have identical neurons, edges, delays and birth weights; only lambda differs. '
        'Expansion increases source fan-out and native return paths; it is not an unchanged-conductance comparison to the original eight-input predictor.')
    return cfg,basis
