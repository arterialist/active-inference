"""Observe neural reference arrivals and independently reconstruct local learning."""
import math

import numpy as np

from . import context_organization as base
from . import crossed_av_world as world
from .magnitude_feedback_probe import ReleaseObserver


def record(net,arm,delay,features,groups,selected,video,audio,*,ticks=364,reverse=False):
    neurons=[net.network.neurons[n] for n in groups['prediction']]
    pools=[net.network.neurons[n] for role,ids in groups.items() if role.startswith('eligibility_reference_') for n in ids]
    initial=np.array([n.prediction_reference.copy() for n in neurons])
    meta=dict(reference_initial=initial,
        reference_ports=np.array([n.prediction_reference_ports for n in neurons]),
        reference_indices=np.array([n.prediction_reference_indices for n in neurons]),
        reference_sources=np.array([[n.synapse_sources[s] for s in n.prediction_reference_ports] for n in neurons]),
        reference_strength=np.array([n.prediction_reference_strength for n in neurons]),
        basal_eta=np.array([n.params.eta_post for n in neurons]),
        pool_ids=np.array([n.id for n in pools]),
        pool_sources=np.array([[n.synapse_sources[s] for s in range(n.params.num_inputs)] for n in pools]),
        pool_weight_initial=np.array([[n.postsynaptic_points[s].u_i.info for s in range(n.params.num_inputs)] for n in pools]))
    traces={k:[] for k in ('reference_arrivals','reference_trace','effective_eligibility','pool_weights')}
    original=net.run_tick
    def tick():
        result=original()
        for key,attr in [('reference_arrivals','prediction_reference_arrivals'),
                         ('reference_trace','prediction_reference'),('effective_eligibility','prediction_eligibility')]:
            traces[key].append(np.array([getattr(n,attr).copy() for n in neurons]))
        traces['pool_weights'].append(np.array([[n.postsynaptic_points[s].u_i.info
            for s in range(n.params.num_inputs)] for n in pools]))
        return result
    net.run_tick=tick
    try:
        with ReleaseObserver(net,groups['context'][0]) as observer:
            data=world.course(net,arm,features,groups,selected,video,audio,ticks=ticks,delay=delay,reverse=reverse)
    finally:
        net.run_tick=original
    return dict(**data,**observer.arrays(),**meta,**{k:np.asarray(v) for k,v in traces.items()})


def verify_learning(data):
    """Reconstruct from recorded arrivals, not the cell's reported eligibility."""
    if any(not np.isfinite(v).all() for v in data.values()):
        raise ValueError('Nonfinite reference record')
    q=data['weights_initial'].copy();x=data['context_initial'].copy();e=data['error_initial'].copy()
    z=data['reference_initial'].copy();idx=data['reference_indices'];g=data['reference_strength']
    basal=data['basal_eta'];d=math.exp(-1/64);de=math.exp(-1/4);residual=0.
    if np.any(basal<=0) or np.any(g<0) or np.any(g>1):raise ValueError('Invalid reference parameters')
    for t,a in enumerate(data['arrivals']):
        x=d*x+(1-d)*a;z=d*z+(1-d)*data['reference_arrivals'][t]
        elig=x-g[:,None]*np.take_along_axis(z,idx,axis=1)
        rate=basal*(1+499*abs(e)/(.01+abs(e)))
        q=np.clip(q+rate[:,None]*e[:,None]*elig,0.,1.)
        residual=max(residual,float(abs(q-data['weights'][t]).max()),
            float(abs(rate-data['eta'][t]).max()),float(abs(elig-data['effective_eligibility'][t]).max()),
            float(abs(z-data['reference_trace'][t]).max()),float(abs(e-data['errors'][t,:,0]).max()))
        e=de*e+(1-de)*data['errors'][t,:,1]
        residual=max(residual,float(abs(e-data['errors'][t,:,2]).max()))
    if not math.isfinite(residual) or residual>2e-12 or np.any(data['eta']<=0):
        raise ValueError(f'Reference learning audit failed: {residual}')
    # Each reference arrives from one declared source through a one-tick cleft.
    # Tick zero can contain inherited in-flight events; later ticks are checked.
    ids=list(data['neuron_ids']);terms=list(map(tuple,data['terminal_ids']))
    for j,sources in enumerate(data['reference_sources']):
        for r,(nid,tid) in enumerate(sources):
            out=data['cells'][:-1,ids.index(nid),base.FIELDS.index('O')]
            terminal=data['terminal_info'][:-1,terms.index((nid,tid))]
            actual=(out*terminal).astype(np.float32).astype(float)
            if not np.array_equal(actual,data['reference_arrivals'][1:,j,r]):
                raise ValueError('Reference arrival differs from neural release')
    return residual
