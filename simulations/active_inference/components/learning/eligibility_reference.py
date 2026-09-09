"""Neural population-reference pathways to zero-throughput learning receptors.

One graded cell pools each specified disjoint bank. Each prediction input is
assigned the reference of its actual source bank, never a stimulus or target
label. No somatic inhibition is added. All added forward/return paths adapt.
"""
from copy import deepcopy
import math

from simulations.paula_loader import ensure_paula_available
ensure_paula_available()
from paula_agent import ckit as k


def append_eligibility_reference(original,banks,predictors,*,enabled=False,strength=1.,rate_scale=1.):
    cfg=deepcopy(original)
    if not enabled:return cfg,{}
    if not math.isfinite(strength) or not 0<=strength<=1 or not math.isfinite(rate_scale) or not 0<rate_scale<=1:
        raise ValueError('Need contrast in 0..1 and positive rate scale at most one')
    nodes={n['id']:n for n in cfg['neurons']};flat=[s for bank in banks.values() for s in bank]
    if not flat or len(set(flat))!=len(flat) or not set(flat)<=nodes.keys():
        raise ValueError('Need disjoint nonempty existing source banks')
    if not predictors or len(set(predictors))!=len(predictors) or not set(predictors)<=nodes.keys():
        raise ValueError('Need unique existing predictors')
    terminals={(p['neuron_id'],p['terminal_id']) for p in cfg['synaptic_points'] if p['type']=='presynaptic'}
    if any((s,k.TERM) not in terminals for s in flat):raise ValueError('Missing source terminal')
    cursor=max(nodes)+1;groups={};reference_of={};pools=[]
    for name,bank in banks.items():
        if not bank or len(bank)>=k.TERM:raise ValueError('Invalid bank size')
        nid=cursor;cursor+=1
        n=k.neuron(nid,lam=2,c=3,eta_post=1e-7,eta_retro=1e-7,delta_decay=.99,
            meta=dict(role='eligibility_reference_'+name,graded_gain=1.,bounded_plasticity=True,
                      retrograde_magnitude_error=True))
        n['params']['num_inputs']=len(bank);cfg['neurons'].append(n)
        cfg['synaptic_points'].append(k.term(nid));groups['eligibility_reference_'+name]=[nid];pools.append(nid)
        # Compensate the declared one-tick dendritic attenuation at birth.
        for sid,source in enumerate(bank):
            cfg['synaptic_points'].append(k.syn(nid,sid,1./(.99*len(bank)),adapt=[0.,0.]))
            cfg['connections'].append(k.conn(source,nid,sid));reference_of[source]=nid
    for nid in predictors:
        node=nodes[nid];ports=node['metadata']['prediction_ports'];lookup={}
        for c in cfg['connections']:
            if c['target_neuron']==nid and c['target_synapse'] in ports:
                if c['target_synapse'] in lookup:raise ValueError('Ambiguous prediction source')
                lookup[c['target_synapse']]=c['source_neuron']
        if set(lookup)!=set(ports) or not set(lookup.values())<=reference_of.keys():
            raise ValueError('Selected input sources must belong to declared banks')
        ref_ports={}
        for pool in pools:
            sid=node['params']['num_inputs'];node['params']['num_inputs']+=1
            if sid>=k.TERM:raise ValueError('Reference exceeds port budget')
            cfg['synaptic_points'].append(k.syn(nid,sid,0.,adapt=[0.,0.]))
            cfg['connections'].append(k.conn(pool,nid,sid));ref_ports[pool]=sid
        node['metadata']['prediction_reference_map']=[[s,ref_ports[reference_of[lookup[s]]]] for s in ports]
        node['metadata']['prediction_reference_strength']=strength
        node['params']['eta_post']*=rate_scale
    cfg.setdefault('metadata',{})['eligibility_reference']=dict(banks=banks,pools=pools,strength=strength,
        rate_scale=rate_scale,limits='Neural delayed spatial reference, not statistical covariance or learned supervisor. '
        'Reference pathways preserve native returns; no direct somatic throughput.')
    return cfg,groups
