"""Append local mean-activity feedback using existing PAULA cells and synapses.

Each pool reads a disjoint group of principal cells with total birth input
conductance one and inhibits only that group. No stimulus identity, fitted
weight, external mean calculation or learning switch enters the network.
Pool size partitions feedback territory; it does not multiply target gain.

This is a circuit hypothesis motivated by inhibitory competition, not a
reproduction of Foldiak's anti-Hebbian rule or Agnes/Vogels co-dependent STDP.
The existing continuously plastic graded cell and magnitude-return extension
are used without a new neuronal equation. Default disabled composition is exact.
"""
from copy import deepcopy
import math

from simulations.paula_loader import ensure_paula_available
ensure_paula_available()
from paula_agent import ckit as k


def append_feedback_competition(original,banks,*,enabled=False,pools_per_bank=4,strength=1.):
    cfg=deepcopy(original)
    if not enabled:return cfg,{}
    if (type(pools_per_bank) is not int or pools_per_bank<1 or not banks or
            not math.isfinite(strength) or not 0<=strength<=1.):
        raise ValueError('Need banks, positive pool count and strength in 0..1')
    flat=[i for bank in banks.values() for i in bank]
    nodes={n['id']:n for n in cfg['neurons']}
    if len(set(flat))!=len(flat) or not set(flat)<=nodes.keys():
        raise ValueError('Feedback territories must be disjoint existing cells')
    terms={(p['neuron_id'],p['terminal_id']) for p in cfg['synaptic_points'] if p['type']=='presynaptic'}
    if any((i,k.TERM) not in terms for i in flat):raise ValueError('Principal terminal missing')
    if any(len(bank)<2*pools_per_bank for bank in banks.values()):
        raise ValueError('Each pool needs at least two principal cells')
    cursor=max(nodes)+1;groups={};territories=[]
    for name,bank in banks.items():
        groups['competition_'+name]=[]
        for pool in range(pools_per_bank):
            members=list(bank[pool::pools_per_bank]);nid=cursor;cursor+=1
            n=k.neuron(nid,lam=2,c=3,eta_post=1e-7,eta_retro=1e-7,delta_decay=.99,
                meta=dict(role='competition_'+name,graded_gain=1.,bounded_plasticity=True,
                          retrograde_magnitude_error=True))
            n['params']['num_inputs']=len(members)
            if len(members)>=k.TERM:raise ValueError('Pool exceeds port budget')
            cfg['neurons'].append(n);cfg['synaptic_points'].append(k.term(nid))
            groups['competition_'+name].append(nid)
            for sid,source in enumerate(members):
                target_port=nodes[source]['params']['num_inputs']
                if target_port>=k.TERM:raise ValueError('Principal cell exceeds port budget')
                nodes[source]['params']['num_inputs']+=1
                cfg['synaptic_points'].extend([k.syn(nid,sid,1./len(members),adapt=[0.,0.]),
                    k.syn(source,target_port,-strength,adapt=[0.,0.])])
                cfg['connections'].extend([k.conn(source,nid,sid),k.conn(nid,source,target_port)])
            territories.append(dict(pool=nid,members=members,input_weight=1./len(members),output_weight=-strength))
    cfg.setdefault('metadata',{})['feedback_competition']=dict(territories=territories,
        pools_per_bank=pools_per_bank,strength=strength,
        limits='Mean feedback within fixed local territories; no learned inhibitory tuning. '
               'All existing learning and return paths remain. Zero output strength is a '
               'wired control, not an unchanged baseline: added inputs still send returns.')
    return cfg,groups
