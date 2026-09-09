"""Neural above/below-population contrast, with matched relay delays.

All inputs are information releases from declared PAULA neurons. One graded
pool receives their mean conductance; matched graded relays carry each channel.
Two rectified cells per channel subtract pool from relay and relay from pool.
No host-computed mean, sensory identity, learned embedding or external tonic
drive enters this module. All postsynaptic and retrograde rates stay positive.

This is phenomenological surround inhibition, not a reconstructed retina.
The surround is global over the supplied list, not a spatially local retinal
receptive field. Native plasticity/return paths can alter that approximation.
Pitkow & Meister 2012 (doi:10.1038/nn.3064) motivates nonlinear decorrelation,
not these particular cells, weights, or timescales.
"""
from copy import deepcopy

from simulations.paula_loader import ensure_paula_available

ensure_paula_available()
from paula_agent import ckit as k


def append_population_contrast(original, source_ids, *, relay_tau=4., contrast_tau=4.,
                               eta_post=1e-7, eta_retro=1e-7):
    if (len(source_ids)<2 or len(set(source_ids))!=len(source_ids) or
            len(source_ids)>=k.TERM or min(relay_tau,contrast_tau)<1 or
            eta_post<=0 or eta_retro<=0):
        raise ValueError('Invalid contrast population or dynamics')
    cfg=deepcopy(original); existing={n['id'] for n in cfg['neurons']}
    terminals={(p['neuron_id'],p['terminal_id']) for p in cfg['synaptic_points'] if p['type']=='presynaptic'}
    if not set(source_ids)<=existing or any((n,k.TERM) not in terminals for n in source_ids):
        raise ValueError('Input channel lacks its declared PAULA terminal')
    cursor=max(existing)+1; groups={}
    for role,count in (('contrast_pool',1),('contrast_relay',len(source_ids)),
                       ('contrast_above',len(source_ids)),('contrast_below',len(source_ids))):
        groups[role]=list(range(cursor,cursor+count));cursor+=count
    ports={n:0 for ids in groups.values() for n in ids}; nodes={};edges=[]
    for role,ids in groups.items():
        for nid in ids:
            n=k.neuron(nid,lam=relay_tau if role in ('contrast_pool','contrast_relay') else contrast_tau,
                c=3,eta_post=eta_post,eta_retro=eta_retro,delta_decay=.99,
                meta={'role':role,'graded_gain':1.,'bounded_plasticity':True,'plasticity_rate_boost':0.})
            nodes[nid]=n;cfg['neurons'].append(n);cfg['synaptic_points'].append(k.term(nid))

    def wire(src,tgt,weight,family):
        sid=ports[tgt];ports[tgt]+=1
        cfg['synaptic_points'].append(k.syn(tgt,sid,weight,1,adapt=[0.,0.]))
        cfg['connections'].append(k.conn(src,tgt,sid));edges.append([src,tgt,sid,family,True])

    pool=groups['contrast_pool'][0]
    for source,relay,above,below in zip(source_ids,groups['contrast_relay'],groups['contrast_above'],groups['contrast_below']):
        wire(source,pool,1/len(source_ids),'surround_observation')
        wire(source,relay,1.,'matched_channel_relay')
        wire(relay,above,1.,'channel_excitation');wire(pool,above,-1.,'surround_inhibition')
        wire(relay,below,-1.,'channel_inhibition');wire(pool,below,1.,'surround_excitation')
    for nid,n in nodes.items():
        while ports[nid]<2:
            sid=ports[nid];ports[nid]+=1
            cfg['synaptic_points'].append(k.syn(nid,sid,0.,1,adapt=[0.,0.]))
        n['params']['num_inputs']=ports[nid]
    cfg['metadata']['population_contrast']=dict(source_ids=list(source_ids),relay_tau=relay_tau,
        contrast_tau=contrast_tau,eta_post=eta_post,eta_retro=eta_retro,
        limitation='Global neural surround, not anatomical retinal reconstruction')
    return cfg,groups,edges
