"""Configure existing PAULA paths for delayed prediction verification.

No new neuron equation or signal queue is installed. Predictor-to-comparator
dendrites carry the requested extra delay, while predictor-to-motor paths stay
unchanged. Birth weights compensate the additional native distance attenuation.
Context amplitude and trace lifetime are explicit construction parameters.
An exponential context trace is not an exact record of past credit.
"""
from copy import deepcopy
import math


def configure_verification(original, groups, *, prediction_lag=64,
                           context_scale=.25, credit_tau=64.):
    if (type(prediction_lag) is not int or not 0 <= prediction_lag <= 128 or
            not math.isfinite(context_scale) or not 0 < context_scale <= 1 or
            not math.isfinite(credit_tau) or credit_tau <= 0):
        raise ValueError('Invalid timing/context configuration')
    if 'opponent_prediction' not in original.get('metadata',{}):
        raise ValueError('This construction requires declared opponent coupling')
    cfg=deepcopy(original)
    nodes={n['id']:n for n in cfg['neurons']}
    sensory=set(groups['vision']+groups['audio'])
    mixed=set(groups['mixed_0']+groups['mixed_1'])
    predictors=set(groups['prediction'])
    comparators=set(groups['error_positive']+groups['error_negative'])
    sources={(c['target_neuron'],c['target_synapse']):c['source_neuron'] for c in cfg['connections']}
    changed=[]
    for p in cfg['synaptic_points']:
        if p['type']!='postsynaptic':continue
        nid,sid=p['neuron_id'],p['synapse_id'];src=sources.get((nid,sid))
        if nid in mixed and src in sensory:
            p['u_i']['info']*=context_scale
        if nid in comparators:
            # Two learning branches participate in the signed residual.
            p['u_i']['info']*=.5
            if src in predictors:
                delta=nodes[nid]['params']['delta_decay']
                p['distance_to_hillock']+=prediction_lag
                p['u_i']['info']*=delta**(-prediction_lag)
                if abs(p['u_i']['info'])>10:
                    raise ValueError('Delay compensation exceeds bounded synapse range')
                changed.append([src,nid,sid,p['distance_to_hillock'],p['u_i']['info']])
    for nid in predictors:
        nodes[nid]['metadata']['prediction_tau_context']=float(credit_tau)
    if len(changed)!=2*len(comparators):
        raise ValueError('Unexpected predictor/comparator topology')
    cfg['metadata']['temporal_verification']=dict(prediction_lag=prediction_lag,
        context_scale=context_scale,credit_tau=credit_tau,comparison_gain=.5,delayed_paths=changed,
        limits='Nominal delay alignment, not learned delay inference. Exponential context history '
               'mixes recent inputs. Birth attenuation compensation drifts under native plasticity.')
    return cfg
