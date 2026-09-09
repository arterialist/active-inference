"""Neural disinhibition of a drive by two phase channels, using existing cells.

One matched drive relay P and two inhibitory comparators Q receive drive D.
Each phase R inhibits its Q. A corresponding output G receives P minus Q.
Ignoring adaptation and attenuation, P=relu(D), Q=relu(D-R), and
G=gain*relu(P-Q)=gain*min(D,R) for nonnegative inputs. This identity is an
operating hypothesis, not a host calculation. Every operation runs in existing
graded PAULA cells, with positive bounded plasticity and native return events.

All dendrites have distance one, decay .99, lambda one. The extra population
layer adds two neural ticks relative to a single additive relay. P has separate
terminals for its two consumers so it does not combine their return histories.
Weak adaptive mismatch can cause leakage; measure it, do not assume exact AND.

Yang, Murray & Wang (2016), doi:10.1038/ncomms12815 motivates disinhibitory
routing. This population-level cancellation circuit does not reproduce their
NMDA/GABA compartment model or claim a particular respiratory anatomy.
"""
from copy import deepcopy
import math

from ..learning.predictive_bridge import k


def append_phase_authorization(original, *, drive, phases, muscles, gain=3., motor_weight=8.):
    cfg = deepcopy(original); nodes = {n['id']:n for n in cfg['neurons']}
    if (len(phases)!=2 or len(muscles)!=2 or len(set(phases))!=2 or len(set(muscles))!=2
            or not set([drive,*phases,*muscles]) <= set(nodes)
            or not all(math.isfinite(x) and 0 < x <= 10 for x in (gain,motor_weight))):
        raise ValueError('Need a drive, two phases/muscles and bounded positive gains')
    terminals = {(p['neuron_id'],p['terminal_id']) for p in cfg['synaptic_points'] if p['type']=='presynaptic'}
    if any((nid,k.TERM) not in terminals for nid in [drive,*phases]):
        raise ValueError('Input lacks an information terminal')
    p,q0,q1,g0,g1 = range(max(nodes)+1,max(nodes)+6)
    added = [p,q0,q1,g0,g1]
    for nid,role in zip(added,('matched_drive','phase_inhibitor_0','phase_inhibitor_1','phase_output_0','phase_output_1')):
        node = k.neuron(nid,lam=1.,c=3,eta_post=1e-7,eta_retro=1e-7,delta_decay=.99,
            meta=dict(role=role,graded_gain=gain if nid in (g0,g1) else 1.,graded_S0=0.,
                      bounded_plasticity=True,plasticity_rate_boost=0.,retrograde_magnitude_error=True))
        node['params']['num_inputs'] = 2; cfg['neurons'].append(node)
        cfg['synaptic_points'] += [k.term(nid), k.syn(nid,0,1.,1,adapt=[0.,0.]),
                                  k.syn(nid,1,0. if nid==p else -1.,1,adapt=[0.,0.])]
    cfg['synaptic_points'].append(k.term(p,k.TERM+1))
    cfg['connections'].append(k.conn(drive,p,0)); ports = []
    for j,(phase,q,out,muscle) in enumerate(zip(phases,(q0,q1),(g0,g1),muscles)):
        cfg['connections'] += [k.conn(drive,q,0),k.conn(phase,q,1),
                               k.conn(p,out,0,stid=k.TERM+j),k.conn(q,out,1)]
        sid = nodes[muscle]['params']['num_inputs']
        if sid >= k.TERM: raise ValueError('Muscle input/terminal collision')
        nodes[muscle]['params']['num_inputs'] += 1
        cfg['synaptic_points'].append(k.syn(muscle,sid,motor_weight,1,adapt=[0.,0.]))
        cfg['connections'].append(k.conn(out,muscle,sid)); ports.append([out,muscle,sid])
    meta = dict(neurons=added,drive=drive,phases=list(phases),matched_drive=p,
                inhibitors=[q0,q1],outputs=[g0,g1],ports=ports,gain=gain,motor_weight=motor_weight,
                extra_latency_ticks=2,limits=__doc__)
    cfg['metadata']['phase_authorization'] = meta
    return cfg,meta
