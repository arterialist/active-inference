"""Add organ feedback to an acquired PAULA motor rhythm without replacing it.

Three existing energy-feedback cells plus an oxygen afferent, graded deficit
comparator and two phase-coincidence relays. Fixed signed dendrites specify the
reference and motor authority. No new neuron equation or host action rule.
The nominal half-reserve wiring reads an acquired tonic terminal, so its actual
reference can move with upstream adaptation. It is not an anatomical reconstruction.
All new cells retain positive bounded local and native retrograde adaptation.
"""
from copy import deepcopy

from ..learning.predictive_bridge import k
from .energy_feedback import append_energy_feedback
from ..motor.sensory_correction import install_on_runtime
from ...core.external_input_state import synchronize_quiescent_external_inputs


MODES = ('feedback', 'oxygen-cut', 'energy-cut', 'tonic')


def append_ventilation_feedback(original, groups, *, mode='feedback'):
    if mode not in MODES or len(groups['muscle']) != 2 or len(groups['cpg']) != 4:
        raise ValueError('Need two muscles, four rhythm cells and a declared mode')
    cfg, energy = append_energy_feedback(original, {'muscles':groups['muscle']},
                                         connected=mode!='energy-cut')
    nodes={n['id']:n for n in cfg['neurons']}
    oxygen, deficit, plus, minus = range(max(nodes)+1,max(nodes)+5)
    added=[oxygen,deficit,plus,minus]
    for nid,role,lam,gain,offset in ((oxygen,'oxygen_afferent',4.,1.,0.),
            (deficit,'oxygen_deficit_comparator',16.,1.,0.),
            (plus,'inspiratory_phase_recruitment',1.,3.,1.),
            (minus,'expiratory_phase_recruitment',1.,3.,1.)):
        node=k.neuron(nid,lam=lam,c=3,eta_post=1e-7,eta_retro=1e-7,delta_decay=.99,
            meta=dict(role=role,graded_gain=gain,graded_S0=offset,bounded_plasticity=True,
                      plasticity_rate_boost=0.,retrograde_magnitude_error=True))
        node['params']['num_inputs']=2;cfg['neurons'].append(node)
        cfg['synaptic_points'].append(k.term(nid))
    all_new=energy['neurons']+added
    for node in cfg['neurons']:
        if node['id'] in all_new: node['metadata']['retrograde_magnitude_error']=True
    weights={oxygen:(1.,0.),deficit:((1/3 if mode=='tonic' else 1.),(0. if mode=='tonic' else -2.)),
             plus:(1.,1.),minus:(1.,1.)}
    for nid,pair in weights.items():
        cfg['synaptic_points'] += [k.syn(nid,s,w,1,adapt=[0.,0.]) for s,w in enumerate(pair)]
    cfg['external_inputs'].append(k.ext(oxygen,0))
    cfg['connections'] += [k.conn(groups['context'][0],deficit,0),k.conn(oxygen,deficit,1)]
    ports=[]
    for relay,phase,muscle in zip((plus,minus),(groups['cpg'][0],groups['cpg'][2]),groups['muscle']):
        cfg['connections'] += [k.conn(phase,relay,0),k.conn(deficit,relay,1)]
        sid=nodes[muscle]['params']['num_inputs']
        if sid>=k.TERM: raise ValueError('Muscle port exceeds terminal budget')
        nodes[muscle]['params']['num_inputs']+=1
        cfg['synaptic_points'].append(k.syn(muscle,sid,0. if mode=='oxygen-cut' else 8.,1,adapt=[0.,0.]))
        cfg['connections'].append(k.conn(relay,muscle,sid));ports.append([relay,muscle,sid])
    meta=dict(mode=mode,energy=energy,oxygen=oxygen,deficit=deficit,relays=[plus,minus],
              neurons=all_new,ports=ports,sensory_ids=[energy['energy'],energy['deficit'],oxygen],
              sensory_delay_ticks=64,
              rule='Deficit is rectified tonic minus twice oxygen afferent. Each relay rectifies '
                   'rhythm plus deficit minus one, with release gain three. Relay-to-muscle weight eight.',
              limits='Specified comparison weights and gains; the acquired tonic terminal changes '
                     'the effective reference. No learned interoceptive preference is claimed. '
                     'Tonic control uses one-third reference without oxygen inhibition, approximately '
                     'one extra rhythm pulse. It is not exact doubling of total muscle output. '
                     'Zero-q controls retain edges and native plastic returns.')
    cfg['metadata']['ventilation_feedback']=meta
    return cfg,meta


def install_ventilation_feedback(net,fresh,original,cfg):
    """Construction-only addition preserving acquired cells and pending events."""
    from types import SimpleNamespace
    old_ext=original['external_inputs'];old_ids={n['id'] for n in original['neurons']}
    if cfg['external_inputs'][:len(old_ext)]!=old_ext:
        raise ValueError('Original sensory interface changed')
    extra=set(fresh.network.external_inputs)-set(net.network.external_inputs)
    if len(extra)!=3 or any(n in old_ids for n,_ in extra):
        raise ValueError('Expected exactly three new organ afferents')
    synchronize_quiescent_external_inputs(net.network)
    install_on_runtime(SimpleNamespace(network=net),fresh,original,dict(cfg,external_inputs=old_ext))
    for key in sorted(extra):
        net.network.external_inputs[key]=deepcopy(fresh.network.external_inputs[key])
        net.network.free_synapses.append(key)
    net.network._ext_vec=None
