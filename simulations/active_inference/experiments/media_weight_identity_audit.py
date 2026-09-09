"""Independently check weight interventions and locate their full-tick effects.

The local-current check covers selected dendritic input potentials, before
intra-neuron delay/attenuation. It is not a reconstruction of total soma current.
No output statistic is an association acceptance criterion.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .association_balance_audit import checked
from .association_route_probe import digest
from .eligibility_media_audit import verify_ledger
from .graded_media_audit import verify_rates
from .media_order_audit import load_state, verify_protocol, reference_projections, temporal_summary, crossed_effect
from .media_order_control import ledger_start
from .media_drive_audit import ReceptorAudit, physical_values


def verify_local_current(arrivals, weights, initial, observed):
    """Native weak-scalar multiplication uses the float32 input buffer here."""
    arrivals, weights, initial, observed = map(np.asarray, (arrivals, weights, initial, observed))
    if (arrivals.shape != weights.shape or arrivals.shape != observed.shape
            or arrivals.ndim != 2 or initial.shape != (arrivals.shape[1],)
            or any(not np.isfinite(a).all() for a in (arrivals, weights, initial, observed))):
        raise ValueError('Invalid local-current ledger')
    before = np.concatenate([initial[None], weights[:-1]], axis=0)
    expected = (arrivals.astype(np.float32)*before.astype(np.float32)).astype(float)
    expected[arrivals <= 0] = 0.
    if not np.array_equal(expected, observed):
        raise ValueError('Selected input potential differs from independent reconstruction')
    return expected


def first_difference(a, b):
    d = np.asarray(b)-np.asarray(a)
    by_tick = np.count_nonzero(d.reshape(len(d), -1), axis=1)
    times = np.flatnonzero(by_tick)
    return dict(first=int(times[0]) if len(times) else None,
                last=int(times[-1]) if len(times) else None,
                different_values_by_tick=by_tick.tolist(), maximum=float(abs(d).max()))


def conditional_cue_effects(samples):
    """Weight-placement effects need not dominate an existing sensory bias."""
    names = ('intact','initial','cycle1','cycle2','cycle3')
    if set(samples) != {(name,cue) for name in names for cue in (0,1)}:
        raise ValueError('Incomplete cue-by-weight intervention')
    shape = samples['intact',0].shape
    if any(v.shape != shape or not np.isfinite(v).all() for v in samples.values()):
        raise ValueError('Invalid cue trajectories')
    d = {name: samples[name,0]-samples[name,1] for name in names}
    return dict(selected_adaptation=d['intact']-d['initial'],
                placement=d['intact']-sum(d[f'cycle{i}'] for i in (1,2,3))/3)


def quiet_receptor_state(cfg, groups, parent):
    """An auditor's initial conditions, not an executable network restore."""
    receptor = ReceptorAudit(cfg,groups,graded_gain=.25)
    states = [parent['neurons'][str(n)] for n in receptor.ids]
    if any(n['S'] != 0 or n['O'] != 0 or any(n['M']) or n['dendritic_queue']
           or n['t_last_fire'] is not None for n in states):
        raise ValueError('This sensory audit requires the recorded quiet graded checkpoint')
    receptor.q = np.array([n['synapses']['0'][0] for n in states])
    receptor.f = np.array([n['F_avg'] for n in states])
    receptor.tick = parent['tick']
    return receptor


def compare(recordings, output):
    roots = [Path(p).resolve() for p in recordings]
    manifests = [json.loads((p/'manifest.json').read_text()) for p in roots]
    courses = [json.loads((Path(m['source'])/'manifest.json').read_text()) for m in manifests]
    verify_protocol({(m['mapping'],m['order']): m for m in courses})
    if len(roots) != 4:
        raise ValueError('Requires four unique histories')
    reference_config = (Path(manifests[0]['source'])/'config.json').read_bytes()
    cfg = json.loads(reference_config); base = courses[0]
    birth = load_state(Path(manifests[0]['source'])/'initial-state.json.gz')
    rows, differences, projections, traces, effect_rows = {}, {}, {}, {}, {}
    maximum = 0.
    for root, m, cm in zip(roots, manifests, courses, strict=True):
        source = Path(m['source']); s = json.loads((root/'summary.json').read_text())
        if (source/'config.json').read_bytes() != reference_config or load_state(source/'initial-state.json.gz') != birth:
            raise ValueError('Different birth state or configuration')
        if any(cm[f] != base[f] for f in ('groups','selected_ports','physical_sources','seed','checkpoints')):
            raise ValueError('Different preparation')
        if any(m[f] != cm[f] for f in ('groups','selected_ports','mapping','order','seed')):
            raise ValueError('Intervention metadata disagrees with its acquisition course')
        if any(digest(p) != h for p,h in m['source_hashes'].items()):
            raise ValueError('Changed source')
        if not all(s[k] for k in ('exact_acquisition','intact_probes_exact','parent_unchanged')):
            raise ValueError('Failed runtime control')
        ports = m['selected_ports']; parent = load_state(source/'checkpoint-16-state.json.gz')
        sensory_parent = quiet_receptor_state(cfg,m['groups'],parent)
        features = []
        for clip in (0,1):
            path = next(Path(p) for p in cm['physical_sources'] if Path(p).name == f'sensory-{clip}.npz')
            with np.load(path) as z:
                features.append({k:z[k] for k in z.files})
        learned = np.array([parent['neurons'][str(n)]['synapses'][str(sid)][0] for n,sid,_ in ports])
        initial = np.array([birth['neurons'][str(n)]['synapses'][str(sid)][0] for n,sid,_ in ports])
        with np.load(root/'intervention-weights.npz') as z:
            qsets = {name: z[name] for name in ('intact','initial','cycle1','cycle2','cycle3')}
            if not np.array_equal(qsets['intact'], learned) or not np.array_equal(qsets['initial'], initial):
                raise ValueError('Wrong learned or reset weights')
            for shift in (1,2,3):
                expected = np.arange(len(ports))
                for nid in sorted({n for n,_,_ in ports}):
                    indices = [i for i,(n,_,_) in enumerate(ports) if n == nid]
                    if len(indices) != 4:
                        raise ValueError('Wrong target input count')
                    for j,i in enumerate(indices):
                        expected[i] = indices[(j-shift)%4]
                if not np.array_equal(z[f'cycle{shift}_permutation'],expected) or not np.array_equal(qsets[f'cycle{shift}'], learned[expected]):
                    raise ValueError('Changed cyclic intervention')
        samples = {}; endpoints = {}
        ns = [n['id'] for n in cfg['neurons']]
        order = [(n, p['synapse_id']) for n in ns for p in cfg['synaptic_points'] if p['type']=='postsynaptic' and p['neuron_id']==n]
        port_index = np.array([order.index((n,sid)) for n,sid,_ in ports])
        baseline_weights = np.array([parent['neurons'][str(n)]['synapses'][str(sid)][0] for n,sid in order])
        for p in s['probes']:
            condition, cue = p['condition'], p['cue']; key = condition, cue
            if key in samples or condition not in qsets or cue not in (None,0,1):
                raise ValueError('Unknown or duplicated branch')
            if p['trial'] != dict(start=parent['tick'],stop=parent['tick']+300,visual_clip=cue,audio_clip=None):
                raise ValueError('Changed probe input declaration')
            with np.load(checked(root,p)) as z:
                d = {k:z[k] for k in z.files}
            state = deepcopy(parent)
            expected_weights = baseline_weights.copy(); expected_weights[port_index] = qsets[condition]
            if not np.array_equal(d['incoming_info_before'],expected_weights):
                raise ValueError('Changed nonselected initial weights')
            for (n,sid,_),q in zip(ports,qsets[condition],strict=True):
                state['neurons'][str(n)]['synapses'][str(sid)][0] = float(q)
            start = ledger_start(state,cfg,ports)
            result = verify_ledger(d,cfg,ports,*start[:4]); maximum = max(maximum,result[4])
            verify_rates(d['cells'],start[4],cfg,ports)
            sensory = deepcopy(sensory_parent)
            sensory.check(d['cells'],physical_values(features,p['trial']))
            if not np.allclose(d['incoming_info_after'][:384:2],sensory.q,atol=2e-12,rtol=0):
                raise ValueError('Physical receptor learning differs')
            verify_local_current(d['arrivals'],d['weights'],qsets[condition],d['selected_local_current'])
            if not np.array_equal(d['incoming_info_after'][port_index],result[0]):
                raise ValueError('Wrong selected weight endpoint')
            samples[key] = d
        if set(samples) != {(k,c) for k in qsets for c in (None,0,1)}:
            raise ValueError('Missing branch')
        history = f"{m['mapping']}/{m['order']}"
        for condition in qsets:
            for role in ('tactile_core','upper_core','mismatch_candidate'):
                ids = np.array(m['groups'][role])-1
                for cue in (None,0,1):
                    d = samples[condition,cue]; x = d['cells'][:,ids,1]
                    blank = samples[condition,None]['cells'][:,ids,1]
                    spikes = (x>0).sum(axis=1); times = np.flatnonzero(spikes)
                    name = f'{history}/{condition}/{cue}/{role}'
                    traces[name+'/output'] = x
                    traces[name+'/cue_minus_blank'] = x-blank
                    rows[name] = dict(events=int(spikes.sum()),after32=int(spikes[32:].sum()),
                        first=int(times[0]) if len(times) else None,last=int(times[-1]) if len(times) else None)
                    if condition != 'intact':
                        control = samples['intact',cue]
                        for label,a,b in [('S',control['cells'][:,ids,0],d['cells'][:,ids,0]),
                                          ('O',control['cells'][:,ids,1],d['cells'][:,ids,1]),
                                          ('M0',control['cells'][:,ids,3],d['cells'][:,ids,3])]:
                            differences[name+'/'+label] = first_difference(a,b)
                cue_diff = samples[condition,0]['cells'][:,ids,1]-samples[condition,1]['cells'][:,ids,1]
                refs = {}
                old_source = Path(cm['source'])
                for basis in ('original','current'):
                    refs[basis] = []
                    for clip in (0,1):
                        path = old_source/f'initial-graded-audio-{clip}.npz' if basis=='original' else source/f'checkpoint-16-context-vNone-a{clip}.npz'
                        with np.load(path) as z:
                            refs[basis].append((z['cells'][32 if basis=='original' else 96:,ids,1]>0).mean(axis=0))
                    for channel,value in reference_projections(cue_diff,refs[basis]).items():
                        if value is not None:
                            name = f'{history}/{condition}/{role}/{basis}/{channel}'
                            projections[name] = temporal_summary(value); traces[name] = value
        for condition in ('initial','cycle1','cycle2','cycle3'):
            for cue in (None,0,1):
                a,b = samples['intact',cue],samples[condition,cue]
                name = f'{history}/{condition}/{cue}'
                for label in ('arrivals','selected_local_current','terminals'):
                    differences[name+'/'+label] = first_difference(a[label],b[label])
                terminal_indices = [i for i,(nid,_) in enumerate(s['terminal_order']) if nid in m['groups']['visual_core']]
                differences[name+'/visual_terminal_info'] = first_difference(
                    a['terminals'][:,terminal_indices,0],b['terminals'][:,terminal_indices,0])
                ids = np.array(m['groups']['visual_core'])-1
                for label,column in (('visual_S',0),('visual_O',1)):
                    differences[name+'/'+label] = first_difference(a['cells'][:,ids,column],b['cells'][:,ids,column])
    # Keep the same birth-state sound observer across assignments and orders.
    original = Path(base['source'])
    for role in ('tactile_core','upper_core','mismatch_candidate'):
        ids = np.array(base['groups'][role])-1
        reference = []
        for cue in (0,1):
            with np.load(original/f'initial-graded-audio-{cue}.npz') as z:
                reference.append((z['cells'][32:,ids,1]>0).mean(axis=0))
        effects = {}
        for mapping,order in (('paired',0),('swapped',0),('paired',1),('swapped',1)):
            history = f'{mapping}/{order}'
            values = {(condition,cue): traces[f'{history}/{condition}/{cue}/{role}/output']
                      for condition in ('intact','initial','cycle1','cycle2','cycle3') for cue in (0,1)}
            effects[mapping,order] = conditional_cue_effects(values)
            for kind,value in effects[mapping,order].items():
                name = f'{history}/{role}/{kind}'
                traces[name] = value
                for channel,p in reference_projections(value,reference).items():
                    if p is not None:
                        traces[name+'/'+channel] = p
                        effect_rows[name+'/'+channel] = temporal_summary(p)
        for kind in ('selected_adaptation','placement'):
            main, interaction = crossed_effect(*(effects[m,o][kind] for m,o in (
                ('paired',0),('swapped',0),('paired',1),('swapped',1))))
            traces[f'{role}/{kind}/assignment'] = main
            traces[f'{role}/{kind}/assignment_by_order'] = interaction
            for label,value in (('assignment',main),('assignment_by_order',interaction)):
                for channel,p in reference_projections(value,reference).items():
                    if p is not None:
                        name = f'{role}/{kind}/{label}/{channel}'
                        traces[name] = p; effect_rows[name] = temporal_summary(p)
    output = Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'trajectories.npz',**traces)
    result = dict(interventions_valid=True, selected_local_currents_exact=True, graded_sensory_trajectories_valid=True,
        max_selected_update_residual=maximum, rows=rows, differences=differences, projections=projections,
        conditional_cue_effects=effect_rows,
        recordings=[str(p) for p in roots],
        audit_source_sha256=digest(__file__),
        limits='Independent selected-current/learning reconstruction; total soma current and all hidden state are not reconstructed. '
               'Same fast state with changed weights is a diagnostic intervention, not a natural learning trajectory. '
               'One graph seed. Cyclic controls preserve weights, not input-weight correlations. '
               'Firing, sensitivity to a shuffle and sound-reference projections do not individually prove associative recall. '
               'Placement effects include birth-weight placement; an initial-weight shuffle control is not present. '
               'Assignment interactions can contain history-dependent gain. Total-preference reversal is not required '
               'for a smaller learned contribution to exist beneath a sensory bias.')
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,nargs=4,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();compare(a.recording,a.output)
