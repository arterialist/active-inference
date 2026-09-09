"""Audit fixed-stimulus, crossed-history neural mismatch and rate responses."""
import argparse
import json
from pathlib import Path

import numpy as np

from .association_route_probe import digest
from .composition_probe import encode
from .predictive_bridge_probe import audit_record
from .predictive_bridge_audit import windows


def expected_inputs(features, groups, trial):
    ids = groups['vision']+groups['touch']; index = {n:i for i,n in enumerate(ids)}
    result = np.zeros((trial['stop']-trial['start'],len(ids)))
    for t,row in enumerate(result):
        for key,role,field in (('visual_clip','vision','visual'),('audio_clip','touch','auditory')):
            clip = trial[key]
            if clip is None:
                continue
            values = features[clip][field][t%int(features[clip]['ticks'])]
            for n,v in zip(groups[role],values):
                if t%4 == n%4 and v > 0:
                    row[index[n]] = 2*v
    return result


def familiarity_benefit(paired, swapped, visual, audio):
    """Positive means smaller response with weights from the familiar history.

    Inputs are the SAME physical event. No difference between audio recordings
    is allowed to stand in for expectation. Applies to full per-channel arrays.
    """
    return swapped-paired if visual == audio else paired-swapped


def contextual_weight_mask(config, bridge):
    selected = {(n['id'],p) for n in config['neurons'] if n['id'] in bridge['prediction']
                for p in n['metadata']['prediction_ports']}
    points = {}
    for p in config['synaptic_points']:
        if p['type'] == 'postsynaptic':
            points.setdefault(p['neuron_id'],[]).append(p['synapse_id'])
    return np.array([(n['id'],p) in selected for n in config['neurons'] for p in points[n['id']]])


def run(paired, swapped, output):
    records, sources, signatures, common_starts = {}, [], [], {}
    maximum = 0.
    for mapping,root in (('paired',Path(paired).resolve()),('swapped',Path(swapped).resolve())):
        m = json.loads((root/'manifest.json').read_text())
        parent = Path(m['parent']); cfg = json.loads((parent/'config.json').read_text())
        if m['mapping'] != mapping:
            raise ValueError('Unexpected acquisition mapping')
        signatures.append((digest(parent/'config.json'),m['physical_sources'],m['order']))
        for p,h in m['source_hashes'].items():
            if digest(p) != h:
                raise ValueError('Changed source: '+p)
        features = []
        for p in sorted(m['physical_sources']):
            with np.load(p) as z:
                features.append({k:z[k] for k in z.files})
        done = json.loads((root/'completion.json').read_text())
        keys = [(e['condition'],e['trial']['visual_clip'],e['trial']['audio_clip']) for e in done['entries']]
        expected = {(c,v,a) for c in ('learned','shuffled') for v in (0,1) for a in (None,0,1)}
        if len(keys) != 12 or set(keys) != expected:
            raise ValueError('Incomplete or duplicate condition')
        index = {n['id']:i for i,n in enumerate(cfg['neurons'])}
        pos = [index[n] for n in m['bridge']['error_positive']]
        neg = [index[n] for n in m['bridge']['error_negative']]
        selected = contextual_weight_mask(cfg,m['bridge'])
        for e in done['entries']:
            p = root/e['file']
            if digest(p) != e['sha256']:
                raise ValueError('Changed raw data')
            with np.load(p) as z:
                data = {k:z[k] for k in z.files}
            trial,c = e['trial'],e['condition']; v,a = trial['visual_clip'],trial['audio_clip']
            length = 32 if a is None else 300
            if trial['start'] != 0 or trial['stop'] != length or len(data['cells']) != length:
                raise ValueError('Unexpected physical timeline')
            if not np.array_equal(expected_inputs(features,m['groups'],trial),data['external_information']):
                raise ValueError('Applied physical input differs')
            maximum = max(maximum,audit_record(data,cfg,m['bridge']))
            if data['start_incoming_info'].shape != selected.shape:
                raise ValueError('Incoming weight column identity mismatch')
            if not np.array_equal(data['start_incoming_info'][selected],data['start_weights'].ravel()):
                raise ValueError('Selected weight column identity mismatch')
            unchanged = data['start_incoming_info'][~selected]
            if 'nonselected_weights' in common_starts and not np.array_equal(unchanged,common_starts['nonselected_weights']):
                raise ValueError('Nonselected initial weights differ across histories')
            common_starts['nonselected_weights'] = unchanged
            with np.load(Path(m['source'])/f'{c}-cue{v}.npz') as golden:
                for k,value in data.items():
                    if k.startswith('start_'):
                        if not np.array_equal(value,golden[k]):
                            raise ValueError('Initial state differs from weight-only transplant')
                        if k not in ('start_weights','start_incoming_info'):
                            if k in common_starts and not np.array_equal(value,common_starts[k]):
                                raise ValueError('Non-weight initial state differs across histories')
                            common_starts[k] = value
                    elif a is None and k not in ('end_incoming_info','external_information'):
                        if not np.array_equal(value,golden[k][:32]):
                            raise ValueError('Replay control differs')
            sources.append(dict(file=str(p),sha256=e['sha256']))
            if a is not None:
                # These are outputs of the actual opponent neurons, not a
                # reconstructed host prediction fed back into the simulation.
                positive = data['cells'][:,pos,1]; negative = data['cells'][:,neg,1]
                records[mapping,c,v,a] = dict(absolute_error=positive+negative,
                    signed_error=positive-negative,rate=data['eta'],error_used=data['error_used'])
    if signatures[0] != signatures[1]:
        raise ValueError('Unmatched graph, media or order')
    arrays,report = {},{}
    for v in (0,1):
        for a in (0,1):
            tag = f'visual{v}_audio{a}'; report[tag] = {}
            for measure in ('absolute_error','signed_error','rate','error_used'):
                for c in ('learned','shuffled'):
                    for mapping in ('paired','swapped'):
                        arrays[f'{tag}_{mapping}_{c}_{measure}'] = records[mapping,c,v,a][measure]
                    benefit = familiarity_benefit(records['paired',c,v,a][measure],records['swapped',c,v,a][measure],v,a)
                    arrays[f'{tag}_{c}_{measure}_benefit'] = benefit
                    if measure in ('absolute_error','rate'):
                        report[tag][f'{c}_{measure}'] = windows(benefit.mean(axis=1))
                effect = arrays[f'{tag}_learned_{measure}_benefit']-arrays[f'{tag}_shuffled_{measure}_benefit']
                arrays[f'{tag}_{measure}_placement_benefit'] = effect
                if measure in ('absolute_error','rate'):
                    report[tag][f'placement_{measure}'] = windows(effect.mean(axis=1))
    output = Path(output).resolve(); output.mkdir(exist_ok=False)
    np.savez_compressed(output/'effects-per-tick.npz',**arrays)
    result = dict(order=signatures[0][2],sources=sources,max_equation_residual=maximum,results=report,
        effects_sha256=digest(output/'effects-per-tick.npz'),
        interpretation='Positive benefit means less actual opponent error or lower local rate with familiar weights for the SAME input. '
                       'Every physical pair is reported; a positive aggregate must not conceal negative cases. '
                       'Rates remain positive and weights continue adapting. One graph, two familiar exemplars, no action or consciousness claim.')
    (output/'summary.json').write_text(encode(result)+'\n')
    print(encode(dict(order=result['order'],max_equation_residual=maximum,cases=4)),flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('paired','swapped','output'):
        p.add_argument('--'+key,type=Path,required=True)
    run(**vars(p.parse_args()))
