"""Verify retrospectively located transient memories in actual PAULA/body runs.

Selection is diagnostic, not acceptance: choose the first acquired weight vector
whose fixed-interface minimum exceeds .001 on all pairings at probe ticks 16..63.
Compare it with the factual weights 32 acquisition ticks later. Both are placed
in the original birth neural state and resting body. No optimizer, new learning
rule, frozen rate or selected-weight edit is used. All four pairings are probed.
This does not claim the acquiring agent reads that memory correctly in its
current activity/body state, or can retain or find the selected state itself.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from . import context_organization as base
from .crossed_av_continuation import expression, isolated_rng
from .crossed_av_continuation_analysis import trajectory
from .opponent_context_analysis import read_record
from .opponent_context import verify_afferents
from .temporal_verification import verify_learning


def select_first(rows, factors, threshold=.001):
    for row in rows:
        bounds = factors[row['key'] + '_bounds']
        passing = np.flatnonzero(bounds[:, :, 0].min(axis=1) > threshold)
        if len(passing):
            tick = int(passing[0])
            if tick + 32 >= row['ticks']:
                raise ValueError('First candidate has no same-episode 32-tick comparison')
            return row, tick
    raise ValueError('No conditional transient candidate')


def install_recorded(net, data, weights):
    """Transfer by actual neuron/source identity, never serialized group order."""
    weights=np.asarray(weights)
    if (weights.shape!=data['context_source_ids'].shape or
            weights.shape[0]!=len(data['prediction_ids']) or
            not np.isfinite(weights).all() or np.any(weights<0) or np.any(weights>1)):
        raise ValueError('Invalid recorded selected weights')
    for j, nid in enumerate(data['prediction_ids']):
        neuron = net.network.neurons[int(nid)]
        if len(neuron.prediction_ports)!=weights.shape[1]:raise ValueError('Selected port count differs')
        for sid, source, value in zip(neuron.prediction_ports, data['context_source_ids'][j], weights[j]):
            if neuron.synapse_sources[sid][0] != source:
                raise ValueError('Selected weight identity differs')
            neuron.postsynaptic_points[sid].u_i.info = float(value)


def run(credit_root, seed, output):
    credit_root, output = Path(credit_root).resolve(), Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free < 3*1024**3:raise OSError('Need 3 GiB reserve')
    s = json.loads((credit_root/'summary.json').read_text())
    if base.digest(Path(__file__).with_name('crossed_av_credit.py')) != s['analysis_source_sha256']:
        raise ValueError('Credit analyzer changed')
    if base.digest(s['audit_summary']) != s['audit_sha256']:raise ValueError('Prior audit changed')
    with np.load(credit_root/'credit-factors.npz') as factors:
        chosen, tick = select_first([r for r in s['episodes'] if r['seed']==seed and not r['reverse']], factors)
        predicted = {t:factors[chosen['key']+'_bounds'][t].copy() for t in (tick,tick+32)}
    a = json.loads(Path(s['audit_summary']).read_text())
    matches=[]
    for candidate in a['sources']:
        path=Path(candidate['root'])/'manifest.json'
        if base.digest(path)!=candidate['manifest_sha256']:raise ValueError('Audited manifest changed')
        metadata=json.loads(path.read_text())
        if metadata['seed']==seed and not metadata['reverse']:matches.append(candidate)
    if len(matches)!=1:raise ValueError('Ambiguous selected condition')
    source=matches[0]
    root = Path(source['root'])
    if base.digest(root/'manifest.json') != source['manifest_sha256']:raise ValueError('Manifest changed')
    if base.digest(root/source['completed_record']) != source['completed_sha256']:raise ValueError('Course changed')
    m = json.loads((root/'manifest.json').read_text())
    if m['seed'] != seed or m['reverse']:raise ValueError('Wrong selected course')
    for path,digest in {**m['source_hashes'],**m['physical_sources']}.items():
        if base.digest(path)!=digest:raise ValueError('Source changed')
    parent = Path(m['parent'])
    for name,digest in m['parent_evidence'].items():
        if base.digest(parent/name)!=digest:raise ValueError('Parent evidence changed')
    progress=json.loads((root/source['completed_record']).read_text())
    row=next(r for r in progress['training'] if all(r[k]==chosen[k] for k in ('block','video','audio')))
    if row['sha256']!=chosen['source_sha256']:raise ValueError('Selected episode changed')
    data=read_record(root,row,m,learning_auditor=verify_learning)
    features=[]
    for clip in (0,1):
        paths=[p for p in m['physical_sources'] if Path(p).name==f'sensory-{clip}.npz']
        if len(paths)!=1:raise ValueError('Ambiguous media')
        with np.load(paths[0]) as z:features.append({k:z[k] for k in ('visual','auditory')})
    output.mkdir()
    manifest=dict(seed=seed,groups=m['groups'],selection=chosen,candidate_tick=tick,later_tick=tick+32,
        selection_rule='First conditional min > .001 on all pairings, probe ticks 16..63; then +32 training ticks.',
        credit_summary_sha256=base.digest(credit_root/'summary.json'),
        credit_factors_sha256=base.digest(credit_root/'credit-factors.npz'),
        producer_sha256=base.digest(__file__),source=str(root),source_record=row,
        source_hashes=m['source_hashes'],physical_sources=m['physical_sources'],
        predicted_bounds={str(t):v.tolist() for t,v in predicted.items()},
        limits=__doc__)
    (output/'manifest.json').write_text(base.encode(manifest)+'\n')
    probes=[];ps=json.loads((parent/'summary.json').read_text())
    # Reconfirm the factual parent expression path before interpreting new q.
    with isolated_rng():learned=base.load_checkpoint(parent/'final.paula',trusted=True).network
    for v,audio in ((0,0),(0,1),(1,0),(1,1)):
        replay=expression(parent/'initial.paula',learned,features,m['groups'],m['selected'],v,audio,False)
        ref=next(r for r in ps['probes'] if (r['kind'],r['weights'],r['presentation'],r['video'],r['audio'])==
                 ('resting','learned','both',v,audio))
        if base.digest(parent/ref['file'])!=ref['sha256']:raise ValueError('Reference changed')
        with np.load(parent/ref['file']) as z:
            if set(z.files)!=set(replay) or any(not np.array_equal(z[k],replay[k]) for k in z.files):
                raise ValueError('Parent expression replay differs')
    for when in (tick,tick+32):
        install_recorded(learned,data,data['weights'][when])
        for v,audio in ((0,0),(0,1),(1,0),(1,1)):
            z=expression(parent/'initial.paula',learned,features,m['groups'],m['selected'],v,audio,False)
            if not np.array_equal(z['weights_initial'],data['weights'][when]):raise ValueError('Wrong transferred weights')
            residuals=dict(learning=verify_learning(z),physics=base.verify_physics(z),afferents=verify_afferents(z))
            name=f'q{when}-v{v}-a{audio}.npz';np.savez_compressed(output/name,**z)
            x=trajectory(z,m['groups'])
            probes.append(dict(file=name,sha256=base.digest(output/name),when=when,video=v,audio=audio,
                residuals=residuals,min_pre_feedback=float(x[16:64,8].min()),
                max_pre_feedback=float(x[16:64,8].max()),
                wrong_ticks=np.flatnonzero(x[:,8]<0).tolist(),min_eta=float(z['eta'].min())))
    result=dict(probes=probes,parent_replay_exact=True,executed_ticks=1152,new_recorded_ticks=768)
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('credit_root',type=Path)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.credit_root,a.seed,a.output)
