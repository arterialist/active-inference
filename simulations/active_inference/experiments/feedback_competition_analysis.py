"""Inspect the whole-loop feedback screen, retaining every input and member.

The factorial transform separates shared, visual, auditory and joint responses
of four factual presentations. It is not a trained decoder. Conditional margins
hold those histories fixed; they are not achievable learned performance bounds
for the feedback-coupled adaptive brain. No fitted weights enter the brain.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import context_organization as base
from .crossed_av_analysis import verify_stimuli
from .crossed_av_capacity import filtered_context, margin_certificate
from .crossed_av_credit import FACTORIAL
from .magnitude_feedback_analysis import verify_returns
from .opponent_context_analysis import read_record
from .temporal_verification import verify_learning


PAIRS=((0,0),(0,1),(1,0),(1,1))
CONDITIONS=('none','wired_zero','feedback')


def representation(arrivals):
    a=np.asarray(arrivals,dtype=float)
    if a.ndim!=3 or a.shape[:2]!=(4,64) or not np.isfinite(a).all() or np.any(a<0):
        raise ValueError('Need four nonnegative finite 64-tick source histories')
    filtered=np.stack([filtered_context(x) for x in a])
    modes=np.einsum('mp,ptn->mtn',FACTORIAL,filtered)
    gram=np.einsum('mtn,ktn->tmk',modes,modes)
    return filtered,modes,gram


def analyze(roots,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    cases=[];records={};provenance={};seeds=set();checked=0
    for root in map(lambda p:Path(p).resolve(),roots):
        m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text())
        if m['seed'] in seeds:raise ValueError('Duplicate seed')
        seeds.add(m['seed'])
        for p,h in {**m['source_hashes'],**m['physical_sources']}.items():
            if base.digest(p)!=h:raise ValueError(f'Source changed: {p}')
        if not s['unchanged_replay_exact'] or s['executed_ticks']!=4368:
            raise ValueError('Missing exact baseline or incomplete screen')
        expected={(c,v,a) for c in CONDITIONS for v,a in PAIRS}
        identities=[tuple(r[k] for k in ('condition','video','audio')) for r in s['records']]
        if len(identities)!=12 or set(identities)!=expected:raise ValueError('Incomplete condition family')
        features=[]
        for clip in (0,1):
            paths=[p for p in m['physical_sources'] if Path(p).name==f'sensory-{clip}.npz']
            if len(paths)!=1:raise ValueError('Ambiguous media')
            with np.load(paths[0]) as z:features.append({k:z[k] for k in ('visual','auditory')})
        source_ids=None
        for condition in CONDITIONS:
            cfg_path=root/f'{condition}.json'
            if base.digest(cfg_path)!=m['config_hashes'][condition]:raise ValueError('Graph changed')
            cfg=json.loads(cfg_path.read_text());groups=m['groups'][condition]
            if any(n['params']['eta_post']<=0 or n['params']['eta_retro']<=0 for n in cfg['neurons']):
                raise ValueError('Frozen adaptation')
            arriving=[];pool_activity=[]
            for v,a in PAIRS:
                row=next(r for r in s['records'] if (r['condition'],r['video'],r['audio'])==(condition,v,a))
                z=read_record(root,row,dict(m,groups=groups),learning_auditor=verify_learning)
                verify_returns(z);verify_stimuli(z,features,row,False)
                if len(z['body'])!=364 or np.any(z['weights_initial']) or np.any(z['weights'][:64]):
                    raise ValueError('Not a fresh pre-feedback representation screen')
                if np.any(z['drive'][:64,194:198]):raise ValueError('Early body feedback')
                ids=list(z['neuron_ids']);first,second=z['context_source_ids']
                if source_ids is None:source_ids=first.copy()
                if not np.array_equal(source_ids,first):raise ValueError('Source ordering differs')
                reorder=[list(second).index(i) for i in first]
                if not np.array_equal(z['arrivals'][:64,0],z['arrivals'][:64,1][:,reorder]):
                    raise ValueError('Opponent contexts differ')
                arriving.append(z['arrivals'][:64,0]);checked+=364
                key=f's{m["seed"]}_{condition}_v{v}_a{a}'
                records[key+'_body']=z['body'];records[key+'_eta']=z['eta']
                records[key+'_terminal_ids']=z['terminal_ids'];records[key+'_terminals']=z['terminal_info']
                for role,members in groups.items():
                    activity=z['cells'][:,[ids.index(n) for n in members],base.FIELDS.index('O')]
                    records[key+'_'+role+'_ids']=np.asarray(members)
                    records[key+'_'+role+'_O']=activity
                    if role=='competition_mixed_0':pool_activity.append(activity)
            incoming=np.stack(arriving);filtered,modes,gram=representation(incoming)
            key=f's{m["seed"]}_{condition}'
            records[key+'_source_ids']=source_ids;records[key+'_arrivals']=incoming
            records[key+'_filtered']=filtered;records[key+'_modes']=modes;records[key+'_gram']=gram
            # Per-tick norms retained; windows are indices into these trajectories.
            norms=np.linalg.norm(modes,axis=2).T;records[key+'_mode_norms']=norms
            signed=filtered*np.array([1.,-1.,-1.,1.])[:,None,None]
            certificates=[]
            for lo,hi in ((16,32),(32,64),(16,64)):
                certificates.append(dict(start=lo,stop=hi,**margin_certificate(signed[:,lo:hi].reshape(-1,signed.shape[2]))))
            cases.append(dict(seed=m['seed'],condition=condition,trace=key,
                norms_mean16_63=norms[16:64].mean(0).tolist(),certificates=certificates,
                active_pool_ticks=[np.flatnonzero(np.any(x>0,axis=1)).tolist() for x in pool_activity]))
        provenance[str(root/'manifest.json')]=base.digest(root/'manifest.json')
        provenance[str(root/'summary.json')]=base.digest(root/'summary.json')
    if not cases:raise ValueError('No evidence')
    output.mkdir();np.savez_compressed(output/'per-tick.npz',**records)
    provenance[str(Path(__file__).resolve())]=base.digest(__file__)
    result=dict(cases=cases,checked_ticks=checked,provenance=provenance,
        modes=['common','visual','audio','joint'],
        limits='Four fresh episodes per condition, not acquired recall. Full member output and '
          'incoming histories retained. Learning/body/afferent/command and context-return audits '
          'are independent; other cells are observed, not independently equation-reconstructed. '
          'Wired zero retains extra retrograde paths. No fitted policy, semantic or consciousness claim.')
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(cases=len(cases),checked_ticks=checked)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.roots,a.output)
