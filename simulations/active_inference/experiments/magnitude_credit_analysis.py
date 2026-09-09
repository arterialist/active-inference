"""Attribute actual selected-weight updates to a fixed audiovisual interface.

No fitted weights or counterfactual brain are installed. The reference is the
four-block resting expression of this same architecture. Full acquisition-tick
by probe-tick effects are retained, with source identity and native arithmetic
checks. Reference-dependent interference is not a closed-loop Jacobian.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import context_organization as base
from .crossed_av_capacity import filtered_context,native_predictor
from .crossed_av_credit import credit_components,differential
from .crossed_av_continuation_analysis import FACTORIAL,PAIR_ORDER
from .opponent_context_analysis import read_record
from .temporal_verification import verify_learning
from .magnitude_feedback_analysis import verify_returns


def update_effects(data,raw_basis):
    """Both opponent branches remain identified before taking their difference."""
    raw_basis=np.asarray(raw_basis,dtype=float)
    if raw_basis.shape!=(4,64,data['weights'].shape[2]) or not np.isfinite(raw_basis).all():
        raise ValueError('Need finite four-pair, 64-tick, source-aligned basis')
    parts=credit_components(data);ids=data['context_source_ids']
    q=differential(data['weights'],ids);initial=differential(data['weights_initial'],ids)
    delta=np.diff(np.concatenate((initial[None],q)),axis=0)
    mode_basis=np.einsum('mp,pkn->mkn',FACTORIAL,raw_basis)
    effects=np.einsum('tn,mkn->tmk',delta,mode_basis,optimize=True)
    # A single probe tick is supplemental attribution; all probe ticks of the
    # total update and the full local factors are saved, not replaced by this.
    factors={key:differential(value,ids) for key,value in parts.items()}
    split=np.stack([np.einsum('tn,mn->tm',factors[k],mode_basis[:,63],optimize=True)
                    for k in parts],axis=1)
    if not np.allclose(split.sum(1),effects[:,:,63],atol=2e-14,rtol=0):
        raise ValueError('Projected update decomposition differs')
    return dict(update_modes=effects,component_modes_at63=split,**factors)


def run(audit,output):
    audit=Path(audit).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    checked=json.loads((audit/'summary.json').read_text());blocks=checked['blocks']
    if blocks<4:raise ValueError('Need at least the four-block reference')
    output.mkdir();records=[];references=[]
    for source in checked['sources']:
        root=Path(source['root']);mp=root/'manifest.json';cp=root/f'completed-block-{blocks}.json'
        if base.digest(mp)!=source['manifest_sha256'] or base.digest(cp)!=source['progress_sha256']:
            raise ValueError('Audited evidence changed')
        m=json.loads(mp.read_text());progress=json.loads(cp.read_text())
        for p,h in {**m['source_hashes'],**m['physical_sources']}.items():
            if base.digest(p)!=h:raise ValueError('Source changed')
        basis=[];ids=None;residual=0.;recorded_refs=[]
        for v,a in PAIR_ORDER:
            row=next(r for r in progress['probes'] if
                (r['blocks'],r['kind'],r['weights'],r['video'],r['audio'])==(4,'resting','learned',v,a))
            z=read_record(root,row,m,learning_auditor=verify_learning)
            if np.any(z['weights'][:64]!=z['weights_initial']) or np.any(z['drive'][:64,194:198]):
                raise ValueError('Not a constant-weight pre-feedback reference')
            if ids is None:ids=z['context_source_ids'].copy()
            if not np.array_equal(ids,z['context_source_ids']):raise ValueError('Input identities differ')
            reindex=[list(ids[1]).index(i) for i in ids[0]]
            if not np.array_equal(z['arrivals'][:64,0],z['arrivals'][:64,1][:,reindex]):
                raise ValueError('Opponents receive different histories')
            for j,nid in enumerate(m['groups']['prediction']):
                y=z['cells'][:64,list(z['neuron_ids']).index(nid),base.FIELDS.index('O')]
                residual=max(residual,float(abs(y-native_predictor(z['arrivals'][:64,j],z['weights_initial'][j])).max()))
            basis.append(filtered_context(z['arrivals'][:64,0]));recorded_refs.append(row)
        if residual>2e-12:raise ValueError('Reference native reconstruction differs')
        basis=np.stack(basis);modes=np.einsum('mp,pkn->mkn',FACTORIAL,basis)
        name=f's{m["seed"]}-basis.npz'
        np.savez_compressed(output/name,raw_basis=basis,source_ids=ids,
            raw_gram=np.einsum('pkn,qkn->kpq',basis,basis),
            mode_gram=np.einsum('pki,qki->kpq',modes,modes))
        references.append(dict(seed=m['seed'],file=name,sha256=base.digest(output/name),
            source_rows=recorded_refs,native_residual=residual,
            mode_norms_16_63=np.linalg.norm(modes[:,16:64],axis=(1,2)).tolist()))
        for row in progress['training']:
            z=read_record(root,row,m,learning_auditor=verify_learning);verify_returns(z)
            if not np.array_equal(ids,z['context_source_ids']):raise ValueError('Training input identities differ')
            data=update_effects(z,basis)
            name=f's{m["seed"]}-{Path(row["file"]).stem}.npz';np.savez_compressed(output/name,**data)
            # These are index values into complete tick-by-probe-tick effects.
            mode63=data['update_modes'][:,:,63]
            records.append(dict(seed=m['seed'],file=name,sha256=base.digest(output/name),
                source=str(root/row['file']),source_sha256=row['sha256'],
                block=row['block'],video=row['video'],audio=row['audio'],ticks=len(mode63),
                net_mode_effect_at63=mode63.sum(0).tolist(),
                absolute_mode_effect_at63=abs(mode63).sum(0).tolist()))
    result=dict(records=records,references=references,audit=str(audit),
        audit_sha256=base.digest(audit/'summary.json'),producer_sha256=base.digest(__file__),
        source_hashes={str(Path(p).resolve()):base.digest(p) for p in
            (__file__,Path(__file__).with_name('crossed_av_credit.py'),Path(__file__).with_name('crossed_av_capacity.py'))},
        mode_order=['common','visual','audio','joint'],component_order=['old_context','new_context','bound_correction'],
        effect_axes=['acquisition_tick','mode','probe_tick'],
        checked_training_ticks=sum(r['ticks'] for r in records),
        limits='Fixed factual four-block reference; changed weights can change upstream return paths. '
               'Projection is an observer, not an implemented regulator, fitted readout, exact '
               'counterfactual brain or proof of a learning timescale. All input ages use factual error.')
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(training_ticks=result['checked_training_ticks'],references=references)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('audit',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.audit,a.output)
