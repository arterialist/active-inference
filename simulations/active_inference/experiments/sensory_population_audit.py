"""Independent per-tick reconstruction of the receptor sensitivity experiment."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from .association_balance_audit import checked
from .media_drive_audit import ReceptorAudit
from .association_route_probe import digest


def compare_recorded_codes(recording, output):
    root,output=Path(recording).resolve(),Path(output).resolve()
    source=json.loads((root/'summary.json').read_text())
    entries={(p['condition'],p['sense'],p['clip'],p['gain']):p for p in source['records']}
    rows=[];differences={}
    for condition in ('baseline','homogeneous','diverse'):
        for sense in ('visual','audio'):
            for gain in (.5,1.,2.):
                pair=[entries[condition,sense,clip,gain] for clip in (0,1)]
                data=[]
                for p in pair:
                    with np.load(checked(root,p)) as z:data.append({k:z[k] for k in z.files})
                delta=(data[0]['cells'][:,:,1]>0)!=(data[1]['cells'][:,:,1]>0)
                ticks=np.flatnonzero(delta.any(axis=1));key=f'{condition}/{sense}/{gain:g}'
                differences[key]=delta
                rows.append(dict(condition=condition,sense=sense,gain=gain,
                    sources=[p['file'] for p in pair],spike_histories_identical=not delta.any(),
                    cellular_fields_identical=np.array_equal(data[0]['cells'],data[1]['cells']),
                    weights_identical=np.array_equal(data[0]['weights'],data[1]['weights']),
                    physical_values_identical=np.array_equal(data[0]['physical_values'],data[1]['physical_values']),
                    differing_cells_by_tick=delta.sum(axis=1).tolist(),
                    first_spike_difference=int(ticks[0]) if len(ticks) else None))
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'spike-differences.npz',**differences)
    result=dict(records=rows,limits='Exact equality of these recorded channels and finite histories, not equality of every intracellular state or a semantic recognition test.')
    (output/'comparisons.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def audit(recording,output):
    root,output=Path(recording).resolve(),Path(output).resolve()
    m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text())
    source=Path(m['source']);old=json.loads((source/'config.json').read_text());sm=json.loads((source/'manifest.json').read_text())
    features=[]
    for clip in (0,1):
        path=Path(sm['source_recording'])/f'sensory-{clip}.npz'
        if digest(path)!=sm['source_files_sha256'][path.name]:raise ValueError('Changed sensory source')
        with np.load(path) as z:features.append({k:z[k] for k in z.files})
    required={(c,k,i,g) for c in ('baseline','homogeneous','diverse') for k in ('visual','audio') for i in (0,1) for g in (.5,1.,2.)}
    if len(s['records'])!=36 or {(p['condition'],p['sense'],p['clip'],p['gain']) for p in s['records']}!=required:
        raise ValueError('Wrong sensory experiment grid')
    rows=[];residual=0.
    for p in s['records']:
        cfg=json.loads((root/f'{p["condition"]}.json').read_text())
        levels=1 if p['condition']=='baseline' else 6
        with np.load(checked(root,p)) as z:
            cells=z['cells'];weights=z['weights'];values=z['physical_values']
        feature=features[p['clip']];expected_values=np.zeros_like(values)
        if p['sense']=='visual':
            expected_values[:,:96]=1-np.clip(p['gain']*(1-feature['visual']),0,1)
            expected_values[:3,:96]=0
        else:
            db=feature['band_db']+20*np.log10(p['gain'])
            expected_values[:,96:]=np.clip((db[:,:,None]-np.array([-65.,-45.,-25.]))/20,0,1).reshape(-1,96)
        if not np.array_equal(values,expected_values):raise ValueError('Wrong physical stimulus')
        if cells.shape!=(len(values),192*levels,8):raise ValueError('Invalid sensory population size')
        for level in range(levels):
            threshold=.6 if p['condition']!='diverse' else m['thresholds'][level]
            local=deepcopy(old)
            for i,n in enumerate(local['neurons'][:192]):
                actual=cfg['neurons'][i*levels+level]
                expected=deepcopy(n['params']);expected.update(r_base=threshold,b_base=threshold+.25)
                if actual['params']!=expected:raise ValueError('Undeclared receptor parameter change')
                n['params']=expected
            a=ReceptorAudit(local,sm['groups'],expected_threshold=threshold)
            data=a.check(cells[:,level::levels],values)
            err=float(abs(data['reconstructed_weights']-weights[:,level::levels]).max());residual=max(residual,err)
            if err>2e-12:raise ValueError('Per-tick receptor plasticity mismatch')
        # Keep the population code per physical receptor and tick. Counts are
        # observer readouts only; no count has been injected into another cell.
        code=(cells[:,:,1]>0).reshape(len(cells),192,levels)
        rows.append(dict(**p,active_cells_by_tick=code.sum(axis=(1,2)).tolist(),
                         active_sensitivities_by_receptor=code.sum(axis=2).tolist()))
    output.mkdir(parents=True,exist_ok=False)
    result=dict(structurally_valid=True,local_update_residual=residual,records=rows,
        limits='Reconstructs every sensory cell and incoming information update. This is population encoding, not invariance, learning or a downstream functional use.')
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.recording,a.output)
