"""Audit the single-terminal gate intervention on all recorded neurons/ticks."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import context_organization as base
from .crossed_av_continuation_analysis import trajectory
from .opponent_context_analysis import read_record
from .temporal_verification import verify_learning


def analyze(roots,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    traces={};cases=[];sources=[];seen=set()
    for root in map(lambda p:Path(p).resolve(),roots):
        m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text())
        if m['seed'] in seen:raise ValueError('Duplicate seed')
        seen.add(m['seed']);source=Path(m['source'])
        if (base.digest(source/'manifest.json')!=m['source_manifest_sha256'] or
                base.digest(source/'summary.json')!=m['source_summary_sha256'] or
                base.digest(Path(__file__).with_name('crossed_av_gate_probe.py'))!=m['producer_sha256']):
            raise ValueError('Source changed')
        if not s['unchanged_replays_exact'] or s['executed_ticks']!=768:raise ValueError('Incomplete experiment')
        identities=[(r['restored'],r['video'],r['audio']) for r in s['probes']]
        expected={(r,v,a) for r in (False,True) for v in (0,1) for a in (0,1)}
        if len(identities)!=8 or set(identities)!=expected:raise ValueError('Missing or duplicate probe')
        for v,a in ((0,0),(0,1),(1,0),(1,1)):
            pair=[]
            for restored in (False,True):
                row=next(r for r in s['probes'] if (r['restored'],r['video'],r['audio'])==(restored,v,a))
                z=read_record(root,row,m,learning_auditor=verify_learning);pair.append(z)
                ids=list(z['neuron_ids']);oi=base.FIELDS.index('O')
                banks=[z['cells'][:,[ids.index(n) for n in m['groups'][role]],oi]
                       for role in ('mixed_0','mixed_1')]
                key=f's{m["seed"]}_r{int(restored)}_v{v}_a{a}'
                for i,bank in enumerate(banks):traces[key+f'_bank{i}']=bank
                traces[key+'_terminal']=z['context_terminal'];traces[key+'_physical']=trajectory(z,m['groups'])
                if z['context_terminal'].shape!=(96,4):raise ValueError('Wrong terminal trace')
                if restored and z['context_terminal'][0,0]!=m['birth_release']:
                    raise ValueError('Terminal not restored to declared birth value')
                if not np.array_equal(z['context_terminal'][1:,0],z['context_terminal'][:-1,1]):
                    raise ValueError('Terminal trajectory discontinuity')
                cases.append(dict(key=key,seed=m['seed'],restored=restored,video=v,audio=a,
                    suppressed_bank_active_ticks=np.flatnonzero(np.any(banks[1]>0,axis=1)).tolist(),
                    release_before=float(z['context_terminal'][0,0]),
                    release_final=float(z['context_terminal'][-1,1]),
                    min_prediction_16_63=float(traces[key+'_physical'][16:64,8].min())))
            for k in ('body_initial','delay_initial','weights_initial','context_initial','error_initial'):
                if not np.array_equal(pair[0][k],pair[1][k]):raise ValueError('Initial matched state differs')
            if not np.array_equal(pair[0]['drive'][:,:196],pair[1]['drive'][:,:196]):
                raise ValueError('Sensory/load evidence differs across intervention')
        sources.append(dict(root=str(root),manifest_sha256=base.digest(root/'manifest.json'),
                            summary_sha256=base.digest(root/'summary.json')))
    if not seen:raise ValueError('No gate evidence')
    output.mkdir();np.savez_compressed(output/'per-tick.npz',**traces)
    result=dict(cases=cases,sources=sources,checked_ticks=len(cases)*96,
        bank_columns='Original manifest group neuron IDs; every member is retained.',
        terminal_columns=['release_before','release_after','retrograde_event_count','arriving_release'],
        physical_columns='Same 15 columns as crossed_av_continuation_analysis.trajectory.',
        limits='Single-terminal restoration diagnoses gate failure, not a durable plasticity repair. '
               'No selected associative weights or body initial state changed. '
               'Terminal updates were reconstructed by the live observer; raw individual return events are not retained.')
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(cases=len(cases),checked_ticks=result['checked_ticks'])),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('roots',type=Path,nargs='+');a=p.parse_args();analyze(a.roots,a.output)
