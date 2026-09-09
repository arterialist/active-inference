"""Audit recorded return events, then inspect complete gate trajectories."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import context_organization as base


def verify_returns(z):
    trace=z['context_terminal'];events=z['retrograde_events'];offsets=z['retrograde_offsets']
    if (trace.ndim!=2 or trace.shape[1]!=4 or len(offsets)!=len(trace)+1 or
            offsets[0]!=0 or offsets[-1]!=len(events) or np.any(np.diff(offsets)<0) or
            events.ndim!=2 or events.shape[1]!=7 or
            not np.array_equal(np.diff(offsets),trace[:,2])):
        raise ValueError('Invalid ordered return record')
    if not np.array_equal(trace[1:,0],trace[:-1,1]):raise ValueError('Terminal continuity lost')
    if len(z['context_before_dtype'])!=len(trace):raise ValueError('Missing arithmetic types')
    for t in range(len(trace)):
        code=int(z['context_before_dtype'][t])
        if code not in (0,32,64):raise ValueError('Unknown scalar type')
        u={0:float,32:np.float32,64:np.float64}[code](trace[t,0])
        for row in events[offsets[t]:offsets[t+1]]:
            u=np.clip(u+1e-7*np.float32(row[3]),-100.,100.)
        if u!=trace[t,1]:raise ValueError(f'Return recurrence differs at local tick {t}')
    return len(trace)


def analyze_gate(roots,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    cases=[];tables={};seen=set()
    for root in map(lambda p:Path(p).resolve(),roots):
        m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text())
        enabled=m['enabled']
        if enabled in seen:raise ValueError('Duplicate gate condition')
        seen.add(enabled)
        for p,h in m['source_hashes'].items():
            if base.digest(p)!=h:raise ValueError('Producer source changed')
        values=[];last=None;count=0
        for row in s['chunks']:
            if row['start']!=count:raise ValueError('Missing or unordered chunk')
            if base.digest(root/row['file'])!=row['sha256']:raise ValueError('Recording changed')
            with np.load(root/row['file']) as z:
                verify_returns(z)
                if len(z['cells'])!=row['ticks']:raise ValueError('Wrong chunk length')
                if last is not None and not np.array_equal(last,z['terminal_initial']):
                    raise ValueError('Cross-chunk terminal discontinuity')
                last=z['terminal_info'][-1].copy()
                if not np.array_equal(z['neuron_ids'],np.arange(1,m['width']+2)):
                    raise ValueError('Unexpected cell columns')
                o=z['cells'][:,:,base.FIELDS.index('O')];mem=z['cells'][:,:,base.FIELDS.index('S')]
                c=z['context_terminal'];d=z['drive'];w=z['weights'][:,1:,1]
                values.append(np.column_stack((np.arange(count,count+len(c)),d[:,0],
                    d[:,1:].min(1),d[:,1:].max(1),o[:,0],c[:,1],c[:,3],c[:,2],
                    o[:,1:].min(1),o[:,1:].max(1),mem[:,1:].min(1),mem[:,1:].max(1),
                    w.min(1),w.max(1))))
                count+=len(c)
        if count!=s['executed_ticks'] or count!=m['ticks']:raise ValueError('Incomplete run')
        table=np.concatenate(values);label='magnitude' if enabled else 'native';tables[label]=table
        negative=np.flatnonzero(table[:,5]<0)
        gate_loss=np.flatnonzero((table[:,0]>=16)&(table[:,0]<12000)&(table[:,9]>0))
        # All failure and withdrawal/recovery ticks remain in the table and raw records.
        cases.append(dict(enabled=enabled,root=str(root),ticks=count,
            manifest_sha256=base.digest(root/'manifest.json'),summary_sha256=base.digest(root/'summary.json'),
            first_negative_terminal_tick=int(negative[0]) if len(negative) else None,
            first_sustained_prefix_gate_leak=int(gate_loss[0]) if len(gate_loss) else None,
            terminal_final=float(table[-1,5]),terminal_min=float(table[:,5].min()),
            max_target_output_gate_on_after_settle=float(table[(table[:,0]>=16)&(table[:,1]>=.5),9].max()),
            max_target_output_gate_off=float(table[(table[:,0]>=12016)&(table[:,0]<13000),9].max())
                if count>12016 else None))
    output.mkdir();np.savez_compressed(output/'per-tick.npz',**tables)
    result=dict(cases=cases,columns=['tick','gate_drive','target_drive_min','target_drive_max',
        'source_O','terminal_info','arriving_release','return_count','target_O_min','target_O_max',
        'target_S_min','target_S_max','target_weight_min','target_weight_max'],
        limits='Controlled 193-cell preparation, not body or memory acceptance. All target cells '
               'are retained in hashed raw records; this table supplements them. No seed effect '
               'is claimed for the deterministic homogeneous inhibitory wiring.')
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(cases),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze_gate(a.roots,a.output)
