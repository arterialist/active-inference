"""Data-only reconstruction of the connected consumer and source back-action."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np

from .association_balance_audit import checked
from .evidence_consumer_audit import ConsumerAudit


def compare_cohort(research_root, output):
    root,output=Path(research_root).resolve(),Path(output).resolve();runs=[]
    for mapping in ('paired','swapped'):
        for seed in (11,23,44,77):
            connected=root/f'20260909_evidence_embedded_{mapping}_seed{seed}'
            isolated=root/f'20260909_evidence_consumer_{mapping}_seed{seed}'
            audited=root/f'20260909_evidence_embedded_audit_{mapping}_seed{seed}'
            a=json.loads((audited/'summary.json').read_text());s=json.loads((connected/'summary.json').read_text())
            reference=json.loads((isolated/'summary.json').read_text())
            refs={(p['checkpoint'],p['state'],p['case']):p for p in reference['probes'] if p['condition']=='evidence_slow'}
            if not a['structurally_valid'] or not s['lower_exact'] or a['seed']!=seed or a['mapping']!=mapping:
                raise ValueError('Invalid or mismatched cohort member')
            rows=[]
            for p in s['probes']:
                old=refs[p['checkpoint'],p['state'],p['case']]
                with np.load(checked(connected,p)) as z:cells=z['consumer_states']
                with np.load(checked(isolated,old)) as z:previous=z['states']
                difference=(cells[:,33:,1]>0)!=(previous[:,33:,1]>0)
                rows.append(dict(checkpoint=p['checkpoint'],state=p['state'],case=p['case'],
                    connected_trace=str(connected/p['file']),isolated_trace=str(isolated/old['file']),
                    contrast_bit_differences_by_tick=difference.sum(axis=1).tolist(),
                    largest_integrator_potential_difference=float(abs(cells[:,:32,0]-previous[:,:32,0]).max())))
            if len(rows)!=112:raise ValueError('Incomplete comparison')
            runs.append(dict(seed=seed,mapping=mapping,probes=rows,trained_terminal_range=a['trained_terminal_range']))
    output.mkdir(parents=True,exist_ok=False)
    result=dict(runs=runs,limits='Exact contrast-spike comparison; graded internal states are allowed to differ and reported. This does not establish embodiment or general composition.')
    (output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


class SourceTerminalAudit:
    def __init__(self):
        self.info = np.ones(192, np.float32)
        self.error = np.zeros(192, np.float32)
        self.active = np.zeros(192, bool)
        self.m = np.zeros(192)
        self.tick = 0

    def check(self, lower, upper):
        for t in range(len(lower['states'])):
            # One return event per bank terminal, from its graded target.
            # The non-spiking target's native timing direction is negative.
            eta = (1e-7*(1+499*self.m/(.01+self.m))).astype(np.float32)
            self.info += np.where(self.active, eta*(-self.error), np.float32(0))
            if not np.array_equal(lower['bank_terminal_info'][t], self.info.astype(float)):
                raise ValueError(f'Source-terminal return update differs at {self.tick}')
            expected = (lower['states'][t,176:,1] > 0)*self.info.astype(float)
            if not np.array_equal(upper['source'][t], expected):
                raise ValueError('Boundary release differs from source spike and terminal')
            x = upper['incoming'][t,:192].astype(np.float32)
            self.error = x-upper['before'][t,:192].astype(np.float32)
            self.active = x > 0
            self.m = lower['states'][t,176:,3].copy()
            self.tick += 1


def split(raw):
    return ({k:raw[k] for k in raw.files if not k.startswith('consumer_')},
            {k[len('consumer_'):]:raw[k] for k in raw.files if k.startswith('consumer_')})


def audit(recording, output):
    root, output = Path(recording).resolve(), Path(output).resolve()
    m = json.loads((root/'manifest.json').read_text())
    s = json.loads((root/'summary.json').read_text())
    cfg = json.loads((root/'config.json').read_text())
    source = Path(m['source']); previous = json.loads((source/'summary.json').read_text())
    if len(s['training']) != 128 or len(s['probes']) != 112 or not s['lower_exact']:
        raise ValueError('Incomplete connected experiment')
    # Recover the consumer's actual configured cells rather than regenerate them.
    consumer_cfg = dict(neurons=deepcopy(cfg['neurons'][368:]),
                        synaptic_points=deepcopy([p for p in cfg['synaptic_points'] if p['neuron_id']>368]))
    for n in consumer_cfg['neurons']: n['id'] -= 368
    for p in consumer_cfg['synaptic_points']: p['neuron_id'] -= 368
    consumer = ConsumerAudit(consumer_cfg); terminals = SourceTerminalAudit()
    parents, results = {}, []
    oldprobes = {(p['checkpoint'],p['state'],p['case']):p for p in previous['probes']}
    if {(p['checkpoint'],p['state'],p['case']) for p in s['probes']} != set(oldprobes):
        raise ValueError('Changed probe grid')
    def check(item, olditem, a, b):
        with np.load(checked(root,item)) as z: lower, upper = split(z)
        with np.load(checked(source,olditem)) as z:
            if any(not np.array_equal(lower[k],z[k]) for k in z.files):
                raise ValueError('Source differs from independently audited bank record')
        b.check(lower,upper); a.check(upper)
        return lower, upper
    for i, item in enumerate(s['training']):
        if consumer.tick != i*96 or terminals.tick != consumer.tick: raise ValueError('Broken acquisition continuity')
        check(item, previous['training'][i], consumer, terminals)
        if i+1 in (32,128): parents[i+1] = deepcopy((consumer,terminals))
    for p in s['probes']:
        a,b = deepcopy(parents[p['checkpoint']])
        lower,upper = check(p,oldprobes[p['checkpoint'],p['state'],p['case']],a,b)
        firing = upper['states'][:,33:,1] > 0
        counts = []
        for cue in (0,1):
            sound = cue if m['mapping']=='paired' else 1-cue
            counts.append(firing[:,np.array(m['masks']['audio'][sound])-33].sum(axis=1).tolist())
        results.append(dict(checkpoint=p['checkpoint'],state=p['state'],case=p['case'],
                            spikes_by_tick_cue=counts,local_update_residual=a.max_error,
                            terminal_range=[float(lower['bank_terminal_info'].min()),float(lower['bank_terminal_info'].max())]))
    output.mkdir(parents=True,exist_ok=False)
    result = dict(seed=m['seed'],mapping=m['mapping'],structurally_valid=True,probes=results,
        trained_terminal_range=[float(terminals.info.min()),float(terminals.info.max())],
        local_update_residual=consumer.max_error,
        limits='Data-only consumer delivery, soma, incoming learning, consumer terminals and source-terminal return updates. Lower activity and incoming learning checked against previously audited isolated records. No embodiment or top-down somatic feedback.')
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args(); audit(a.recording,a.output)
