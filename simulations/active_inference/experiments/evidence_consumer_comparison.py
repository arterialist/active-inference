"""Trace-linked observations of candidate persistence and event switching.

This is an offline observer, never a controller or a generic acceptance score.
An early competing candidate is retained in the report, not counted as an error
before that volley's sensory evidence can reach the contrast cells.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from .association_balance_audit import checked
from .association_route_probe import digest

LATENCY = 6  # Receptor stimulus -> evidence bank 3 -> contrast output 3.


def observations(probe, schedule):
    counts = np.asarray(probe['spikes_by_tick_cue'], dtype=int)
    if counts.shape != (2, len(schedule)) or (counts < 0).any():
        raise ValueError('Invalid cue activity trace')
    case = probe['case']
    result = dict(case=case, condition=probe['condition'], checkpoint=probe['checkpoint'],
                  state=probe['state'], spike_ticks=[np.flatnonzero(x).tolist() for x in counts],
                  last_recorded_tick=len(schedule)-1)
    if case.startswith('transition-'):
        first, gap = map(int, re.fullmatch(r'transition-first([01])-gap(\d+)', case).groups())
        start = 32+gap
        old = np.flatnonzero(counts[first])
        new = np.flatnonzero(counts[1-first])
        after = new[new >= start]
        result.update(event_start=start, earliest_new_response=start+LATENCY,
            first_new_response=int(after[0]) if len(after) else None,
            last_old_response=int(old[-1]) if len(old) else None,
            old_responses_after_new_available=old[old >= start+LATENCY].tolist(),
            old_responses_after_first_new=old[old >= after[0]].tolist() if len(after) else None,
            next_response_latency=int(after[0]-start) if len(after) else None,
            input_withdrawal_tick=start+25,
            new_response_at_final_tick=bool(counts[1-first,-1]))
        return result
    cue = int(re.search(r'cue([01])', case).group(1))
    result['cue'] = cue
    windows = []
    for base in (0, 8, 16, 24):
        active = np.flatnonzero(schedule[base:base+8].any(axis=1))
        if not len(active): continue
        start = base+int(active[-1])+LATENCY
        stop = base+8+LATENCY
        own, other = counts[cue,start:stop], counts[1-cue,start:stop]
        windows.append(dict(start=start, stop_exclusive=stop,
            own_spikes_by_tick=own.tolist(), other_spikes_by_tick=other.tolist(),
            margin=int(own.sum()-other.sum())))
    result['completed_volley_windows'] = windows
    result['competing_spike_ticks'] = np.flatnonzero(counts[1-cue]).tolist()
    result['final_tick_active'] = bool(counts[:,-1].any())
    return result


def compare(research_root, output):
    root, output = Path(research_root).resolve(), Path(output).resolve()
    records, provenance = [], []
    for mapping in ('paired', 'swapped'):
        for seed in (11, 23, 44, 77):
            audit_root = root/f'20260909_evidence_consumer_audit_{mapping}_seed{seed}'
            recording = root/f'20260909_evidence_consumer_{mapping}_seed{seed}'
            a = json.loads((audit_root/'summary.json').read_text())
            m = json.loads((recording/'manifest.json').read_text())
            s = json.loads((recording/'summary.json').read_text())
            source = Path(m['source'])
            source_summary = json.loads((source/'summary.json').read_text())
            schedule_path = source/'schedules.npz'
            if digest(schedule_path) != source_summary['schedules_sha256']:
                raise ValueError('Changed sensory schedules')
            if not a['structurally_valid'] or a['seed'] != seed or a['mapping'] != mapping:
                raise ValueError('Wrong or invalid audited run')
            original = {(p['condition'],p['checkpoint'],p['state'],p['case']):p for p in s['probes']}
            if len(original) != 336 or len(a['probes']) != 336:
                raise ValueError('Incomplete source cohort')
            with np.load(schedule_path) as schedules:
                for p in a['probes']:
                    item = original[p['condition'],p['checkpoint'],p['state'],p['case']]
                    # Verify the auditor's observer labels against actual per-cell output.
                    with np.load(checked(recording,item)) as z:
                        firing = z['states'][:,33:,1] > 0
                    wanted = []
                    for cue in (0,1):
                        sound = cue if mapping == 'paired' else 1-cue
                        wanted.append(firing[:,np.array(m['masks']['audio'][sound])-33].sum(axis=1).tolist())
                    if wanted != p['spikes_by_tick_cue']: raise ValueError('Observer differs from raw cells')
                    row = observations(p, schedules[p['case']])
                    records.append(dict(seed=seed, mapping=mapping, trace=str(recording/item['file']), **row))
            provenance.append(dict(audit=str(audit_root), audit_summary_sha256=digest(audit_root/'summary.json'),
                                   recording=str(recording), recording_summary_sha256=digest(recording/'summary.json')))
    output.mkdir(parents=True, exist_ok=False)
    result = dict(records=records, provenance=provenance, latency_ticks=LATENCY,
        interpretation='Finite, isolated replay. Complete-volley windows are explicitly latency-aligned observer intervals, not a learned boundary or universal behavioral deadline. Spike ticks and source paths are retained; final activity is censored, not proof of indefinite persistence.')
    (output/'observations.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--research-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); compare(a.research_root, a.output)
