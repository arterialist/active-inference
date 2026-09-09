"""Independently reconstruct recorded native output-information updates.

Network.run_tick delivers returns before ticking cells. Both return adaptation
and incoming learning therefore see the preceding completed cell modulation.
The present records use float32 native error/terminal arithmetic. These checks
do not reconstruct error generation or per-event terminal modulation updates.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def information_updates(events, cells, initial_modulation, config, start_tick):
    events, cells = np.asarray(events), np.asarray(cells)
    if events.ndim != 2 or events.shape[1] != 11 or not np.isfinite(events).all():
        raise ValueError('Malformed return events')
    indices = {n['id']: i for i, n in enumerate(config['neurons'])}
    neurons = {n['id']: n for n in config['neurons']}
    predicted, wrong_clock, rates = [], [], []
    for row in events:
        t = int(row[0])-start_tick
        nid = int(row[3])
        if not 0 <= t < len(cells) or nid not in neurons:
            raise ValueError('Undeclared event time or recipient')
        n, i = neurons[nid], indices[nid]
        md = n['metadata']; index = md.get('plasticity_rate_index', 0)
        if index not in (0, 1):
            raise ValueError('Recorded cells contain only two modulation channels')
        previous = initial_modulation[i, index] if t == 0 else cells[t-1, i, 3+index]
        current = cells[t, i, 3+index]
        def rate(m):
            m = max(0., float(m))
            return n['params']['eta_retro']*(1.+md.get('plasticity_rate_boost', 0.)*m/(md.get('plasticity_rate_half_saturation', .1)+m))
        eta, wrong_eta = rate(previous), rate(current)
        if eta <= 0:
            raise ValueError('Adaptation was not positive')
        # Model error vectors arise from float32 input buffers. Once a native
        # return updates a terminal, numpy weak-scalar promotion keeps info in
        # float32. Verify that the observed native values have this precision.
        if any(float(np.float32(row[j])) != row[j] for j in (5, 6, 7)):
            raise ValueError('This recording is not the declared float32 path')
        def update(e):
            return float(np.clip(np.float32(row[5])+np.float32(e)*np.float32(row[7]), -100., 100.))
        predicted.append(update(eta)); wrong_clock.append(update(wrong_eta)); rates.append(eta)
    return np.asarray(predicted), np.asarray(wrong_clock), np.asarray(rates)


def audit(recording, output):
    root, output = Path(recording).resolve(), Path(output).resolve()
    m = json.loads((root/'manifest.json').read_text())
    s = json.loads((root/'summary.json').read_text())
    source = Path(m['source'])
    cfg = json.loads((source/'config.json').read_text())
    with gzip.open(source/'trained-state.json.gz', 'rt') as f:
        parent = json.load(f)
    initial = np.array([parent['neurons'][str(n['id'])]['M'] for n in cfg['neurons']])
    rows, traces = [], {}
    for branch in s['branches']:
        path = root/branch['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != branch['sha256']:
            raise ValueError('Changed branch')
        with np.load(path) as z:
            events, cells = z['retro_events'], z['cells']
        if branch['cut']:
            if len(events):
                raise ValueError('Cut events were delivered')
            continue
        expected, wrong, rates = information_updates(events, cells, initial, cfg, branch['trial']['start'])
        if not len(events) or not np.array_equal(expected, events[:, 6]):
            raise ValueError('Native return information update differs')
        key = 'reset' if branch['reset'] else 'learned'
        traces[key+'/events'] = events
        traces[key+'/effective_return_rate'] = rates
        traces[key+'/predicted_information'] = expected
        traces[key+'/incorrect_current_tick_information'] = wrong
        rows.append(dict(branch=key, events=len(events), exact=True,
                         wrong_clock_different_events=int(np.count_nonzero(wrong != events[:, 6])),
                         wrong_clock_max_residual=float(np.max(np.abs(wrong-events[:, 6])))))
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'event-reconstruction.npz', **traces)
    result = dict(recording=str(root), branches=rows,
                  limits='Only delivered selected-event output-information updates. Error vectors are inputs to this audit, '
                         'not independently reconstructed; per-event output modulation is not recorded. '
                         'Clock and float32 arithmetic are specific to this runtime and these records.')
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    audit(a.recording, a.output)
