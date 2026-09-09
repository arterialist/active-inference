"""Inspect the adaptive whole across scarcity and refeeding, without fitting scores."""
import argparse
import json
from pathlib import Path

import numpy as np

from .association_route_probe import digest
from .composition_probe import encode
from .hierarchical_body_audit import describe
from .metabolic_population_probe import audit_energy
from .proprioceptive_learning_audit import intervals
from .sensory_motor_probe import body_measures, audit_motor
from .temporal_body_probe import audit_history
from .predictive_bridge_probe import audit_record
from .hierarchical_body_perturbation import audit_intervention


def run(roots, output):
    output = Path(output).resolve()
    if len(roots) != 4 or output.exists():
        raise ValueError('Need four complete courses and a new output')
    records, report, sources = {}, {}, {}
    for root in map(Path, roots):
        m = json.loads((root/'manifest.json').read_text())
        cfg = json.loads((root/'config.json').read_text())
        summary = json.loads((root/'summary.json').read_text())
        key = bool(m['energy_feedback']['connected']), bool(m['fed'])
        if key in records:
            raise ValueError('Duplicated condition')
        raw = root/'closed-loop.npz'
        if digest(raw) != summary['raw_sha256'] or any(digest(p) != h for p, h in m['source_hashes'].items()):
            raise ValueError('Recording or source changed')
        with np.load(raw) as z:
            d = {k: z[k] for k in z.files}
        np.testing.assert_array_equal(d['neuron_ids'], m['neuron_ids'])
        audit_energy(d, cfg, m); audit_motor(d, cfg, m['motor'])
        audit_history(d, cfg, m['basis']); audit_record(d, cfg, m['bridge'])
        audit_intervention(d, cfg, cut=False, torque=0., gain=m['gain'], start=256, stop=304)
        body_measures(d)
        index = {int(n): i for i, n in enumerate(d['neuron_ids'])}
        cpg = d['cells'][:, [index[n] for n in m['motor']['cpg']], 1]
        alarm = d['cells'][:, index[m['energy_feedback']['alarm']], 1]
        phase0 = np.flatnonzero(cpg[:, 0] > 0)
        cycles = []
        for start, stop in zip(phase0[:-1], phase0[1:]):
            # Boundaries are actual phase-0 events, not a fitted or imposed clock.
            cycles.append(dict(start=int(start), stop_exclusive=int(stop),
                energy_start=float(d['energy_before'][start, 0]), energy_end=float(d['energy_after'][stop-1, 0]),
                work_j=float(d['positive_work_j'][start:stop].sum()),
                forward_change_m=float(d['forward_progress'][stop-1]-d['forward_progress'][start-1]),
                joint_range_rad=np.ptp(d['joint_position'][start:stop], axis=0),
                peak_actuation=float(d['actuator_ctrl'][start:stop].max()),
                predictor_weight_change=float(np.max(np.abs(d['weights'][stop-1]-d['weights'][start])))))
        e = d['energy_after']; first = lambda x: int(np.flatnonzero(x)[0]) if np.any(x) else None
        report[str(key)] = dict(minimum_energy_j=float(e[:, 0].min()),
            first_unmet_demand=first(e[:, 4] > 0), final_organs=e[-1],
            completely_zero_actuation_intervals=intervals(np.all(d['actuator_ctrl'] == 0, axis=1)),
            phase0_events=phase0, cycles=cycles, minimum_prediction_rate=float(d['eta'].min()))
        records[key] = {k: d[k] for k in ('actuator_ctrl', 'joint_position', 'energy_after', 'energy_afferents', 'forward_progress')}
        records[key].update(cpg=cpg.copy(), alarm=alarm.copy(), prediction_error=d['error_arrival'].copy())
        sources[str(raw.resolve())] = digest(raw)
        del d
    if set(records) != {(c, f) for c in (False, True) for f in (False, True)}:
        raise ValueError('Incomplete factorial')
    contrasts, traces = {}, {}
    for connected in (False, True):
        a, b = records[connected, True], records[connected, False]
        for k in a:
            np.testing.assert_array_equal(a[k][:1024], b[k][:1024])
            delta = a[k]-b[k]
            label = f'refeeding/connected{connected}/{k}'
            contrasts[label] = describe(delta); traces[label] = delta
    for key, r in records.items():
        for k, value in r.items():
            traces[f'{key}/{k}'] = value
    output.mkdir(exist_ok=False)
    np.savez_compressed(output/'regimes-per-tick.npz', **traces)
    result = dict(source_hash=digest(__file__), raw_hashes=sources, conditions=report,
        contrasts=contrasts, interpretation='The fixed neural feedback is a provisional component. '
        'The question is whether its composition with ongoing predictive learning preserves viable, useful activity. '
        'Cadence preservation, energy preservation and forward progress are separate observations. '
        'No sleep, torpor, autonomous feeding, learned mode selection or consciousness claim.')
    (output/'summary.json').write_text(encode(result)+'\n')
    print(encode(dict(contrasts=contrasts)), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots', nargs=4, type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(**vars(p.parse_args()))
