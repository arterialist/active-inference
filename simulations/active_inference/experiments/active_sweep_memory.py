"""Acquired-weight intervention in the continuing action-dependent sweep.

Reset only selected predictive information weights to their zero birth values.
Retain body, neural state, delayed inputs, queued signals, terminal coefficients,
error/context traces and positive adaptation. This tests the contribution of
these retained weights, not all memory in the organism. Neither branch receives
a task score or host policy. Reload a reserialized intact state for exact replay.
"""
import argparse
import json
from pathlib import Path
import random
import shutil

import numpy as np

from . import context_organization as base
from .active_sweep_probe import PhysicalDelay, record, verify
from .active_sweep_analysis import mechanics, verify_predictive_arrivals
from .crossed_av_continuation import isolated_rng
from ..components.body.loaded_hinge import LoadedHinge, DT


def restore(neural, physical):
    saved = base.load_checkpoint(neural, trusted=True)
    random.setstate(saved.python_rng)
    np.random.set_state(saved.numpy_rng)
    body = LoadedHinge(.8)
    with np.load(physical) as z:
        body.restore(z['state'], crossings=int(z['gate'][0]), next_gate=int(z['gate'][1]))
        delay = PhysicalDelay(z['delay'])
    if abs(body.data.time - saved.network.current_tick*DT) > 1e-10:
        raise ValueError('Neural and physical clocks disagree')
    for n in saved.network.network.neurons.values():
        if n.params.eta_post <= 0 or n.params.eta_retro <= 0:
            raise ValueError('Basal adaptation must remain positive')
    return saved.network, body, delay


def reset_selected(net, groups):
    changed = []
    for nid in groups['prediction']:
        n = net.network.neurons[nid]
        for sid in n.prediction_ports:
            point = n.postsynaptic_points[sid]
            changed.append((nid, sid, point.u_i.info))
            point.u_i.info = 0.
    return np.asarray(changed)


def exact_prefix(full, prefix):
    """Compare every recorder field, including ragged returns and delay state."""
    if set(full) != set(prefix):
        raise ValueError('Replay fields differ')
    n = len(prefix['body'])
    constants = {'neuron_ids', 'terminal_ids', 'physical_parameters'}
    for key, value in prefix.items():
        if key.endswith('_initial') or key in constants:
            expected = full[key]
        elif key == 'delay_final':
            delay = PhysicalDelay(full['delay_initial'])
            for raw in full['raw_afferents'][:n]:
                delay.step(raw)
            expected = delay.state()
        elif key == 'retrograde_offsets':
            expected = full[key][:n+1]
        elif key == 'retrograde_events':
            expected = full[key][:full['retrograde_offsets'][n]]
        else:
            expected = full[key][:n]
        if not np.array_equal(expected, value):
            raise ValueError('Executable replay differs: '+key)


def run(parent, output, ticks=512):
    parent, output = (Path(p).resolve() for p in (parent, output))
    if output.exists():
        raise FileExistsError(output)
    if type(ticks) is not int or not 512 <= ticks <= 1024:
        raise ValueError('Use 512..1024 continuing ticks, at least three motor cycles')
    if shutil.disk_usage(output.parent).free < 3*1024**3:
        raise OSError('Need 3 GiB reserve')
    m = json.loads((parent/'manifest.json').read_text())
    s = json.loads((parent/'summary.json').read_text())
    row = next(r for r in s['rows'] if r['condition'] == 'loaded_fused')
    hashes = dict(m['source_hashes'])
    hashes[str(Path(__file__).resolve())] = base.digest(__file__)
    for p, h in hashes.items():
        if base.digest(p) != h:
            raise ValueError('Source changed: '+p)
    for field in ('checkpoint', 'physical'):
        if base.digest(parent/row[field]) != row[field+'_sha256']:
            raise ValueError('Acquired state changed')
    if base.digest(parent/row['file']) != row['sha256']:
        raise ValueError('Acquisition record changed')
    config = parent/'loaded_fused.json'
    if base.digest(config) != m['config_hashes']['loaded_fused']:
        raise ValueError('Graph changed')
    cfg = json.loads(config.read_text()); groups = m['groups']['loaded_fused']
    if base.digest(m['physical_source']) != m['physical_sha256']:
        raise ValueError('Media changed')
    with np.load(m['physical_source']) as z:
        features = {k:z[k] for k in ('visual', 'auditory')}
    with np.load(parent/row['file']) as z:
        acquired = {k:z[k][-1] for k in ('weights', 'cells', 'terminal_info', 'physical_states', 'gate')}
        acquired['delay'] = z['delay_final']
    output.mkdir(); rows = []; intact = None
    for condition in ('intact', 'reset', 'replay'):
        with isolated_rng():
            checkpoint = output/'roundtrip.paula' if condition == 'replay' else parent/row['checkpoint']
            net, body, delay = restore(checkpoint, parent/row['physical'])
            if not np.array_equal(base.cellular(list(net.network.neurons.values())), acquired['cells']):
                raise ValueError('Acquired cellular endpoint disagrees with checkpoint')
            if condition == 'intact':
                base.save_checkpoint(net, output/'roundtrip.paula', sources=list(hashes))
            changed = reset_selected(net, groups) if condition == 'reset' else None
            data = record(net, body, delay, features, groups, ticks=96 if condition == 'replay' else ticks)
            verify(data, groups, features); verify_predictive_arrivals(data, cfg, groups); mechanics(data)
            if condition == 'intact':
                for key, old in [('weights_initial','weights'), ('terminal_initial','terminal_info'),
                                 ('body_initial','physical_states'), ('delay_initial','delay')]:
                    if not np.array_equal(data[key], acquired[old]):
                        raise ValueError('Acquired initial field differs: '+key)
                if not np.array_equal(data['gate_initial'], acquired['gate'][:2]):
                    raise ValueError('Acquired gate history differs')
                intact = data
            elif condition == 'replay':
                exact_prefix(intact, data)
            else:
                for key in ('body_initial','delay_initial','context_initial','error_initial','terminal_initial','gate_initial'):
                    if not np.array_equal(data[key], intact[key]):
                        raise ValueError('Undeclared initial intervention: '+key)
                if np.any(data['weights_initial']):
                    raise ValueError('Selected reset was not applied')
                np.savez_compressed(output/'intervention.npz', selected_before=changed)
            path = output/f'{condition}.npz'; np.savez_compressed(path, **data)
            item = dict(condition=condition, file=path.name, sha256=base.digest(path), ticks=len(data['body']))
            if condition != 'replay':
                final = output/f'{condition}-final.paula'; base.save_checkpoint(net, final, sources=list(hashes))
                physical = output/f'{condition}-final-body.npz'
                np.savez_compressed(physical, state=body.state(), delay=delay.state(), gate=[body.crossings, body.next_gate])
                item.update(checkpoint=final.name, checkpoint_sha256=base.digest(final),
                            physical=physical.name, physical_sha256=base.digest(physical))
            rows.append(item)
        print(base.encode(dict(seed=m['seed'], condition=condition, ticks=item['ticks'], crossings=body.crossings)), flush=True)
    if any(base.digest(p) != h for p,h in hashes.items()):
        raise ValueError('Source changed during run')
    manifest = dict(parent=str(parent), parent_manifest_sha256=base.digest(parent/'manifest.json'),
                    parent_summary_sha256=base.digest(parent/'summary.json'), seed=m['seed'], groups=groups,
                    source_hashes=hashes, start_tick=row['ticks'], ticks=ticks, limits=__doc__)
    (output/'manifest.json').write_text(base.encode(manifest)+'\n')
    result = dict(rows=rows, executed_ticks=2*ticks+96, exact_replay_ticks=96)
    (output/'summary.json').write_text(base.encode(result)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('parent', type=Path); p.add_argument('output', type=Path)
    a = p.parse_args(); run(a.parent, a.output)
