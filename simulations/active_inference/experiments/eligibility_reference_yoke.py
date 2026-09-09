"""Separate bodily reafference from internal feedback after a learning intervention.

Diagnostic only: replay the intact contrast branch's physical afferents into a
native-eligibility branch. Its real MuJoCo body still follows its own PAULA
muscles. The actual raw sensors and their real delay queue are retained beside
the substituted input. No weights, neural returns, learning or motor output
are clamped. This is not an agent policy or an acquisition condition.
"""
import argparse
import inspect
import json
from pathlib import Path
import shutil

import numpy as np

from . import context_organization as base
from .body_state_memory_analysis import intervals, mechanics
from .crossed_av_analysis import verify_stimuli
from .crossed_av_continuation import isolated_rng, restore_acquired
from .eligibility_reference_intervention_analysis import onset, changed_ticks
from .eligibility_reference_probe import record, verify_learning
from .eligibility_reference_pool_audit import audit_pools
from .magnitude_feedback_analysis import verify_returns
from .opponent_context import AfferentDelay, verify_afferents
from .opponent_context_analysis import read_record


class YokedAfferents:
    """Retain the actual delay state, substitute only the delivered sample."""
    def __init__(self, delay, samples):
        self.delay = delay
        self.samples = np.asarray(samples, dtype=float).copy()
        if (self.samples.ndim != 2 or self.samples.shape[1] != 4
                or not np.isfinite(self.samples).all() or np.any(self.samples < 0)):
            raise ValueError('Need finite nonnegative physical afferents')
        self.cursor = 0

    def state(self):
        return self.delay.state()

    def step(self, raw):
        if self.cursor >= len(self.samples):
            raise ValueError('Yoked afferent course exhausted')
        self.delay.step(raw)
        sample = self.samples[self.cursor].copy()
        self.cursor += 1
        return sample


def verify_yoked(data, reference, groups):
    """Reconstruct actual sensors independently; then verify the declared substitution."""
    for field in ('body_initial', 'delay_initial', 'neuron_ids'):
        if not np.array_equal(data[field], reference[field]):
            raise ValueError('Yoked initial physical state or identities differ')
    if not np.array_equal(data['drive'], reference['drive']):
        raise ValueError('Yoked sensory replay differs')
    # The ordinary sensor auditor is still used, with the actual delayed sensor
    # stream reconstructed from raw recordings. It independently replays physics
    # to verify those raw recordings and the final real delay state.
    delay = AfferentDelay(len(data['delay_initial']), data['delay_initial'])
    actual = np.array([delay.step(raw) for raw in data['raw_afferents']])
    physical = dict(data, drive=data['drive'].copy())
    physical['drive'][:, 194:198] = actual
    verify_afferents(physical)
    verify_learning(data); verify_returns(data); mechanics(data)
    ids = list(data['neuron_ids'])
    muscles = data['cells'][:, [ids.index(n) for n in groups['muscle']], base.FIELDS.index('O')]
    if not np.array_equal(muscles[:, 0]-muscles[:, 1], data['body'][:, 3]):
        raise ValueError('Yoked body command differs from neural muscles')
    return actual


def load(path, digest):
    if base.digest(path) != digest:
        raise ValueError(f'Record changed: {path}')
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def run(parent, output):
    parent, output = (Path(p).resolve() for p in (parent, output))
    if output.exists():
        raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free < 3*1024**3:
        raise OSError('Need 3 GiB reserve')
    m = json.loads((parent/'manifest.json').read_text())
    s = json.loads((parent/'summary.json').read_text())
    acquired = Path(m['parent']); g = m['groups']; checkpoint = m['checkpoint']
    hashes = dict(m['source_hashes'])
    for obj in (run, onset, verify_yoked, verify_afferents, mechanics):
        p = str(Path(inspect.getfile(obj)).resolve()); hashes[p] = base.digest(p)
    for name in ('manifest.json', 'summary.json'):
        hashes[str(parent/name)] = base.digest(parent/name)
    for p, h in {**hashes, **m['physical_sources']}.items():
        if base.digest(p) != h:
            raise ValueError(f'Source changed: {p}')
    for name in ('neural', 'physical'):
        if base.digest(acquired/checkpoint[name]) != checkpoint[name+'_sha256']:
            raise ValueError('Acquired checkpoint changed')
    pm = json.loads((acquired/'manifest.json').read_text())
    cfg_path = acquired/'contrast.json'
    if base.digest(cfg_path) != pm['config_hashes']['contrast']:
        raise ValueError('Source graph changed')
    cfg = json.loads(cfg_path.read_text()); hashes[str(cfg_path)] = base.digest(cfg_path)
    features = []
    for clip in (0, 1):
        paths = [p for p in m['physical_sources'] if Path(p).name == f'sensory-{clip}.npz']
        if len(paths) != 1:
            raise ValueError('Ambiguous sensory source')
        with np.load(paths[0]) as z:
            features.append({k: z[k] for k in ('visual', 'auditory')})
    output.mkdir(); rows = []
    for v, a in ((0, 0), (0, 1), (1, 0), (1, 1)):
        originals = {}
        original_rows = {}
        for condition in ('intact', 'native'):
            matches = [r for r in s['rows'] if (r['condition'], r['video'], r['audio']) == (condition, v, a)]
            if len(matches) != 1:
                raise ValueError('Missing or duplicated parent case')
            row = matches[0]; original_rows[condition] = row
            originals[condition] = read_record(parent, row, m, learning_auditor=verify_learning)
            if len(originals[condition]['body']) != 96:
                raise ValueError('Wrong reference duration')
        for condition in ('closed', 'yoked'):
            with isolated_rng():
                net, arm, delay = restore_acquired(acquired/checkpoint['neural'], acquired/checkpoint['physical'])
                for nid in g['prediction']:
                    n = net.network.neurons[nid]
                    n.prediction_reference_strength = 0.
                    n.metadata['prediction_reference_strength'] = 0.
                if condition == 'yoked':
                    delay = YokedAfferents(delay, originals['intact']['drive'][:, 194:198])
                data = record(net, arm, delay, features, g, m['selected'], v, a, ticks=96)
            audit_pools(data, cfg)
            verify_stimuli(data, features, dict(video=v, audio=a), False)
            if condition == 'closed':
                old = originals['native']
                if set(old) != set(data) or any(not np.array_equal(old[k], data[k]) for k in old):
                    raise ValueError('Native closed-loop replay differs')
            else:
                verify_yoked(data, originals['intact'], g)
            name = f'{condition}-v{v}-a{a}.npz'
            np.savez_compressed(output/name, **data)
            rows.append(dict(file=name, sha256=base.digest(output/name), condition=condition,
                             video=v, audio=a, references=original_rows))
        print(base.encode(dict(seed=m['seed'], video=v, audio=a, stage='pair-complete')), flush=True)
    if any(base.digest(p) != h for p, h in hashes.items()):
        raise ValueError('Source changed during probe')
    manifest = dict(parent=str(parent), seed=m['seed'], groups=g, source_hashes=hashes,
                    physical_sources=m['physical_sources'], config=str(cfg_path), limits=__doc__)
    (output/'manifest.json').write_text(base.encode(manifest)+'\n')
    result = dict(rows=rows, executed_ticks=768, exact_replay_ticks=384)
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(seed=m['seed'], **{k: result[k] for k in ('executed_ticks', 'exact_replay_ticks')})), flush=True)


def analyze(roots, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    arrays = {}; cases = []; seeds = set(); sources = {}
    fields = ('cells', 'weights', 'errors', 'eta', 'terminal_info', 'body', 'drive')
    for root in (Path(p).resolve() for p in roots):
        m = json.loads((root/'manifest.json').read_text()); s = json.loads((root/'summary.json').read_text())
        if m['seed'] in seeds:
            raise ValueError('Duplicate seed')
        seeds.add(m['seed']); parent = Path(m['parent']); g = m['groups']
        for p, h in {**m['source_hashes'], **m['physical_sources']}.items():
            if base.digest(p) != h:
                raise ValueError(f'Source changed: {p}')
        if s['executed_ticks'] != 768 or s['exact_replay_ticks'] != 384:
            raise ValueError('Incomplete run')
        identities = [(r['condition'], r['video'], r['audio']) for r in s['rows']]
        expected = {(c, v, a) for c in ('closed', 'yoked') for v in (0, 1) for a in (0, 1)}
        if len(identities) != len(expected) or set(identities) != expected:
            raise ValueError('Incomplete case family')
        cfg = json.loads(Path(m['config']).read_text())
        for v, a in ((0, 0), (0, 1), (1, 0), (1, 1)):
            branches = {}
            for c in ('closed', 'yoked'):
                row = next(r for r in s['rows'] if (r['condition'], r['video'], r['audio']) == (c, v, a))
                z = load(root/row['file'], row['sha256'])
                if len(z['body']) != 96:
                    raise ValueError('Wrong duration')
                audit_pools(z, cfg); branches[c] = z
            refs = {c: load(parent/r['file'], r['sha256']) for c, r in row['references'].items()}
            old = refs['native']; closed = branches['closed']; yoked = branches['yoked']
            if set(old) != set(closed) or any(not np.array_equal(old[k], closed[k]) for k in old):
                raise ValueError('Closed reference replay differs')
            verify_afferents(closed); verify_learning(closed); verify_returns(closed); mechanics(closed)
            verify_afferents(refs['intact']); verify_learning(refs['intact']); verify_returns(refs['intact'])
            for field in ('weights_initial', 'context_initial', 'error_initial', 'reference_initial',
                          'terminal_initial', 'pool_weight_initial', 'basal_eta', 'reference_strength'):
                if not np.array_equal(closed[field], yoked[field]):
                    raise ValueError('Unmatched neural intervention state')
            actual = verify_yoked(yoked, refs['intact'], g)
            name = f's{m["seed"]}_v{v}_a{a}'
            arrays[name+'_actual_delayed_afferents'] = actual
            comparisons = {}
            for label, left, right in (('internal_effect', yoked, refs['intact']),
                                       ('reafferent_effect', closed, yoked)):
                comparisons[label] = {}
                for f in fields:
                    mask = changed_ticks(left[f], right[f]); arrays[name+'_'+label+'_'+f+'_changed'] = mask
                    comparisons[label][f] = dict(first=onset(left[f], right[f]), intervals=intervals(mask))
                # Retain actual signed values, not just nonzero/equality masks.
                arrays[name+'_'+label+'_errors'] = left['errors']-right['errors']
                arrays[name+'_'+label+'_pose'] = left['body'][:, 1]-right['body'][:, 1]
                ids = list(left['neuron_ids'])
                comparisons[label]['population_output'] = {
                    role: onset(left['cells'][:, [ids.index(n) for n in members], base.FIELDS.index('O')],
                                right['cells'][:, [ids.index(n) for n in members], base.FIELDS.index('O')])
                    for role, members in g.items()}
            cases.append(dict(seed=m['seed'], video=v, audio=a, comparisons=comparisons))
        for name in ('manifest.json', 'summary.json'):
            sources[str(root/name)] = base.digest(root/name)
    if len(seeds) < 4:
        raise ValueError('Need four graph seeds')
    output.mkdir(); np.savez_compressed(output/'per-tick.npz', **arrays)
    result = dict(cases=cases, seeds=sorted(seeds), executed_ticks=768*len(seeds),
                  exact_replay_ticks=384*len(seeds), sources=sources, producer_sha256=base.digest(__file__),
                  limits='Controlled sensory substitution, not natural behavior. Internal effect includes all remaining neural paths; '
                  'it does not isolate one synapse. The reafferent contrast is conditional on native eligibility. '
                  'Divergence times refer to this 96-tick acquired-state probe, not general delays or eventual absence.')
    (output/'summary.json').write_text(base.encode(result)+'\n')
    print(base.encode(dict(seeds=sorted(seeds), cases=len(cases), executed_ticks=result['executed_ticks'])), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots', nargs='+', type=Path); p.add_argument('--output', required=True, type=Path)
    p.add_argument('--analyze', action='store_true'); a = p.parse_args()
    if a.analyze:
        analyze(a.roots, a.output)
    elif len(a.roots) == 1:
        run(a.roots[0], a.output)
    else:
        p.error('One parent intervention directory required for a run')
