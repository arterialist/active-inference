"""Data-only checks for contact-specific versus mean-pooled PAULA returns.

The architecture builder is deliberately not imported. This audit checks its
actual configuration against the original graph, then compares recorded ticks.
No observer output is used by the running network.
"""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wiring_checks(original, candidate, selected, mode):
    if mode not in ('contact', 'mean_pooled'):
        raise ValueError('Unknown architecture')
    selected = set(selected)
    old_neurons = {n['id']: n for n in original['neurons']}
    new_neurons = {n['id']: n for n in candidate['neurons']}
    old_terms = {(p['neuron_id'], p['terminal_id']): p
                 for p in original['synaptic_points'] if p['type'] == 'presynaptic'}
    new_terms = {(p['neuron_id'], p['terminal_id']): p
                 for p in candidate['synaptic_points'] if p['type'] == 'presynaptic'}
    ids = [(p['neuron_id'], p.get('terminal_id', p.get('synapse_id')))
           for p in candidate['synaptic_points']]
    fanout = Counter(c['source_neuron'] for c in original['connections'])
    new_fanout = Counter((c['source_neuron'], c['source_terminal'])
                        for c in candidate['connections'])
    expected_neurons = deepcopy(original['neurons'])
    if mode == 'mean_pooled':
        for n in expected_neurons:
            if n['id'] in selected:
                n['params']['eta_retro'] /= fanout[n['id']]
    checks = dict(
        same_neuron_ids=len(old_neurons) == len(new_neurons) == len(candidate['neurons'])
                        and old_neurons.keys() == new_neurons.keys(),
        unique_point_ids=len(ids) == len(set(ids)),
        only_declared_neuron_change=candidate['neurons'] == expected_neurons,
        same_incoming_points=[p for p in original['synaptic_points'] if p['type'] == 'postsynaptic']
                             == [p for p in candidate['synaptic_points'] if p['type'] == 'postsynaptic'],
        original_terminals_preserved=all(new_terms.get(k) == p for k, p in old_terms.items()),
        same_edge_count=len(original['connections']) == len(candidate['connections']),
        only_declared_terminal_routing=True,
        same_initial_per_edge_release=True,
        selected_contact_fanout=True,
        only_declared_added_terminals=True,
        adaptation_positive=all(n['params']['eta_post'] > 0 and n['params']['eta_retro'] > 0
                                for n in candidate['neurons']))
    for before, after in zip(original['connections'], candidate['connections']):
        expected = dict(before)
        if mode == 'contact' and before['source_neuron'] in selected:
            expected['source_terminal'] = after['source_terminal']
        checks['only_declared_terminal_routing'] &= expected == after
        a = old_terms.get((before['source_neuron'], before['source_terminal']))
        b = new_terms.get((after['source_neuron'], after['source_terminal']))
        checks['same_initial_per_edge_release'] &= a is not None and b is not None and (
            {k: v for k, v in a.items() if k != 'terminal_id'}
            == {k: v for k, v in b.items() if k != 'terminal_id'})
    added = new_terms.keys() - old_terms.keys()
    if mode == 'contact':
        checks['selected_contact_fanout'] = all(v == 1 for (n, _), v in new_fanout.items() if n in selected)
        checks['only_declared_added_terminals'] = all(n in selected and new_fanout[n, t] == 1 for n, t in added)
        checks['exact_terminal_capacity'] = len(new_terms) == len(old_terms) + sum(fanout[n] for n in selected) - sum(n in selected for n, t in old_terms if (n, t) in new_fanout)
    else:
        checks['exact_terminal_capacity'] = candidate['synaptic_points'] == original['synaptic_points']
    # Compare all remaining fields, including metadata, not an allowlist of
    # familiar simulation fields that would miss a newly added parameter.
    remaining = deepcopy(candidate)
    remaining['neurons'] = original['neurons']
    remaining['connections'] = original['connections']
    remaining['synaptic_points'] = original['synaptic_points']
    expected_metadata = dict(original.get('metadata', {}))
    expected_metadata['contact_terminal_sources' if mode == 'contact' else 'mean_pooled_return_sources'] = sorted(selected)
    checks['only_declared_metadata'] = candidate.get('metadata') == expected_metadata
    if 'metadata' in original:
        remaining['metadata'] = original['metadata']
    else:
        remaining.pop('metadata', None)
    checks['all_other_configuration_exact'] = remaining == original
    return {k: bool(v) for k, v in checks.items()}


def audit_prepared(path):
    root = Path(path).resolve()
    m = json.loads((root/'manifest.json').read_text())
    source = Path(m['prepared_from'])
    old = json.loads((source/'manifest.json').read_text())
    for p, h in {**m['source_hashes'], **m['prepared_source_files']}.items():
        if digest(p) != h:
            raise ValueError(f'Changed preparation source: {p}')
    mode = m['configuration_intervention']['mode']
    checks = wiring_checks(json.loads((source/'config.json').read_text()),
                           json.loads((root/'config.json').read_text()),
                           old['groups']['visual_core'], mode)
    for key in ('groups', 'selected_ports', 'source_recording', 'source_files_sha256', 'seed', 'clip_ticks'):
        checks['unchanged_'+key] = m[key] == old[key]
    if not all(checks.values()):
        raise ValueError(checks)
    return dict(mode=mode, source=str(source), prepared=str(root), checks=checks)


def initial_state_checks(original, candidate, original_config, candidate_config):
    remaining = deepcopy(candidate)
    if original['neurons'].keys() != candidate['neurons'].keys():
        return dict(same_cells=False)
    for nid in remaining['neurons']:
        remaining['neurons'][nid]['terminals'] = original['neurons'][nid]['terminals']
    same_release = len(original_config['connections']) == len(candidate_config['connections'])
    for before, after in zip(original_config['connections'], candidate_config['connections']):
        a = original['neurons'][str(before['source_neuron'])]['terminals'][str(before['source_terminal'])]
        b = candidate['neurons'][str(after['source_neuron'])]['terminals'][str(after['source_terminal'])]
        same_release &= a == b
    expected = {(str(p['neuron_id']), str(p['terminal_id'])) for p in candidate_config['synaptic_points'] if p['type'] == 'presynaptic'}
    actual = {(nid, tid) for nid, n in candidate['neurons'].items() for tid in n['terminals']}
    return dict(all_other_recorded_initial_state_exact=remaining == original,
                initial_terminal_state_per_edge_exact=same_release,
                loaded_terminal_ids_exact=expected == actual)


def compare_architectures(cohorts, output):
    """Preserve full per-tick cross-architecture differences, including controls.

    Each cohort must already have passed media_order_audit.compare. This adds
    original-graph checks, cross-architecture physical-history equality and
    descriptive recruitment timing. It does not introduce an acceptance score.
    """
    from .media_order_audit import verify_protocol
    if set(cohorts) != {'shared', 'contact', 'mean_pooled'}:
        raise ValueError('Need all three architectures')
    manifests, locations, summaries, prepared = {}, {}, {}, {}
    for architecture, paths in cohorts.items():
        histories = {}
        for path in paths:
            root = Path(path).resolve()
            m = json.loads((root/'manifest.json').read_text())
            key = architecture, m['mapping'], m['order']
            if key in manifests:
                raise ValueError('Duplicate history')
            manifests[key], locations[key] = m, root
            summaries[key] = json.loads((root/'summary.json').read_text())
            histories[key[1:]] = m
        verify_protocol(histories)
        sources = {m['source'] for m in histories.values()}
        if len(sources) != 1:
            raise ValueError('Changed architecture source within cohort')
        if architecture != 'shared':
            prepared[architecture] = audit_prepared(sources.pop())
    original = manifests['shared', 'paired', 0]['source']
    if any(p['source'] != original for p in prepared.values()):
        raise ValueError('Different original graph')
    for key, m in manifests.items():
        base = manifests['shared', *key[1:]]
        for field in ('seed', 'groups', 'selected_ports', 'physical_sources', 'trials', 'repeats'):
            if m[field] != base[field]:
                raise ValueError(f'Unmatched {field}')
    with gzip.open(locations['shared', 'paired', 0]/'initial-state.json.gz', 'rt') as f:
        original_state = json.load(f)
    original_cfg = json.loads((locations['shared', 'paired', 0]/'config.json').read_text())
    birth_checks = {}
    for key, root in locations.items():
        with gzip.open(root/'initial-state.json.gz', 'rt') as f:
            state = json.load(f)
        checks = initial_state_checks(original_state, state, original_cfg,
                                      json.loads((root/'config.json').read_text()))
        if not all(checks.values()):
            raise ValueError(checks)
        birth_checks['/'.join(map(str, key))] = checks
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    traces, timing = {}, {}
    for mapping in ('paired', 'swapped'):
        for order in (0, 1):
            baseline = {}
            for architecture in ('shared', 'contact', 'mean_pooled'):
                key = architecture, mapping, order
                root, summary = locations[key], summaries[key]
                for p in summary['probes']:
                    if digest(root/p['file']) != p['sha256']:
                        raise ValueError('Changed probe record')
                    case = p['state'], p['expression'], p['sense'], p['clip']
                    with np.load(root/p['file']) as z:
                        values = z['cells'][:, :, 1].copy()
                    if not np.isfinite(values).all():
                        raise ValueError('Nonfinite output')
                    if architecture == 'shared':
                        baseline[case] = values
                    elif values.shape != baseline[case].shape:
                        raise ValueError('Changed output shape')
                    name = '/'.join(map(str, (architecture, mapping, order, *case)))
                    if architecture != 'shared':
                        traces[name+'/output_minus_shared'] = values - baseline[case]
                    for role in ('visual_core', 'tactile_core', 'upper_core'):
                        v = values[:, np.array(manifests[key]['groups'][role])-1] > 0
                        spikes = v.sum(axis=1)
                        traces[name+'/'+role+'/spikes_by_tick'] = spikes
                        where = np.flatnonzero(spikes)
                        timing[name+'/'+role] = dict(total=int(spikes.sum()), first=int(where[0]) if len(where) else None,
                            last=int(where[-1]) if len(where) else None, active_ticks=len(where),
                            onset=int(spikes[:32].sum()), post32=int(spikes[32:].sum()))
    np.savez_compressed(output/'trajectories.npz', **traces)
    result = dict(preparations=prepared, initial_state_checks=birth_checks, timing=timing,
                  sources={k: [str(Path(p).resolve()) for p in v] for k, v in cohorts.items()},
                  analysis_sha256=digest(__file__),
                  limits='One graph seed and two recordings. Full tick differences retained. '
                  'Firing recruitment is not sound identity or recall. Initial per-edge drive is matched, '
                  'but active adaptation allows architecture trajectories to diverge even during initial controls. '
                  'Mean-rate normalization is not an exact closed-loop dose match. No embodiment is tested.')
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for mode in ('shared', 'contact', 'mean_pooled'):
        p.add_argument('--'+mode.replace('_', '-'), type=Path, nargs=4, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    compare_architectures({k: getattr(a, k) for k in ('shared', 'contact', 'mean_pooled')}, a.output)
