"""Test learned audiovisual weights through their neural comparison circuit.

Cross both visual and auditory clips, both acquisition mappings, learned versus
within-neuron shuffled weights, and both acquisition orders. Only contextual
weights enter an otherwise original executable state. All adaptation remains
active. No familiarity label or host error enters the network. This is an
isolated neural assay, not an embodied task or a consciousness test.
"""
import argparse
import inspect
import json
from pathlib import Path
import random
import shutil

import numpy as np

from .association_route_probe import digest
from .composition_probe import encode
from .predictive_bridge_probe import audit_record
from .predictive_weight_transplant import install_weights, recording_contrast_groups
from .sensory_timing_transfer import record_timing
from ..core.runtime_checkpoint import load_checkpoint
from neuron.neuron import setup_neuron_logger


def audiovisual_trials():
    """Identical physical trials for every acquired weight history."""
    return [dict(start=0, stop=300, visual_clip=v, audio_clip=a)
            for v in (0, 1) for a in (0, 1)]


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    m = json.loads((source/'manifest.json').read_text())
    parent = Path(m['source'])
    pm = json.loads((parent/'manifest.json').read_text())
    cfg = json.loads((parent/'config.json').read_text())
    done = json.loads((source/'completion.json').read_text())
    hashes = {**m['source_hashes'], **pm['source_hashes'], **pm['physical_sources']}
    for p, h in hashes.items():
        if digest(p) != h:
            raise ValueError('Changed source: ' + p)
    references = {}
    for row in done['entries']:
        if row['condition'] not in ('learned', 'shuffled'):
            continue
        path = source/row['file']
        if digest(path) != row['sha256']:
            raise ValueError('Changed transplant reference')
        references[row['condition'], row['clip']] = path
        hashes[str(path)] = row['sha256']
    if len(references) != 4:
        raise ValueError('Incomplete acquired weight references')
    for condition in ('learned', 'shuffled'):
        with np.load(references[condition, 0]) as a, np.load(references[condition, 1]) as b:
            if not np.array_equal(a['start_weights'], b['start_weights']):
                raise ValueError('Weights must not depend on the probe cue')
    for obj in (run, record_timing, audit_record, load_checkpoint, install_weights):
        p = Path(inspect.getfile(obj)).resolve()
        hashes[str(p)] = digest(p)
    for p in (source/'manifest.json', parent/'manifest.json', parent/'config.json'):
        hashes[str(p)] = digest(p)
    features = []
    for p in sorted(pm['physical_sources']):
        with np.load(p) as z:
            features.append({k: z[k] for k in z.files})
    contrast = recording_contrast_groups(pm['contrast'])
    if shutil.disk_usage(output.parent).free < 2*1024**3:
        raise OSError('Keep at least 2 GiB available before starting')
    output.mkdir(exist_ok=False)
    manifest = dict(source=str(source), parent=str(parent), mapping=m['mapping'], order=m['order'],
        source_hashes=hashes, physical_sources=pm['physical_sources'], groups=pm['groups'],
        bridge=pm['bridge'], fields=pm['fields'], trials=audiovisual_trials(),
        external_neuron_ids=pm['groups']['vision']+pm['groups']['touch'],
        contrast_weight_neuron_order=[n for ids in contrast.values() for n in ids],
        hypothesis='For the SAME physical audiovisual event, familiar acquired weights reduce '
                   'opponent neural error relative to violated acquired weights; weight shuffling '
                   'tests feature placement. Report every event and time course, not only a pooled effect.',
        scope='1761 cells, one base graph, two familiar physical clips, two acquisition orders. '
              'No label or host error enters the brain. Positive adaptation throughout. '
              'Error and learning-rate responses are not yet neural action, global regulation or consciousness.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    setup_neuron_logger('CRITICAL')
    entries, used = [], 0
    for condition in ('learned', 'shuffled'):
        # Exact native visual-only controls exercise the new recorder entrypoint.
        trials = [dict(start=0, stop=32, visual_clip=v, audio_clip=None) for v in (0, 1)]
        trials += audiovisual_trials()
        for trial in trials:
            if used > 240*1024**2 or shutil.disk_usage(output).free < 1536*1024**2:
                raise OSError('Bounded recording reserve reached')
            v, a = trial['visual_clip'], trial['audio_clip']
            branch = load_checkpoint(parent/'initial.neural-checkpoint', trusted=True)
            net = branch.network
            if net.current_tick != 0:
                raise ValueError('Expected initial executable state')
            with np.load(references[condition, v]) as z:
                install_weights(net, pm['bridge'], z['start_weights'])
            ambient = random.getstate(), np.random.get_state()
            random.setstate(branch.python_rng)
            np.random.set_state(branch.numpy_rng)
            try:
                data = record_timing(net, features, pm['groups'], pm['bridge'], contrast, trial, 'native')
            finally:
                random.setstate(ambient[0]); np.random.set_state(ambient[1])
            residual = audit_record(data, cfg, pm['bridge'])
            with np.load(references[condition, v]) as z:
                for key, value in data.items():
                    if key.startswith('start_'):
                        expected = z[key]
                    elif a is None and key not in ('external_information', 'end_incoming_info'):
                        expected = z[key][:32]
                    else:
                        continue
                    if not np.array_equal(value, expected):
                        raise ValueError('Unexpected initial state or replay: ' + key)
            path = output/f'{condition}-visual{v}-audio{a}.npz'
            np.savez_compressed(path, **data)
            used += path.stat().st_size
            entries.append(dict(file=path.name, sha256=digest(path), condition=condition, trial=trial,
                                equation_residual=residual))
            (output/'progress.json').write_text(encode(dict(entries=entries, bytes=used))+'\n')
            print(encode(dict(file=path.name, bytes=used, equation_residual=residual)), flush=True)
            if used > 280*1024**2:
                raise OSError('Hard recording cap reached')
    if any(digest(p) != h for p, h in hashes.items()):
        raise ValueError('Source changed during assay')
    result = dict(entries=entries, bytes=used, ticks=sum(e['trial']['stop'] for e in entries))
    (output/'completion.json').write_text(encode(result)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    run(**vars(p.parse_args()))
