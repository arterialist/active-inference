"""Does acquired cross-sensory weight structure affect regulatory responses?

Focus on both visual pairings with sound 1, which gave opposite familiarity
contrasts in the complete physical factorial. Restore selected weights to birth
values while preserving learned fast state, every other weight and adaptation.
This is a temporary diagnostic intervention, not a candidate brain mechanism.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from neuron.neuron import setup_neuron_logger
from ..core.runtime_checkpoint import load_checkpoint
from .association_route_probe import digest
from .composition_probe import encode
from .eligibility_association_probe import dynamic_snapshot
from .eligibility_media_probe import record
from .media_order_audit import load_state
from .media_weight_identity import PathObserver, set_selected_weights
from .runtime_checkpoint_probe import CheckpointDriver


def run(source, output):
    setup_neuron_logger('CRITICAL')
    source, output = Path(source).resolve(), Path(output).resolve()
    m = json.loads((source/'manifest.json').read_text())
    course = Path(m['course'])
    cm = json.loads((course/'manifest.json').read_text())
    checkpoint = Path(m['source'])/m['checkpoint']['file']
    hashes = {**m['source_hashes'], str(source/'manifest.json'): digest(source/'manifest.json'),
              str(source/'summary.json'): digest(source/'summary.json'),
              str(Path(__file__).resolve()): digest(__file__)}
    if any(digest(p) != h for p, h in hashes.items()):
        raise ValueError('Changed source')
    parent = load_state(course/'checkpoint-16-state.json.gz')
    birth = load_state(course/'initial-state.json.gz')
    ports = m['selected_ports']
    q0 = np.array([birth['neurons'][str(n)]['synapses'][str(s)][0] for n, s, _ in ports])
    features = []
    for clip in (0, 1):
        p = next(Path(p) for p in cm['physical_sources'] if Path(p).name == f'sensory-{clip}.npz')
        with np.load(p) as z:
            features.append({k: z[k] for k in z.files})
    output.mkdir(parents=True, exist_ok=False)
    (output/'manifest.json').write_text(encode(dict(source=str(source), source_hashes=hashes,
        intervention='selected_info_to_birth', pairs=[[0, 1], [1, 1]],
        limits='Sound-1 pathway localization only. Learned fast state remains; no whole-memory erasure.'))+'\n')
    rows = []
    for reset, visual in ((False, 0), (True, 0), (True, 1)):
        if shutil.disk_usage(output).free < 1024**3:
            raise OSError('Less than 1 GiB free')
        branch = load_checkpoint(checkpoint, trusted=True)
        net = branch.network
        if json.loads(dynamic_snapshot(net)) != parent:
            raise ValueError('Wrong restored state')
        if reset:
            set_selected_weights(net, ports, q0)
        cells = list(net.network.neurons.values())
        syns = [p for n in cells for p in n.postsynaptic_points.values()]
        observer = PathObserver(cells, syns, net, ports)
        trial = dict(start=net.current_tick, stop=net.current_tick+300, visual_clip=visual, audio_clip=1)
        d = record(net, CheckpointDriver(branch), cells, syns, features, m['groups'], trial, ports, observer)
        d.update(selected_potential=np.asarray(observer.potentials), terminals=np.asarray(observer.releases))
        d['selected_local_current'] = np.where(d['arrivals'] > 0, d['selected_potential'], 0.)
        if not reset:
            with np.load(source/'visual-0-audio-1.npz') as z:
                if set(d) != set(z.files) or any(not np.array_equal(d[k], z[k]) for k in d):
                    raise ValueError('Intact paired-input control differs')
        else:
            name = f'reset-visual-{visual}-audio-1.npz'
            np.savez_compressed(output/name, **d)
            rows.append(dict(visual=visual, audio=1, trial=trial, file=name, sha256=digest(output/name)))
        print(encode(dict(reset=reset, visual=visual)), flush=True)
        del branch, net, cells, syns, observer, d
    if any(digest(p) != h for p, h in hashes.items()):
        raise ValueError('Source changed')
    s = dict(probes=rows, intact_control_exact=True, acquisition_ticks=0, new_probe_ticks=600, validation_ticks=300)
    (output/'summary.json').write_text(encode(s)+'\n')
    return s


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    run(a.source, a.output)
