"""Verify a trusted runtime checkpoint during active real-media PAULA dynamics.

This is a serialization/continuation experiment, not a learning result. It uses
the full current media graph, keeps adaptation active, and compares all recorded
fields and selected learning ledgers after reload without acquisition replay.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from simulations.active_inference.core.runtime_checkpoint import save_checkpoint, load_checkpoint
from simulations.active_inference.experiments.association_route_probe import digest
from simulations.active_inference.experiments.composition_probe import encode
from simulations.active_inference.experiments.eligibility_association_probe import dynamic_snapshot
from simulations.active_inference.experiments.eligibility_media_probe import record
from simulations.active_inference.experiments.multimodal_pairing_probe import fresh
from simulations.active_inference.experiments.population_state_branch import TickDriver
from neuron.extensions.experimental.graded_eligibility import GradedEligibilityNeuron


class CheckpointDriver:
    def __init__(self, branch):
        self.branch = branch

    def do_tick(self):
        return self.branch.step()


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    m = json.loads((source/'manifest.json').read_text())
    if any(digest(p) != h for p, h in m['source_hashes'].items()):
        raise ValueError('Source runtime changed')
    features, inputs = [], []
    for clip in (0, 1):
        p = next(Path(p) for p in m['physical_sources'] if Path(p).name == f'sensory-{clip}.npz')
        if digest(p) != m['physical_sources'][str(p)]:
            raise ValueError('Physical source changed')
        inputs.append(p)
        with np.load(p) as z:
            features.append({k: z[k] for k in z.files})
    output.mkdir(parents=True, exist_ok=False)
    net, core, members, syns = fresh(source/'config.json', m['seed'], GradedEligibilityNeuron)
    warmup = dict(start=0, stop=64, visual_clip=0, audio_clip=0)
    data = record(net, core, members, syns, features, m['groups'], warmup, m['selected_ports'])
    np.savez_compressed(output/'warmup.npz', **data)
    queues = sum(map(len, net.presynaptic_wheel))+sum(map(len, net.retrograde_wheel))
    if not queues:
        raise ValueError('Checkpoint must contain in-flight neural signals')
    before = dynamic_snapshot(net); started = time.perf_counter()
    checkpoint = output/'active.neural-checkpoint'
    save_checkpoint(net, checkpoint, sources=[source/'config.json', *inputs])
    save_seconds = time.perf_counter()-started
    started = time.perf_counter()
    restored = load_checkpoint(checkpoint, trusted=True)
    restore_seconds = time.perf_counter()-started
    if dynamic_snapshot(restored.network) != before or dynamic_snapshot(net) != before:
        raise ValueError('Checkpoint changed recorded neural state')
    other = list(restored.network.network.neurons.values())
    other_syns = [p for n in other for p in n.postsynaptic_points.values()]
    for a, b in zip(members, other, strict=True):
        if type(a) is not type(b) or type(a.S) is not type(b.S) or a.input_buffer.dtype != b.input_buffer.dtype:
            raise ValueError('Neural class or numeric types changed')
        for sid, syn in a.postsynaptic_points.items():
            if type(syn.u_i.info) is not type(b.postsynaptic_points[sid].u_i.info):
                raise ValueError('Synaptic scalar type changed')
    trial = dict(start=64, stop=192, visual_clip=1, audio_clip=0)
    actual = record(net, TickDriver(net), members, syns, features, m['groups'], trial, m['selected_ports'])
    replay = record(restored.network, CheckpointDriver(restored), other, other_syns, features, m['groups'], trial, m['selected_ports'])
    exact = {k: np.array_equal(v, replay[k]) for k, v in actual.items()}
    if not all(exact.values()) or dynamic_snapshot(net) != dynamic_snapshot(restored.network):
        raise ValueError('Restored continuation differs')
    np.savez_compressed(output/'uninterrupted.npz', **actual)
    np.savez_compressed(output/'restored.npz', **replay)
    # Saving a branch must retain its private RNG stream, not ambient RNG.
    save_checkpoint(restored, output/'continued.neural-checkpoint', sources=[source/'config.json', *inputs])
    second = load_checkpoint(output/'continued.neural-checkpoint', trusted=True)
    if second.python_rng != restored.python_rng or not np.array_equal(second.numpy_rng[1], restored.numpy_rng[1]):
        raise ValueError('Continued branch RNG was not retained')
    result = dict(source=str(source), neurons=len(members), in_flight_signals=queues,
                  ticks=64+2*128, save_seconds=save_seconds, restore_seconds=restore_seconds,
                  checkpoint_bytes=checkpoint.stat().st_size, fields_exact=exact, final_recorded_state_exact=True,
                  neuron_and_synapse_scalar_types_exact=True, continued_branch_rng_exact=True,
                  limits='One active full-graph prefix. No body or environmental state is checkpointed. '
                  'Trusted executable local format, not a safe interchange format for untrusted uploads. '
                  'Same Python/library/source versions required; logging sinks are intentionally not preserved.')
    (output/'summary.json').write_text(encode(result)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); print(encode(run(a.source, a.output)))
