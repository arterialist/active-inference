"""Localize long-acquisition recruitment without freezing adaptive dynamics.

Replay each recorded acquisition once to create exact executable checkpoints.
Subsequent branches reset selected input weights or cycle them among the four
existing inputs to each target. Each cycle preserves the target's weight
multiset, but not its source-specific weighted drive. This is a diagnostic
intervention, not a learning mechanism or evidence of biological rewiring.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import numpy as np

from simulations.active_inference.core.runtime_checkpoint import save_checkpoint, load_checkpoint
from .association_balance_audit import checked
from .association_route_probe import digest
from .composition_probe import encode
from .eligibility_association_probe import dynamic_snapshot
from .eligibility_media_probe import record
from .eligibility_media_audit import verify_ledger
from .graded_media_audit import verify_rates
from .media_order_audit import load_state
from .media_order_control import ledger_start
from .multimodal_pairing_probe import fresh, WeightObserver
from .runtime_checkpoint_probe import CheckpointDriver
from neuron.extensions.experimental.graded_eligibility import GradedEligibilityNeuron


def cycle_weights(ports, weights, shift):
    """Fixed, label-blind cyclic controls cover every nonidentity offset."""
    weights = np.asarray(weights)
    if (weights.shape != (len(ports),) or not np.isfinite(weights).all()
            or len({(n, sid) for n, sid, _ in ports}) != len(ports)
            or type(shift) is not int or shift not in (1, 2, 3)):
        raise ValueError('Invalid four-input cycle')
    targets = {}
    for i, (nid, sid, src) in enumerate(ports):
        targets.setdefault(nid, []).append(i)
    permutation = np.arange(len(ports))
    for indices in targets.values():
        if len(indices) != 4 or len({ports[i][2] for i in indices}) != 4:
            raise ValueError('Requires four distinct visual sources per target')
        permutation[indices] = np.roll(indices, shift)
    return weights[permutation].copy(), permutation


def set_selected_weights(net, ports, weights):
    """Preserve scalar types and every other recorded field at intervention."""
    if len(weights) != len(ports) or not np.isfinite(weights).all():
        raise ValueError('Malformed weights')
    if len({(n, sid) for n, sid, _ in ports}) != len(ports):
        raise ValueError('Repeated target port')
    expected = json.loads(dynamic_snapshot(net))
    for (nid, sid, src), weight in zip(ports, weights, strict=True):
        n = net.network.neurons[nid]
        if (sid not in n.eligibility_ports or n.synapse_sources[sid][0] != src
                or not 0 <= weight <= n.eligibility_cap or n.postsynaptic_points[sid].u_i.plast != 0):
            raise ValueError('Invalid selected excitatory pathway')
    for (nid, sid, _), weight in zip(ports, weights, strict=True):
        syn = net.network.neurons[nid].postsynaptic_points[sid]
        syn.u_i.info = type(syn.u_i.info)(weight)
        expected['neurons'][str(nid)]['synapses'][str(sid)][0] = float(syn.u_i.info)
    if json.loads(dynamic_snapshot(net)) != expected:
        raise ValueError('Undeclared recorded state changed')
    return expected


class PathObserver(WeightObserver):
    def __init__(self, neurons, synapses, net, ports):
        super().__init__(neurons, synapses)
        self.selected = [net.network.neurons[n].postsynaptic_points[s] for n, s, _ in ports]
        self.potentials, self.releases = [], []

    def __call__(self):
        super().__call__()
        self.potentials.append([p.potential for p in self.selected])
        self.releases.append([[p.u_o.info, *p.u_o.mod] for p in self.terminals])


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    m = json.loads((source/'manifest.json').read_text())
    s = json.loads((source/'summary.json').read_text())
    if m['checkpoints'] != [4, 16]:
        raise ValueError('Requires recorded four/sixteen course')
    hashes = {**m['source_hashes'], **m['source_files'], **m['physical_sources'],
              str(source/'manifest.json'): digest(source/'manifest.json'),
              str(source/'summary.json'): digest(source/'summary.json'),
              str(Path(__file__).resolve()): digest(__file__)}
    if any(digest(p) != h for p, h in hashes.items()):
        raise ValueError('Changed source')
    if shutil.disk_usage(output.parent).free < 1536*1024**2:
        raise OSError('Need 1.5 GiB headroom; acquisition records are reused, not copied')
    features = []
    for clip in (0, 1):
        p = next(Path(p) for p in m['physical_sources'] if Path(p).name == f'sensory-{clip}.npz')
        with np.load(p) as z:
            features.append({k: z[k] for k in z.files})
    cfg = json.loads((source/'config.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter(); ports = m['selected_ports']
    manifest = dict(source=str(source), groups=m['groups'], selected_ports=ports,
        source_hashes=hashes, mapping=m['mapping'], order=m['order'], seed=m['seed'],
        conditions=['intact', 'initial', 'cycle1', 'cycle2', 'cycle3'], cues=[None, 0, 1],
        probe_checkpoint=16, probe_ticks=300,
        limits='One graph seed, four learning histories. Cycles preserve each target weight multiset, '
               'not weighted input current, timing correlations or return errors. Fast state is retained; '
               'all learning and feedback remain active. No whole-memory erasure claim.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    net, core, cells, syns = fresh(source/'config.json', m['seed'], GradedEligibilityNeuron)
    birth = load_state(source/'initial-state.json.gz')
    if json.loads(dynamic_snapshot(net)) != birth:
        raise ValueError('Birth state changed')
    health = WeightObserver(cells, syns); checkpoints = []
    for i, item in enumerate(s['training']):
        data = record(net, core, cells, syns, features, m['groups'], item['trial'], ports, health)
        with np.load(checked(source, item)) as z:
            if set(z.files) != set(data) or any(not np.array_equal(data[k], z[k]) for k in z.files):
                raise ValueError('Full-field acquisition replay differs')
        del data
        if (i+1) % 4 == 0:
            repeats = (i+1)//4
            print(encode(dict(stage='replay', repeats=repeats, seconds=round(time.perf_counter()-began, 2))), flush=True)
            if repeats in m['checkpoints']:
                state_file = next(x for x in s['states'] if x['repeats'] == repeats)
                if json.loads(dynamic_snapshot(net)) != load_state(checked(source, state_file)):
                    raise ValueError('Checkpoint endpoint differs')
                path = output/f'checkpoint-{repeats}.neural-checkpoint'
                save_checkpoint(net, path, sources=list(hashes)+[source/'config.json'])
                checkpoints.append(dict(repeats=repeats, file=path.name, sha256=digest(path)))
    parent = dynamic_snapshot(net)
    learned = np.array([net.network.neurons[n].postsynaptic_points[sid].u_i.info for n, sid, _ in ports])
    initial = np.array([birth['neurons'][str(n)]['synapses'][str(sid)][0] for n, sid, _ in ports])
    qsets = dict(intact=learned, initial=initial); permutations = {}
    for shift in (1, 2, 3):
        qsets[f'cycle{shift}'], permutations[f'cycle{shift}'] = cycle_weights(ports, learned, shift)
    np.savez_compressed(output/'intervention-weights.npz', **qsets, **{k+'_permutation': v for k, v in permutations.items()})
    rows = []; residual = 0.
    for condition, q in qsets.items():
        for cue in (None, 0, 1):
            if shutil.disk_usage(output).free < 1024**3:
                raise OSError('Below 1 GiB; preserve records and stop')
            restored = load_checkpoint(output/'checkpoint-16.neural-checkpoint', trusted=True)
            branch = restored.network
            if dynamic_snapshot(branch) != parent:
                raise ValueError('Restored parent differs')
            start = set_selected_weights(branch, ports, q)
            cells = list(branch.network.neurons.values())
            syns = [p for n in cells for p in n.postsynaptic_points.values()]
            if any(n.params.eta_post <= 0 or n.params.eta_retro <= 0 for n in cells):
                raise ValueError('Frozen adaptation')
            observer = PathObserver(cells, syns, branch, ports)
            trial = dict(start=branch.current_tick, stop=branch.current_tick+300, visual_clip=cue, audio_clip=None)
            data = record(branch, CheckpointDriver(restored), cells, syns, features, m['groups'], trial, ports, observer)
            values = ledger_start(start, cfg, ports)
            result = verify_ledger(data, cfg, ports, *values[:4]); residual = max(residual, result[4])
            verify_rates(data['cells'], values[4], cfg, ports)
            if condition == 'intact':
                ref = next(p for p in s['probes'] if p['checkpoint']==16 and p['visual']==cue
                           and p['audio'] is None and p['kind']==('context' if cue is None else 'recall'))
                with np.load(checked(source, ref)) as z:
                    for k in data:
                        if k == 'incoming_info_after' and cue is None:
                            continue  # Reference blank lasts 364 ticks, this probe 300.
                        expected = z[k] if k.startswith('incoming_info_') else z[k][:300]
                        if not np.array_equal(data[k], expected):
                            raise ValueError(f'Restored intact probe differs: {k}')
            data.update(selected_potential=np.asarray(observer.potentials), terminals=np.asarray(observer.releases))
            data['selected_local_current'] = np.where(data['arrivals'] > 0, data['selected_potential'], 0.)
            name = f'{condition}-cue-{cue}.npz'; np.savez_compressed(output/name, **data)
            spikes = (data['cells'][:, np.array(m['groups']['tactile_core'])-1, 1] > 0).sum(axis=1)
            rows.append(dict(condition=condition, cue=cue, trial=trial, file=name, sha256=digest(output/name)))
            if dynamic_snapshot(net) != parent:
                raise ValueError('Probe changed parent')
            print(encode(dict(stage='probe', condition=condition, cue=cue, events=int(spikes.sum()))), flush=True)
            del restored, branch, data, observer
    if any(digest(p) != h for p, h in hashes.items()):
        raise ValueError('Runtime changed during execution')
    result = dict(checkpoints=checkpoints, probes=rows, exact_acquisition=True, intact_probes_exact=True,
        parent_unchanged=True, max_selected_update_residual=residual, seconds=time.perf_counter()-began,
        ticks=net.current_tick+len(rows)*300, weight_health_fields=WeightObserver.fields,
        terminal_order=[(n.id, tid) for n in net.network.neurons.values() for tid in n.presynaptic_points])
    (output/'summary.json').write_text(encode(result)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.source, a.output)
