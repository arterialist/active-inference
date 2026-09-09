"""Read-only current accounting and surgical pathway interventions.

Replays recorded sensory inputs into the original PAULA graph. The observer
never writes neural state. It reconstructs each observed cell's membrane from
delayed synaptic arrivals, verifies firing, and attributes local weight credit.
It is process-local, not safe to share with a concurrent simulation thread.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from .audiovisual_population import media_inputs
from .composition_probe import Neuron, encode, fingerprint, k
from .population_hierarchy import cellular, FIELDS
from neuron.extensions.experimental.plasticity_rate import PlasticityRateNeuron


def cut_incoming(config, edges, targets, families):
    """Keep dendritic ports and t_ref bounds; remove only named connections."""
    result = deepcopy(config)
    removed = {(src, tgt, sid) for src, tgt, sid, family, present in edges
               if present and tgt in targets and family in families}
    result["connections"] = [c for c in result["connections"]
        if (c["source_neuron"], c["target_neuron"], c["target_synapse"]) not in removed]
    return result, sorted(removed)


class CurrentLedger:
    def __init__(self, ids, edges, ticks):
        self.ids = list(ids)
        self.index = {nid: i for i, nid in enumerate(ids)}
        self.families = sorted({e[3] for e in edges}) + ["unassigned"]
        self.family_index = {f: i for i, f in enumerate(self.families)}
        self.ports = {(tgt, sid): self.family_index[f] for _, tgt, sid, f, _ in edges}
        shape = (ticks, len(ids), len(self.families))
        self.arrivals = np.zeros(shape)
        self.membrane = np.zeros(shape)
        self.credit_positive = np.zeros(shape, dtype=np.uint16)
        self.credit_negative = np.zeros(shape, dtype=np.uint16)
        self.weight_delta = np.zeros(shape)
        self.scalar = np.zeros((ticks, len(ids), 4))
        self.residual = np.zeros(shape[1:])
        self.max_error = 0.

    @contextmanager
    def observe(self):
        original = Neuron.tick
        ledger = self

        def tick(neuron, external_inputs, current_tick, dt=1.):
            if neuron.id not in ledger.index:
                return original(neuron, external_inputs, current_tick, dt)
            i, t = ledger.index[neuron.id], current_tick
            old_S, old_fire = neuron.S, neuron.t_last_fire
            active = [(int(sid), neuron.postsynaptic_points[sid].u_i.info)
                      for sid in np.flatnonzero(neuron.input_buffer[:, 0] > 0)
                      if sid in neuron.postsynaptic_points]
            # This graph has strictly positive dendritic delays. Inputs added
            # inside tick cannot arrive until a later tick.
            assert min(neuron.distances.values()) >= 1 if isinstance(neuron.distances, dict) else min(neuron.distances) >= 1
            native_current = 0.
            for arrival, _, amplitude, sid in sorted(neuron.propagation_queue):
                if arrival <= t:
                    family = ledger.ports.get((neuron.id, sid), len(ledger.families)-1)
                    value = amplitude * neuron.params.delta_decay ** neuron.distances[sid]
                    native_current += value
                    ledger.arrivals[t, i, family] += value
            alpha = dt/neuron.params.lambda_param
            ledger.residual[i] = (1-alpha)*ledger.residual[i] + alpha*ledger.arrivals[t, i]
            # Preserve native operation order and scalar dtype for firing.
            # The family accounting is float64; native buffered values may be
            # float32, so its conservation residual includes rounding.
            predicted_S = old_S + alpha*(-old_S + native_current)
            if not np.isfinite(predicted_S) or abs(predicted_S) >= 1000:
                raise AssertionError("Ledger scope excludes numerical clamps")
            result = original(neuron, external_inputs, current_tick, dt)
            threshold = neuron.b if t-old_fire <= neuron.params.c else neuron.r
            if abs(predicted_S) < .005:
                threshold = neuron.r
            predicted_fire = predicted_S >= threshold and t-old_fire >= neuron.params.c
            assert predicted_fire == (neuron.O > 0), (t, neuron.id, predicted_S, threshold)
            error = max(abs(ledger.residual[i].sum()-predicted_S),
                        abs(neuron.S-(0. if predicted_fire else predicted_S)))
            ledger.max_error = max(ledger.max_error, error)
            assert error < 1e-5, (t, neuron.id, error)
            ledger.membrane[t, i] = ledger.residual[i]
            ledger.scalar[t, i] = [predicted_S, threshold, neuron.O, neuron.t_ref]
            if predicted_fire:
                ledger.residual[i] = 0.
            # This is the native t_ref credit decision, not an STDP claim.
            positive = t-neuron.t_last_fire <= neuron.t_ref
            for sid, before in active:
                family = ledger.ports.get((neuron.id, sid), len(ledger.families)-1)
                counter = ledger.credit_positive if positive else ledger.credit_negative
                counter[t, i, family] += 1
                ledger.weight_delta[t, i, family] += neuron.postsynaptic_points[sid].u_i.info-before
            return result

        Neuron.tick = tick
        try:
            yield self
        finally:
            Neuron.tick = original

    def save(self, path):
        np.savez_compressed(path, neuron_ids=self.ids, families=self.families,
            arrivals=self.arrivals, membrane_before_reset=self.membrane,
            scalar=self.scalar, scalar_fields=["S_before_reset", "threshold", "O", "t_ref"],
            credit_positive=self.credit_positive, credit_negative=self.credit_negative,
            weight_delta=self.weight_delta)


def run_probe(recording, output, ticks=192):
    recording, output = Path(recording), Path(output)
    manifest = json.loads((recording/"manifest.json").read_text())
    config = json.loads((recording/"config.json").read_text())
    with np.load(recording/"sensory-features.npz") as f:
        features = {key: f[key] for key in f.files}
    if not 1 <= ticks <= manifest["trials"][0]["stop"]:
        raise ValueError("Probe must stay inside the original first silent-video trial")
    with np.load(recording/"ticks-000000.npz") as raw:
        reference = raw["cells"][:ticks].copy()
    output.mkdir(parents=True, exist_ok=False)
    groups, edges = manifest["groups"], manifest["edges"]
    targets = set(groups["tactile_core"])
    conditions = {"intact": [], "crossmodal_cut": ["crossmodal"],
                  "descending_cut": ["descending"], "both_cut": ["crossmodal", "descending"]}
    reports = {}
    for condition, families in conditions.items():
        started = time.perf_counter()
        modified, removed = cut_incoming(config, edges, targets, families)
        directory = output/condition
        directory.mkdir()
        (directory/"config.json").write_text(encode(modified)+"\n")
        random.seed(manifest["seed"])
        np.random.seed(manifest["seed"])
        net, core = k.load(str(directory/"config.json"), neuron_class=PlasticityRateNeuron)
        net.record_history = False
        neurons = list(net.network.neurons.values())
        states = []
        ledger = CurrentLedger(groups["tactile_core"], edges, ticks)
        with ledger.observe():
            for t in range(ticks):
                for nid, amplitude in media_inputs(features, groups, manifest["trials"][0], t):
                    net.set_external_input(nid, 0, amplitude)
                result = core.do_tick()
                if "error" in result or net.current_tick != t+1:
                    raise RuntimeError(result)
                states.append(cellular(neurons))
        states = np.stack(states)
        exact = bool(np.array_equal(states, reference)) if condition == "intact" else None
        if condition == "intact" and not exact:
            raise AssertionError("Instrumented baseline differs from the recorded experiment")
        ledger.save(directory/"current-ledger.npz")
        np.savez_compressed(directory/"cells.npz", cells=states, fields=FIELDS)
        spikes = states[:, :, 1] > 0
        hits = np.argwhere(spikes[:, np.array(groups["tactile_core"])-1])
        first = None
        if len(hits):
            t, i = map(int, hits[0])
            first = {"tick": t, "neuron": ledger.ids[i],
                "S_before_reset": ledger.scalar[t, i, 0], "threshold": ledger.scalar[t, i, 1],
                "membrane_by_family": dict(zip(ledger.families, ledger.membrane[t, i]))}
        reports[condition] = {"removed_connections": len(removed), "exact_baseline_replay": exact,
            "max_membrane_accounting_error": ledger.max_error, "first_auditory_spike": first,
            "spikes": {g: int(spikes[:, np.array(ids)-1].sum()) for g, ids in groups.items()},
            "auditory_current_by_family": dict(zip(ledger.families, ledger.arrivals.sum(axis=(0, 1)))),
            "positive_credit_events": dict(zip(ledger.families, ledger.credit_positive.sum(axis=(0, 1)))),
            "negative_credit_events": dict(zip(ledger.families, ledger.credit_negative.sum(axis=(0, 1)))),
            "postsynaptic_weight_delta": dict(zip(ledger.families, ledger.weight_delta.sum(axis=(0, 1)))),
            "seconds": time.perf_counter()-started}
        print(encode({"condition": condition, **reports[condition]}), flush=True)
    sources = fingerprint()
    sources[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {"source_recording": str(recording.resolve()), "ticks": ticks, "conditions": reports,
        "source_hashes": sources, "scope": "Early passive response, one graph seed. Cuts have network-wide indirect effects; current attribution is local algebra, not independent causal contribution. No neuronal dynamics were changed by the observer."}
    (output/"summary.json").write_text(encode(report)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=192)
    args = parser.parse_args()
    run_probe(args.recording, args.output, args.ticks)
