"""Exact replay with read-only accounting of one native legacy learning rule.

The observer runs inside the existing plasticity-rate extension, so it records
the effective eta, not the basal value restored after tick. It changes no state.
Use a separate process: the temporary class-method observer is not thread safe.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np

from .composition_probe import Neuron, encode
from .multimodal_pairing_probe import episode, fresh
from .population_hierarchy import weight_values


def legacy_update(weight, eta, direction, error_magnitude, lower=-100., upper=100.):
    raw = eta*direction*error_magnitude*weight
    factor = max(0., 1.-weight/10.) if raw > 0 else 1.
    decay = eta*.02*weight
    unclipped = weight+raw*factor-decay
    return {"raw_delta": raw, "soft_factor": factor, "decay": decay,
            "unclipped_weight": unclipped, "expected_weight": min(upper, max(lower, unclipped))}


@contextmanager
def observe_port(neuron_id, synapse_id, rows):
    original = Neuron.tick

    def tick(neuron, external_inputs, current_tick, dt=1.):
        if neuron.id != neuron_id:
            return original(neuron, external_inputs, current_tick, dt)
        if neuron.params.plasticity_mode != "legacy_multiplicative" or neuron.params.weight_decay_tau != 0 or neuron._ablation:
            raise ValueError("Observer scope is native legacy learning without ablations or passive decay")
        synapse = neuron.postsynaptic_points[synapse_id]
        before = float(synapse.u_i.info)
        incoming = neuron.input_buffer[synapse_id].copy()
        magnitude = float(np.linalg.norm(np.array([
            incoming[0]-synapse.u_i.info, incoming[1]-synapse.u_i.plast, *incoming[2:]])))
        eta = neuron.params.eta_post
        result = original(neuron, external_inputs, current_tick, dt)
        direction = 1. if current_tick-neuron.t_last_fire <= neuron.t_ref else -1.
        calculation = legacy_update(before, eta, direction, magnitude, neuron.params.w_min, neuron.params.w_max)
        active = bool(incoming[0] > 0)
        expected = calculation["expected_weight"] if active else before
        after = float(synapse.u_i.info)
        error = abs(after-expected)
        if error > 1e-10:
            raise AssertionError((current_tick, before, after, expected, error))
        rows.append({"tick": current_tick, "active": active, "weight_before": before,
                     "weight_after": after, "eta_effective": eta,
                     "input_info": float(incoming[0]), "error_magnitude": magnitude,
                     "credit_direction": direction, "t_ref": float(neuron.t_ref),
                     "last_spike": float(neuron.t_last_fire) if np.isfinite(neuron.t_last_fire) else -1.,
                     "has_spiked": bool(np.isfinite(neuron.t_last_fire)), "spiked": bool(neuron.O > 0),
                     "M0": float(neuron.M_vector[0]), **calculation,
                     "sign_changed": bool(before*after < 0), "accounting_error": error})
        return result

    Neuron.tick = tick
    try:
        yield
    finally:
        Neuron.tick = original


def run(recording, output, neuron_id, synapse_id, ticks):
    recording, output = Path(recording).resolve(), Path(output).resolve()
    manifest = json.loads((recording/"manifest.json").read_text())
    if manifest.get("weight_dynamics", "native") != "native":
        raise ValueError("This observer reconstructs native legacy updates only, not a replacement weight rule")
    if not 1 <= ticks <= manifest["trials"][-1]["stop"]:
        raise ValueError("Tick limit must fall within the recorded training")
    for name, expected in manifest["source_hashes"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed: {name}")
    features = []
    for clip in (0, 1):
        with np.load(recording/f"sensory-{clip}.npz") as raw:
            features.append({key: raw[key] for key in raw.files})
    net, core, neurons, synapses = fresh(recording/"config.json", manifest["seed"])
    if neuron_id not in net.network.neurons or synapse_id not in net.network.neurons[neuron_id].postsynaptic_points:
        raise ValueError("Requested postsynaptic port does not exist")
    output.mkdir(parents=True, exist_ok=False)
    rows, checks = [], []
    with observe_port(neuron_id, synapse_id, rows):
        for index, trial in enumerate(manifest["trials"]):
            if trial["start"] >= ticks:
                break
            shortened = {**trial, "stop": min(ticks, trial["stop"])}
            cells = episode(net, core, neurons, features, manifest["groups"], shortened)
            with np.load(recording/f"experience-{index:03d}.npz") as ref:
                exact_cells = bool(np.array_equal(cells, ref["cells"][:len(cells)]))
                exact_weights = (bool(np.array_equal(weight_values(synapses), ref["incoming_info_after"]))
                                 if shortened["stop"] == trial["stop"] else None)
            if not exact_cells or exact_weights is False:
                raise AssertionError("Observer replay differs from original training")
            checks.append({"episode": index, "exact_cells": exact_cells, "exact_weights": exact_weights})
            print(encode({"episode": index, "tick": net.current_tick, "exact": True,
                          "weight": rows[-1]["weight_after"]}), flush=True)
    columns = list(rows[0])
    np.savez_compressed(output/"synapse-ticks.npz", columns=columns,
                        values=np.array([[row[key] for key in columns] for row in rows]))
    first_flip = next((row for row in rows if row["sign_changed"]), None)
    first_bound = next((row for row in rows if row["active"] and row["unclipped_weight"] != row["expected_weight"]), None)
    summary = {"source_recording": str(recording), "source_manifest_sha256": hashlib.sha256((recording/"manifest.json").read_bytes()).hexdigest(),
               "observer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "neuron": neuron_id, "synapse": synapse_id, "ticks": ticks,
               "checks": checks, "first_clamp": first_bound, "first_sign_change": first_flip,
               "max_accounting_error": max(row["accounting_error"] for row in rows),
               "limits": "One retrospectively selected synapse. Reconstructs arithmetic and exact replay, not independent causal attribution or a population-wide biological claim. Calculation columns on inactive ticks are hypothetical; only active rows apply the rule."}
    (output/"summary.json").write_text(encode(summary)+"\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neuron", type=int, required=True)
    parser.add_argument("--synapse", type=int, required=True)
    parser.add_argument("--ticks", type=int, required=True)
    args = parser.parse_args()
    run(args.recording, args.output, args.neuron, args.synapse, args.ticks)
