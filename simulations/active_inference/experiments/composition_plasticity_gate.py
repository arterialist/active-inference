"""Nonzero basal plasticity with a PAULA neuron delivering a rate modulator.

The ninth cell is externally stimulated as an experimental preparation. It is
not yet an autonomous supervisor. Its release, transmission, receptor filtering
and rate effect are neural/local. No Python code sets rates during the run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from .composition_probe import build, encode, fingerprint, k, observe_inputs, snapshot, summarize
from .composition_analysis import expected_ring_ticks
from neuron.extensions.experimental.plasticity_rate import PlasticityRateNeuron


MODES = ("basal", "burst", "tonic", "receptor_cut")


def build_gated(destination, seed, mode):
    # Reuse the fixed composition motif. Keep its source config as provenance.
    build(seed, "slow", "shared", destination / "source_config.json")
    config = json.loads((destination / "source_config.json").read_text())
    for cell in config["neurons"]:
        cell["params"]["eta_post"] = cell["params"]["eta_retro"] = 1e-5
        cell["metadata"].update(plasticity_rate_boost=0. if mode == "receptor_cut" else 999.,
                                plasticity_rate_index=0, plasticity_rate_half_saturation=.1)
    # Channel 0 is unused in this isolated motif. Do not repurpose a live
    # agent's stress/reward channels or pretend its current wire has a third one.
    for synapse in config["synaptic_points"]:
        if synapse.get("synapse_id") == 2:
            synapse["u_i"]["adapt"] = [1., 0.]
    config["external_inputs"] = [x for x in config["external_inputs"] if x["target_synapse"] != 2]
    source = k.neuron(9, r=.6, b=1.2, c=2, lam=1, eta_post=1e-5, eta_retro=1e-5,
                      delta_decay=.95, meta={"role": "plasticity_modulator"})
    source["params"]["num_inputs"] = 3
    config["neurons"].append(source)
    terminal = k.term(9, mod=[1., 0.])
    terminal["u_o"]["info"] = 0.  # Pure modulatory transmission; no hidden excitation.
    config["synaptic_points"] += [k.syn(9, i, 3., 1) for i in range(3)] + [terminal]
    config["connections"] += [k.conn(9, i, 2) for i in range(1, 9)]
    config["external_inputs"] += [k.ext(9, i) for i in range(3)]
    final = destination / "config.json"
    final.write_text(encode(config)+"\n")
    random.seed(seed)
    np.random.seed(seed)
    net, core = k.load(str(final), neuron_class=PlasticityRateNeuron)
    net.record_history = False
    return net, core


def run(destination, seed, phase, mode, ticks=4200):
    destination.mkdir(parents=True, exist_ok=False)
    net, core = build_gated(destination, seed, mode)
    sources = fingerprint()
    sources[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {"seed": seed, "phase": phase, "mode": mode, "ticks": ticks,
        "source_hashes": sources, "rng_after_build": random.getstate(),
        "resolved": {str(i): {"class": type(n).__module__+"."+type(n).__name__,
                     "parameters": asdict(n.params), "metadata": n.metadata}
                     for i, n in net.network.neurons.items()},
        "modulator_stimulus": "9:0 amplitude5, even ticks [200,280) burst or [200,T) tonic",
        "scope": "Isolated rate-gate test; no autonomous supervisor, new memory acquisition, body or consciousness test",
        "channel": "Previously unused index0; no additional wire dimension in this preparation"}
    (destination / "manifest.json").write_text(encode(manifest)+"\n")
    spikes = {i: [] for i in net.network.neurons}
    digest = hashlib.sha256()
    initial = snapshot(net)
    peak_multiplier = 1.
    rate_totals = {i: 0. for i in range(1, 9)}
    started = time.perf_counter()
    with observe_inputs() as inputs, gzip.open(destination / "ticks.jsonl.gz", "wt", compresslevel=3) as stream:
        stream.write(encode({"stage": "initial", "state": initial})+"\n")
        for t in range(ticks):
            external = []
            if t == 3:
                external += [[1, 1, 5.], [4+phase, 1, 5.]]
            if t % 2 == 0 and t >= 200 and (mode == "tonic" or (mode in ("burst", "receptor_cut") and t < 280)):
                external.append([9, 0, 5.])
            for nid, sid, amplitude in external:
                net.set_external_input(nid, sid, amplitude)
            inputs.clear()
            result = core.do_tick()
            if "error" in result or net.current_tick != t+1:
                raise RuntimeError(result)
            state = snapshot(net)
            rates = {}
            for i, n in net.network.neurons.items():
                assert n.params.eta_post == n.params.eta_retro == 1e-5
                rates[str(i)] = {"used_multiplier": n.last_tick_rate_multiplier,
                                 "next_multiplier": n.rate_multiplier(), "basal": 1e-5}
                if i <= 8:
                    peak_multiplier = max(peak_multiplier, n.last_tick_rate_multiplier)
                    rate_totals[i] += 1e-5*n.last_tick_rate_multiplier
                    assert inputs[str(i)][2][0] == 0., "Modulatory route injected excitation"
                if n.O > 0:
                    spikes[i].append(t)
            state["plasticity_rates"] = rates
            digest.update(encode(state).encode())
            stream.write(encode({"stage": "completed", "executed_tick": t, "external": external,
                                "delivered_inputs": inputs, "state": state})+"\n")
    after = fingerprint()
    after[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert sources == after
    result = summarize(spikes, ticks, phase)
    result.update(state_digest=digest.hexdigest(), elapsed_seconds=time.perf_counter()-started,
        rng_final=random.getstate(), final_state=state, peak_multiplier=peak_multiplier,
        rate_integrals=rate_totals,
        ring_exact=all(spikes[i] == expected_ring_ticks(i, phase, ticks) for i in range(1, 7)),
        readout_exact=spikes[7+phase] == list(range(6, ticks, 21)) and not spikes[8-phase],
        recurrent_weight_changes={str(i): state["neurons"][str(i)]["synapses"]["0"][0]
             - initial["neurons"][str(i)]["synapses"]["0"][0] for i in range(1, 7)})
    (destination / "summary.json").write_text(encode(result)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 44, 77, 101])
    parser.add_argument("--ticks", type=int, default=4200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "invocation.json").write_text(encode({"seeds": args.seeds, "ticks": args.ticks,
                                                          "phases": [0, 1], "modes": MODES})+"\n")
    for seed in args.seeds:
        for phase in (0, 1):
            for mode in MODES:
                name = f"seed{seed}_{mode}_phase{phase}"
                result = run(args.output / name, seed, phase, mode, args.ticks)
                print(encode({"run": name, "ring_exact": result["ring_exact"],
                              "readout_exact": result["readout_exact"],
                              "peak_multiplier": result["peak_multiplier"]}), flush=True)


if __name__ == "__main__":
    main()
