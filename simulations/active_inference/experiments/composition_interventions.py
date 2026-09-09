"""Counterfactual gain interventions on the unchanged PAULA composition probe.

These are experimental manipulations, explicitly NOT a neural controller.
Each branch starts from the same seed/configuration and has its entire history
recorded. No claim of arbitrary checkpoint restoration is made.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
from pathlib import Path
import random
import time

from .composition_probe import build, encode, fingerprint, observe_inputs, snapshot, summarize


CONDITIONS = (("none", 0), ("freeze", 160), ("freeze", 400),
              ("restore_gains", 400), ("restore_gains_and_reseed", 400))


def run(destination, seed, phase, condition, intervention_tick, ticks=1260):
    destination.mkdir(parents=True, exist_ok=False)
    net, core = build(seed, "both", "shared", destination / "config.json")
    def state():
        result = snapshot(net)
        result["learning_rates"] = {str(i): [n.params.eta_post, n.params.eta_retro]
                                     for i, n in net.network.neurons.items()}
        return result
    initial = state()
    sources = fingerprint()
    sources[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {"seed": seed, "phase": phase, "condition": condition,
        "intervention_tick": intervention_tick, "ticks": ticks,
        "source_hashes": sources, "rng_after_build": random.getstate(),
        "resolved_initial": {str(i): asdict(n.params) for i, n in net.network.neurons.items()},
        "scope": "External causal experiment, not an agent, supervisor or biological fix",
        "reseed_warning": "The positive control supplies the original phase cue again. It is not memory retrieval."}
    (destination / "manifest.json").write_text(encode(manifest)+"\n")
    digest = hashlib.sha256()
    spikes = {i: [] for i in net.network.neurons}
    started = time.perf_counter()
    with observe_inputs() as inputs, gzip.open(destination / "ticks.jsonl.gz", "wt", compresslevel=3) as stream:
        stream.write(encode({"stage": "initial", "state": initial})+"\n")
        for t in range(ticks):
            action = None
            if condition != "none" and t == intervention_tick:
                action = {"condition": condition, "tick": t, "changes": []}
                for i, n in net.network.neurons.items():
                    if condition == "freeze":
                        action["changes"].append([i, "learning_rates", [n.params.eta_post, n.params.eta_retro], [0., 0.]])
                        n.params.eta_post = n.params.eta_retro = 0.
                    else:
                        for j, s in n.postsynaptic_points.items():
                            value = initial["neurons"][str(i)]["synapses"][str(j)][0]
                            action["changes"].append([i, "synapse", j, s.u_i.info, value])
                            s.u_i.info = value
                        for j, p in n.presynaptic_points.items():
                            value = initial["neurons"][str(i)]["terminals"][str(j)][0]
                            action["changes"].append([i, "terminal", j, p.u_o.info, value])
                            p.u_o.info = value
            external = []
            if t == 3 or (condition == "restore_gains_and_reseed" and t == intervention_tick):
                external = [[1, 1, 5.], [4+phase, 1, 5.]]
                for nid, sid, strength in external:
                    net.set_external_input(nid, sid, strength)
            inputs.clear()
            result = core.do_tick()
            if "error" in result or net.current_tick != t+1:
                raise RuntimeError(result)
            current = state()
            digest.update(encode(current).encode())
            for i, n in net.network.neurons.items():
                if n.O > 0:
                    spikes[i].append(t)
            stream.write(encode({"stage": "completed", "executed_tick": t, "external": external,
                "intervention": action, "delivered_inputs": inputs, "state": current})+"\n")
    after = fingerprint()
    after[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert after == sources, "Source changed during intervention run"
    summary = summarize(spikes, ticks, phase)
    summary.update(state_digest=digest.hexdigest(), elapsed_seconds=time.perf_counter()-started,
                   rng_final=random.getstate(), final_state=state())
    (destination / "summary.json").write_text(encode(summary)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 44, 77, 101])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "invocation.json").write_text(encode({"seeds": args.seeds, "conditions": CONDITIONS,
        "ticks": 1260, "phases": [0, 1], "mode": "both", "attachment": "shared"})+"\n")
    for seed in args.seeds:
        for phase in (0, 1):
            for condition, tick in CONDITIONS:
                name = f"seed{seed}_{condition}_at{tick}_phase{phase}"
                result = run(args.output / name, seed, phase, condition, tick)
                print(encode({"run": name, "late": result["windows"][-1]}), flush=True)


if __name__ == "__main__":
    main()
