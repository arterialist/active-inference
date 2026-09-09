"""Native-PAULA relational-state composition experiment, one worker only.

Run from active-inference: python -m simulations.active_inference.experiments.composition_probe
  --output .live/research/UNIQUE_NAME --ticks 840 --seeds 11 --modes frozen both

This is an isolated mechanistic preparation, not an embodied acceptance test.
No trained decoder, supervisor, reward injection or neuron extension is used.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import gzip
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
from simulations.paula_loader import ensure_paula_available

PAULA_ROOT = ensure_paula_available()
from neuron.neuron import Neuron
from neuron import network as network_module
from paula_agent import ckit as k

RATES = {"frozen": (0., 0.), "post": (.01, 0.), "retro": (0., .01),
         "both": (.01, .01), "slow": (.001, .001)}
RING_IDS = tuple(range(1, 7))


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(type(value).__name__)


def encode(value):
    return json.dumps(value, default=plain, allow_nan=False, sort_keys=True, separators=(",", ":"))


def fingerprint():
    files = sorted((PAULA_ROOT / "neuron").rglob("*.py"))
    files += [PAULA_ROOT / "paula_agent/ckit.py", Path(__file__).resolve()]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def build(seed, mode, attachment, destination, *, cut_consumer_retro=False, tref_frozen=False):
    """Two three-cell synfire rings and two ordinary PAULA coincidence consumers.

    Each ring cell has three actual ports: recurrent, birth and perturbation.
    This also gives non-inverted t_ref bounds, unlike a one-port preparation.
    Both output terminals exist in all conditions; only wiring changes.
    """
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    post, retro = RATES[mode]
    neurons, synapses, connections, external = [], [], [], []
    for nid in RING_IDS:
        neurons.append(k.neuron(nid, r=.6, b=1.2, c=6, lam=3,
            eta_post=post, eta_retro=retro, delta_decay=.95,
            meta={"role": "ring_a" if nid < 4 else "ring_b"}))
        synapses += [k.syn(nid, 0, float(rng.uniform(3.8, 4.2)), 6),
                     k.syn(nid, 1, 3., 1), k.syn(nid, 2, 1., 1),
                     k.term(nid, tid=900), k.term(nid, tid=901)]
        external += [k.ext(nid, 1), k.ext(nid, 2)]
    for base in (1, 4):
        for i in range(3):
            connections.append(k.conn(base+i, base+(i+1) % 3, 0, stid=900))
    for nid in (7, 8):
        neurons.append(k.neuron(nid, r=1.4, b=2., c=8, lam=2,
            eta_post=post, eta_retro=retro, delta_decay=.95,
            meta={"role": "consumer_aligned" if nid == 7 else "consumer_shifted"}))
        synapses += [k.syn(nid, 0, 2., 1), k.syn(nid, 1, 2., 1),
                     k.syn(nid, 2, 1., 1), k.term(nid)]
        external += [k.ext(nid, 2)]
    if attachment != "none":
        terminal = 900 if attachment == "shared" else 901
        for consumer, b_cell in ((7, 4), (8, 5)):
            connections += [k.conn(1, consumer, 0, stid=terminal),
                            k.conn(b_cell, consumer, 1, stid=terminal)]
    temporary = Path(k.build(neurons, synapses, connections, external))
    try:
        config = json.loads(temporary.read_text())
        destination.write_text(encode(config) + "\n")
        net, core = k.load(str(destination), graded=False)
    finally:
        # Delete only the exact temporary config created above by ckit.
        temporary.unlink()
    net.record_history = False
    for n in net.network.neurons.values():
        if tref_frozen:
            n._ablation.add("tref_frozen")
        if cut_consumer_retro and n.id in (7, 8):
            n._ablation.add("retrograde_disabled")
        assert type(n) is Neuron
        assert n.lower_t_ref_bound <= n.t_ref <= n.upper_t_ref_bound
    return net, core


def snapshot(net):
    def wheel(slots):
        return [[s.arrival_tick, type(s.event).__name__, s.event]
                for slot in slots for s in slot]
    return {"tick": net.current_tick, "neurons": {str(nid): {
        "S": n.S, "O": n.O, "r": n.r, "b": n.b,
        "t_ref": n.t_ref, "F_avg": n.F_avg, "M": n.M_vector,
        "t_last_fire": None if not np.isfinite(n.t_last_fire) else n.t_last_fire,
        "dendritic_queue": sorted(n.propagation_queue),
        "synapses": {str(sid): [p.u_i.info, p.u_i.plast, p.u_i.adapt, p.potential]
                     for sid, p in n.postsynaptic_points.items()},
        "terminals": {str(tid): [p.u_o.info, p.u_o.mod, p.u_i_retro]
                      for tid, p in n.presynaptic_points.items()},
        } for nid, n in net.network.neurons.items()},
        "presynaptic_wheel": wheel(net.presynaptic_wheel),
        "retrograde_wheel": wheel(net.retrograde_wheel)}


@contextmanager
def observe_inputs(enabled=True):
    """Read-only probe around the unchanged base method; no subclass dynamics.

    Restored on exit. --unobserved permits an instrumentation equivalence test.
    Only use in this single-worker process, never in a running live server.
    """
    original = Neuron.tick
    inputs = {}
    def observed(n, external_inputs, current_tick, dt=1.):
        inputs[str(n.id)] = n.input_buffer.tolist()
        return original(n, external_inputs, current_tick, dt)
    if enabled:
        Neuron.tick = observed
    try:
        yield inputs
    finally:
        Neuron.tick = original


def summarize(spikes, ticks, phase):
    windows = []
    for start in range(0, ticks, 105):
        stop = min(start+105, ticks)
        counts = {str(n): sum(start <= t < stop for t in ts) for n, ts in spikes.items()}
        winner = 7 if phase == 0 else 8
        loser = 15-winner
        denominator = counts[str(winner)] + counts[str(loser)]
        windows.append({"start": start, "stop": stop, "spikes": counts,
                        "expected_consumer": winner,
                        "selectivity": None if not denominator else
                        (counts[str(winner)]-counts[str(loser)])/denominator,
                        "both_rings_active": all(counts[str(i)] > 0 for i in RING_IDS)})
    return {"windows": windows, "spike_ticks": spikes,
            "last_ring_spike": {str(i): max(spikes[i], default=None) for i in RING_IDS},
            "scope": "Finite isolated preparation; selectivity is descriptive, not acceptance."}


def run(destination, seed, mode, attachment, phase, ticks, *,
        cut_consumer_retro=False, tref_frozen=False, observed=True):
    destination.mkdir(parents=True, exist_ok=False)
    net, core = build(seed, mode, attachment, destination / "config.json",
                      cut_consumer_retro=cut_consumer_retro, tref_frozen=tref_frozen)
    before_sources = fingerprint()
    manifest = {"seed": seed, "mode": mode, "attachment": attachment, "phase": phase,
                "ticks": ticks, "cut_consumer_retro": cut_consumer_retro,
                "tref_frozen": tref_frozen, "observed_inputs": observed,
                "python": platform.python_version(), "numpy": np.__version__,
                "source_hashes": before_sources, "rng_after_build": random.getstate(),
                "cleft_delay_range": [network_module.MIN_CONNECTION_SIGNAL_TRAVEL_TICKS,
                                      network_module.MAX_CONNECTION_SIGNAL_TRAVEL_TICKS],
                "resolved": {str(i): {"parameters": asdict(n.params),
                    "class": type(n).__module__+"."+type(n).__name__,
                    "ablations": sorted(n._ablation)} for i, n in net.network.neurons.items()},
                "birth": [[3, 1, 1, 5.], [3, 4+phase, 1, 5.]],
                "limitations": ["No supervisor yet", "No body", "No consciousness inference",
                    "No externally injected modulators; endogenous terminal mod starts at zero",
                    "ckit beta/gamma retained; delta_decay set to base model .95",
                    "Frozen finite-range wiring with seed-specific recurrent gains in [3.8,4.2]"]}
    (destination / "manifest.json").write_text(encode(manifest)+"\n")
    spikes = {i: [] for i in net.network.neurons}
    digest = hashlib.sha256()
    started = time.perf_counter()
    with observe_inputs(observed) as inputs, gzip.open(destination / "ticks.jsonl.gz", "wt", compresslevel=3) as stream:
        stream.write(encode({"stage": "initial", "state": snapshot(net)})+"\n")
        for t in range(ticks):
            inputs.clear()
            external = []
            if t == 3:
                external = [[1, 1, 5.], [4+phase, 1, 5.]]
                for nid, sid, strength in external:
                    net.set_external_input(nid, sid, strength)
            result = core.do_tick()
            if "error" in result or result.get("paused") or net.current_tick != t+1:
                raise RuntimeError(f"tick {t}: {result}")
            state = snapshot(net)
            digest.update(encode(state).encode())
            for i, n in net.network.neurons.items():
                if n.O > 0:
                    spikes[i].append(t)
            stream.write(encode({"stage": "completed", "executed_tick": t,
                "external": external, "delivered_inputs": inputs if observed else None,
                "state": state})+"\n")
    if fingerprint() != before_sources:
        raise RuntimeError("Source files changed during this run; record is not a fixed-model experiment")
    summary = summarize(spikes, ticks, phase)
    summary.update({"state_digest": digest.hexdigest(), "elapsed_seconds": time.perf_counter()-started,
                    "trace_bytes": (destination / "ticks.jsonl.gz").stat().st_size,
                    "final_state": snapshot(net), "rng_final": random.getstate()})
    (destination / "summary.json").write_text(encode(summary)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=840)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11])
    parser.add_argument("--modes", choices=RATES, nargs="+", default=["frozen", "post", "retro", "both"])
    parser.add_argument("--attachments", choices=["none", "shared", "separate"], nargs="+", default=["none", "shared", "separate"])
    parser.add_argument("--phases", type=int, choices=[0, 1], nargs="+", default=[0, 1])
    parser.add_argument("--cut-consumer-retro", action="store_true")
    parser.add_argument("--tref-frozen", action="store_true")
    parser.add_argument("--unobserved", action="store_true")
    args = parser.parse_args()
    if args.ticks < 1:
        parser.error("ticks must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "invocation.json").write_text(encode(vars(args) | {"output": str(args.output)})+"\n")
    for seed in args.seeds:
        for mode in args.modes:
            for attachment in args.attachments:
                for phase in args.phases:
                    name = f"seed{seed}_{mode}_{attachment}_phase{phase}"
                    result = run(args.output / name, seed, mode, attachment, phase, args.ticks,
                        cut_consumer_retro=args.cut_consumer_retro, tref_frozen=args.tref_frozen,
                        observed=not args.unobserved)
                    print(encode({"run": name, "late": result["windows"][-1],
                                  "seconds": round(result["elapsed_seconds"], 2)}), flush=True)


if __name__ == "__main__":
    main()
