"""Exploratory PAULA recall--mismatch--plasticity feedback preparation.

Protocol declared before the first run: two initially equivalent cues, two
outcomes, two plastic outcome-prediction cells, and two inhibitory-comparison
modulator cells. The environment pairs one cue with one outcome, then the other
cue with the other outcome. It never tells the circuit which trial/phase it is
in, sets a plasticity rate, resets neural state, or supplies a prediction error.

Question: can acquired recall inhibit a neural signal that accelerated its
acquisition, and can a second association be acquired while retaining the first?
This is an isolated, hand-wired eight-cell mechanism, NOT autonomous architecture
discovery, embodiment, a demonstration of ALERM's objective, or consciousness.
All post- and presynaptic basal learning rates remain positive throughout.

Controls: cut prediction-to-comparator feedback, cut the rate receptor while
retaining basal adaptation, or separate cues/outcomes with matched pulse counts.
The native comparator is an outcome-minus-prediction motif, not a complete
bidirectional prediction-error system: omission errors are outside this test.
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

from .composition_probe import encode, fingerprint, k, observe_inputs, snapshot
from neuron.extensions.experimental.plasticity_rate import PlasticityRateNeuron


MODES = ("closed_loop", "feedback_cut", "receptor_cut", "unpaired", "yoked", "yoked_shifted")
VARIANTS = ("original", "matched_ports", "outcome_competition", "shared_modulator", "corrective", "corrective_window", "corrective_window_slow_retro", "corrective_slow_retro", "corrective_window_resolved_retro", "corrective_resolved_retro")
BASAL = 1e-5
TRIAL_TICKS = 160


def build_association(destination, seed, mode, boost=9999., variant="original"):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    neurons, synapses, connections, external = [], [], [], []
    for nid in range(1, 9):
        prediction = nid in (5, 6)
        role = "cue" if nid < 3 else "outcome" if nid < 5 else "prediction" if prediction else "comparator"
        neurons.append(k.neuron(nid, r=.6, b=1.2, c=2,
            lam=1 if nid <= 4 else 2, eta_post=BASAL,
            # Slower adaptation of the prediction's outgoing regulator pathway,
            # still strictly positive and still neuromodulated. No weight clamp
            # or disabled retrograde path: this is a PAULA parameter experiment.
            # The 1e-9 variant is retained as a *failed precision control*: in
            # NumPy float32 arithmetic its unmodulated terminal updates vanish.
            # 1e-7 exceeds the observed resolution threshold near q=1.
            eta_retro=(1e-9 if prediction and variant.endswith("slow_retro") else
                       1e-7 if prediction and variant.endswith("resolved_retro") else BASAL),
            w_tref=[-100., 0.] if prediction and variant.startswith("corrective_window") else None,
            delta_decay=.95, meta={"role": role,
                "plasticity_rate_boost": boost if prediction and mode != "receptor_cut" else 0.,
                "plasticity_rate_index": 0, "plasticity_rate_half_saturation": .1}))
        terminal = k.term(nid, mod=[1., 0.] if nid >= 7 else [0., 0.])
        terminal["u_o"]["info"] = 0. if nid >= 7 else 1.5 if nid <= 2 else 1.
        synapses.append(terminal)
        if nid <= 4:
            # Second port is a declared unused perturbation port; two ports
            # keep the ordinary model's t_ref bounds non-inverted.
            synapses += [k.syn(nid, 0, 3., adapt=[0., 0.]), k.syn(nid, 1, 0., adapt=[0., 0.])]
            external += [k.ext(nid, 0), k.ext(nid, 1)]
        elif prediction:
            synapses += [k.syn(nid, sid, float(rng.uniform(.4275, .4725)), adapt=[0., 0.]) for sid in (0, 1)]
            synapses += [k.syn(nid, 2, 3., adapt=[0., 0.]), k.syn(nid, 3, 1., adapt=[1., 0.])]
            if variant != "original":
                # Matched unused port in the controls keeps t_ref bounds equal.
                # This motif assumes mutually exclusive sensory outcomes; it
                # does NOT encode which cue predicts which outcome.
                synapses.append(k.syn(nid, 4, -6., adapt=[0., 0.]))
            if variant == "outcome_competition" or variant.startswith("corrective"):
                connections.append(k.conn(9-nid, nid, 4))
            connections += [k.conn(1, nid, 0), k.conn(2, nid, 1), k.conn(nid-2, nid, 2)]
            if mode.startswith("yoked"):
                external.append(k.ext(nid, 3))
            else:
                connections.append(k.conn(nid+2, nid, 3))
                if variant == "shared_modulator" or variant.startswith("corrective"):
                    connections.append(k.conn(13-nid, nid, 3))
        else:
            synapses += [k.syn(nid, 0, 3., adapt=[0., 0.]), k.syn(nid, 1, -4., adapt=[0., 0.])]
            connections.append(k.conn(nid-4, nid, 0))
            if mode != "feedback_cut":
                connections.append(k.conn(nid-2, nid, 1))
    temporary = Path(k.build(neurons, synapses, connections, external))
    try:
        config = json.loads(temporary.read_text())
        destination.write_text(encode(config)+"\n")
        net, core = k.load(str(destination), neuron_class=PlasticityRateNeuron)
    finally:
        temporary.unlink()
    net.record_history = False
    return net, core


def schedule(training_trials=20, mapping=0, order=0, probe_repeats=3, challenge="standard"):
    """External protocol only. Returned labels are never supplied to neurons."""
    trials = []
    def add(phase, cue, paired):
        trials.append({"index": len(trials), "start": len(trials)*TRIAL_TICKS,
            "stop": (len(trials)+1)*TRIAL_TICKS, "phase": phase, "cue": cue,
            "outcome": None if cue is None else cue ^ mapping, "paired": paired,
            "cue_period": 2, "cue_stop": 36})
    for phase, trained in (("before", None), ("after_first", order), ("after_second", 1-order)):
        if trained is not None:
            for _ in range(training_trials):
                add("train_first" if phase == "after_first" else "train_second", trained, True)
        for _ in range(probe_repeats):
            for cue in (0, 1):
                add(phase, cue, False)
    # The last block also tests survival of many unrewarded recalls, with no
    # state reset/freeze. A long silent interval is represented explicitly.
    for _ in range(10):
        add("silence", None, False)
        trials[-1]["outcome"] = None
    for _ in range(probe_repeats):
        for cue in (0, 1):
            add("retention", cue, False)
    if challenge == "reversal":
        for _ in range(training_trials):
            for cue in (order, 1-order):
                add("reversal_train", cue, True)
                trials[-1]["outcome"] ^= 1
        for _ in range(probe_repeats):
            for cue in (0, 1):
                add("reversal_probe", cue, False)
                trials[-1]["outcome"] ^= 1
    elif challenge in ("timing_transfer", "timing_recovery"):
        periods = (3, 4, 6, 2) if challenge == "timing_recovery" else (3, 4, 6)
        for period in periods:
            for _ in range(probe_repeats):
                for cue in (0, 1):
                    add(f"transfer_period{period}", cue, False)
                    trials[-1].update(cue_period=period, cue_stop=18*period)
    elif challenge != "standard":
        raise ValueError(challenge)
    return trials


def stimuli(trial, t, mode):
    rel = t-trial["start"]
    pulses = []
    if trial["cue"] is not None and 0 <= rel < trial["cue_stop"] and rel % trial["cue_period"] == 0:
        pulses.append([1+trial["cue"], 0, 5.])
    onset = 80 if mode == "unpaired" else 24
    if trial["paired"] and onset <= rel < onset+12 and rel % 2 == 0:
        pulses.append([3+trial["outcome"], 0, 5.])
    return pulses


def summarize_trial(trial, spikes, neurons):
    start, stop = trial["start"], trial["stop"]
    outcome = trial["outcome"]
    # The pre-US window is conservative: it ends at the first *external*
    # outcome pulse, before its transduction and dendritic delays.
    counts = {str(n): sum(start <= t < stop for t in ts) for n, ts in spikes.items()}
    early = {str(n): sum(start <= t < start+24 for t in spikes[n]) for n in (5, 6)}
    return {**trial, "spikes": counts, "pre_us_prediction_spikes": early,
        "correct_prediction_spikes": None if outcome is None else counts[str(5+outcome)],
        "wrong_prediction_spikes": None if outcome is None else counts[str(6-outcome)],
        "cue_weights": {str(n): [neurons[n].postsynaptic_points[s].u_i.info for s in (0, 1)] for n in (5, 6)}}


def run(destination, seed=11, mode="closed_loop", mapping=0, order=0,
        training_trials=20, boost=9999., observed=True, challenge="standard", reference=None, variant="original"):
    if mode not in MODES or variant not in VARIANTS or mapping not in (0, 1) or order not in (0, 1) or training_trials <= 0:
        raise ValueError("Invalid experimental condition")
    destination.mkdir(parents=True, exist_ok=False)
    net, core = build_association(destination / "config.json", seed, mode, boost, variant)
    trials = schedule(training_trials, mapping, order, challenge=challenge)
    replay = {}
    if mode.startswith("yoked"):
        if reference is None:
            raise ValueError("Yoked controls require an explicit closed-loop reference")
        ref_manifest = json.loads((reference / "manifest.json").read_text())
        assert ref_manifest["mode"] == "closed_loop"
        assert ref_manifest["variant"] == variant
        assert ref_manifest["seed"] == seed and ref_manifest["mapping"] == mapping and ref_manifest["order"] == order
        assert ref_manifest["trials"] == trials, "Reference and replay protocol differ"
        shift = 60 if mode == "yoked_shifted" else 0
        # Replay actual pure-modulator packets, not a fitted/scheduled learning
        # rate. Preserve counts; rotate arrivals within their original trials.
        with gzip.open(reference / "ticks.jsonl.gz", "rt") as f:
            for line in f:
                row = json.loads(line)
                if row["stage"] != "completed":
                    continue
                t = row["executed_tick"]
                for nid in (5, 6):
                    port = row["delivered_inputs"][str(nid)][3]
                    assert port[0] == port[1] == 0.
                    if any(port[2:]):
                        target_tick = (t//TRIAL_TICKS)*TRIAL_TICKS + (t % TRIAL_TICKS+shift) % TRIAL_TICKS
                        replay.setdefault(target_tick, []).append([nid, 3, 0., port[2:]])
    sources = fingerprint()
    sources[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {"seed": seed, "mode": mode, "mapping": mapping, "order": order,
        "training_trials": training_trials, "boost": boost, "basal": BASAL,
        "variant": variant,
        "challenge": challenge, "reference": None if reference is None else str(reference.resolve()),
        "reference_trace_sha256": None if reference is None else hashlib.sha256((reference / "ticks.jsonl.gz").read_bytes()).hexdigest(),
        "observed": observed, "trials": trials, "source_hashes": sources,
        "resolved": {str(i): {"class": type(n).__module__+"."+type(n).__name__,
            "parameters": asdict(n.params), "metadata": n.metadata} for i, n in net.network.neurons.items()},
        "scope": "Exploratory isolated recall--mismatch--plasticity loop; no energy budget, body, omission error, or consciousness test"}
    (destination / "manifest.json").write_text(encode(manifest)+"\n")
    digest = hashlib.sha256()
    spikes = {i: [] for i in net.network.neurons}
    reports = []
    started = time.perf_counter()
    with observe_inputs(observed) as inputs, gzip.open(destination / "ticks.jsonl.gz", "wt", compresslevel=3) as stream:
        stream.write(encode({"stage": "initial", "state": snapshot(net)})+"\n")
        for trial in trials:
            rates = {i: 0. for i in (5, 6)}
            for t in range(trial["start"], trial["stop"]):
                external = stimuli(trial, t, mode)
                for nid, sid, amplitude in external:
                    net.set_external_input(nid, sid, amplitude)
                replayed = replay.get(t, [])
                for nid, sid, amplitude, mod in replayed:
                    net.set_external_input(nid, sid, amplitude, mod=np.array(mod))
                inputs.clear()
                result = core.do_tick()
                if "error" in result or net.current_tick != t+1:
                    raise RuntimeError(result)
                state = snapshot(net)
                state["plasticity_rates"] = {}
                for i, n in net.network.neurons.items():
                    assert n.params.eta_post == BASAL > 0
                    assert n.params.eta_retro == manifest["resolved"][str(i)]["parameters"]["eta_retro"] > 0
                    state["plasticity_rates"][str(i)] = {
                        "used_multiplier": n.last_tick_rate_multiplier,
                        "next_multiplier": n.rate_multiplier(), "basal": BASAL,
                        "basal_retro": n.params.eta_retro}
                    if i in rates:
                        rates[i] += BASAL*n.last_tick_rate_multiplier
                        if observed:
                            assert inputs[str(i)][3][0] == 0., "Modulator supplied excitation"
                    if n.O > 0:
                        spikes[i].append(t)
                digest.update(encode(state).encode())
                stream.write(encode({"stage": "completed", "executed_tick": t,
                    "trial": trial["index"], "external": external,
                    "replayed_modulator": replayed,
                    "delivered_inputs": dict(inputs), "state": state})+"\n")
            report = summarize_trial(trial, spikes, net.network.neurons)
            report["rate_integrals"] = rates
            reports.append(report)
    after = fingerprint()
    after[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert sources == after, "Source changed during recording"
    probes = [r for r in reports if r["phase"] in ("after_second", "retention")]
    challenge_probes = [r for r in reports if r["phase"] == "reversal_probe" or r["phase"].startswith("transfer")]
    result = {"trials": reports, "spike_ticks": spikes, "state_digest": digest.hexdigest(),
        "ticks": trials[-1]["stop"], "elapsed_seconds": time.perf_counter()-started,
        "both_associations_retained": all(r["correct_prediction_spikes"] >= 2 and r["wrong_prediction_spikes"] == 0 for r in probes),
        "challenge_passed": None if not challenge_probes else all(r["correct_prediction_spikes"] >= 2 and r["wrong_prediction_spikes"] == 0 for r in challenge_probes),
        "initially_naive": all(r["spikes"]["5"] == r["spikes"]["6"] == 0 for r in reports if r["phase"] == "before"),
        "scope": manifest["scope"]}
    (destination / "summary.json").write_text(encode(result)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--mode", choices=MODES, default="closed_loop")
    parser.add_argument("--mapping", type=int, choices=(0, 1), default=0)
    parser.add_argument("--order", type=int, choices=(0, 1), default=0)
    parser.add_argument("--training-trials", type=int, default=20)
    parser.add_argument("--boost", type=float, default=9999.)
    parser.add_argument("--unobserved", action="store_true")
    parser.add_argument("--challenge", choices=("standard", "reversal", "timing_transfer", "timing_recovery"), default="standard")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--variant", choices=VARIANTS, default="original")
    args = parser.parse_args()
    result = run(args.output, args.seed, args.mode, args.mapping, args.order,
        args.training_trials, args.boost, not args.unobserved, args.challenge, args.reference, args.variant)
    print(encode({k: v for k, v in result.items() if k not in ("trials", "spike_ticks")}))


if __name__ == "__main__":
    main()
