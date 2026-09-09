"""Fixed, single-worker confirmation after exploratory association recordings.

No parameter search. Writes its complete case list before any simulation runs;
every condition records full ticks and receives an independent equation audit.
Live agents and embodied acceptance suites are not started.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

from .adaptive_association import run
from .association_analysis import audit
from .composition_probe import encode


def cases(suite="initial"):
    if suite == "resolved":
        return [
            *[dict(seed=seed, mapping=mapping, order=order, variant="corrective_window_resolved_retro", challenge="reversal")
              for seed in (13, 29, 47, 71, 97) for mapping in (0, 1) for order in (0, 1)],
            *[dict(seed=seed, order=order, variant="corrective_window_resolved_retro", challenge="reversal")
              for seed, order in ((44, 0), (44, 1), (101, 0))],
            *[dict(seed=seed, variant="corrective_resolved_retro", challenge="reversal")
              for seed in (13, 29, 47, 71, 97)],
            dict(seed=29, variant="corrective_window_resolved_retro", challenge="reversal", training_trials=100, name="long_exposure"),
            dict(seed=29, variant="corrective_window_resolved_retro", challenge="timing_transfer", name="timing_transfer"),
            dict(seed=29, variant="corrective_window_resolved_retro", challenge="reversal", observed=False, name="unobserved"),
        ]
    if suite == "timescale":
        # New initial-weight seeds, not inspected in the exploratory work.
        conditions = [dict(seed=seed, mapping=mapping, order=order,
            variant="corrective_window_slow_retro", challenge="reversal")
            for seed in (7, 19, 37, 61, 89) for mapping in (0, 1) for order in (0, 1)]
        # Exact failed conditions from the earlier confirmation; these are
        # targeted replications, not held-out evidence.
        conditions += [dict(seed=seed, order=order, variant="corrective_window_slow_retro", challenge="reversal")
                       for seed, order in ((44, 0), (44, 1), (101, 0))]
        # Does the timing-window intervention remain necessary after slowing
        # retrograde adaptation? All other wiring/parameters are matched.
        conditions += [dict(seed=seed, variant="corrective_slow_retro", challenge="reversal")
                       for seed in (7, 19, 37, 61, 89)]
        conditions += [dict(seed=19, variant="corrective_window_slow_retro", challenge="reversal",
                            training_trials=100, name="long_exposure"),
                       dict(seed=19, variant="corrective_window_slow_retro", challenge="timing_transfer", name="timing_transfer"),
                       dict(seed=19, variant="corrective_window_slow_retro", challenge="reversal", observed=False, name="unobserved")]
        return conditions
    conditions = []
    # Held fixed after seed11 discovery. Perturb initial cue weights, invert
    # the learned mapping, and invert acquisition order without changing wiring.
    for seed in (11, 23, 44, 77, 101):
        for mapping in (0, 1):
            for order in (0, 1):
                conditions.append(dict(seed=seed, mapping=mapping, order=order,
                    variant="corrective_window", challenge="reversal"))
        # Mechanistic controls matched for the added fifth input port.
        for variant in ("matched_ports", "outcome_competition", "shared_modulator", "corrective"):
            conditions.append(dict(seed=seed, variant=variant, challenge="reversal"))
        conditions.append(dict(seed=seed, variant="corrective_window", mode="receptor_cut", challenge="reversal"))
    # Same neural-packet replay and a 60-tick within-trial time shift. Reference
    # is generated first; the replay is a causal control, not an agent design.
    for mode in ("closed_loop", "yoked", "yoked_shifted", "feedback_cut", "unpaired"):
        conditions.append(dict(seed=23, variant="original", mode=mode,
            challenge="standard", reference_name="original_reference" if mode.startswith("yoked") else None,
            name="original_reference" if mode == "closed_loop" else f"original_{mode}"))
    # Transfer failure boundaries and input-instrumentation equivalence.
    conditions.append(dict(seed=23, variant="corrective_window", challenge="timing_transfer", name="timing_transfer"))
    conditions.append(dict(seed=23, variant="corrective_window", challenge="reversal", observed=False, name="unobserved"))
    return conditions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("initial", "timescale", "resolved"), default="initial")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    conditions = cases(args.suite)
    (args.output / "protocol.json").write_text(encode({"cases": conditions,
        "default_training_trials_per_cue": 20, "boost": 9999., "workers": 1,
        "status": "Exploratory circuit selected before this confirmation batch; no population inference from five seeds",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})+"\n")
    results = []
    started = time.perf_counter()
    for index, condition in enumerate(conditions):
        spec = dict(condition)
        name = spec.pop("name", f"case{index:02d}_seed{spec['seed']}_{spec['variant']}")
        reference = spec.pop("reference_name", None)
        if reference is not None:
            spec["reference"] = args.output / reference
        result = run(args.output / name, **spec)
        checked = audit(args.output / name) if spec.get("observed", True) else None
        if checked is not None:
            (args.output / name / "audit.json").write_text(encode(checked)+"\n")
        report = {"name": name, "condition": condition,
            "both_associations_retained": result["both_associations_retained"],
            "challenge_passed": result["challenge_passed"],
            "initially_naive": result["initially_naive"],
            "state_digest": result["state_digest"], "audit": checked}
        results.append(report)
        print(encode({k: v for k, v in report.items() if k not in ("audit", "state_digest")}), flush=True)
    observer_seed = {"initial": 23, "timescale": 19, "resolved": 29}[args.suite]
    observer_variant = {"initial": "corrective_window", "timescale": "corrective_window_slow_retro", "resolved": "corrective_window_resolved_retro"}[args.suite]
    target = next(r for r in results if r["condition"] == dict(seed=observer_seed, mapping=0, order=0, variant=observer_variant, challenge="reversal"))
    assert target["state_digest"] == results[-1]["state_digest"], "Input observer changed circuit state"
    (args.output / "batch_summary.json").write_text(encode({"runs": results,
        "elapsed_seconds": time.perf_counter()-started, "observer_equivalence": True})+"\n")


if __name__ == "__main__":
    main()
