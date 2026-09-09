"""Embodied causal test: toxin teaching makes a neutral food odour avoidable.

The teaching source deliberately separates conditioned stimulus and
unconditioned stimulus: a food source sits in front of the head while a toxin
contact occurs at the body.  The later probe contains only that food source.
Thus the normal toxin sensor/reflex cannot explain a change in the probe
trajectory.  The opt-in MB->lateral-horn route must carry the causal chain:

    physical food odour + physical toxin contact -> STG_T/DAN -> KC--MBON
    plasticity -> AVOID -> LH_LEFT/LH_RIGHT -> opposite TL/TR -> relay,
    graded muscle, MuJoCo body.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from simulations.active_inference import aif_agent3d as ag


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NEURON_MODEL = ROOT.parent / "neuron-model"
FAR = [100.0, 100.0]
TRAIN_FOOD = [-0.8, 0.0]
PROBES = {
    11: [-1.1, 0.75],
    23: [-1.1, -0.75],
    44: [-1.1, 0.75],
    77: [-1.1, -0.75],
}
BUILD = {
    "w_km0": 0.4,
    "w_kc_max": 2.0,
    "w_av_mbon": 6.0,
    "mb_lh": True,
    "r_lh": 0.8,
}
CASES = {
    "taught": BUILD,
    # Preserve toxin contact, STG_T, and KC--MBON learning, but sever just the
    # learned-output projection onto the LH-like turn population.
    "lh_output_ablation": {**BUILD, "w_lh_avoid": 0.0},
    "teaching_ablation": {**BUILD, "w_trig": 0.0},
}


def _write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _reset_body(world) -> None:
    world.data.qpos[world.jx] = 0.0
    world.data.qpos[world.jy] = 0.0
    world.data.qpos[world.jyaw] = 0.0
    world._in = [False] * len(world._in)
    world.pending_event = None
    mujoco.mj_forward(world.model, world.data)


def _row(current, phase: str, source: list[float]) -> dict[str, object]:
    world = current.world
    x, y, yaw = world.pose()
    return {
        "phase": phase,
        "neural_tick": current.t,
        "physics_time": float(world.data.time),
        "pose": [x, y, yaw],
        "source_distance": float(np.hypot(x - source[0], y - source[1])),
        "food_eaten": int(world.eaten),
        "toxin_hits": int(world.tox_hits),
        "sensor_drives": dict(getattr(current, "last_sensor_drives", {})),
        "stg_t": int(current.nb[ag.STG_T].O > 0),
        "mbon": int(sum(current.nb[nid].O > 0 for nid in ag.MBONP)),
        "avoid": int(current.nb[ag.AVOID].O > 0),
        "lh_left": int(sum(current.nb[nid].O > 0 for nid in ag.LH_LEFT)),
        "lh_right": int(sum(current.nb[nid].O > 0 for nid in ag.LH_RIGHT)),
        "turn_left": int(current.nb[ag.TL].O > 0),
        "turn_right": int(current.nb[ag.TR].O > 0),
        "muscles": [
            float(current.nb[ag.MLp].S),
            float(current.nb[ag.MLr].S),
            float(current.nb[ag.MRp].S),
            float(current.nb[ag.MRr].S),
        ],
        "actuator_ctrl": [float(v) for v in world.data.ctrl],
    }


def _train(agent, trials: int, train_steps: int, substeps: int) -> list[dict[str, object]]:
    world = agent.world
    world.foods = [TRAIN_FOOD.copy()] + [FAR.copy() for _ in range(6)]
    world.toxins = [[0.0, 0.0]] + [FAR.copy() for _ in range(5)]
    world._sync_mocap()
    rows: list[dict[str, object]] = []
    for trial in range(trials):
        _reset_body(world)

        def capture(current):
            row = _row(current, "teaching", TRAIN_FOOD)
            row["trial"] = trial
            rows.append(row)

        ag.run_episode(
            agent,
            steps=train_steps,
            sub=substeps,
            vision=False,
            render_every=10**9,
            render_ticks=0,
            log_every=10**9,
            tick_hook=capture,
        )
    return rows


def _probe(agent, source: list[float], steps: int, substeps: int) -> list[dict[str, object]]:
    world = agent.world
    world.foods = [source.copy()] + [FAR.copy() for _ in range(6)]
    world.toxins = [FAR.copy() for _ in range(6)]
    world._sync_mocap()
    _reset_body(world)
    rows: list[dict[str, object]] = []

    def capture(current):
        rows.append(_row(current, "food_only_probe", source))

    ag.run_episode(
        agent,
        steps=steps,
        sub=substeps,
        vision=False,
        render_every=10**9,
        render_ticks=0,
        log_every=10**9,
        tick_hook=capture,
    )
    return rows


def run_case(
    seed: int, name: str, build: dict[str, float | bool], trials: int, train_steps: int, probe_steps: int, substeps: int
) -> dict[str, object]:
    np.random.seed(seed)
    agent = ag.AIFAgent3D(seed=seed, **build)
    agent.birth()
    source = list(PROBES.get(seed, PROBES[11]))
    training = _train(agent, trials, train_steps, substeps)
    probe = _probe(agent, source, probe_steps, substeps)
    return {
        "condition": name,
        "seed": seed,
        "build": build,
        "source": source,
        "trials": trials,
        "training_trace": training,
        "probe_trace": probe,
    }


def _sum(rows: list[dict[str, object]], field: str) -> int:
    return int(sum(int(row[field]) for row in rows))


def _summary(record: dict[str, object]) -> dict[str, object]:
    teaching = record["training_trace"]
    probe = record["probe_trace"]
    assert isinstance(teaching, list) and isinstance(probe, list)
    distances = [float(row["source_distance"]) for row in probe]
    return {
        "teaching_toxin_entries": max((int(row["toxin_hits"]) for row in teaching), default=0),
        "teaching_stg_t": _sum(teaching, "stg_t"),
        "probe_toxin_entries": max((int(row["toxin_hits"]) for row in probe), default=0),
        "food_eaten": int(probe[-1]["food_eaten"]) if probe else 0,
        "food_identity_drive": max((float(row["sensor_drives"].get("odor_identity_food", 0.0)) for row in probe), default=0.0),
        "probe_mbon": _sum(probe, "mbon"),
        "probe_avoid": _sum(probe, "avoid"),
        "probe_lh_left": _sum(probe, "lh_left"),
        "probe_lh_right": _sum(probe, "lh_right"),
        "probe_turn_balance": _sum(probe, "turn_left") - _sum(probe, "turn_right"),
        "minimum_distance": min(distances, default=float("inf")),
        "final_distance": distances[-1] if distances else float("inf"),
    }


def _accept(summaries: dict[str, dict[str, dict[str, object]]], trials: int) -> list[str]:
    failures: list[str] = []
    for seed, by_case in summaries.items():
        if set(by_case) != set(CASES):
            failures.append(f"seed {seed}: incomplete conditions")
            continue
        taught = by_case["taught"]
        output_ablated = by_case["lh_output_ablation"]
        ablated = by_case["teaching_ablation"]
        if any(row["teaching_toxin_entries"] < trials for row in (taught, output_ablated, ablated)):
            failures.append(f"seed {seed}: physical teaching contact missing")
        if taught["teaching_stg_t"] == 0 or output_ablated["teaching_stg_t"] == 0 or ablated["teaching_stg_t"] != 0:
            failures.append(f"seed {seed}: sting-trigger ablation did not isolate teaching")
        if any(row["food_identity_drive"] <= 0.0 for row in (taught, output_ablated, ablated)):
            failures.append(f"seed {seed}: food-only probe did not deliver conditioned stimulus")
        if taught["probe_mbon"] <= 0 or taught["probe_avoid"] <= 0:
            failures.append(f"seed {seed}: learned MBON/AVOID response absent")
        if output_ablated["probe_mbon"] <= 0 or output_ablated["probe_avoid"] <= 0:
            failures.append(f"seed {seed}: output ablation altered upstream learned MB response")
        if ablated["probe_mbon"] != 0 or ablated["probe_avoid"] != 0:
            failures.append(f"seed {seed}: teaching ablation retained learned output")
        taught_lh = taught["probe_lh_left"] + taught["probe_lh_right"]
        output_ablated_lh = output_ablated["probe_lh_left"] + output_ablated["probe_lh_right"]
        if taught_lh <= 1.5 * output_ablated_lh:
            failures.append(f"seed {seed}: AVOID->LH output ablation did not reduce lateral-horn recruitment")
        if taught["food_eaten"] != 0 or taught["minimum_distance"] < 0.7:
            failures.append(f"seed {seed}: taught agent entered the conditioned food source")
        for name, row in (("LH-output", output_ablated), ("teaching", ablated)):
            if row["food_eaten"] < 1 or row["minimum_distance"] >= 0.7:
                failures.append(f"seed {seed}: {name} ablation failed to approach and collect food")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=4)
    parser.add_argument("--probe-steps", type=int, default=80)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 44, 77])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-cases", type=int, help="write this many missing condition records, then stop")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume and args.output:
        raise SystemExit("use either --output or --resume, not both")
    output = args.resume or args.output or HERE / "results" / f"embodied_mb_food_avoidance_causal_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    manifest_path = output / "manifest.json"
    if args.resume:
        if not manifest_path.exists() or json.loads(manifest_path.read_text()).get("experiment") != "embodied_mb_food_avoidance_causal":
            raise SystemExit("--resume requires this experiment's manifest")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write(manifest_path, {
            "experiment": "embodied_mb_food_avoidance_causal",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": {"active_inference": _revision(ROOT), "neuron_model": _revision(NEURON_MODEL)},
            "source_fingerprints": {
                "aif_agent3d.py": _sha(HERE.parent / "aif_agent3d.py"),
                "embodied_mb_food_avoidance_causal.py": _sha(Path(__file__).resolve()),
            },
            "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "mujoco": mujoco.__version__},
            "build": BUILD,
            "conditions": CASES,
            "seeds": args.seeds,
            "probes": {str(seed): PROBES.get(seed, PROBES[11]) for seed in args.seeds},
            "trials": args.trials,
            "train_steps": args.train_steps,
            "probe_steps": args.probe_steps,
            "substeps": args.substeps,
            "scope": "physical neutral-food odour plus separate toxin contact teaches an MB route that reverses later food-only approach; not a general multi-odour policy",
        })

    processed = 0
    for seed in args.seeds:
        for name, build in CASES.items():
            record_path = output / f"{name}_seed{seed}.json"
            if record_path.exists():
                continue
            record = run_case(seed, name, build, args.trials, args.train_steps, args.probe_steps, args.substeps)
            _write(record_path, record)
            processed += 1
            if args.max_cases is not None and processed >= args.max_cases:
                break
        if args.max_cases is not None and processed >= args.max_cases:
            break

    summaries: dict[str, dict[str, dict[str, object]]] = {}
    for seed in args.seeds:
        rows: dict[str, dict[str, object]] = {}
        for name in CASES:
            record_path = output / f"{name}_seed{seed}.json"
            if record_path.exists():
                rows[name] = _summary(json.loads(record_path.read_text()))
        summaries[str(seed)] = rows
    _write(output / "summary.json", summaries)
    failures = _accept(summaries, args.trials)
    complete = all(set(rows) == set(CASES) for rows in summaries.values())
    _write(output / "acceptance.json", {"complete": complete, "passed": complete and not failures, "failures": failures})
    print(f"Wrote MB food-avoidance evidence to {output}")
    for seed, rows in summaries.items():
        for name, row in rows.items():
            print(f"seed={seed} {name}: food={row['food_eaten']} MBON={row['probe_mbon']} AVOID={row['probe_avoid']} LH={row['probe_lh_left']}/{row['probe_lh_right']} min_d={row['minimum_distance']:.3f}")
    if not complete:
        print("INCOMPLETE: resume to run missing cases")
        return 0
    if failures:
        print("FAIL: " + " | ".join(failures), file=sys.stderr)
        return 1
    print("PASS: physical toxin teaching causally reverses later food-only approach through MB->LH->turn circuitry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
