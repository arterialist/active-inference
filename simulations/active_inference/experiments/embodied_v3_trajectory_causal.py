"""Tick-level embodied acceptance for the strict V3 composition.

The older V3 fixture compared final energy and aggregate spike totals.  That
was useful for the SLEEP ablation, but it could pass while the body repeatedly
turned away from a reachable meal or while the strict arbiter stayed in
FORAGE with no exploratory output.  This harness keeps the complete neural
and physical trace and checks the trajectory phases themselves:

* an odour-free body must express the energy-backed EXPLORE route and move;
* a reachable food source must be approached and physically consumed;
* a food/toxin conflict must still collect the food without entering the
  toxin's contact radius (the V3 high-threshold toxin transducer);
* a delayed meal must recruit SLEEP, lower hunger, and reduce realized motor
  drive rather than merely changing a final scalar.

Every case is the real strict ``InteroceptiveV3Agent`` and the ordinary
MuJoCo ``run_episode`` loop.  The host schedules only fixed environmental
fixtures; it never selects a mode, turn, or pose.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from simulations.active_inference import aif_agent3d as ag
from simulations.active_inference.agents.interoceptive_v3 import InteroceptiveV3Agent


HERE = Path(__file__).resolve().parent
FAR = [100.0, 100.0]
MEAL_BODY_STEP = 12

CASES = (
    "no_food_exploration",
    "no_food_exploration_ablation",
    "food_approach",
    "food_toxin_conflict",
    "food_toxin_threshold_ablation",
    "meal_rest",
)


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _world(agent: InteroceptiveV3Agent, *, food, toxin, seed: int) -> None:
    world = ag.w3.World3D(seed=seed, n_food=1, n_tox=1, arena=8.0)
    world.foods = [list(map(float, food))]
    world.toxins = [list(map(float, toxin))]
    # A consumed source must not respawn into the measured route.
    world._rp = lambda _lo, _hi: FAR.copy()
    world._sync_mocap()
    agent.world = world
    agent.img = world.retina()


def _mode_spikes(current, group) -> int:
    return int(sum(int(current.nb[n].O > 0) for n in group if n in current.nb))


def _row(current, target=None) -> dict:
    x, y, yaw = current.world.pose()
    pose = np.asarray([x, y], dtype=float)
    food_distance = None
    toxin_distance = None
    if target is not None:
        food_distance = float(np.linalg.norm(pose - np.asarray(target, dtype=float)))
    if current.world.toxins:
        toxin_distance = float(min(np.linalg.norm(pose - np.asarray(p, dtype=float))
                                   for p in current.world.toxins))
    modes = {
        "FORAGE": _mode_spikes(current, ag.ar.MODE[0]),
        "EXPLORE": _mode_spikes(current, ag.ar.MODE[2]),
        "SLEEP": _mode_spikes(current, ag.ar.SLEEP_MODE),
    }
    winner = max(modes, key=modes.get) if max(modes.values()) else "none"
    return {
        "neural_tick": int(current.t),
        "pose": {"x": float(x), "y": float(y), "yaw": float(yaw)},
        "speed": float(current.world.speed()),
        "food_distance": food_distance,
        "toxin_distance": toxin_distance,
        "food_eaten_total": int(current.world.eaten),
        "toxin_hits_total": int(current.world.tox_hits),
        "metabolic": dict(current.world.metabolic_state()),
        "hunger_spikes": _mode_spikes(current, ag.ar.HUNGER),
        "mode_spikes": modes,
        "winner": winner,
        "search_spike": int(current.nb[ag.SEARCH].O > 0),
        "steer_spike": int(current.nb[ag.STEER].O > 0),
        "actuator_abs_sum": float(sum(abs(float(v)) for v in current.world.data.ctrl)),
        "sensor_current": dict(getattr(current, "last_sensor_drives", {})),
    }


def run_case(name: str, seed: int, steps: int, substeps: int) -> dict:
    if name == "no_food_exploration":
        agent = InteroceptiveV3Agent(seed=seed)
        _world(agent, food=FAR, toxin=FAR, seed=seed)
        target = None
    elif name == "no_food_exploration_ablation":
        agent = InteroceptiveV3Agent(seed=seed, w_explore_energy=0.0)
        _world(agent, food=FAR, toxin=FAR, seed=seed)
        target = None
    elif name == "food_approach":
        agent = InteroceptiveV3Agent(seed=seed)
        target = [-1.0, 0.0]
        _world(agent, food=target, toxin=FAR, seed=seed)
    elif name in ("food_toxin_conflict", "food_toxin_threshold_ablation"):
        agent = InteroceptiveV3Agent(
            seed=seed,
            **({"metabolic_toxin_threshold": 0.0}
               if name == "food_toxin_threshold_ablation" else {}),
        )
        target = [-3.0, 0.0]
        _world(agent, food=target, toxin=[-3.0, 1.5], seed=seed)
    elif name == "meal_rest":
        agent = InteroceptiveV3Agent(seed=seed)
        _world(agent, food=FAR, toxin=FAR, seed=seed)
        target = None
    else:  # pragma: no cover - argparse and callers constrain names
        raise ValueError(name)

    agent.birth()
    inserted = False
    trace: list[dict] = []

    def observe(current) -> None:
        nonlocal inserted
        if name == "meal_rest" and not inserted and current.t >= MEAL_BODY_STEP * substeps:
            x, y, _ = current.world.pose()
            current.world.foods[0] = [float(x), float(y)]
            current.world._sync_mocap()
            inserted = True
        trace.append(_row(current, target=target))

    # Keep the long conflict route long enough for a real approach, while the
    # other cases remain quick enough for routine regression runs.
    if name in ("food_toxin_conflict", "food_toxin_threshold_ablation"):
        local_steps, local_sub = max(steps, 100), max(substeps, 16)
    else:
        local_steps, local_sub = steps, substeps
    ag.run_episode(
        agent, steps=local_steps, sub=local_sub, vision=False,
        render_every=10**9, render_ticks=0, log_every=10**9, tick_hook=observe,
    )
    return {
        "case": name,
        "seed": seed,
        "steps": local_steps,
        "substeps": local_sub,
        "initial_pose": dict(zip(("x", "y", "yaw"), map(float, trace[0]["pose"].values()))) if trace else {},
        "final_pose": dict(zip(("x", "y", "yaw"), map(float, trace[-1]["pose"].values()))) if trace else {},
        "trace": trace,
    }


def summarize(record: dict) -> dict:
    rows = record["trace"]
    if not rows:
        raise ValueError(f"{record['case']} produced no tick trace")
    first = rows[0]
    last = rows[-1]
    meal = next((i for i, row in enumerate(rows) if row["food_eaten_total"]), None)
    post = rows[(meal + 16) if meal is not None else len(rows):]
    pre = rows[:meal] if meal is not None else rows

    def total(rows_, key):
        return int(sum(row["mode_spikes"][key] for row in rows_))

    displacement = float(np.hypot(
        last["pose"]["x"] - first["pose"]["x"],
        last["pose"]["y"] - first["pose"]["y"],
    ))
    distances = [row["food_distance"] for row in rows if row["food_distance"] is not None]
    toxin_distances = [row["toxin_distance"] for row in rows if row["toxin_distance"] is not None]
    return {
        "case": record["case"],
        "food_eaten": int(last["food_eaten_total"]),
        "toxin_hits": int(last["toxin_hits_total"]),
        "minimum_food_distance": float(min(distances)) if distances else None,
        "minimum_toxin_distance": float(min(toxin_distances)) if toxin_distances else None,
        "displacement": displacement,
        "explore_spikes": total(rows, "EXPLORE"),
        "forage_spikes": total(rows, "FORAGE"),
        "sleep_spikes": total(rows, "SLEEP"),
        "explore_winner_ticks": int(sum(row["winner"] == "EXPLORE" for row in rows)),
        "sleep_winner_ticks": int(sum(row["winner"] == "SLEEP" for row in rows)),
        "search_spikes": int(sum(row["search_spike"] for row in rows)),
        "pre_hunger_mean": float(np.mean([row["hunger_spikes"] for row in pre])) if pre else 0.0,
        "post_hunger_mean": float(np.mean([row["hunger_spikes"] for row in post])) if post else 0.0,
        "pre_motor": float(sum(row["actuator_abs_sum"] for row in pre)) if pre else 0.0,
        "post_motor": float(sum(row["actuator_abs_sum"] for row in post)) if post else 0.0,
        "final_energy": float(last["metabolic"]["energy_store"]),
        "final_gut_load": float(last["metabolic"]["gut_load"]),
        "meal_tick": int(rows[meal]["neural_tick"]) if meal is not None else None,
        "ticks_recorded": len(rows),
    }


def accept(summary: dict[str, dict]) -> list[str]:
    failures: list[str] = []
    explore = summary["no_food_exploration"]
    explore_off = summary["no_food_exploration_ablation"]
    approach = summary["food_approach"]
    conflict = summary["food_toxin_conflict"]
    conflict_off = summary["food_toxin_threshold_ablation"]
    meal = summary["meal_rest"]

    if explore["explore_winner_ticks"] < 4 or explore["search_spikes"] < 4:
        failures.append("odour-free V3 never expressed the neural EXPLORE/search route")
    if explore["displacement"] <= 0.05:
        failures.append("odour-free V3 did not move through the ordinary motor path")
    if explore_off["explore_spikes"] >= max(4, explore["explore_spikes"] // 2):
        failures.append("energy-to-EXPLORE ablation did not remove most exploratory activity")

    if approach["food_eaten"] < 1 or approach["minimum_food_distance"] is None or approach["minimum_food_distance"] > 0.70:
        failures.append("strict V3 did not approach and physically consume a reachable food source")
    if conflict["food_eaten"] < 1:
        failures.append("strict V3 still turned away from the safe side of a food/toxin conflict")
    if conflict["toxin_hits"] != 0 or (conflict["minimum_toxin_distance"] is not None and conflict["minimum_toxin_distance"] <= 0.55):
        failures.append("strict V3 entered the toxin contact radius during the conflict route")
    if conflict_off["food_eaten"] >= 1:
        failures.append("toxin-threshold ablation did not expose the original conflict turn-away")

    if meal["meal_tick"] is None:
        failures.append("the scheduled meal was not physically consumed")
    if meal["sleep_winner_ticks"] < 8 or meal["sleep_spikes"] < 30:
        failures.append("delayed meal did not recruit sustained PAULA SLEEP")
    if meal["post_hunger_mean"] >= meal["pre_hunger_mean"]:
        failures.append("body-derived gut load did not lower the PAULA hunger state")
    if meal["post_motor"] >= meal["pre_motor"] * 0.8:
        failures.append("SLEEP did not reduce realized motor drive in the post-meal window")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps <= 0 or args.substeps <= 0:
        raise SystemExit("--steps and --substeps must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or HERE / "results" / f"embodied_v3_trajectory_causal_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "manifest.json", {
        "experiment": "embodied_v3_trajectory_causal",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed, "steps": args.steps, "substeps": args.substeps,
        "cases": CASES, "full_tick_traces": True,
        "scope": "strict V3 PAULA + metabolic body + MuJoCo trajectory phases",
    })
    records = {}
    summaries = {}
    for case in CASES:
        record = run_case(case, args.seed, args.steps, args.substeps)
        records[case] = record
        summaries[case] = summarize(record)
        _write(output / f"{case}.json", record)
        print(case, summaries[case], flush=True)
    _write(output / "summary.json", summaries)
    failures = accept(summaries)
    _write(output / "acceptance.json", {"passed": not failures, "failures": failures})
    if failures:
        print("FAIL: " + " | ".join(failures))
        return 1
    print(f"PASS: strict V3 trajectory acceptance written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
