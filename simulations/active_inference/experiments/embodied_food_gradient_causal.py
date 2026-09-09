"""Causal food-gradient steering test in the integrated PAULA/MuJoCo agent.

The two tests use an odour-only food source at opposite lateral positions.
Vision is deliberately disabled.  Physical odour concentrations are the sole
sensory bridge; PAULA food-sensor populations, opponent turn populations, and
graded muscle states remain on the agent's normal control path.  The
``w_sd=0`` control severs only the food-sensor-to-turn synapses, leaving the
same field, sensors, CPG, muscles, and body intact.
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
CASES = {
    # This is an isolation experiment for the innate food-gradient primitive;
    # disable the separately accepted learned-valence/LH route in every arm so
    # w_sd remains the only food-to-turn causal path under test.
    "food_positive_y": {"source": [-1.0, 1.0], "build": {"mb_lh": False}, "gain": 5.0},
    "food_negative_y": {"source": [-1.0, -1.0], "build": {"mb_lh": False}, "gain": 5.0},
    "food_path_ablation": {"source": [-1.0, 1.0], "build": {"mb_lh": False, "w_sd": 0.0}, "gain": 5.0},
}


def _revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_food_only_world(agent, source: list[float]) -> None:
    """Keep the existing physics model but make every non-test source inert."""
    agent.world.foods = [list(source)] + [FAR.copy() for _ in range(6)]
    agent.world.toxins = [FAR.copy() for _ in range(6)]
    agent.world._sync_mocap()


def run_case(seed: int, name: str, case: dict, steps: int, substeps: int) -> dict:
    np.random.seed(seed)
    agent = ag.AIFAgent3D(seed=seed, **case["build"])
    _install_food_only_world(agent, case["source"])
    agent.birth()
    trace = []

    def capture(current):
        x, y, yaw = current.world.pose()
        trace.append({
            "neural_tick": current.t,
            "sensory_current": dict(current.last_sensor_drives),
            "food_sensor_spikes": {
                "left": sum(int(current.nb[nid].O > 0) for nid in ag.FLp),
                "right": sum(int(current.nb[nid].O > 0) for nid in ag.FRp),
            },
            "turn_spikes": {"TL": int(current.nb[ag.TL].O > 0), "TR": int(current.nb[ag.TR].O > 0)},
            "muscle_state": {
                "MLp": float(current.nb[ag.MLp].S), "MLr": float(current.nb[ag.MLr].S),
                "MRp": float(current.nb[ag.MRp].S), "MRr": float(current.nb[ag.MRr].S),
            },
            "actuator_ctrl": {name: float(current.world.data.ctrl[aid]) for name, aid in current.world.act_id.items()},
            "pose": {"x": float(x), "y": float(y), "yaw": float(yaw)},
        })

    start = agent.world.pose()
    ag.run_episode(
        agent, steps=steps, sub=substeps, render_every=10**9, render_ticks=0,
        vision=False, gain=case["gain"], log_every=10**9, tick_hook=capture,
    )
    end = agent.world.pose()
    return {
        "condition": name, "seed": seed, "build": case["build"], "source": case["source"],
        "vision_enabled": False, "steps": steps, "neural_substeps_per_body_step": substeps,
        "initial_pose": {"x": float(start[0]), "y": float(start[1]), "yaw": float(start[2])},
        "final_pose": {"x": float(end[0]), "y": float(end[1]), "yaw": float(end[2])},
        "tick_trace": trace,
    }


def _summary(record: dict) -> dict:
    trace = record["tick_trace"]
    food_left = sum(row["food_sensor_spikes"]["left"] for row in trace)
    food_right = sum(row["food_sensor_spikes"]["right"] for row in trace)
    tl = sum(row["turn_spikes"]["TL"] for row in trace)
    tr = sum(row["turn_spikes"]["TR"] for row in trace)
    first, last = record["initial_pose"], record["final_pose"]
    return {
        "food_sensor_left_spikes": food_left,
        "food_sensor_right_spikes": food_right,
        "turn_TL_spikes": tl,
        "turn_TR_spikes": tr,
        "turn_imbalance_TL_minus_TR": tl - tr,
        "max_abs_actuator_control": max(abs(value) for row in trace for value in row["actuator_ctrl"].values()),
        "body_displacement": float(np.hypot(last["x"] - first["x"], last["y"] - first["y"])),
    }


def _accept(summaries: dict[str, dict[str, dict]]) -> list[str]:
    failures = []
    for seed, rows in summaries.items():
        positive, negative, ablated = rows["food_positive_y"], rows["food_negative_y"], rows["food_path_ablation"]
        if positive["food_sensor_left_spikes"] + positive["food_sensor_right_spikes"] < 100:
            failures.append(f"seed {seed}: positive-y food did not activate the physical food sensors")
        if negative["food_sensor_left_spikes"] + negative["food_sensor_right_spikes"] < 100:
            failures.append(f"seed {seed}: negative-y food did not activate the physical food sensors")
        if positive["turn_imbalance_TL_minus_TR"] > -10:
            failures.append(f"seed {seed}: positive-y food did not drive TR over TL")
        if negative["turn_imbalance_TL_minus_TR"] < 10:
            failures.append(f"seed {seed}: negative-y food did not drive TL over TR")
        if ablated["food_sensor_left_spikes"] + ablated["food_sensor_right_spikes"] < 100:
            failures.append(f"seed {seed}: food-path ablation removed the sensor rather than its output path")
        if ablated["turn_TL_spikes"] or ablated["turn_TR_spikes"]:
            failures.append(f"seed {seed}: w_sd=0 still drove a food turn population")
        if min(positive["max_abs_actuator_control"], negative["max_abs_actuator_control"]) <= 0.0:
            failures.append(f"seed {seed}: neural food steering did not reach the physical actuator path")
    return failures


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10, help="agent steps per trial")
    parser.add_argument("--substeps", type=int, default=8, help="neural/physics ticks per agent step")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 44, 77])
    parser.add_argument("--output", type=Path, help="new directory for manifest and raw traces")
    parser.add_argument("--resume", type=Path, help="complete missing per-seed traces in an interrupted output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0 or args.substeps <= 0:
        raise SystemExit("--steps and --substeps must be positive")
    if args.resume and args.output:
        raise SystemExit("use either --output or --resume, not both")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.resume or args.output or HERE / "results" / f"embodied_food_gradient_causal_{timestamp}"
    if args.resume:
        manifest_path = output / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"--resume requires an existing manifest: {manifest_path}")
        prior = json.loads(manifest_path.read_text())
        if prior.get("experiment") != "embodied_food_gradient_causal":
            raise SystemExit(f"--resume directory belongs to {prior.get('experiment')!r}, not this experiment")
        if prior.get("steps") != args.steps or prior.get("substeps") != args.substeps:
            raise SystemExit("--resume requires the same --steps and --substeps as its manifest")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "manifest.json", {
            "experiment": "embodied_food_gradient_causal", "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": {"active_inference": _revision(ROOT), "neuron_model": _revision(NEURON_MODEL)},
            "source_fingerprints": {
                "aif_agent3d.py": _sha256(HERE.parent / "aif_agent3d.py"),
                "embodied_food_gradient_causal.py": _sha256(Path(__file__).resolve()),
            },
            "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "mujoco": mujoco.__version__},
            "seeds": args.seeds, "steps": args.steps, "substeps": args.substeps, "conditions": CASES,
            "scope": "nonvisual physical odour -> PAULA food sensors -> opponent turn neurons -> normal graded-muscle/MuJoCo path; not long-horizon food collection",
            "primary_evidence": "tick_trace in every per-condition-per-seed JSON file",
        })
    summaries = {}
    for seed in args.seeds:
        summaries[str(seed)] = {}
        for name, case in CASES.items():
            record_path = output / f"{name}_seed{seed}.json"
            if args.resume and record_path.exists():
                record = json.loads(record_path.read_text())
            else:
                record = run_case(seed, name, case, args.steps, args.substeps)
                _write_json(record_path, record)
            summaries[str(seed)][name] = _summary(record)
    _write_json(output / "summary.json", summaries)
    failures = _accept(summaries)
    _write_json(output / "acceptance.json", {"passed": not failures, "failures": failures})
    print(f"Wrote embodied food-gradient causal evidence to {output}")
    for seed, rows in summaries.items():
        print(f"seed={seed}: " + "; ".join(
            f"{name} sensor={row['food_sensor_left_spikes'] + row['food_sensor_right_spikes']} turn={row['turn_imbalance_TL_minus_TR']:+d}"
            for name, row in rows.items()
        ))
    if failures:
        print("FAIL: " + " | ".join(failures), file=sys.stderr)
        return 1
    print("PASS: the nonvisual physical food gradient causally drives signed PAULA turn populations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
