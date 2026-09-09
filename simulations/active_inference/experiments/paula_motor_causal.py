"""Causal acceptance experiment for the PAULA-to-MuJoCo motor primitive.

The body has no kinematic velocity or yaw command.  A PAULA synfire CPG
activates graded PAULA muscle cells, whose membrane state is the sole input to
the MuJoCo actuators.  The steering currents used here are an open-loop
descending-neural-input probe: they do not choose an action or control body
pose.  The test records the complete neural-to-body path once per body step.

Run from the active-inference root:

    uv run python -m simulations.active_inference.experiments.paula_motor_causal
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

from simulations.active_inference import nmrower2 as motor


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NEURON_MODEL = ROOT.parent / "neuron-model"
STEER_CURRENT = 0.025
CASES = {
    "forward": {"steer_left": 0.0, "steer_right": 0.0, "build": {}},
    "left_turn": {"steer_left": STEER_CURRENT, "steer_right": 0.0, "build": {}},
    "right_turn": {"steer_left": 0.0, "steer_right": STEER_CURRENT, "build": {}},
    "cpg_to_muscle_ablation": {"steer_left": 0.0, "steer_right": 0.0, "build": {"w_cpg": 0.0}},
    "nmj_ablation": {"steer_left": 0.0, "steer_right": 0.0, "build": {"muscle_gain": 0.0}},
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


def _wrap_degrees(angle: float) -> float:
    return float(np.degrees((angle + np.pi) % (2.0 * np.pi) - np.pi))


def run_case(seed: int, name: str, case: dict, steps: int, substeps: int) -> dict:
    """Run one physical trial and retain every allowed bridge variable."""
    np.random.seed(seed)
    rower = motor.NMRower2(**case["build"])
    x0, y0, _ = rower.pose()
    heading0 = rower.heading()
    trace = []
    for _ in range(steps):
        rower.step(case["steer_left"], case["steer_right"], sub=substeps, tick_trace=trace)
    x1, y1, _ = rower.pose()
    return {
        "condition": name,
        "seed": seed,
        "build": case["build"],
        "steer_left_current": case["steer_left"],
        "steer_right_current": case["steer_right"],
        "steps": steps,
        "neural_substeps_per_body_step": substeps,
        "initial_pose": {"x": x0, "y": y0, "heading": heading0},
        "final_pose": {"x": x1, "y": y1, "heading": rower.heading()},
        "displacement": float(np.hypot(x1 - x0, y1 - y0)),
        "heading_change_degrees": _wrap_degrees(rower.heading() - heading0),
        "tick_trace": trace,
    }


def _summary(record: dict) -> dict:
    trace = record["tick_trace"]
    return {
        "displacement": record["displacement"],
        "heading_change_degrees": record["heading_change_degrees"],
        "cpg_spikes": int(sum(sum(row["cpg_spikes"].values()) for row in trace)),
        "max_abs_muscle_state": float(max((abs(value) for row in trace for value in row["muscle_state"].values()), default=0.0)),
        "max_abs_actuator_control": float(max((abs(value) for row in trace for value in row["actuator_ctrl"].values()), default=0.0)),
    }


def _accept(summaries: dict[str, dict[str, dict]]) -> list[str]:
    failures = []
    for seed, rows in summaries.items():
        forward, left, right = rows["forward"], rows["left_turn"], rows["right_turn"]
        if forward["displacement"] < 2.0 or abs(forward["heading_change_degrees"]) > 10.0:
            failures.append(f"seed {seed}: forward CPG did not produce stable locomotion")
        if left["displacement"] < 1.0 or left["heading_change_degrees"] > -30.0:
            failures.append(f"seed {seed}: left descending drive did not turn left")
        if right["displacement"] < 1.0 or right["heading_change_degrees"] < 30.0:
            failures.append(f"seed {seed}: right descending drive did not turn right")
        for name in ("cpg_to_muscle_ablation", "nmj_ablation"):
            ablation = rows[name]
            if ablation["displacement"] > 0.05 or ablation["max_abs_actuator_control"] > 1e-12:
                failures.append(f"seed {seed}: {name} still moved the body")
            if ablation["cpg_spikes"] == 0:
                failures.append(f"seed {seed}: {name} removed the CPG rather than its output path")
    return failures


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1800, help="MuJoCo body steps per trial")
    parser.add_argument("--substeps", type=int, default=6, help="PAULA ticks per body step")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 44, 77])
    parser.add_argument("--output", type=Path, help="new directory for manifest and raw traces")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0 or args.substeps <= 0:
        raise SystemExit("--steps and --substeps must be positive")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or HERE / "results" / f"paula_motor_causal_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "manifest.json", {
        "experiment": "paula_motor_causal",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": {"active_inference": _revision(ROOT), "neuron_model": _revision(NEURON_MODEL)},
        "source_fingerprints": {
            "nmrower2.py": _sha256(HERE.parent / "nmrower2.py"),
            "paula_motor_causal.py": _sha256(Path(__file__).resolve()),
        },
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "mujoco": mujoco.__version__},
        "seeds": args.seeds,
        "steps": args.steps,
        "neural_substeps_per_body_step": args.substeps,
        "conditions": CASES,
        "scope": "open-loop descending-current probe of a PAULA CPG, graded muscles, and MuJoCo body; not an odour-navigation or behavioural-selection claim",
        "primary_evidence": "tick_trace in every per-condition-per-seed JSON file",
    })
    summaries: dict[str, dict[str, dict]] = {}
    for seed in args.seeds:
        summaries[str(seed)] = {}
        for name, case in CASES.items():
            record = run_case(seed, name, case, args.steps, args.substeps)
            _write_json(output / f"{name}_seed{seed}.json", record)
            summaries[str(seed)][name] = _summary(record)
    _write_json(output / "summary.json", summaries)
    failures = _accept(summaries)
    _write_json(output / "acceptance.json", {"passed": not failures, "failures": failures})
    print(f"Wrote PAULA motor causal evidence to {output}")
    for seed, rows in summaries.items():
        print("seed=" + seed + "; ".join(
            f" {name}: dist={row['displacement']:.2f} heading={row['heading_change_degrees']:+.1f}"
            for name, row in rows.items()
        ))
    if failures:
        print("FAIL: " + " | ".join(failures), file=sys.stderr)
        return 1
    print("PASS: PAULA CPG, graded muscles, and the NMJ transducer are causally required for locomotion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
