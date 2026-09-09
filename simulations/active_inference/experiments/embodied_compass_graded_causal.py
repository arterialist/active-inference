"""Causal evidence for continuous PAULA gyro-to-ring compass tracking.

The opt-in route contains only PAULA circuitry after the sanctioned raw
MuJoCo gyroscope transducer: paired graded afferents form a delayed
full-stroke notch, then their continuous release drives the existing local
P-EN gates.  It is tested in two contexts: declared currents into existing
PAULA turn neurons, and ordinary unscripted PAULA CPG locomotion.  No pose,
heading, target, or motor command is supplied by Python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from simulations.active_inference import aif_agent3d as ag


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NEURON_MODEL = ROOT.parent / "neuron-model"
TURN_START = 80
TURN_STOP = 480
BURN_IN = 200
BUILD = {
    "vestibular_notch_graded": True,
    # Measured MuJoCo yaw-stroke period is 44 neural ticks.  These are
    # experimental route parameters, not defaults for the organism.
    "vop_notch_graded_window": 44,
    "vop_notch_graded_input_gain": 1000.0,
    "vop_notch_graded_pen_gain": 3.0,
    "turn_probe_ports": True,
    # Keep as-yet-unaccepted home outputs out of the compass measurement.
    "w_opp": 0.0,
    "w_cpu1": 0.0,
    "w_musf_mode": 0.0,
}
CASES = {
    "tl_neural_turn": {"turn_neuron": "TL", "build": BUILD},
    "tr_neural_turn": {"turn_neuron": "TR", "build": BUILD},
    "graded_to_pen_ablation": {
        "turn_neuron": "TL",
        "build": {**BUILD, "vop_notch_graded_pen_gain": 0.0},
    },
    "free_gait_tracking": {"turn_neuron": None, "build": BUILD},
    "free_gait_to_pen_ablation": {
        "turn_neuron": None,
        "build": {**BUILD, "vop_notch_graded_pen_gain": 0.0},
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _initial_yaw(seed: int) -> float:
    return float(np.radians((seed * 137) % 360 - 180))


def _ring_heading(spikes: np.ndarray) -> float:
    vector = np.sum(spikes * np.exp(1j * ag.cc.PHI))
    if abs(vector) < 1e-12:
        raise RuntimeError("compass ring was silent during graded-route measurement")
    return float(np.angle(vector))


def run_case(seed: int, name: str, case: dict, steps: int, substeps: int) -> dict:
    np.random.seed(seed)
    agent = ag.AIFAgent3D(seed=seed, **case["build"])
    agent.world = ag.w3.World3D(seed=seed, n_food=0, n_tox=0, arena=8.0)
    agent.world.data.qpos[agent.world.jyaw] = _initial_yaw(seed)
    mujoco.mj_forward(agent.world.model, agent.world.data)
    agent.img = agent.world.retina()
    agent.birth()

    target = getattr(ag, case["turn_neuron"]) if case["turn_neuron"] else None
    stimulus_active = False
    ring_window: deque[np.ndarray] = deque(maxlen=24)
    trace: list[dict] = []

    def neural_input(current) -> None:
        nonlocal stimulus_active
        stimulus_active = target is not None and TURN_START <= current.t < TURN_STOP
        if stimulus_active:
            current.net.set_external_input(target, ag.TURN_PROBE_SYN[target], 3.0)

    def capture(current) -> None:
        ring_window.append(np.asarray([current.nb[nid].O > 0 for nid in ag.cc.RING], dtype=float))
        x, y, yaw = current.world.pose()
        trace.append({
            "neural_tick": current.t,
            "turn_probe_active": stimulus_active,
            "physical_yaw_rate": float(current.world.yaw_rate()),
            "pose": {"x": float(x), "y": float(y), "yaw": float(yaw)},
            "ring_heading_radians": _ring_heading(np.sum(ring_window, axis=0)),
            "turn_spikes": {
                "TL": int(current.nb[ag.TL].O > 0),
                "TR": int(current.nb[ag.TR].O > 0),
            },
            "graded_vestibular_release": {
                "CCW": float(current.nb[ag.VEST_NET_CCW].O),
                "CW": float(current.nb[ag.VEST_NET_CW].O),
            },
            "shift_spikes": {
                "CL": sum(int(current.nb[nid].O > 0) for column in ag.cc.CL for nid in column),
                "CR": sum(int(current.nb[nid].O > 0) for column in ag.cc.CR for nid in column),
            },
            "actuator_ctrl": {
                label: float(current.world.data.ctrl[actuator])
                for label, actuator in current.world.act_id.items()
            },
        })

    start = agent.world.pose()
    _, modes = ag.run_episode(
        agent,
        steps=steps,
        sub=substeps,
        vision=False,
        render_every=10**9,
        render_ticks=0,
        log_every=10**9,
        tick_hook=capture,
        neural_input_hook=neural_input,
    )
    end = agent.world.pose()
    return {
        "condition": name,
        "seed": seed,
        "build": case["build"],
        "turn_neuron": case["turn_neuron"],
        "vision_enabled": False,
        "objects": {"food": 0, "toxin": 0},
        "steps": steps,
        "neural_substeps_per_body_step": substeps,
        "initial_pose": {"x": float(start[0]), "y": float(start[1]), "yaw": float(start[2])},
        "final_pose": {"x": float(end[0]), "y": float(end[1]), "yaw": float(end[2])},
        "mode_steps": modes,
        "tick_trace": trace,
    }


def _delta(rows: list[dict], key: str) -> float:
    if key == "body":
        values = np.unwrap(np.asarray([row["pose"]["yaw"] for row in rows]))
    else:
        values = np.unwrap(np.asarray([row["ring_heading_radians"] for row in rows]))
    return float(np.degrees(values[-1] - values[0]))


def _summary(record: dict) -> dict:
    trace = record["tick_trace"]
    active = [row for row in trace if row["turn_probe_active"]]
    rows = active if active else [row for row in trace if row["neural_tick"] >= BURN_IN]
    if len(rows) < 2:
        raise ValueError("measurement window was not sampled")
    body_delta = _delta(rows, "body")
    ring_delta = _delta(rows, "ring")
    return {
        "measurement": "turn_probe" if active else "post_burn_free_gait",
        "body_yaw_degrees": body_delta,
        "ring_yaw_degrees": ring_delta,
        "tracking_ratio": ring_delta / body_delta if abs(body_delta) > 1e-12 else None,
        "turn_spikes": {name: int(sum(row["turn_spikes"][name] for row in rows)) for name in ("TL", "TR")},
        "mean_graded_vestibular_release": {
            name: float(np.mean([row["graded_vestibular_release"][name] for row in rows]))
            for name in ("CCW", "CW")
        },
        "shift_spikes": {name: int(sum(row["shift_spikes"][name] for row in rows)) for name in ("CL", "CR")},
        "max_abs_actuator_control": float(max(abs(value) for row in rows for value in row["actuator_ctrl"].values())),
    }


def _accept(summaries: dict[str, dict[str, dict]]) -> list[str]:
    failures: list[str] = []
    for seed, rows in summaries.items():
        for label, expected in (("tl_neural_turn", 1), ("tr_neural_turn", -1)):
            row = rows[label]
            body, ring, ratio = row["body_yaw_degrees"], row["ring_yaw_degrees"], row["tracking_ratio"]
            if expected * body < 30.0:
                failures.append(f"seed {seed} {label}: PAULA turn did not rotate the MuJoCo body")
            if expected * ring < 15.0:
                failures.append(f"seed {seed} {label}: recurrent ring did not follow the physical turn")
            if ratio is None or not (0.25 <= abs(ratio) <= 1.5):
                failures.append(f"seed {seed} {label}: ring/body gain outside the calibrated range")
            turn = "TL" if label.startswith("tl") else "TR"
            if row["turn_spikes"][turn] < 50 or row["max_abs_actuator_control"] <= 0.0:
                failures.append(f"seed {seed} {label}: turn did not use the PAULA motor path")
        ablated = rows["graded_to_pen_ablation"]
        if ablated["body_yaw_degrees"] < 30.0 or ablated["turn_spikes"]["TL"] < 50:
            failures.append(f"seed {seed}: graded-to-P-EN cut changed the physical turn course")
        if abs(ablated["ring_yaw_degrees"]) > 15.0:
            failures.append(f"seed {seed}: graded-to-P-EN cut still moved the ring")
        intact, cut = rows["free_gait_tracking"], rows["free_gait_to_pen_ablation"]
        if abs(intact["body_yaw_degrees"]) < 20.0:
            failures.append(f"seed {seed}: normal PAULA gait lacked enough post-burn physical yaw")
        if intact["body_yaw_degrees"] * intact["ring_yaw_degrees"] <= 0.0:
            failures.append(f"seed {seed}: free-gait ring moved opposite the physical body")
        if intact["tracking_ratio"] is None or not (0.25 <= abs(intact["tracking_ratio"]) <= 1.25):
            failures.append(f"seed {seed}: free-gait ring/body gain outside acceptance range")
        if abs(cut["body_yaw_degrees"] - intact["body_yaw_degrees"]) > 5.0:
            failures.append(f"seed {seed}: graded-to-P-EN cut changed free gait")
        if abs(cut["ring_yaw_degrees"]) > 5.0:
            failures.append(f"seed {seed}: cut route still moved the post-burn ring")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 44, 77])
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-cases", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0 or args.substeps <= 0 or args.max_cases is not None and args.max_cases <= 0:
        raise SystemExit("--steps, --substeps, and --max-cases must be positive")
    if args.output and args.resume:
        raise SystemExit("use either --output or --resume, not both")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.resume or args.output or HERE / "results" / f"embodied_compass_graded_causal_{stamp}"
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest.get("experiment") != "embodied_compass_graded_causal":
            raise SystemExit("--resume directory belongs to another experiment")
        if manifest.get("steps") != args.steps or manifest.get("substeps") != args.substeps or manifest.get("seeds") != args.seeds:
            raise SystemExit("--resume requires the manifest's exact steps, substeps, and seeds")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "manifest.json", {
            "experiment": "embodied_compass_graded_causal",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": {"active_inference": _revision(ROOT), "neuron_model": _revision(NEURON_MODEL)},
            "source_fingerprints": {
                "aif_agent3d.py": _sha256(HERE.parent / "aif_agent3d.py"),
                "world3d.py": _sha256(HERE.parent / "world3d.py"),
                "embodied_compass_graded_causal.py": _sha256(Path(__file__).resolve()),
            },
            "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "mujoco": mujoco.__version__},
            "seeds": args.seeds,
            "steps": args.steps,
            "substeps": args.substeps,
            "conditions": CASES,
            "turn_probe": {"start_neural_tick": TURN_START, "stop_neural_tick": TURN_STOP, "current": 3.0},
            "free_gait_burn_in_neural_ticks": BURN_IN,
            "scope": "PAULA turn/motor -> MuJoCo raw yaw -> graded PAULA delay-line/P-EN/ring, including ordinary CPG gait; excludes path integration, HOME, and homing",
        })
    attempts = 0
    for seed in args.seeds:
        for name in args.cases:
            path = output / f"{name}_seed{seed}.json"
            if path.exists():
                continue
            if args.max_cases is not None and attempts >= args.max_cases:
                break
            record = run_case(seed, name, CASES[name], args.steps, args.substeps)
            _write_json(path, record)
            row = _summary(record)
            print(f"seed={seed} {name}: body={row['body_yaw_degrees']:+.1f}deg ring={row['ring_yaw_degrees']:+.1f}deg gain={row['tracking_ratio']:+.2f}", flush=True)
            attempts += 1
        if args.max_cases is not None and attempts >= args.max_cases:
            break
    complete = all((output / f"{name}_seed{seed}.json").exists() for seed in args.seeds for name in CASES)
    if not complete:
        print("INCOMPLETE: resume to run missing cases")
        return 0
    summaries = {str(seed): {name: _summary(json.loads((output / f"{name}_seed{seed}.json").read_text())) for name in CASES} for seed in args.seeds}
    _write_json(output / "summary.json", summaries)
    failures = _accept(summaries)
    _write_json(output / "acceptance.json", {"passed": not failures, "failures": failures})
    print(f"Wrote graded compass causal evidence to {output}")
    if failures:
        print("FAIL: " + " | ".join(failures), file=sys.stderr)
        return 1
    print("PASS: raw physical yaw causally drives continuous PAULA recurrent-compass tracking")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
