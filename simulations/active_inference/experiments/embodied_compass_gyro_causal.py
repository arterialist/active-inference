"""Physical causal calibration of the nonvisual PAULA gyro-to-compass route.

The normal agent body is held in an empty MuJoCo arena.  A fixed experimental
current enters its *existing* TL or TR turn neuron for one preregistered
window; that current is expressed through the normal relay, graded-muscle and
MuJoCo route.  The only feedback to the compass is raw physical yaw rate.

The opt-in vestibular route implements a full-stroke PAULA delay-line notch:
signed gyro current is integrated and cancelled against its delayed copy,
then sparse opponent vestibular events gate the local P-EN shift circuit.
There is no pose/heading injection, host turn, target selection, or visual
input.  The port stimulus is solely an isolated component calibration of the
shared neural motor and recurrent compass paths.
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
BUILD = {
    "vestibular_notch": True,
    # Empirically chosen only in the declared, physical TL/TR calibration
    # course below.  It is not the default organism configuration yet.
    "vop_notch_r": 0.02,
    "vop_notch_pen_gain": 2.25,
    "turn_probe_ports": True,
    # Isolate compass movement from the as-yet unaccepted home controller.
    "w_opp": 0.0,
    "w_cpu1": 0.0,
    "w_musf_mode": 0.0,
}
CASES = {
    "tl_neural_turn": {"turn_neuron": "TL", "build": BUILD},
    "tr_neural_turn": {"turn_neuron": "TR", "build": BUILD},
    "notch_to_pen_ablation": {
        "turn_neuron": "TL",
        "build": {**BUILD, "vop_notch_pen_gain": 0.0},
    },
}


def _revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _initial_yaw(seed: int) -> float:
    """Different physical orientations; all assessment uses yaw deltas."""
    return float(np.radians((seed * 137) % 360 - 180))


def _heading_from_ring(spikes: np.ndarray) -> float:
    vector = np.sum(spikes * np.exp(1j * ag.cc.PHI))
    if abs(vector) < 1e-12:
        raise RuntimeError("compass ring was silent during calibration")
    return float(np.angle(vector))


def run_case(seed: int, name: str, case: dict, steps: int, substeps: int) -> dict:
    np.random.seed(seed)
    agent = ag.AIFAgent3D(seed=seed, **case["build"])
    agent.world = ag.w3.World3D(seed=seed, n_food=0, n_tox=0, arena=8.0)
    agent.world.data.qpos[agent.world.jyaw] = _initial_yaw(seed)
    mujoco.mj_forward(agent.world.model, agent.world.data)
    agent.img = agent.world.retina()
    agent.birth()

    target = getattr(ag, case["turn_neuron"])
    stimulus_active = False
    ring_window: deque[np.ndarray] = deque(maxlen=24)
    trace: list[dict] = []

    def neural_input(current) -> None:
        nonlocal stimulus_active
        stimulus_active = TURN_START <= current.t < TURN_STOP
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
            "ring_heading_radians": _heading_from_ring(np.sum(ring_window, axis=0)),
            "turn_spikes": {"TL": int(current.nb[ag.TL].O > 0), "TR": int(current.nb[ag.TR].O > 0)},
            "vestibular_events": {
                "CCW": int(current.nb[ag.VEST_CCW].O > 0),
                "CW": int(current.nb[ag.VEST_CW].O > 0),
            },
            "notch_state": {
                "CCW": float(current.nb[ag.VEST_NET_CCW].S),
                "CW": float(current.nb[ag.VEST_NET_CW].S),
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
        agent, steps=steps, sub=substeps, vision=False,
        render_every=10**9, render_ticks=0, log_every=10**9,
        tick_hook=capture, neural_input_hook=neural_input,
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


def _summary(record: dict) -> dict:
    trace = record["tick_trace"]
    active = [row for row in trace if row["turn_probe_active"]]
    if len(active) < 2:
        raise ValueError("turn-probe window was not sampled")

    def delta(rows: list[dict], key: str) -> float:
        if key == "yaw":
            values = np.unwrap(np.asarray([row["pose"]["yaw"] for row in rows]))
        else:
            values = np.unwrap(np.asarray([row["ring_heading_radians"] for row in rows]))
        return float(np.degrees(values[-1] - values[0]))

    body_delta = delta(active, "yaw")
    ring_delta = delta(active, "ring")
    return {
        "active_window_body_yaw_degrees": body_delta,
        "active_window_ring_yaw_degrees": ring_delta,
        "active_window_tracking_ratio": ring_delta / body_delta if abs(body_delta) > 1e-12 else None,
        "turn_spikes": {
            name: int(sum(row["turn_spikes"][name] for row in active)) for name in ("TL", "TR")
        },
        "vestibular_events": {
            name: int(sum(row["vestibular_events"][name] for row in active)) for name in ("CCW", "CW")
        },
        "shift_spikes": {
            name: int(sum(row["shift_spikes"][name] for row in active)) for name in ("CL", "CR")
        },
        "max_abs_actuator_control": float(max(
            abs(value) for row in active for value in row["actuator_ctrl"].values()
        )),
    }


def _accept(summaries: dict[str, dict[str, dict]]) -> list[str]:
    failures: list[str] = []
    for seed, rows in summaries.items():
        tl, tr, ablated = rows["tl_neural_turn"], rows["tr_neural_turn"], rows["notch_to_pen_ablation"]
        for label, row, expected in (("TL", tl, 1), ("TR", tr, -1)):
            body = row["active_window_body_yaw_degrees"]
            ring = row["active_window_ring_yaw_degrees"]
            ratio = row["active_window_tracking_ratio"]
            if expected * body < 30.0:
                failures.append(f"seed {seed} {label}: neural turn did not rotate the MuJoCo body")
            if expected * ring < 15.0:
                failures.append(f"seed {seed} {label}: PAULA compass did not move in the physical turn direction")
            if ratio is None or not (0.25 <= abs(ratio) <= 1.5):
                failures.append(f"seed {seed} {label}: active-window ring/body gain outside calibrated range")
            if row["turn_spikes"][label] < 50 or row["max_abs_actuator_control"] <= 0.0:
                failures.append(f"seed {seed} {label}: prescribed neural turn bypassed the normal motor path")
            if sum(row["vestibular_events"].values()) < 10:
                failures.append(f"seed {seed} {label}: raw physical gyro did not reach PAULA vestibular events")
        if ablated["active_window_body_yaw_degrees"] < 30.0 or ablated["turn_spikes"]["TL"] < 50:
            failures.append(f"seed {seed}: P-EN-output ablation changed the physical neural turn course")
        if abs(ablated["active_window_ring_yaw_degrees"]) > 15.0:
            failures.append(f"seed {seed}: zeroed notch-to-P-EN output still moved the ring")
        if sum(ablated["vestibular_events"].values()) < 10:
            failures.append(f"seed {seed}: P-EN-output ablation removed physical gyro representation")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 44, 77])
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-cases", type=int, help="run at most this many unfinished seed/condition pairs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0 or args.substeps <= 0:
        raise SystemExit("--steps and --substeps must be positive")
    if args.max_cases is not None and args.max_cases <= 0:
        raise SystemExit("--max-cases must be positive")
    if args.output and args.resume:
        raise SystemExit("use either --output or --resume, not both")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.resume or args.output or HERE / "results" / f"embodied_compass_gyro_causal_{stamp}"
    if args.resume:
        manifest_path = output / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"--resume requires an existing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("experiment") != "embodied_compass_gyro_causal":
            raise SystemExit("--resume directory belongs to another experiment")
        if manifest.get("steps") != args.steps or manifest.get("substeps") != args.substeps:
            raise SystemExit("--resume requires the manifest's --steps and --substeps")
        if manifest.get("seeds") != args.seeds:
            raise SystemExit("--resume requires the manifest's exact --seeds set and order")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "manifest.json", {
            "experiment": "embodied_compass_gyro_causal",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": {"active_inference": _revision(ROOT), "neuron_model": _revision(NEURON_MODEL)},
            "source_fingerprints": {
                "aif_agent3d.py": _sha256(HERE.parent / "aif_agent3d.py"),
                "world3d.py": _sha256(HERE.parent / "world3d.py"),
                "embodied_compass_gyro_causal.py": _sha256(Path(__file__).resolve()),
            },
            "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "mujoco": mujoco.__version__},
            "seeds": args.seeds, "steps": args.steps, "substeps": args.substeps,
            "conditions": CASES,
            "turn_probe": {"start_neural_tick": TURN_START, "stop_neural_tick": TURN_STOP, "current": 3.0},
            "scope": "prescribed PAULA TL/TR current -> PAULA relay/graded muscle -> MuJoCo yaw-rate -> PAULA full-stroke notch/comparator -> P-EN/ring; excludes autonomous turn selection, path integration, HOME, and homing",
            "primary_evidence": "every per-condition-per-seed JSON retains physical, vestibular, ring, and motor tick traces",
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
            print(
                f"seed={seed} {name}: body={row['active_window_body_yaw_degrees']:+.1f}deg "
                f"ring={row['active_window_ring_yaw_degrees']:+.1f}deg "
                f"gain={row['active_window_tracking_ratio']:+.2f}",
                flush=True,
            )
            attempts += 1
        if args.max_cases is not None and attempts >= args.max_cases:
            break

    complete = all((output / f"{name}_seed{seed}.json").exists() for seed in args.seeds for name in CASES)
    if not complete:
        print("INCOMPLETE: resume to run missing cases")
        return 0
    summaries = {
        str(seed): {name: _summary(json.loads((output / f"{name}_seed{seed}.json").read_text())) for name in CASES}
        for seed in args.seeds
    }
    _write_json(output / "summary.json", summaries)
    failures = _accept(summaries)
    _write_json(output / "acceptance.json", {"passed": not failures, "failures": failures})
    print(f"Wrote compass gyro causal evidence to {output}")
    if failures:
        print("FAIL: " + " | ".join(failures), file=sys.stderr)
        return 1
    print("PASS: raw physical yaw causally drives signed PAULA recurrent-compass motion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
