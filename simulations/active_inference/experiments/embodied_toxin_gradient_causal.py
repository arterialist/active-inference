"""Causal lateral-toxin steering test in the integrated PAULA/MuJoCo agent.

Vision is disabled.  A physical short-range toxin field is placed at either
lateral side of the body; the normal toxin sensor populations feed crossed
opponent turn populations.  ``w_tox=0`` removes only those sensor-to-turn
synapses while retaining identical sensors, CPG, muscles, and body physics.
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
    "toxin_positive_y": {"source": [-2.0, 1.0], "build": {}},
    "toxin_negative_y": {"source": [-2.0, -1.0], "build": {}},
    "toxin_path_ablation": {"source": [-2.0, 1.0], "build": {"w_tox": 0.0}},
}


def _revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_toxin_only_world(agent, source: list[float]) -> None:
    agent.world.foods = [FAR.copy() for _ in range(7)]
    agent.world.toxins = [list(source)] + [FAR.copy() for _ in range(5)]
    agent.world._sync_mocap()


def run_case(seed: int, name: str, case: dict, steps: int, substeps: int) -> dict:
    np.random.seed(seed)
    agent = ag.AIFAgent3D(seed=seed, **case["build"])
    _install_toxin_only_world(agent, case["source"])
    agent.birth()
    trace = []

    def capture(current):
        x, y, yaw = current.world.pose()
        trace.append({
            "neural_tick": current.t,
            "sensory_current": dict(current.last_sensor_drives),
            "toxin_sensor_spikes": {"left": sum(int(current.nb[n].O > 0) for n in ag.TXL), "right": sum(int(current.nb[n].O > 0) for n in ag.TXR)},
            "turn_spikes": {"TL": int(current.nb[ag.TL].O > 0), "TR": int(current.nb[ag.TR].O > 0)},
            "muscle_state": {"MLp": float(current.nb[ag.MLp].S), "MLr": float(current.nb[ag.MLr].S), "MRp": float(current.nb[ag.MRp].S), "MRr": float(current.nb[ag.MRr].S)},
            "actuator_ctrl": {label: float(current.world.data.ctrl[aid]) for label, aid in current.world.act_id.items()},
            "pose": {"x": float(x), "y": float(y), "yaw": float(yaw)},
        })

    start = agent.world.pose()
    ag.run_episode(agent, steps=steps, sub=substeps, render_every=10**9, render_ticks=0, vision=False, log_every=10**9, tick_hook=capture)
    end = agent.world.pose()
    return {"condition": name, "seed": seed, "build": case["build"], "source": case["source"], "vision_enabled": False, "steps": steps, "neural_substeps_per_body_step": substeps, "initial_pose": {"x": float(start[0]), "y": float(start[1]), "yaw": float(start[2])}, "final_pose": {"x": float(end[0]), "y": float(end[1]), "yaw": float(end[2])}, "tick_trace": trace}


def _summary(record: dict) -> dict:
    trace = record["tick_trace"]
    left = sum(row["toxin_sensor_spikes"]["left"] for row in trace)
    right = sum(row["toxin_sensor_spikes"]["right"] for row in trace)
    tl = sum(row["turn_spikes"]["TL"] for row in trace)
    tr = sum(row["turn_spikes"]["TR"] for row in trace)
    initial, final = record["initial_pose"], record["final_pose"]
    return {"toxin_sensor_left_spikes": left, "toxin_sensor_right_spikes": right, "turn_TL_spikes": tl, "turn_TR_spikes": tr, "turn_imbalance_TL_minus_TR": tl-tr, "max_abs_actuator_control": max(abs(value) for row in trace for value in row["actuator_ctrl"].values()), "body_displacement": float(np.hypot(final["x"]-initial["x"], final["y"]-initial["y"]))}


def _accept(summaries: dict[str, dict[str, dict]]) -> list[str]:
    failures = []
    for seed, rows in summaries.items():
        positive, negative, ablated = rows["toxin_positive_y"], rows["toxin_negative_y"], rows["toxin_path_ablation"]
        if positive["toxin_sensor_left_spikes"] + positive["toxin_sensor_right_spikes"] < 100 or negative["toxin_sensor_left_spikes"] + negative["toxin_sensor_right_spikes"] < 100:
            failures.append(f"seed {seed}: lateral toxin did not reach physical toxin sensors")
        if positive["turn_imbalance_TL_minus_TR"] < 6 or negative["turn_imbalance_TL_minus_TR"] > -6:
            failures.append(f"seed {seed}: mirroring toxin did not reverse opponent turn output")
        if ablated["toxin_sensor_left_spikes"] + ablated["toxin_sensor_right_spikes"] < 100:
            failures.append(f"seed {seed}: toxin-path ablation removed sensory activity")
        if ablated["turn_TL_spikes"] or ablated["turn_TR_spikes"]:
            failures.append(f"seed {seed}: w_tox=0 still drove toxin turn populations")
        if min(positive["max_abs_actuator_control"], negative["max_abs_actuator_control"]) <= 0.0:
            failures.append(f"seed {seed}: toxin steering did not retain the normal motor path")
    return failures


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 44, 77])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="complete an interrupted output directory without overwriting traces")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0 or args.substeps <= 0:
        raise SystemExit("--steps and --substeps must be positive")
    if args.resume and args.output:
        raise SystemExit("use either --output or --resume, not both")
    output = args.resume or args.output or HERE / "results" / f"embodied_toxin_gradient_causal_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if args.resume:
        manifest = output / "manifest.json"
        if not manifest.exists() or json.loads(manifest.read_text()).get("experiment") != "embodied_toxin_gradient_causal":
            raise SystemExit("--resume requires this experiment's existing manifest")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "manifest.json", {"experiment": "embodied_toxin_gradient_causal", "created_utc": datetime.now(timezone.utc).isoformat(), "source_revision": {"active_inference": _revision(ROOT), "neuron_model": _revision(NEURON_MODEL)}, "source_fingerprints": {"aif_agent3d.py": _sha256(HERE.parent / "aif_agent3d.py"), "embodied_toxin_gradient_causal.py": _sha256(Path(__file__).resolve())}, "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "mujoco": mujoco.__version__}, "seeds": args.seeds, "steps": args.steps, "substeps": args.substeps, "conditions": CASES, "scope": "nonvisual lateral physical toxin -> PAULA toxin sensors -> crossed opponent-turn populations -> normal MuJoCo motor path; excludes head-on toxin response", "primary_evidence": "tick_trace in every per-condition-per-seed JSON file"})
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
    print(f"Wrote embodied toxin-gradient causal evidence to {output}")
    for seed, rows in summaries.items():
        print(f"seed={seed}: " + "; ".join(f"{name} sensor={row['toxin_sensor_left_spikes']+row['toxin_sensor_right_spikes']} turn={row['turn_imbalance_TL_minus_TR']:+d}" for name, row in rows.items()))
    if failures:
        print("FAIL: " + " | ".join(failures), file=sys.stderr)
        return 1
    print("PASS: the nonvisual physical toxin gradient causally drives signed PAULA avoidance turns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
