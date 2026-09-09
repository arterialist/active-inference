"""Causal physical wiring test for the opt-in Stone PI home-output route.

The body is deliberately empty: its own PAULA CPG makes it move, speed is
transduced into the graded Stone memory, and that memory must drive OPP, the
heading-gated CPU1 comparator, HOME, and the normal relay/muscle path.  The
control leaves the same memory cells, body, proprioception and CPG intact but
zeros *only* Stone's two output projections.  It therefore tests expression
of the memory in the composed MuJoCo organism, not a Python home policy.

This is not yet a homing-success claim.  The recurrent compass must still be
validated under a physical outbound-and-return trajectory before this route
can be credited with accurate return-to-origin navigation.
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
CASES = {
    "stone_home_output": {"build": {"pistone": True, "pistone_opp": True, "graded": True}},
    "stone_outputs_zero": {
        "build": {
            "pistone": True, "pistone_opp": True, "graded": True,
            "w_opp_stone": 0.0, "w_cpu1_stone": 0.0,
        }
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


def run_case(seed: int, name: str, case: dict, steps: int, substeps: int) -> dict:
    np.random.seed(seed)
    agent = ag.AIFAgent3D(seed=seed, **case["build"])
    # No object, visual, or host-side destination signal participates.  The
    # only spatial reference is the birth origin stored by the physics body.
    agent.world = ag.w3.World3D(seed=seed, n_food=0, n_tox=0, arena=6.0)
    agent.img = agent.world.retina()
    agent.birth()
    trace: list[dict] = []

    def capture(current):
        x, y, yaw = current.world.pose()
        trace.append({
            "neural_tick": current.t,
            "pose": {"x": float(x), "y": float(y), "yaw": float(yaw)},
            "distance_from_birth": float(np.hypot(x, y)),
            "proprioceptive_speed_current": float(min(1.5, current.world.speed() * current.k_prop)),
            "stone_memory_S": [float(current.nb[nid].S) for nid in ag.pstone.MEM],
            "stone_memory_release": [float(current.nb[nid].O) for nid in ag.pstone.MEM],
            "opponent_spikes": [int(current.nb[nid].O > 0) for nid in ag.nv.OPP],
            "mode_spikes": {
                mode: sum(int(current.nb[nid].O > 0) for nid in ag.ar.MODE[index])
                for index, mode in enumerate(ag.MODES)
            },
            "cpu1_spikes": {
                "left": sum(int(current.nb[nid].O > 0) for nid in ag.nv.HL),
                "right": sum(int(current.nb[nid].O > 0) for nid in ag.nv.HR),
            },
            "home_turn_spikes": {"HTL": int(current.nb[ag.HTL].O > 0), "HTR": int(current.nb[ag.HTR].O > 0)},
            "muscle_state": {
                "MLp": float(current.nb[ag.MLp].S), "MLr": float(current.nb[ag.MLr].S),
                "MRp": float(current.nb[ag.MRp].S), "MRr": float(current.nb[ag.MRr].S),
            },
            "actuator_ctrl": {
                key: float(current.world.data.ctrl[value]) for key, value in current.world.act_id.items()
            },
        })

    start = agent.world.pose()
    _, modes = ag.run_episode(
        agent, steps=steps, sub=substeps, vision=False,
        render_every=10**9, render_ticks=0, log_every=10**9, tick_hook=capture,
    )
    end = agent.world.pose()
    return {
        "condition": name, "seed": seed, "build": case["build"],
        "vision_enabled": False, "objects": {"food": 0, "toxin": 0},
        "steps": steps, "neural_substeps_per_body_step": substeps,
        "initial_pose": {"x": float(start[0]), "y": float(start[1]), "yaw": float(start[2])},
        "final_pose": {"x": float(end[0]), "y": float(end[1]), "yaw": float(end[2])},
        "mode_steps": modes, "tick_trace": trace,
    }


def _summary(record: dict) -> dict:
    trace = record["tick_trace"]
    yaw = np.unwrap(np.asarray([row["pose"]["yaw"] for row in trace]))
    first, last = record["initial_pose"], record["final_pose"]
    return {
        "stone_release_total": float(sum(sum(row["stone_memory_release"]) for row in trace)),
        "stone_memory_span_final": float(np.ptp(trace[-1]["stone_memory_S"])),
        "opponent_spikes": int(sum(sum(row["opponent_spikes"]) for row in trace)),
        "home_mode_spikes": int(sum(row["mode_spikes"]["HOME"] for row in trace)),
        "home_turn_spikes": int(sum(sum(row["home_turn_spikes"].values()) for row in trace)),
        "cpu1_spikes": int(sum(sum(row["cpu1_spikes"].values()) for row in trace)),
        "body_displacement": float(np.hypot(last["x"] - first["x"], last["y"] - first["y"])),
        "maximum_distance_from_birth": float(max(row["distance_from_birth"] for row in trace)),
        "yaw_excursion_degrees": float(np.degrees(np.ptp(yaw))),
        "motor_asymmetry_L_minus_R": float(sum(
            (row["muscle_state"]["MLp"] + row["muscle_state"]["MLr"])
            - (row["muscle_state"]["MRp"] + row["muscle_state"]["MRr"])
            for row in trace
        )),
        "max_abs_actuator_control": float(max(
            abs(value) for row in trace for value in row["actuator_ctrl"].values()
        )),
    }


def _accept(summaries: dict[str, dict[str, dict]]) -> list[str]:
    failures = []
    for seed, rows in summaries.items():
        full, control = rows["stone_home_output"], rows["stone_outputs_zero"]
        if min(full["stone_release_total"], control["stone_release_total"]) <= 1.0:
            failures.append(f"seed {seed}: physical speed did not drive the graded Stone memory")
        if full["stone_memory_span_final"] <= 0.01:
            failures.append(f"seed {seed}: Stone memory was not spatially differentiated")
        if full["opponent_spikes"] <= 0 or full["home_mode_spikes"] <= 0 or full["home_turn_spikes"] <= 0:
            failures.append(f"seed {seed}: intact Stone route did not reach OPP, HOME, and home-turn neurons")
        if control["opponent_spikes"] != 0 or control["home_mode_spikes"] != 0:
            failures.append(f"seed {seed}: zeroed Stone output still reached OPP or the HOME population")
        # CPU1 retains its heading-only ring inputs.  A rare isolated turn-neuron
        # spike is therefore possible even with its memory source at zero; it
        # does not constitute an expressed home vector.  Require a strong
        # causal separation rather than falsely treating that baseline noise as
        # a complete neural route.
        if control["home_turn_spikes"] > max(4, full["home_turn_spikes"] // 10):
            failures.append(f"seed {seed}: zeroed Stone output did not suppress home-turn expression")
        if min(full["max_abs_actuator_control"], control["max_abs_actuator_control"]) <= 0.0:
            failures.append(f"seed {seed}: physical trial bypassed the normal MuJoCo actuator path")
    return failures


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 44, 77])
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0 or args.substeps <= 0:
        raise SystemExit("--steps and --substeps must be positive")
    if args.output and args.resume:
        raise SystemExit("use either --output or --resume, not both")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.resume or args.output or HERE / "results" / f"embodied_stone_home_causal_{stamp}"
    if args.resume:
        manifest_path = output / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"--resume requires an existing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("experiment") != "embodied_stone_home_causal":
            raise SystemExit("--resume directory belongs to another experiment")
        if manifest.get("steps") != args.steps or manifest.get("substeps") != args.substeps:
            raise SystemExit("--resume requires the manifest's --steps and --substeps")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "manifest.json", {
            "experiment": "embodied_stone_home_causal", "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": {"active_inference": _revision(ROOT), "neuron_model": _revision(NEURON_MODEL)},
            "source_fingerprints": {
                "aif_agent3d.py": _sha256(HERE.parent / "aif_agent3d.py"),
                "cx_navigator.py": _sha256(HERE.parent / "cx_navigator.py"),
                "pi_stone.py": _sha256(HERE.parent / "pi_stone.py"),
                "embodied_stone_home_causal.py": _sha256(Path(__file__).resolve()),
            },
            "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "mujoco": mujoco.__version__},
            "seeds": args.seeds, "steps": args.steps, "substeps": args.substeps, "conditions": CASES,
            "scope": "empty physical world -> PAULA CPG/body speed -> graded Stone PI memory -> OPP/HOME/CPU1/home-turn -> ordinary relays, muscles, and MuJoCo; excludes recurrent-compass fidelity and return-to-origin success",
            "primary_evidence": "every per-condition-per-seed JSON retains physical and neural tick traces",
        })
    records: dict[str, dict[str, dict]] = {}
    for seed in args.seeds:
        records[str(seed)] = {}
        for name in args.cases:
            path = output / f"{name}_seed{seed}.json"
            record = json.loads(path.read_text()) if args.resume and path.exists() else run_case(
                seed, name, CASES[name], args.steps, args.substeps
            )
            _write_json(path, record)
            records[str(seed)][name] = _summary(record)
            row = records[str(seed)][name]
            print(
                f"seed={seed} case={name}: release={row['stone_release_total']:.2f} "
                f"opp={row['opponent_spikes']} home={row['home_mode_spikes']} turn={row['home_turn_spikes']}",
                flush=True,
            )
    complete = all((output / f"{name}_seed{seed}.json").exists() for seed in args.seeds for name in CASES)
    if not complete:
        print("Partial evidence saved; resume remaining seed/condition files before acceptance.")
        return 0
    all_records = {
        str(seed): {name: _summary(json.loads((output / f"{name}_seed{seed}.json").read_text())) for name in CASES}
        for seed in args.seeds
    }
    _write_json(output / "summary.json", all_records)
    failures = _accept(all_records)
    _write_json(output / "acceptance.json", {"passed": not failures, "failures": failures})
    print(f"Wrote Stone home-output causal evidence to {output}")
    if failures:
        print("FAIL: " + " | ".join(failures), file=sys.stderr)
        return 1
    print("PASS: the graded Stone memory is causally expressed through the embodied neural home-output route")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
