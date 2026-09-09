"""Causal acceptance test for symmetric, head-on toxin escape.

The normal crossed toxin turn circuit is intentionally symmetric in this
geometry, so it cannot choose left versus right.  The accepted route is the
neural temporal-rise bank (TRISE): toxin sensor population -> TPOOL -> delayed
rise neurons -> STEER -> relay/muscle asymmetry -> MuJoCo turn.  No host-side
turn, pose, velocity, or escape state is supplied.
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
from simulations.active_inference.experiments.matrix_protocol import (
    hardest_completion_steps,
    protocol_metadata,
    require_protocol,
    selected_worlds,
)
from simulations.active_inference.experiments.embodied_recording import (
    EmbodiedVideoRecorder,
    artifact_directory,
)
from simulations.active_inference.live.versions import get_version


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NEURON_MODEL = ROOT.parent / "neuron-model"
FAR = [100.0, 100.0]
WORLD_SPECS = {
    "near_01": {"source": [-0.65, 0.0], "label": "near symmetric head-on 1", "completion_steps": 150},
    "near_02": {"source": [-0.75, 0.0], "label": "near symmetric head-on 2", "completion_steps": 160},
    "near_03": {"source": [-0.85, 0.0], "label": "near symmetric head-on 3", "completion_steps": 170},
    "mid_04": {"source": [-0.95, 0.0], "label": "mid symmetric head-on 1", "completion_steps": 180},
    "mid_05": {"source": [-1.05, 0.0], "label": "mid symmetric head-on 2", "completion_steps": 200},
    "mid_06": {"source": [-1.15, 0.0], "label": "mid symmetric head-on 3", "completion_steps": 220},
    "far_07": {"source": [-1.25, 0.0], "label": "far symmetric head-on 1", "completion_steps": 240},
    "far_08": {"source": [-1.35, 0.0], "label": "far symmetric head-on 2", "completion_steps": 260},
    "far_09": {"source": [-1.45, 0.0], "label": "far symmetric head-on 3", "completion_steps": 280},
    "far_10": {"source": [-1.60, 0.0], "label": "far symmetric head-on 4", "completion_steps": 320},
}
WORLD_NAMES = tuple(WORLD_SPECS)
CASES = {
    "headon_trise_full": {"build": {}},
    "trise_to_steer_ablation": {"build": {"trise": True, "w_trise": 0.0}},
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


def _install_headon_hazard(agent, world: str) -> None:
    agent.world.foods = [FAR.copy() for _ in range(7)]
    agent.world.toxins = [list(WORLD_SPECS[world]["source"])] + [FAR.copy() for _ in range(5)]
    agent.world.respawn_food = False
    agent.world._sync_mocap()


def run_case(seed: int, name: str, case: dict, steps: int, substeps: int, version: str, world: str,
             artifact_root: Path | None = None) -> dict:
    np.random.seed(seed)
    profile = get_version(version)
    agent = ag.AIFAgent3D(seed=seed, components=profile.components, **case["build"])
    _install_headon_hazard(agent, world)
    agent.birth()
    recorder = EmbodiedVideoRecorder(
        agent, artifact_directory(artifact_root, name, seed),
        total_ticks=steps * substeps, substeps=substeps,
    )
    trace = []

    def capture(current):
        x, y, yaw = current.world.pose()
        trace.append({
            "neural_tick": current.t,
            "sensory_current": dict(current.last_sensor_drives),
            "toxin_sensor_spikes": {
                "left": sum(int(current.nb[nid].O > 0) for nid in ag.TXL),
                "right": sum(int(current.nb[nid].O > 0) for nid in ag.TXR),
            },
            "trise_spikes": sum(int(current.nb[nid].O > 0) for nid in ag.TRISE),
            "steer_spike": int(current.nb[ag.STEER].O > 0),
            "turn_spikes": {"TL": int(current.nb[ag.TL].O > 0), "TR": int(current.nb[ag.TR].O > 0)},
            "muscle_state": {
                "MLp": float(current.nb[ag.MLp].S), "MLr": float(current.nb[ag.MLr].S),
                "MRp": float(current.nb[ag.MRp].S), "MRr": float(current.nb[ag.MRr].S),
            },
            "actuator_ctrl": {key: float(current.world.data.ctrl[value]) for key, value in current.world.act_id.items()},
            "pose": {"x": float(x), "y": float(y), "yaw": float(yaw)},
            "source_distance": float(np.hypot(x - WORLD_SPECS[world]["source"][0], y - WORLD_SPECS[world]["source"][1])),
        })
        recorder.observe(current)

    start = agent.world.pose()
    _, modes = ag.run_episode(
        agent, steps=steps, sub=substeps, vision=False, gain=5.0, tox_gain=6.0,
        render_every=10**9, render_ticks=0, log_every=10**9, tick_hook=capture,
    )
    end = agent.world.pose()
    video_artifacts = recorder.finalize()
    return {
        "condition": name, "seed": seed, "version": profile.id, "world": world,
        "component_profile": dict(agent.component_profile), "neuron_count": len(agent.nb),
        "build_parameters": dict(agent.build_parameters),
        "build": case["build"], "source": list(WORLD_SPECS[world]["source"]),
        "world_fixture": {"food_count": len(agent.world.foods), "toxin_count": len(agent.world.toxins),
                           "respawn_food": bool(agent.world.respawn_food), "fixed_source": list(WORLD_SPECS[world]["source"])},
        "vision_enabled": False, "steps": steps, "neural_substeps_per_body_step": substeps,
        "initial_pose": {"x": float(start[0]), "y": float(start[1]), "yaw": float(start[2])},
        "final_pose": {"x": float(end[0]), "y": float(end[1]), "yaw": float(end[2])},
        "toxin_hits": agent.world.tox_hits, "mode_steps": modes, "tick_trace": trace,
        "video_artifacts": video_artifacts,
    }


def _summary(record: dict) -> dict:
    trace = record["tick_trace"]
    first, last = record["initial_pose"], record["final_pose"]
    return {
        "world": record["world"],
        "toxin_hits": record["toxin_hits"],
        "toxin_sensor_spikes": sum(row["toxin_sensor_spikes"]["left"] + row["toxin_sensor_spikes"]["right"] for row in trace),
        "trise_spikes": sum(row["trise_spikes"] for row in trace),
        "steer_spikes": sum(row["steer_spike"] for row in trace),
        "turn_imbalance_TL_minus_TR": sum(row["turn_spikes"]["TL"] - row["turn_spikes"]["TR"] for row in trace),
        "minimum_source_distance": min(row["source_distance"] for row in trace),
        "final_source_distance": trace[-1]["source_distance"],
        "body_displacement": float(np.hypot(last["x"] - first["x"], last["y"] - first["y"])),
        "max_abs_actuator_control": max(abs(value) for row in trace for value in row["actuator_ctrl"].values()),
        "trace_ticks": len(trace),
        "expected_trace_ticks": int(record["steps"] * record["neural_substeps_per_body_step"]),
        "strict_topology": bool(record["component_profile"].get("strict")),
        "component_names": list(record["component_profile"].get("components", ())),
        "neuron_count": int(record["neuron_count"]),
        "fixture_no_respawn": record["world_fixture"].get("respawn_food") is False,
        "fixture_food_count": int(record["world_fixture"].get("food_count", -1)),
        "fixture_toxin_count": int(record["world_fixture"].get("toxin_count", -1)),
    }


EXPECTED_NEURONS = {"v1": 95, "v2": 299, "v3": 345, "v4": 373}


def _accept(summaries: dict[str, dict[str, dict[str, dict]]], version: str) -> list[str]:
    failures = []
    expected_components = set(get_version(version).components)
    for world, by_seed in summaries.items():
        for seed, rows in by_seed.items():
            full, ablated = rows["headon_trise_full"], rows["trise_to_steer_ablation"]
            prefix = f"{world} seed {seed}"
            for label, row in rows.items():
                if row["trace_ticks"] != row["expected_trace_ticks"]:
                    failures.append(f"{prefix} {label}: incomplete tick trace ({row['trace_ticks']}/{row['expected_trace_ticks']})")
                if (not row["strict_topology"] or row["neuron_count"] != EXPECTED_NEURONS[version]
                        or set(row["component_names"]) != expected_components):
                    failures.append(f"{prefix} {label}: trace was not produced by strict {version} topology ({row['neuron_count']} neurons)")
                if not row["fixture_no_respawn"] or row["fixture_food_count"] != 7 or row["fixture_toxin_count"] != 6:
                    failures.append(f"{prefix} {label}: head-on fixture was changed or permits respawn")
            if full["toxin_sensor_spikes"] < 6000 or ablated["toxin_sensor_spikes"] < 6000:
                failures.append(f"{prefix}: symmetric physical toxin did not drive the toxin sensors")
            if full["trise_spikes"] < 100 or ablated["trise_spikes"] < 100:
                failures.append(f"{prefix}: TRISE did not detect the head-on temporal rise")
            if full["toxin_hits"] != 0 or full["minimum_source_distance"] <= 0.55:
                failures.append(f"{prefix}: intact TRISE route did not prevent head-on toxin contact")
            if full["steer_spikes"] < ablated["steer_spikes"] + 20:
                failures.append(f"{prefix}: TRISE output did not increase the neural escape drive")
            if min(full["max_abs_actuator_control"], ablated["max_abs_actuator_control"]) <= 0.0:
                failures.append(f"{prefix}: head-on trial did not reach the ordinary MuJoCo actuator path")
    return failures


def _inconclusive(summaries: dict[str, dict[str, dict[str, dict]]]) -> list[str]:
    """A non-contacting control cannot prove that TRISE prevented contact."""
    return [
        f"{world} seed {seed}: TRISE ablation never entered the contact zone "
        "(fixture is inconclusive for this replicate)"
        for world, by_seed in summaries.items()
        for seed, rows in by_seed.items()
        if rows["trise_to_steer_ablation"]["toxin_hits"] < 1
        or rows["trise_to_steer_ablation"]["minimum_source_distance"] >= 0.55
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=None,
                        help="horizon; defaults to the hardest selected world")
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 44, 77, 101])
    parser.add_argument("--version", choices=("v1", "v2", "v3", "v4"), default="v1",
                        help="strict agent profile under test")
    parser.add_argument("--world", choices=("all", *WORLD_NAMES), default="all",
                        help="symmetric toxin distance fixture; 'all' runs near/mid/far")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES), help="conditions to run or resume")
    parser.add_argument("--protocol", action="store_true",
                        help="require the ten-world/five-seed embodied matrix contract")
    parser.add_argument("--output", type=Path, help="new directory for manifest and raw traces")
    parser.add_argument("--resume", type=Path, help="continue a previously created evidence directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    worlds = selected_worlds(WORLD_SPECS, args.world)
    args.steps = hardest_completion_steps(WORLD_SPECS, worlds) if args.steps is None else int(args.steps)
    if args.steps <= 0 or args.substeps <= 0:
        raise SystemExit("--steps and --substeps must be positive")
    if tuple(args.cases) != tuple(CASES):
        raise SystemExit("acceptance requires both causal conditions; use the full --cases set")
    if args.protocol:
        try:
            require_protocol(world_specs=WORLD_SPECS, worlds=worlds, seeds=args.seeds,
                             steps=args.steps, requested_world=args.world)
        except ValueError as exc:
            raise SystemExit(str(exc))
    if args.output and args.resume:
        raise SystemExit("use either --output or --resume, not both")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.resume or args.output or HERE / "results" / f"embodied_headon_toxin_causal_{timestamp}"
    if args.resume:
        manifest = output / "manifest.json"
        if not manifest.exists():
            raise SystemExit(f"--resume requires an existing manifest: {manifest}")
        prior = json.loads(manifest.read_text())
        if prior.get("experiment") != "embodied_headon_toxin_causal":
            raise SystemExit("--resume directory belongs to another experiment")
        if (prior.get("steps") != args.steps or prior.get("substeps") != args.substeps
                or prior.get("version") != args.version or prior.get("world") != args.world):
            raise SystemExit("--resume requires the manifest's version, world, steps, and substeps")
        if prior.get("seeds") != args.seeds or prior.get("cases") != CASES:
            raise SystemExit("--resume requires the manifest's exact seeds and causal conditions")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "manifest.json", {
            "experiment": "embodied_headon_toxin_causal", "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": {"active_inference": _revision(ROOT), "neuron_model": _revision(NEURON_MODEL)},
            "source_fingerprints": {
                "aif_agent3d.py": _sha256(HERE.parent / "aif_agent3d.py"),
                "embodied_headon_toxin_causal.py": _sha256(Path(__file__).resolve()),
            },
            "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "mujoco": mujoco.__version__},
            "version": args.version, "components": list(get_version(args.version).components),
            "effective_build_parameters": True,
            "record_visual_artifacts": True,
            "world": args.world, "worlds": worlds, "world_catalog": WORLD_SPECS,
            "seeds": args.seeds, "steps": args.steps, "substeps": args.substeps, "conditions": CASES,
            "acceptance_protocol": protocol_metadata(world_specs=WORLD_SPECS, worlds=worlds,
                                                       seeds=args.seeds, steps=args.steps,
                                                       required=args.protocol, requested_world=args.world),
            "scope": "symmetric head-on toxin -> PAULA toxin sensors -> TPOOL/TRISE temporal-rise bank -> STEER -> normal relay/muscle/MuJoCo path; excludes visual and learned-valence routes",
            "primary_evidence": "every per-condition-per-seed JSON contains raw neural/physical tick traces",
        })
    records: dict[str, dict[str, dict[str, dict]]] = {}
    for world in worlds:
        world_output = output / world if len(worlds) > 1 else output
        world_output.mkdir(parents=True, exist_ok=True)
        records[world] = {}
        for seed in args.seeds:
            records[world][str(seed)] = {}
            for name in args.cases:
                path = world_output / f"{name}_seed{seed}.json"
                if args.resume and path.exists():
                    record = json.loads(path.read_text())
                else:
                    record = run_case(seed, name, CASES[name], args.steps, args.substeps, args.version, world,
                                      artifact_root=world_output)
                    _write_json(path, record)
                records[world][str(seed)][name] = _summary(record)
                row = records[world][str(seed)][name]
                print(f"world={world} seed={seed} case={name}: hits={row['toxin_hits']} min={row['minimum_source_distance']:.3f} trise={row['trise_spikes']} steer={row['steer_spikes']}", flush=True)
    complete = all(
        (output / world / f"{name}_seed{seed}.json" if len(worlds) > 1 else output / f"{name}_seed{seed}.json").exists()
        for world in worlds for seed in args.seeds for name in CASES
    )
    if not complete:
        print("Partial evidence saved; resume the remaining seed/condition files before acceptance.")
        return 0
    _write_json(output / "summary.json", records)
    failures = _accept(records, args.version)
    inconclusive = _inconclusive(records)
    status = "failed" if failures else ("inconclusive" if inconclusive else "passed")
    protocol = json.loads((output / "manifest.json").read_text()).get("acceptance_protocol", {})
    protocol_failures = list(protocol.get("failures", [])) if protocol.get("required") else []
    if protocol_failures:
        status = "failed"
    _write_json(output / "acceptance.json", {"passed": not failures and not inconclusive and not protocol_failures, "status": status,
                                                "failures": failures + protocol_failures, "inconclusive": inconclusive,
                                                "version": args.version, "worlds": worlds,
                                                "protocol_compliant": bool(protocol.get("compliant", False)),
                                                "full_tick_traces": True, "strict_topology": True})
    print(f"Wrote embodied head-on toxin causal evidence to {output}")
    if failures or protocol_failures:
        print("FAIL: " + " | ".join(failures + protocol_failures), file=sys.stderr)
        return 1
    if inconclusive:
        print("INCONCLUSIVE: " + " | ".join(inconclusive), file=sys.stderr)
        return 1
    print(f"PASS: strict {args.version} TRISE route is causally required across {len(worlds)} symmetric hazards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
