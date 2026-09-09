"""Short-horizon physical food-collection acceptance test.

The source is deliberately lateral to the baseline trajectory.  The full
PAULA food-sensor-to-opponent-turn route must collect it; ``w_sd=0`` leaves
the same physical odour, CPG, muscles, and MuJoCo body but cannot steer into
the source.  No Python path following or position update is used.
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
# A suite of fixed, non-respawning odour fields ordered from the smallest
# lateral displacement to the longest/cross-body approach.  The ten worlds
# are deliberately declared rather than sampled from a trajectory: the
# protocol's order is part of the evidence about increasing constraint.
WORLD_SPECS = {
    "shallow_left": {"source": [-1.0, 1.0], "label": "shallow left target", "completion_steps": 180},
    "shallow_right": {"source": [-1.0, -1.0], "label": "shallow right target", "completion_steps": 180},
    "near_left": {"source": [-1.2, 1.0], "label": "near left target", "completion_steps": 200},
    "near_right": {"source": [-1.2, -1.0], "label": "near right target", "completion_steps": 200},
    "mid_left": {"source": [-1.5, 1.0], "label": "mid left target", "completion_steps": 220},
    "mid_right": {"source": [-1.5, -1.0], "label": "mid right target", "completion_steps": 220},
    "far_left": {"source": [-1.8, 1.0], "label": "far left target", "completion_steps": 260},
    "far_right": {"source": [-1.8, -1.0], "label": "far right target", "completion_steps": 260},
    "deep_left": {"source": [-2.2, 1.1], "label": "deep left target", "completion_steps": 300},
    "deep_right": {"source": [-2.2, -1.1], "label": "deep right target", "completion_steps": 300},
}
WORLD_NAMES = tuple(WORLD_SPECS)
CASES = {
    "food_collection": {"build": {}},
    "food_sensor_to_turn_ablation": {"build": {"w_sd": 0.0}},
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


def _install_food_world(agent, world: str) -> None:
    source = list(WORLD_SPECS[world]["source"])
    agent.world.foods = [source.copy()] + [FAR.copy() for _ in range(6)]
    agent.world.toxins = [FAR.copy() for _ in range(6)]
    # This is a one-target causal fixture.  A respawn would turn an easy
    # collection into an unbounded reward stream and could make a control look
    # successful after the original target was missed.
    agent.world.respawn_food = False
    agent.world._sync_mocap()


def run_case(seed: int, name: str, case: dict, steps: int, substeps: int, version: str, world: str,
             artifact_root: Path | None = None) -> dict:
    np.random.seed(seed)
    profile = get_version(version)
    agent = ag.AIFAgent3D(seed=seed, components=profile.components, **case["build"])
    _install_food_world(agent, world)
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
            "food_sensor_spikes": {
                "left": sum(int(current.nb[nid].O > 0) for nid in ag.FLp),
                "right": sum(int(current.nb[nid].O > 0) for nid in ag.FRp),
            },
            "turn_spikes": {"TL": int(current.nb[ag.TL].O > 0), "TR": int(current.nb[ag.TR].O > 0)},
            "muscle_state": {
                "MLp": float(current.nb[ag.MLp].S), "MLr": float(current.nb[ag.MLr].S),
                "MRp": float(current.nb[ag.MRp].S), "MRr": float(current.nb[ag.MRr].S),
            },
            "actuator_ctrl": {key: float(current.world.data.ctrl[value]) for key, value in current.world.act_id.items()},
            "pose": {"x": float(x), "y": float(y), "yaw": float(yaw)},
            "food_eaten_total": current.world.eaten,
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
        "condition": name, "seed": seed, "version": profile.id,
        "component_profile": dict(agent.component_profile), "neuron_count": len(agent.nb),
        "build_parameters": dict(agent.build_parameters),
        "build": case["build"], "world": world, "source": list(WORLD_SPECS[world]["source"]),
        "world_fixture": {"food_count": len(agent.world.foods), "toxin_count": len(agent.world.toxins),
                           "respawn_food": bool(agent.world.respawn_food),
                           "fixed_target": list(WORLD_SPECS[world]["source"]), "far_placeholder": FAR},
        "vision_enabled": False, "steps": steps, "neural_substeps_per_body_step": substeps,
        "initial_pose": {"x": float(start[0]), "y": float(start[1]), "yaw": float(start[2])},
        "final_pose": {"x": float(end[0]), "y": float(end[1]), "yaw": float(end[2])},
        "food_eaten": agent.world.eaten, "mode_steps": modes, "tick_trace": trace,
        "video_artifacts": video_artifacts,
    }


def _summary(record: dict) -> dict:
    trace = record["tick_trace"]
    first, last = record["initial_pose"], record["final_pose"]
    food_left = sum(row["food_sensor_spikes"]["left"] for row in trace)
    food_right = sum(row["food_sensor_spikes"]["right"] for row in trace)
    tl = sum(row["turn_spikes"]["TL"] for row in trace)
    tr = sum(row["turn_spikes"]["TR"] for row in trace)
    return {
        "world": record["world"],
        "food_eaten": record["food_eaten"],
        "food_sensor_spikes": food_left + food_right,
        "turn_imbalance_TL_minus_TR": tl - tr,
        "first_collection_tick": next((row["neural_tick"] for row in trace if row["food_eaten_total"] > 0), None),
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
            full, ablated = rows["food_collection"], rows["food_sensor_to_turn_ablation"]
            prefix = f"{world} seed {seed}"
            for label, row in rows.items():
                if row["trace_ticks"] != row["expected_trace_ticks"]:
                    failures.append(f"{prefix} {label}: incomplete tick trace ({row['trace_ticks']}/{row['expected_trace_ticks']})")
                if (not row["strict_topology"] or row["neuron_count"] != EXPECTED_NEURONS[version]
                        or set(row["component_names"]) != expected_components):
                    failures.append(f"{prefix} {label}: trace was not produced by strict {version} topology ({row['neuron_count']} neurons)")
                if not row["fixture_no_respawn"] or row["fixture_food_count"] != 7 or row["fixture_toxin_count"] != 6:
                    failures.append(f"{prefix} {label}: food fixture was changed or permits respawn")
            if full["food_sensor_spikes"] < 500 or ablated["food_sensor_spikes"] < 500:
                failures.append(f"{prefix}: fixed source did not drive both food sensor populations")
            if full["food_eaten"] != 1 or full["first_collection_tick"] is None:
                failures.append(f"{prefix}: intact route did not collect exactly the one fixed target")
            if ablated["food_eaten"] != 0:
                failures.append(f"{prefix}: w_sd=0 control collected the target")
            if min(full["max_abs_actuator_control"], ablated["max_abs_actuator_control"]) <= 0.0:
                failures.append(f"{prefix}: test did not reach the normal MuJoCo actuator path")
    return failures


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=None,
                        help="horizon; defaults to the hardest selected world")
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 44, 77, 101])
    parser.add_argument("--version", choices=("v1", "v2", "v3", "v4"), default="v1",
                        help="strict agent profile under test")
    parser.add_argument("--world", choices=("all", *WORLD_NAMES), default="all",
                        help="deterministic food-field fixture; 'all' runs mirrored and distance-varied targets")
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
    output = args.resume or args.output or HERE / "results" / f"embodied_food_collection_causal_{timestamp}"
    if args.resume:
        manifest = output / "manifest.json"
        if not manifest.exists():
            raise SystemExit(f"--resume requires an existing manifest: {manifest}")
        prior = json.loads(manifest.read_text())
        if prior.get("experiment") != "embodied_food_collection_causal":
            raise SystemExit("--resume directory belongs to another experiment")
        if (prior.get("steps") != args.steps or prior.get("substeps") != args.substeps
                or prior.get("version") != args.version or prior.get("world") != args.world):
            raise SystemExit("--resume requires the manifest's version, world, steps, and substeps")
        if prior.get("seeds") != args.seeds or prior.get("cases") != CASES:
            raise SystemExit("--resume requires the manifest's exact seeds and causal conditions")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "manifest.json", {
            "experiment": "embodied_food_collection_causal", "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": {"active_inference": _revision(ROOT), "neuron_model": _revision(NEURON_MODEL)},
            "source_fingerprints": {
                "aif_agent3d.py": _sha256(HERE.parent / "aif_agent3d.py"),
                "embodied_food_collection_causal.py": _sha256(Path(__file__).resolve()),
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
            "scope": "mirrored and distance-varied fixed physical food odours -> PAULA food sensors/opponent turn -> normal CPG/muscle/MuJoCo path -> physical contact; excludes visual, learned value, and long-life foraging claims",
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
                print(f"world={world} seed={seed} case={name}: eaten={row['food_eaten']} sensor={row['food_sensor_spikes']} turn={row['turn_imbalance_TL_minus_TR']:+d}", flush=True)
    complete = all(
        (output / world / f"{name}_seed{seed}.json" if len(worlds) > 1 else output / f"{name}_seed{seed}.json").exists()
        for world in worlds for seed in args.seeds for name in args.cases
    )
    if not complete:
        print("Partial evidence saved; resume the remaining seed/condition files before acceptance.")
        return 0
    _write_json(output / "summary.json", records)
    failures = _accept(records, args.version)
    protocol = json.loads((output / "manifest.json").read_text()).get("acceptance_protocol", {})
    protocol_failures = list(protocol.get("failures", [])) if protocol.get("required") else []
    _write_json(output / "acceptance.json", {"passed": not failures and not protocol_failures, "failures": failures + protocol_failures,
                                                "version": args.version, "worlds": worlds,
                                                "protocol_compliant": bool(protocol.get("compliant", False)),
                                                "full_tick_traces": True, "strict_topology": True})
    print(f"Wrote embodied food-collection causal evidence to {output}")
    if failures or protocol_failures:
        print("FAIL: " + " | ".join(failures + protocol_failures), file=sys.stderr)
        return 1
    print(f"PASS: strict {args.version} food route is causally required across {len(worlds)} deterministic environments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
