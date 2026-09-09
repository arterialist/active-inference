"""Causal embodied test of hunger-driven FORAGE -> EXPLORE action selection.

An initially odour-free MuJoCo world lets the neural hunger ladder select
FORAGE.  At a preregistered tick, one physical food item is delivered at the
body and consumed by the normal world-contact detector; the ordinary food US
then drains the neural hunger ladder.  With hunger low, the uncertainty drive
selects EXPLORE.  The intact circuit uses the existing ENGINE -> SEARCH curved
search primitive only under EXPLORE: a FORAGE population inhibits SEARCH, while
the EXPLORE winner releases it.  No host-side branch chooses a mode or motor
command.

The control zeros *only* the FORAGE -> SEARCH projection.  It therefore keeps
the same scheduled physical food delivery, contact, STG food event, hunger
state and spiking WTA transition, but fails to suppress search while FORAGE
wins.  This establishes that the arbiter changes the embodied motor outcome,
rather than merely changing a decoded mode label.
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
from simulations.active_inference.agents.interoceptive_v3 import InteroceptiveV3Agent


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NEURON_MODEL = ROOT.parent / "neuron-model"
FAR = [100.0, 100.0]
WORLD_SPECS = {
    "early_0": {"insert_tick": 64, "yaw_offset": -90.0, "label": "early meal / -90° heading", "completion_steps": 80},
    "early_1": {"insert_tick": 96, "yaw_offset": -45.0, "label": "early meal / -45° heading", "completion_steps": 80},
    "early_2": {"insert_tick": 128, "yaw_offset": 0.0, "label": "early meal / forward heading", "completion_steps": 80},
    "mid_3": {"insert_tick": 160, "yaw_offset": 45.0, "label": "mid meal / +45° heading", "completion_steps": 90},
    "mid_4": {"insert_tick": 192, "yaw_offset": 90.0, "label": "mid meal / +90° heading", "completion_steps": 96},
    "mid_5": {"insert_tick": 224, "yaw_offset": 135.0, "label": "mid meal / +135° heading", "completion_steps": 100},
    "late_6": {"insert_tick": 256, "yaw_offset": 180.0, "label": "late meal / 180° heading", "completion_steps": 104},
    "late_7": {"insert_tick": 320, "yaw_offset": -135.0, "label": "late meal / -135° heading", "completion_steps": 112},
    "late_8": {"insert_tick": 384, "yaw_offset": -90.0, "label": "late meal / -90° heading", "completion_steps": 120},
    "late_9": {"insert_tick": 448, "yaw_offset": 0.0, "label": "late meal / forward heading", "completion_steps": 128},
}
WORLD_NAMES = tuple(WORLD_SPECS)
CASES = {
    "mode_gated_exploration": {"build": {"w_forage_steer": -2.0}},
    "forage_search_output_ablation": {"build": {"w_forage_steer": 0.0}},
}


def _initial_yaw(seed: int, world: str = "early_2") -> float:
    """Deterministic but distinct physical orientation for each replicate."""
    return float(np.radians((seed * 137) % 360 - 180 + WORLD_SPECS[world]["yaw_offset"]))


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


def _install_odour_free_world(agent, seed: int, world: str) -> None:
    """Keep every source physically absent until the fixed food-delivery tick."""
    agent.world = ag.w3.World3D(seed=seed, n_food=7, n_tox=6, arena=8.0)
    agent.world.foods = [FAR.copy() for _ in range(7)]
    agent.world.toxins = [FAR.copy() for _ in range(6)]
    # Keep the one trial meal from respawning into the later exploratory world.
    agent.world._rp = lambda _rlo, _rhi: FAR.copy()
    agent.world.respawn_food = False
    # The world is isotropic but the MuJoCo body is not: this prevents all
    # repetitions from being the identical world-frame trajectory.
    agent.world.data.qpos[agent.world.jyaw] = _initial_yaw(seed, world)
    mujoco.mj_forward(agent.world.model, agent.world.data)
    agent.world._sync_mocap()


def run_case(seed: int, name: str, case: dict, steps: int, substeps: int, version: str, world: str,
             artifact_root: Path | None = None) -> dict:
    np.random.seed(seed)
    profile = get_version(version)
    # This is a V3-only harness.  Use the versioned entrypoint so its strict
    # mode-drive calibration and topology contract are exercised together.
    agent = InteroceptiveV3Agent(seed=seed, **case["build"])
    _install_odour_free_world(agent, seed, world)
    agent.img = agent.world.retina()
    agent.birth()
    recorder = EmbodiedVideoRecorder(
        agent, artifact_directory(artifact_root, name, seed),
        total_ticks=steps * substeps, substeps=substeps,
    )
    trace: list[dict] = []
    inserted = False

    def capture(current) -> None:
        nonlocal inserted
        # This is a fixed external world event, independent of every neural or
        # body state.  The object is then consumed by World3D's ordinary
        # physical radius check on the following MuJoCo step.
        insert_tick = int(WORLD_SPECS[world]["insert_tick"])
        if not inserted and current.t >= insert_tick:
            x, y, _ = current.world.pose()
            current.world.foods[0] = [float(x), float(y)]
            current.world._sync_mocap()
            inserted = True
        x, y, yaw = current.world.pose()
        def mode_spikes(index: int) -> int:
            # Strict V3 intentionally omits the path-integration HOME column;
            # recording it as zero makes the absent population explicit rather
            # than crashing or silently substituting a legacy full brain.
            return sum(int(current.nb[nid].O > 0) for nid in ag.ar.MODE[index] if nid in current.nb)
        trace.append({
            "neural_tick": current.t,
            "food_inserted": inserted,
            "food_eaten_total": current.world.eaten,
            "sensory_current": dict(current.last_sensor_drives),
            "hunger_spikes": sum(int(current.nb[nid].O > 0) for nid in ag.ar.HUNGER),
            "mode_spikes": {
                mode: mode_spikes(index)
                for index, mode in enumerate(ag.MODES)
            },
            "search_spike": int(current.nb[ag.SEARCH].O > 0),
            # Keep the separate hazard gate in the trace: it must not become
            # an accidental substitute for the arbiter's search channel.
            "hazard_steer_spike": int(current.nb[ag.STEER].O > 0),
            "turn_spikes": {"TL": int(current.nb[ag.TL].O > 0), "TR": int(current.nb[ag.TR].O > 0)},
            "muscle_state": {
                "MLp": float(current.nb[ag.MLp].S), "MLr": float(current.nb[ag.MLr].S),
                "MRp": float(current.nb[ag.MRp].S), "MRr": float(current.nb[ag.MRr].S),
            },
            "actuator_ctrl": {
                key: float(current.world.data.ctrl[value]) for key, value in current.world.act_id.items()
            },
            "pose": {"x": float(x), "y": float(y), "yaw": float(yaw)},
        })
        recorder.observe(current)

    start = agent.world.pose()
    _, modes = ag.run_episode(
        agent, steps=steps, sub=substeps, vision=False, render_every=10**9,
        render_ticks=0, log_every=10**9, tick_hook=capture,
    )
    end = agent.world.pose()
    video_artifacts = recorder.finalize()
    return {
        "condition": name,
        "seed": seed,
        "version": profile.id,
        "world": world,
        "component_profile": dict(agent.component_profile),
        "neuron_count": len(agent.nb),
        "build_parameters": dict(agent.build_parameters),
        "build": case["build"],
        "vision_enabled": False,
        "scenario": {
            "food_before_insert": "absent (all sources at a far coordinate)",
            "scheduled_food_insert_neural_tick": int(WORLD_SPECS[world]["insert_tick"]),
            "world_label": WORLD_SPECS[world]["label"],
            "post_contact_respawn": "far coordinate",
        },
        "steps": steps,
        "neural_substeps_per_body_step": substeps,
        "trace_ticks": len(trace),
        "expected_trace_ticks": int(steps * substeps),
        "world_fixture": {
            "food_count": len(agent.world.foods),
            "toxin_count": len(agent.world.toxins),
            "respawn_food": bool(agent.world.respawn_food),
            "all_sources_far_before_scheduled_insert": True,
        },
        "initial_pose": {"x": float(start[0]), "y": float(start[1]), "yaw": float(start[2])},
        "final_pose": {"x": float(end[0]), "y": float(end[1]), "yaw": float(end[2])},
        "mode_steps": modes,
        "tick_trace": trace,
        "video_artifacts": video_artifacts,
    }


def _yaw_delta(rows: list[dict]) -> float:
    yaw = np.unwrap(np.asarray([row["pose"]["yaw"] for row in rows], dtype=float))
    return float(np.degrees(yaw[-1] - yaw[0])) if len(yaw) > 1 else 0.0


def _actuator_asymmetry(rows: list[dict]) -> float:
    """Total physical left/right paddle-force imbalance over one window."""
    return float(sum(abs(
        row["actuator_ctrl"]["plp"] + row["actuator_ctrl"]["plr"]
        - row["actuator_ctrl"]["prp"] - row["actuator_ctrl"]["prr"]
    ) for row in rows))


def _lateral_displacement(rows: list[dict], initial_yaw: float) -> float:
    """Endpoint excursion on the body's initial local left/right axis."""
    delta = np.asarray([
        rows[-1]["pose"]["x"] - rows[0]["pose"]["x"],
        rows[-1]["pose"]["y"] - rows[0]["pose"]["y"],
    ])
    forward = initial_yaw + np.pi  # World3D hull convention: forward is -x.
    left = np.asarray([-np.sin(forward), np.cos(forward)])
    return float(np.dot(delta, left))


def _summary(record: dict) -> dict:
    trace = record["tick_trace"]
    first_food = next((i for i, row in enumerate(trace) if row["food_eaten_total"] > 0), None)
    if first_food is None:
        raise ValueError("scheduled physical food was not consumed")
    # Avoid the source's brief contact/US transient.  Before it, the field is
    # absent, so any search suppression can only be the FORAGE neural state.
    pre = trace[40:max(40, first_food - 20)]
    # The food-drain pulse lasts 40 body steps x 8 neural ticks.  The later
    # window measures the stable satiety/exploration state after that pulse.
    post = trace[min(len(trace), first_food + 160):]
    if not pre or not post:
        raise ValueError("trial horizon does not contain pre- and post-meal windows")
    initial_yaw = float(record["initial_pose"]["yaw"])

    def spikes(rows: list[dict], key: str) -> int:
        return int(sum(row["mode_spikes"][key] for row in rows))

    return {
        "world": record["world"],
        "first_food_tick": trace[first_food]["neural_tick"],
        "pre": {
            "hunger_mean": float(np.mean([row["hunger_spikes"] for row in pre])),
            "mode_spikes": {mode: spikes(pre, mode) for mode in ag.MODES},
            "search_spikes": int(sum(row["search_spike"] for row in pre)),
            "hazard_steer_spikes": int(sum(row["hazard_steer_spike"] for row in pre)),
            "net_yaw_degrees": _yaw_delta(pre),
            "actuator_left_right_asymmetry": _actuator_asymmetry(pre),
            "lateral_displacement": _lateral_displacement(pre, initial_yaw),
            "max_abs_actuator_control": float(max(abs(value) for row in pre for value in row["actuator_ctrl"].values())),
        },
        "post": {
            "hunger_mean": float(np.mean([row["hunger_spikes"] for row in post])),
            "mode_spikes": {mode: spikes(post, mode) for mode in ag.MODES},
            "search_spikes": int(sum(row["search_spike"] for row in post)),
            "hazard_steer_spikes": int(sum(row["hazard_steer_spike"] for row in post)),
            "net_yaw_degrees": _yaw_delta(post),
            "actuator_left_right_asymmetry": _actuator_asymmetry(post),
            "lateral_displacement": _lateral_displacement(post, initial_yaw),
            "max_abs_actuator_control": float(max(abs(value) for row in post for value in row["actuator_ctrl"].values())),
        },
        "trace_ticks": int(record["trace_ticks"]),
        "expected_trace_ticks": int(record["expected_trace_ticks"]),
        "strict_topology": bool(record["component_profile"].get("strict")),
        "component_names": list(record["component_profile"].get("components", ())),
        "neuron_count": int(record["neuron_count"]),
        "fixture_no_respawn": record["world_fixture"].get("respawn_food") is False,
        "fixture_food_count": int(record["world_fixture"].get("food_count", -1)),
        "fixture_toxin_count": int(record["world_fixture"].get("toxin_count", -1)),
    }


def _accept(summaries: dict[str, dict[str, dict[str, dict]]]) -> list[str]:
    failures: list[str] = []
    expected_components = set(get_version("v3").components)
    for world, by_seed in summaries.items():
        for seed, rows in by_seed.items():
            intact, ablated = rows["mode_gated_exploration"], rows["forage_search_output_ablation"]
            for label, row in rows.items():
                pre, post = row["pre"], row["post"]
                if (not row["strict_topology"] or row["neuron_count"] != 345
                        or set(row["component_names"]) != expected_components
                        or row["trace_ticks"] != row["expected_trace_ticks"]):
                    failures.append(
                        f"{world} seed {seed} {label}: incomplete trace or non-strict V3 topology "
                        f"({row['trace_ticks']}/{row['expected_trace_ticks']}, {row['neuron_count']} neurons)"
                    )
                if (not row["fixture_no_respawn"] or row["fixture_food_count"] != 7
                        or row["fixture_toxin_count"] != 6):
                    failures.append(f"{world} seed {seed} {label}: arbiter fixture was changed or permits respawn")
                if pre["hunger_mean"] <= 1.0 or post["hunger_mean"] >= 0.25:
                    failures.append(f"{world} seed {seed} {label}: physical meal did not switch the neural hunger state")
                if pre["mode_spikes"]["FORAGE"] <= pre["mode_spikes"]["EXPLORE"]:
                    failures.append(f"{world} seed {seed} {label}: FORAGE did not win before the meal")
                if post["mode_spikes"]["EXPLORE"] <= post["mode_spikes"]["FORAGE"]:
                    failures.append(f"{world} seed {seed} {label}: EXPLORE did not win after the meal")
                if min(pre["max_abs_actuator_control"], post["max_abs_actuator_control"]) <= 0.0:
                    failures.append(f"{world} seed {seed} {label}: mode trial bypassed the normal actuator path")
            # The ablation must restore the search population and the physical
            # paddle-force asymmetry.  Endpoint lateral displacement remains a
            # diagnostic because short stroke windows can cancel yaw even when
            # the realized actuator forces differ.
            if intact["pre"]["search_spikes"] > 6:
                failures.append(f"{world} seed {seed}: FORAGE did not suppress the intact search population")
            if ablated["pre"]["search_spikes"] <= intact["pre"]["search_spikes"] + 12:
                failures.append(f"{world} seed {seed}: output ablation did not restore FORAGE-period search spikes")
            if ablated["pre"]["actuator_left_right_asymmetry"] <= intact["pre"]["actuator_left_right_asymmetry"] + 30.0:
                failures.append(f"{world} seed {seed}: output ablation did not restore physical paddle-force asymmetry")
            if intact["post"]["search_spikes"] <= 40:
                failures.append(f"{world} seed {seed}: EXPLORE did not release the intact search population")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=None,
                        help="horizon; defaults to the hardest selected world")
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 44, 77, 101])
    parser.add_argument("--version", choices=("v3",), default="v3",
                        help="strict interoceptive agent profile under test")
    parser.add_argument("--world", choices=("all", *WORLD_NAMES), default="all",
                        help="meal timing/heading fixture; 'all' runs early, standard, and late meals")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--protocol", action="store_true",
                        help="require the ten-world/five-seed embodied matrix contract")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-cases", type=int, help="run at most this many unfinished seed/condition pairs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    worlds = selected_worlds(WORLD_SPECS, args.world)
    args.steps = hardest_completion_steps(WORLD_SPECS, worlds) if args.steps is None else int(args.steps)
    if args.steps <= 0 or args.substeps <= 0:
        raise SystemExit("--steps and --substeps must be positive")
    if args.protocol:
        try:
            require_protocol(world_specs=WORLD_SPECS, worlds=worlds, seeds=args.seeds,
                             steps=args.steps, requested_world=args.world)
        except ValueError as exc:
            raise SystemExit(str(exc))
    if args.max_cases is not None and args.max_cases <= 0:
        raise SystemExit("--max-cases must be positive")
    if args.output and args.resume:
        raise SystemExit("use either --output or --resume, not both")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.resume or args.output or HERE / "results" / f"embodied_arbiter_explore_causal_{stamp}"
    if args.resume:
        manifest_path = output / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"--resume requires an existing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("experiment") != "embodied_arbiter_explore_causal":
            raise SystemExit("--resume directory belongs to another experiment")
        if (manifest.get("steps") != args.steps or manifest.get("substeps") != args.substeps
                or manifest.get("world") != args.world):
            raise SystemExit("--resume requires the manifest's --steps and --substeps")
        if manifest.get("seeds") != args.seeds or manifest.get("version") != args.version:
            raise SystemExit("--resume requires the manifest's exact --seeds set and order")
        if manifest.get("cases") != CASES:
            raise SystemExit("--resume requires the manifest's complete causal condition set")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "manifest.json", {
            "experiment": "embodied_arbiter_explore_causal",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": {"active_inference": _revision(ROOT), "neuron_model": _revision(NEURON_MODEL)},
            "source_fingerprints": {
                "aif_agent3d.py": _sha256(HERE.parent / "aif_agent3d.py"),
                "aif_arbiter.py": _sha256(HERE.parent / "aif_arbiter.py"),
                "embodied_config.py": _sha256(HERE.parent / "embodied_config.py"),
                "embodied_arbiter_explore_causal.py": _sha256(Path(__file__).resolve()),
            },
            "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "mujoco": mujoco.__version__},
            "seeds": args.seeds, "steps": args.steps, "substeps": args.substeps,
            "version": args.version, "components": list(get_version(args.version).components),
            "effective_build_parameters": True,
            "record_visual_artifacts": True,
            "world": args.world,
            "worlds": worlds,
            "world_catalog": WORLD_SPECS,
            "acceptance_protocol": protocol_metadata(world_specs=WORLD_SPECS, worlds=worlds,
                                                       seeds=args.seeds, steps=args.steps,
                                                       required=args.protocol, requested_world=args.world),
            "initial_yaw_degrees_by_seed": {str(seed): {world: float(np.degrees(_initial_yaw(seed, world))) for world in WORLD_NAMES} for seed in args.seeds},
            "conditions": CASES,
            "scope": "odour-free physical world -> neural hunger/FORAGE -> scheduled physical food contact -> hunger drain/EXPLORE -> PAULA SEARCH/relay/muscle/MuJoCo; hazard STEER remains a separate PAULA escape route; excludes path integration, HOME, vision, and broad active-inference claims",
            "primary_evidence": "every per-condition-per-seed JSON retains neural and physical tick traces",
        })

    attempts = 0
    summaries = {}
    for world in worlds:
        world_output = output / world if len(worlds) > 1 else output
        world_output.mkdir(parents=True, exist_ok=True)
        summaries[world] = {}
        for seed in args.seeds:
            summaries[world][str(seed)] = {}
            for name in args.cases:
                path = world_output / f"{name}_seed{seed}.json"
                if path.exists():
                    record = json.loads(path.read_text())
                else:
                    if args.max_cases is not None and attempts >= args.max_cases:
                        break
                    record = run_case(seed, name, CASES[name], args.steps, args.substeps, args.version, world,
                                      artifact_root=world_output)
                    _write_json(path, record)
                    attempts += 1
                row = _summary(record)
                summaries[world][str(seed)][name] = row
                print(
                    f"world={world} seed={seed} {name}: pre F/E={row['pre']['mode_spikes']['FORAGE']}/{row['pre']['mode_spikes']['EXPLORE']} "
                    f"SEARCH={row['pre']['search_spikes']} yaw={row['pre']['net_yaw_degrees']:+.1f}; "
                    f"post F/E={row['post']['mode_spikes']['FORAGE']}/{row['post']['mode_spikes']['EXPLORE']} "
                    f"SEARCH={row['post']['search_spikes']}", flush=True,
                )
            if args.max_cases is not None and attempts >= args.max_cases:
                break
        if args.max_cases is not None and attempts >= args.max_cases:
            break

    complete = all(
        (output / world / f"{name}_seed{seed}.json" if len(worlds) > 1 else output / f"{name}_seed{seed}.json").exists()
        for world in worlds for seed in args.seeds for name in CASES
    )
    if not complete:
        print("INCOMPLETE: resume to run missing cases")
        return 0
    _write_json(output / "summary.json", summaries)
    failures = _accept(summaries)
    protocol = json.loads((output / "manifest.json").read_text()).get("acceptance_protocol", {})
    protocol_failures = list(protocol.get("failures", [])) if protocol.get("required") else []
    _write_json(output / "acceptance.json", {"passed": not failures and not protocol_failures, "failures": failures + protocol_failures,
                                                "version": args.version, "worlds": worlds,
                                                "protocol_compliant": bool(protocol.get("compliant", False)),
                                                "full_tick_traces": True, "strict_topology": True})
    print(f"Wrote arbiter causal evidence to {output}")
    if failures or protocol_failures:
        print("FAIL: " + " | ".join(failures + protocol_failures), file=sys.stderr)
        return 1
    print("PASS: neural hunger selection causally gates exploratory body search through the PAULA arbiter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
