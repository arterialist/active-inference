"""Full-tick causal acceptance test for the constraint-driven V4 obstacle delta.

The V4 catalyst is deliberately deterministic: the harness can run a
full-width physical wall, an L-corner, an alternating chicane, or a compact
maze, and every fixture has no food at all.  Unchanged V3 therefore has no way
to turn the challenge into a food-search problem or to ``solve`` it by
stumbling onto a random respawn.
V4 adds only the bilateral body afferent and the PAULA obstacle reflex.  Every
neural tick records the physical distance/current, obstacle populations,
descending commands, actuator state, contact state, and pose.

The acceptance question is correspondingly narrow: does V4 approach the
constraint, detect it before contact, and deflect laterally without collision;
do the two independent ablations restore the prior collision/stall?  This is
an embodied causal test of the V4 delta, not a claim that this small reflex is
a general-purpose navigation planner.  ``--world all`` runs the required wall
and corner cases plus the chicane/maze route-depth diagnostics.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

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
V3 = get_version("v3").components
V4 = get_version("v4").components
CASES = (
    "v3_prior",
    "v4_intact",
    "v4_body_afferent_ablation",
    "v4_reflex_ablation",
)
WORLD_SPECS = {
    "wall_short": {"label": "short wall", "tier": "required", "description": "small first wall", "required_segments": 0, "completion_steps": 180},
    "wall_medium": {"label": "medium wall", "tier": "required", "description": "wider first wall", "required_segments": 0, "completion_steps": 200},
    "wall_offset_left": {"label": "left-offset wall", "tier": "required", "description": "wall with asymmetric left extent", "required_segments": 0, "completion_steps": 210},
    "wall_offset_right": {"label": "right-offset wall", "tier": "required", "description": "wall with asymmetric right extent", "required_segments": 0, "completion_steps": 210},
    "head_on_wall": {
        "label": "full-width wall", "tier": "required",
        "description": "single deterministic wall; validates the primitive contact-free reflex",
        "required_segments": 0, "completion_steps": 240,
    },
    "corner_small": {"label": "small L-corner", "tier": "required", "description": "two-segment small corner", "required_segments": 1, "completion_steps": 260},
    "corner": {
        "label": "L-corner",
        "tier": "required",
        "description": "two-segment corner; tests a compound asymmetric obstacle",
        "required_segments": 0,
        "completion_steps": 280,
    },
    "chicane_short": {"label": "short chicane", "tier": "required", "description": "two alternating bars", "required_segments": 1, "completion_steps": 300},
    "chicane": {
        "label": "alternating chicane",
        "tier": "required",
        "description": "three alternating bars; tests repeated bilateral encounters",
        "required_segments": 2,
        "completion_steps": 340,
    },
    "maze": {
        "label": "compact maze",
        "tier": "required",
        "description": "four alternating bars plus a cross-cap; tests route-depth limits",
        "required_segments": 4,
        "completion_steps": 400,
    },
}
WORLD_NAMES = tuple(WORLD_SPECS)


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _world(seed: int, world: str):
    return ag.w3.World3D(n_food=0, n_tox=0, arena=7.0, seed=seed, barrier=world)


def _make(case: str, seed: int):
    if case == "v3_prior":
        return ag.AIFAgent3D(seed=seed, components=V3)
    if case == "v4_reflex_ablation":
        return ag.AIFAgent3D(
            seed=seed,
            components=V4,
            obstacle_turn_gain=0.0,
            obstacle_onset_turn_gain=0.0,
            obstacle_brake_gain=0.0,
            obstacle_wall_gain=0.0,
        )
    return ag.AIFAgent3D(seed=seed, components=V4)


def _zero_obstacle_input(current) -> None:
    for nid in ag.OBL + ag.OBR:
        current.net.set_external_input(nid, 0, 0.0)


def _json_distance(values: dict, key: str):
    """Return finite distance metadata without emitting JSON ``Infinity``."""
    try:
        value = float(values.get(key, float("inf")))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def run_case(case: str, seed: int, steps: int, substeps: int, world: str,
             artifact_root: Path | None = None) -> dict:
    agent = _make(case, seed)
    agent.world = _world(seed, world)
    agent.img = agent.world.retina()
    agent.birth()
    recorder = EmbodiedVideoRecorder(
        agent, artifact_directory(artifact_root, case, seed),
        total_ticks=steps * substeps, substeps=substeps,
    )
    trace: list[dict] = []

    def neural_ablation(current) -> None:
        if case == "v4_body_afferent_ablation":
            _zero_obstacle_input(current)

    def capture(current) -> None:
        x, y, yaw = current.world.pose()
        obs = dict(current.last_obstacle_afferents)
        # ``last_obstacle_afferents`` is the physical transducer sample.  The
        # body-ablation hook zeros the PAULA input ports after this sample is
        # taken, so retaining both the physical sample and the downstream
        # spike raster makes the ablation auditable rather than inferential.
        trace.append({
            "neural_tick": int(current.t),
            "obstacle_afferents": {
                "left": float(obs.get("left", 0.0)),
                "right": float(obs.get("right", 0.0)),
                "left_onset": float(obs.get("left_onset", 0.0)),
                "right_onset": float(obs.get("right_onset", 0.0)),
                "distance_left": _json_distance(obs, "distance_left"),
                "distance_right": _json_distance(obs, "distance_right"),
                "contact": float(obs.get("contact", 0.0)),
            },
            "obstacle_spikes": {
                "left": int(sum(current.nb[n].O > 0 for n in ag.OBL if n in current.nb)),
                "right": int(sum(current.nb[n].O > 0 for n in ag.OBR if n in current.nb)),
                "left_onset": int(sum(current.nb[n].O > 0 for n in ag.OBDL if n in current.nb)),
                "right_onset": int(sum(current.nb[n].O > 0 for n in ag.OBDR if n in current.nb)),
            },
            "reflex_spikes": {
                "left": int(current.nb[ag.OBS_LEFT].O > 0) if ag.OBS_LEFT in current.nb else 0,
                "right": int(current.nb[ag.OBS_RIGHT].O > 0) if ag.OBS_RIGHT in current.nb else 0,
                "brake": int(current.nb[ag.OBS_BRAKE].O > 0) if ag.OBS_BRAKE in current.nb else 0,
                "wall": int(current.nb[ag.OBS_WALL].O > 0) if ag.OBS_WALL in current.nb else 0,
            },
            "turn_spikes": {"TL": int(current.nb[ag.TL].O > 0), "TR": int(current.nb[ag.TR].O > 0)},
            "actuator_abs_sum": float(sum(abs(current.world.data.ctrl[v]) for v in current.world.act_id.values())),
            "pose": {"x": float(x), "y": float(y), "yaw": float(yaw)},
            "contact_count": int(current.world.data.ncon),
            "food_eaten": int(current.world.eaten),
        })
        recorder.observe(current)

    start = agent.world.pose()
    ag.run_episode(
        agent, steps=steps, sub=substeps, vision=False,
        render_every=10**9, render_ticks=0, log_every=10**9,
        tick_hook=capture, neural_input_hook=neural_ablation,
    )
    end = agent.world.pose()
    video_artifacts = recorder.finalize()
    return {
        "case": case, "seed": seed, "version": "v3" if case == "v3_prior" else "v4",
        "world": world, "steps": steps, "substeps": substeps,
        "component_profile": dict(agent.component_profile), "neuron_count": len(agent.nb),
        "build_parameters": dict(agent.build_parameters),
        "initial_pose": dict(zip(("x", "y", "yaw"), map(float, start))),
        "final_pose": dict(zip(("x", "y", "yaw"), map(float, end))),
        "food_eaten": int(agent.world.eaten), "toxin_hits": int(agent.world.tox_hits),
        "video_artifacts": video_artifacts,
        "world_fixture": {
            "food_count": len(agent.world.foods),
            "toxin_count": len(agent.world.toxins),
            "respawn_food": bool(agent.world.respawn_food),
            "barrier": world,
        },
        "barriers": [dict(item) for item in agent.world.barriers],
        "trace": trace,
    }


def summarize(record: dict) -> dict:
    trace = record["trace"]
    initial = record["initial_pose"]
    poses = [r["pose"] for r in trace]
    distances = [
        d for r in trace for d in (
            r["obstacle_afferents"].get("distance_left"),
            r["obstacle_afferents"].get("distance_right"),
        ) if d is not None
    ]
    contact_ticks = [r["neural_tick"] for r in trace if r["contact_count"] > 0]
    obstacle_ticks = [
        r["neural_tick"] for r in trace
        if r["obstacle_spikes"]["left"] + r["obstacle_spikes"]["right"] > 0
    ]
    reflex_ticks = [
        r["neural_tick"] for r in trace
        if sum(r["reflex_spikes"].values()) > 0
    ]
    xs = [float(p["x"]) for p in poses]
    ys = [float(p["y"]) for p in poses]
    minimum_x = min(xs, default=initial["x"])
    barriers = record.get("barriers", [])
    reached_segments = sum(
        1 for barrier in barriers
        if minimum_x <= float(barrier.get("x", 0.0)) + 0.25
    )
    return {
        "world": record.get("world", "head_on_wall"),
        "barrier_count": int(len(barriers)),
        "food_eaten": record["food_eaten"],  # sanity: this fixture intentionally contains no food
        "max_contact_count": int(max((r["contact_count"] for r in trace), default=0)),
        "contact_ticks": int(len(contact_ticks)),
        "first_contact_tick": min(contact_ticks) if contact_ticks else None,
        "obstacle_onset_tick": min(obstacle_ticks) if obstacle_ticks else None,
        "reflex_onset_tick": min(reflex_ticks) if reflex_ticks else None,
        "obstacle_sensor_spikes": int(sum(r["obstacle_spikes"]["left"] + r["obstacle_spikes"]["right"] for r in trace)),
        "obstacle_onset_spikes": int(sum(r["obstacle_spikes"]["left_onset"] + r["obstacle_spikes"]["right_onset"] for r in trace)),
        "reflex_spikes": int(sum(sum(r["reflex_spikes"].values()) for r in trace)),
        "turn_imbalance_TL_minus_TR": int(sum(r["turn_spikes"]["TL"] - r["turn_spikes"]["TR"] for r in trace)),
        # Forward is -x in this body.  This measures approach to the wall, not
        # eventual target attainment.  Lateral deflection is the direct body
        # consequence the V4 reflex is authorized to produce.
        "forward_progress": float(max(0.0, initial["x"] - minimum_x)),
        "minimum_x": float(minimum_x),
        "reached_segments": int(reached_segments),
        "lateral_deflection": float(max((abs(y - initial["y"]) for y in ys), default=0.0)),
        "path_length": float(sum(np.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) for i in range(1, len(xs)))),
        "minimum_whisker_distance": float(min(distances)) if distances else None,
        "actuator_abs_sum": float(sum(r["actuator_abs_sum"] for r in trace)),
        "full_tick_trace": True,
        "trace_ticks": len(trace),
        "expected_trace_ticks": int(record["steps"] * record["substeps"]),
        "strict_topology": bool(record["component_profile"].get("strict")),
        "component_names": list(record["component_profile"].get("components", ())),
        "neuron_count": int(record["neuron_count"]),
        "fixture_no_respawn": record["world_fixture"].get("respawn_food") is False,
        "fixture_food_count": int(record["world_fixture"].get("food_count", -1)),
        "fixture_toxin_count": int(record["world_fixture"].get("toxin_count", -1)),
    }


def accept(summaries: dict[str, dict[str, dict]], world: str = "head_on_wall") -> list[str]:
    """Return failures for the safety contract of one geometry.

    The primitive and corner are required V4 capabilities.  Chicane/maze are
    intentionally diagnostic: their safety failures are reported, while route
    depth is surfaced as a limitation instead of being hidden by a binary
    food-eaten result.
    """
    failures: list[str] = []
    expected_components = {
        "v3_prior": set(V3),
        "v4_intact": set(V4),
        "v4_body_afferent_ablation": set(V4),
        "v4_reflex_ablation": set(V4),
    }
    for seed, rows in summaries.items():
        prior, intact = rows["v3_prior"], rows["v4_intact"]
        body_off, reflex_off = rows["v4_body_afferent_ablation"], rows["v4_reflex_ablation"]
        if any(row["food_eaten"] != 0 for row in rows.values()):
            failures.append(f"seed {seed}: deterministic {world} fixture unexpectedly contains/eats food")
        for label, row in rows.items():
            expected_neurons = 345 if label == "v3_prior" else 373
            if (not row["strict_topology"] or row["neuron_count"] != expected_neurons
                    or set(row["component_names"]) != expected_components[label]):
                failures.append(f"seed {seed}: {label} is not the declared strict topology ({row['neuron_count']} neurons)")
            if row["trace_ticks"] != row["expected_trace_ticks"]:
                failures.append(f"seed {seed}: {label} has an incomplete tick trace")
            if (not row["fixture_no_respawn"] or row["fixture_food_count"] != 0
                    or row["fixture_toxin_count"] != 0):
                failures.append(f"seed {seed}: {label} obstacle fixture contains mutable food/toxin sources")
        if prior["max_contact_count"] < 1:
            failures.append(f"seed {seed}: unchanged V3 did not collide/stall in {world}")
        if intact["max_contact_count"] != 0:
            failures.append(f"seed {seed}: intact V4 contacted {world}")
        if intact["forward_progress"] < 0.6:
            failures.append(f"seed {seed}: intact V4 never approached the physical constraint")
        if intact["lateral_deflection"] < 0.25:
            failures.append(f"seed {seed}: intact V4 did not produce lateral deflection")
        if intact["obstacle_sensor_spikes"] < 20 or intact["reflex_spikes"] < 10:
            failures.append(f"seed {seed}: V4 causal sensor/reflex path was silent")
        if body_off["max_contact_count"] < 1:
            failures.append(f"seed {seed}: body-afferent ablation still avoided the wall")
        if reflex_off["max_contact_count"] < 1:
            failures.append(f"seed {seed}: reflex ablation still avoided {world}")
        if intact["actuator_abs_sum"] <= body_off["actuator_abs_sum"] * 0.05:
            failures.append(f"seed {seed}: body ablation did not preserve the ordinary motor substrate")
    return failures


def route_limitations(summaries: dict[str, dict[str, dict]], world: str) -> list[str]:
    """Describe the deliberate higher-complexity route-depth probe."""
    required = int(WORLD_SPECS[world]["required_segments"])
    if not required:
        return []
    limitations: list[str] = []
    for seed, rows in summaries.items():
        reached = rows["v4_intact"]["reached_segments"]
        if reached < required:
            limitations.append(
                f"seed {seed}: intact V4 reached {reached}/{required} expected {world} segments"
            )
    return limitations


def run_world(world: str, seeds: list[int], steps: int, substeps: int, output: Path) -> tuple[dict, dict]:
    summaries: dict[str, dict[str, dict]] = {}
    output.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        summaries[str(seed)] = {}
        for case in CASES:
            record = run_case(case, seed, steps, substeps, world, artifact_root=output)
            _write(output / f"{case}_seed{seed}.json", record)
            summaries[str(seed)][case] = summarize(record)
            print(world, seed, case, summaries[str(seed)][case], flush=True)
    failures = accept(summaries, world)
    limitations = route_limitations(summaries, world)
    result = {
        "world": world,
        "label": WORLD_SPECS[world]["label"],
        "tier": WORLD_SPECS[world]["tier"],
        "passed": not failures,
        "failures": failures,
        "route_limitations": limitations,
    }
    _write(output / "summary.json", summaries)
    _write(output / "acceptance.json", result)
    return summaries, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=None,
                        help="horizon; defaults to the hardest selected world")
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 44, 77, 101])
    parser.add_argument("--version", choices=("v4",), default="v4",
                        help="strict V4 delta under test; V3 is retained only as the unchanged prior")
    parser.add_argument("--world", choices=("all", *WORLD_NAMES), default="all",
                        help="geometry fixture to run; 'all' runs the ten-world suite")
    parser.add_argument("--protocol", action="store_true",
                        help="require the ten-world/five-seed embodied matrix contract")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    worlds = selected_worlds(WORLD_SPECS, args.world)
    args.steps = hardest_completion_steps(WORLD_SPECS, worlds) if args.steps is None else int(args.steps)
    if args.version != "v4":
        raise SystemExit("obstacle_detour is a V4-only acceptance harness")
    if args.protocol:
        try:
            require_protocol(world_specs=WORLD_SPECS, worlds=worlds, seeds=args.seeds,
                             steps=args.steps, requested_world=args.world)
        except ValueError as exc:
            raise SystemExit(str(exc))
    if args.steps <= 0 or args.substeps <= 0:
        raise SystemExit("steps and substeps must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or HERE / "results" / f"embodied_obstacle_detour_causal_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "manifest.json", {
        "experiment": "embodied_obstacle_detour_causal", "created_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": args.seeds, "steps": args.steps, "substeps": args.substeps,
        "version": args.version, "components": list(V4), "prior_components": list(V3),
        "effective_build_parameters": True,
        "record_visual_artifacts": True,
        "cases": CASES, "world": args.world, "worlds": worlds,
        "world_catalog": WORLD_SPECS,
        "acceptance_protocol": protocol_metadata(world_specs=WORLD_SPECS, worlds=worlds,
                                                   seeds=args.seeds, steps=args.steps,
                                                   required=args.protocol, requested_world=args.world),
        "constraint": "deterministic physical geometry; no food source",
        "full_tick_traces": True,
        "causal_chain": "geometry -> bilateral whisker distances/currents -> obstacle PAULA populations -> TL/TR/relays -> muscles -> MuJoCo contact/deflection/progress",
    })
    results: dict[str, dict] = {}
    for world in worlds:
        world_output = output / world if len(worlds) > 1 else output
        _summaries, result = run_world(world, args.seeds, args.steps, args.substeps, world_output)
        results[world] = result
    required_failures = [
        f"{world}: {failure}"
        for world, result in results.items()
        if result["tier"] == "required"
        for failure in result["failures"]
    ]
    diagnostic_safety_failures = [
        f"{world}: {failure}"
        for world, result in results.items()
        if result["tier"] == "diagnostic"
        for failure in result["failures"]
    ]
    diagnostic_limitations = [
        f"{world}: {limitation}"
        for world, result in results.items()
        for limitation in result["route_limitations"]
    ]
    protocol = json.loads((output / "manifest.json").read_text()).get("acceptance_protocol", {})
    protocol_failures = list(protocol.get("failures", [])) if protocol.get("required") else []
    suite = {
        # Route-depth limitations are informative for the diagnostic worlds;
        # a safety/causal failure is still a real failure and must make the
        # run red, including when a diagnostic world is selected alone.
        "passed": not required_failures and not diagnostic_safety_failures,
        "failures": required_failures + diagnostic_safety_failures + protocol_failures,
        "protocol_compliant": bool(protocol.get("compliant", False)),
        "diagnostic_limitations": diagnostic_limitations,
        "worlds": results,
    }
    _write(output / "acceptance.json", suite)
    if required_failures or diagnostic_safety_failures or protocol_failures:
        print("FAIL:", " | ".join(required_failures + diagnostic_safety_failures + protocol_failures))
        return 1
    if diagnostic_limitations:
        print("PASS: required V4 geometries; diagnostic route limits:", " | ".join(diagnostic_limitations))
    else:
        print("PASS: V4 geometry suite is causally necessary and embodied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
