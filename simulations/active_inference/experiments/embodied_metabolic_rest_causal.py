"""Embodied causal test for V3 metabolic interoception and PAULA SLEEP.

The altered environment turns a meal into delayed nutrition: food contact
loads the body's gut state, digestion proceeds over time, and realized motion
consumes usable energy.  The prior V2 agent is unchanged.  V3 receives only
body afferent currents and uses one PAULA WTA to arbitrate FORAGE, EXPLORE,
and SLEEP.  No Python branch selects a mode or motor output.

Controls:

* ``v3_sleep_output_ablation`` removes only SLEEP's inhibitory output to the
  motor/search route;
* ``v3_metabolic_afferent_ablation`` zeros only the V3 body afferents before
  each PAULA tick, leaving the physical body and all other circuits intact.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mujoco

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
from simulations.active_inference.agents.interoceptive_v3 import InteroceptiveV3Agent
from simulations.active_inference.agents.memory_v2 import MemoryV2Agent
from simulations.active_inference.embodied_config import DEFAULT_EMBODIED_CONFIG
from simulations.active_inference.live.versions import get_version


HERE = Path(__file__).resolve().parent
FAR = [100.0, 100.0]
WORLD_SPECS = {
    "meal_04": {"meal_body_step": 4, "label": "very early delayed meal", "completion_steps": 60},
    "meal_06": {"meal_body_step": 6, "label": "early delayed meal", "completion_steps": 60},
    "meal_08": {"meal_body_step": 8, "label": "early delayed meal 2", "completion_steps": 64},
    "meal_10": {"meal_body_step": 10, "label": "early-mid delayed meal", "completion_steps": 68},
    "meal_12": {"meal_body_step": 12, "label": "standard delayed meal", "completion_steps": 72},
    "meal_14": {"meal_body_step": 14, "label": "mid delayed meal", "completion_steps": 72},
    "meal_16": {"meal_body_step": 16, "label": "mid-late delayed meal", "completion_steps": 76},
    "meal_18": {"meal_body_step": 18, "label": "late delayed meal 1", "completion_steps": 76},
    "meal_20": {"meal_body_step": 20, "label": "late delayed meal 2", "completion_steps": 80},
    "meal_24": {"meal_body_step": 24, "label": "hardest delayed meal", "completion_steps": 88},
}
WORLD_NAMES = tuple(WORLD_SPECS)
CASES = (
    "v2_prior",
    "v3_intact",
    "v3_sleep_output_ablation",
    "v3_metabolic_afferent_ablation",
)


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _prepare_world(agent, seed: int) -> None:
    agent.world = ag.w3.World3D(seed=seed, n_food=7, n_tox=6, arena=8.0)
    agent.world.foods = [FAR.copy() for _ in agent.world.foods]
    agent.world.toxins = [FAR.copy() for _ in agent.world.toxins]
    agent.world._rp = lambda _lo, _hi: FAR.copy()
    agent.world.respawn_food = False
    agent.world._sync_mocap()
    agent.img = agent.world.retina()


def _agent_for(case: str, seed: int):
    if case == "v2_prior":
        return MemoryV2Agent(seed=seed)
    if case == "v3_sleep_output_ablation":
        cfg = DEFAULT_EMBODIED_CONFIG.with_overrides(w_sleep_veto=0.0)
        return InteroceptiveV3Agent(seed=seed, config=cfg)
    return InteroceptiveV3Agent(seed=seed)


def _ablate_metabolic_afferents(agent) -> None:
    for ids in (
        ag.met.GUT_AFFERENTS,
        ag.met.ENERGY_AFFERENTS,
        ag.met.LOW_ENERGY_AFFERENTS,
        ag.met.DIGESTION_AFFERENTS,
    ):
        for nid in ids:
            agent.net.set_external_input(nid, 0, 0.0)


def run_case(case: str, seed: int, steps: int, substeps: int, world_name: str,
             artifact_root: Path | None = None) -> dict:
    agent = _agent_for(case, seed)
    _prepare_world(agent, seed)
    agent.birth()
    recorder = EmbodiedVideoRecorder(
        agent, artifact_directory(artifact_root, case, seed),
        total_ticks=steps * substeps, substeps=substeps,
    )
    inserted = False
    trace: list[dict] = []

    def neural_ablation(current) -> None:
        if case == "v3_metabolic_afferent_ablation":
            _ablate_metabolic_afferents(current)

    def observe(current) -> None:
        nonlocal inserted
        # Insert the meal after a normal body step.  The ordinary radius-based
        # contact detector consumes it on the next physical step; the event is
        # not a host-side reward or mode command.
        meal_body_step = int(WORLD_SPECS[world_name]["meal_body_step"])
        if not inserted and current.t >= meal_body_step * substeps:
            x, y, _ = current.world.pose()
            current.world.foods[0] = [float(x), float(y)]
            current.world._sync_mocap()
            inserted = True
        def spikes(group):
            # The unchanged V2 control deliberately has no arbiter neurons;
            # missing populations are a structural ablation, not a harness
            # error.  V3 records all three strict mode groups plus SLEEP.
            return sum(int(current.nb[n].O > 0) for n in group if n in current.nb)

        trace.append({
            "neural_tick": current.t,
            "meal_inserted": inserted,
            "food_eaten": int(current.world.eaten),
            "metabolic": dict(current.world.metabolic_state()),
            "afferents": dict(current.last_metabolic_afferents),
            "mode_spikes": {
                "FORAGE": spikes(ag.ar.MODE[0]),
                "HOME": spikes(ag.ar.MODE[1]),
                "EXPLORE": spikes(ag.ar.MODE[2]),
                "SLEEP": spikes(ag.ar.SLEEP_MODE),
            },
            "actuator_abs_sum": sum(float(abs(current.world.data.ctrl[v])) for v in current.world.act_id.values()),
            "pose": {key: float(value) for key, value in zip(("x", "y", "yaw"), current.world.pose())},
        })
        recorder.observe(current)

    start = agent.world.pose()
    ag.run_episode(
        agent,
        steps=steps,
        sub=substeps,
        vision=False,
        render_every=10**9,
        render_ticks=0,
        log_every=10**9,
        tick_hook=observe,
        neural_input_hook=neural_ablation,
    )
    end = agent.world.pose()
    video_artifacts = recorder.finalize()
    return {
        "case": case,
        "version": "v2" if case == "v2_prior" else "v3",
        "world": world_name,
        "component_profile": dict(agent.component_profile),
        "neuron_count": len(agent.nb),
        "build_parameters": dict(agent.build_parameters),
        "world_fixture": {"food_count": len(agent.world.foods), "toxin_count": len(agent.world.toxins),
                           "respawn_food": bool(agent.world.respawn_food)},
        "seed": seed,
        "steps": steps,
        "substeps": substeps,
        "meal_body_step": int(WORLD_SPECS[world_name]["meal_body_step"]),
        "initial_pose": dict(zip(("x", "y", "yaw"), map(float, start))),
        "final_pose": dict(zip(("x", "y", "yaw"), map(float, end))),
        "final_metabolic": dict(agent.world.metabolic_state()),
        "trace": trace,
        "video_artifacts": video_artifacts,
    }


def summarize(record: dict) -> dict:
    trace = record["trace"]
    meal = next((i for i, row in enumerate(trace) if row["food_eaten"] > 0), None)
    if meal is None:
        raise ValueError(f"{record['case']} seed {record['seed']} did not consume the scheduled meal")
    post = trace[meal + 12:]
    return {
        "meal_trace_index": meal,
        "meal_neural_tick": trace[meal]["neural_tick"],
        "post_sleep_spikes": int(sum(row["mode_spikes"]["SLEEP"] for row in post)),
        "post_explore_spikes": int(sum(row["mode_spikes"]["EXPLORE"] for row in post)),
        "post_actuator_abs_sum": float(sum(row["actuator_abs_sum"] for row in post)),
        "final_energy": float(record["final_metabolic"]["energy_store"]),
        "final_gut_load": float(record["final_metabolic"]["gut_load"]),
        "strict_topology": bool(record["component_profile"].get("strict")),
        "component_names": list(record["component_profile"].get("components", ())),
        "neuron_count": int(record["neuron_count"]),
        "trace_ticks": len(trace),
        "expected_trace_ticks": int(record["steps"] * record["substeps"]),
        "fixture_no_respawn": record["world_fixture"].get("respawn_food") is False,
        "fixture_food_count": int(record["world_fixture"].get("food_count", -1)),
        "fixture_toxin_count": int(record["world_fixture"].get("toxin_count", -1)),
    }


def accept(summaries: dict[str, dict[str, dict[str, dict]]]) -> list[str]:
    failures: list[str] = []
    expected_components = {
        "v2_prior": set(get_version("v2").components),
        "v3_intact": set(get_version("v3").components),
        "v3_sleep_output_ablation": set(get_version("v3").components),
        "v3_metabolic_afferent_ablation": set(get_version("v3").components),
    }
    for world, by_seed in summaries.items():
      for seed, rows in by_seed.items():
        prefix = f"{world} seed {seed}"
        if any(not row["strict_topology"] or row["neuron_count"] != (299 if case == "v2_prior" else 345)
               or set(row["component_names"]) != expected_components[case]
               or row["trace_ticks"] != row["expected_trace_ticks"] or not row["fixture_no_respawn"]
               or row["fixture_food_count"] != 7 or row["fixture_toxin_count"] != 6
               for case, row in rows.items()):
            failures.append(f"{prefix}: incomplete trace or non-strict V3 topology")
        prior = rows["v2_prior"]
        intact = rows["v3_intact"]
        sleep_off = rows["v3_sleep_output_ablation"]
        aff_off = rows["v3_metabolic_afferent_ablation"]
        if intact["post_sleep_spikes"] < 30:
            failures.append(f"{prefix}: V3 did not sustain PAULA SLEEP after digestion began")
        if intact["final_energy"] <= prior["final_energy"] + 0.05:
            failures.append(f"{prefix}: V3 did not preserve more usable energy than unchanged V2")
        if intact["final_energy"] <= sleep_off["final_energy"] + 0.05:
            failures.append(f"{prefix}: SLEEP output ablation did not reduce usable energy")
        if intact["final_energy"] <= aff_off["final_energy"] + 0.05:
            failures.append(f"{prefix}: metabolic afferent ablation did not reduce usable energy")
        if sleep_off["post_actuator_abs_sum"] <= intact["post_actuator_abs_sum"] + 20.0:
            failures.append(f"{prefix}: SLEEP output ablation did not restore active motor drive")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=None,
                        help="horizon; defaults to the hardest selected world")
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 44, 77, 101])
    parser.add_argument("--version", choices=("v3",), default="v3",
                        help="strict interoceptive agent profile under test")
    parser.add_argument("--world", choices=("all", *WORLD_NAMES), default="all",
                        help="meal timing fixture; 'all' runs early, standard, and late meals")
    parser.add_argument("--protocol", action="store_true",
                        help="require the ten-world/five-seed embodied matrix contract")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    worlds = selected_worlds(WORLD_SPECS, args.world)
    args.steps = hardest_completion_steps(WORLD_SPECS, worlds) if args.steps is None else int(args.steps)
    if args.version != "v3":
        raise SystemExit("metabolic_rest is a V3-only acceptance harness")
    if args.protocol:
        try:
            require_protocol(world_specs=WORLD_SPECS, worlds=worlds, seeds=args.seeds,
                             steps=args.steps, requested_world=args.world)
        except ValueError as exc:
            raise SystemExit(str(exc))
    if args.steps <= max(spec["meal_body_step"] for spec in WORLD_SPECS.values()) + 5 or args.substeps <= 0:
        raise SystemExit("steps must leave a post-meal window and substeps must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or HERE / "results" / f"embodied_metabolic_rest_causal_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "manifest.json", {
        "experiment": "embodied_metabolic_rest_causal",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": args.seeds,
        "version": args.version,
        "components": list(get_version(args.version).components),
        "effective_build_parameters": True,
        "record_visual_artifacts": True,
        "world": args.world,
        "worlds": worlds,
        "world_catalog": WORLD_SPECS,
        "acceptance_protocol": protocol_metadata(world_specs=WORLD_SPECS, worlds=worlds,
                                                   seeds=args.seeds, steps=args.steps,
                                                   required=args.protocol, requested_world=args.world),
        "steps": args.steps,
        "substeps": args.substeps,
        "cases": CASES,
        "scope": "delayed gut digestion and energy cost -> PAULA FORAGE/EXPLORE/SLEEP -> graded muscle/MuJoCo",
        "full_tick_traces": True,
    })
    summaries: dict[str, dict[str, dict]] = {}
    for world in worlds:
        world_output = output / world if len(worlds) > 1 else output
        world_output.mkdir(parents=True, exist_ok=True)
        summaries[world] = {}
        for seed in args.seeds:
            summaries[world][str(seed)] = {}
            for case in CASES:
                record = run_case(case, seed, args.steps, args.substeps, world, artifact_root=world_output)
                _write(world_output / f"{case}_seed{seed}.json", record)
                summaries[world][str(seed)][case] = summarize(record)
                print(world, seed, case, summaries[world][str(seed)][case], flush=True)
    _write(output / "summary.json", summaries)
    failures = accept(summaries)
    protocol = json.loads((output / "manifest.json").read_text()).get("acceptance_protocol", {})
    protocol_failures = list(protocol.get("failures", [])) if protocol.get("required") else []
    _write(output / "acceptance.json", {"passed": not failures and not protocol_failures, "failures": failures + protocol_failures,
                                          "version": args.version, "worlds": worlds,
                                          "protocol_compliant": bool(protocol.get("compliant", False)),
                                          "full_tick_traces": True, "strict_topology": True})
    if failures or protocol_failures:
        print("FAIL:", " | ".join(failures + protocol_failures))
        return 1
    print("PASS: V3 metabolic afferents and PAULA SLEEP causally preserve energy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
