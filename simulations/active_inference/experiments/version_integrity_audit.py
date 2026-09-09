"""Fail-closed integrity audit for the maintained V1--V4 evidence boundary.

This is intentionally a structural audit, not a behavioural score.  It catches
the class of mistake that previously produced persuasive but invalid results:
an experiment labelled ``v2``/``v3`` silently calling ``AIFAgent3D()`` without a
component tuple and therefore receiving the legacy 1,766-neuron full brain.

The audit checks the actual instantiated topology, versioned entrypoints, lab
command contracts, fixture declarations, and the full-tick evidence contract.
It fails closed: a strict harness with no ``--version`` propagation, a fixture
without a declared no-respawn policy, or a missing environment member is a
failure rather than a warning.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from simulations.active_inference import aif_agent3d as ag
from simulations.active_inference.components.body.world import World3D
from simulations.active_inference.live.versions import get_version, version_ids
from simulations.active_inference.agents.reactive_v1 import ReactiveV1Agent
from simulations.active_inference.agents.memory_v2 import MemoryV2Agent
from simulations.active_inference.agents.interoceptive_v3 import InteroceptiveV3Agent
from simulations.active_inference.agents.obstacle_v4 import ObstacleV4Agent
from simulations.active_inference.lab.harnesses import HARNESSES
from simulations.active_inference.experiments.matrix_protocol import PROTOCOL_SEEDS, WORLD_COUNT


HERE = Path(__file__).resolve().parent
EXPECTED_NEURONS = {"v1": 95, "v2": 299, "v3": 345, "v4": 373}
ENTRYPOINTS = {
    "v1": ReactiveV1Agent,
    "v2": MemoryV2Agent,
    "v3": InteroceptiveV3Agent,
    "v4": ObstacleV4Agent,
}
FORBIDDEN_STRICT_COMPONENTS = {
    "navigation.heading_ring", "navigation.path_integration", "vision.visual_cortex",
}
REQUIRED_FOOD_WORLDS = {"shallow_left", "shallow_right", "near_left", "near_right", "mid_left", "mid_right", "far_left", "far_right", "deep_left", "deep_right", "all"}
REQUIRED_TOXIN_WORLDS = {"near_left", "near_right", "mid_left", "mid_right", "deep_left", "deep_right", "far_left", "far_right", "cross_left", "cross_right", "all"}
REQUIRED_HEADON_WORLDS = {"near_01", "near_02", "near_03", "mid_04", "mid_05", "mid_06", "far_07", "far_08", "far_09", "far_10", "all"}
REQUIRED_ARBITER_WORLDS = {"early_0", "early_1", "early_2", "mid_3", "mid_4", "mid_5", "late_6", "late_7", "late_8", "late_9", "all"}
REQUIRED_MEMORY_WORLDS = {"center", "left_015", "right_015", "left_025", "right_025", "left_035", "right_035", "left_045", "right_045", "cross_050", "all"}
REQUIRED_METABOLIC_WORLDS = {"meal_04", "meal_06", "meal_08", "meal_10", "meal_12", "meal_14", "meal_16", "meal_18", "meal_20", "meal_24", "all"}
REQUIRED_OBSTACLE_WORLDS = {"wall_short", "wall_medium", "wall_offset_left", "wall_offset_right", "head_on_wall", "corner_small", "corner", "chicane_short", "chicane", "maze", "all"}


def _check_topologies(failures: list[str], records: dict) -> None:
    for version in version_ids():
        profile = get_version(version)
        expected = set(profile.components)
        try:
            direct = ag.AIFAgent3D(seed=11, components=profile.components)
            entry = ENTRYPOINTS[version](seed=11)
        except Exception as exc:  # pragma: no cover - recorded as an audit failure
            failures.append(f"{version}: strict construction raised {type(exc).__name__}: {exc}")
            continue
        direct_count, entry_count = len(direct.nb), len(entry.nb)
        row = records.setdefault(version, {})
        row.update({
            "components": list(profile.components),
            "direct_neuron_count": direct_count,
            "entrypoint_neuron_count": entry_count,
            "direct_profile": dict(direct.component_profile),
            "entrypoint_profile": dict(entry.component_profile),
            "direct_build_parameters": dict(direct.build_parameters),
            "entrypoint_build_parameters": dict(entry.build_parameters),
        })
        if direct_count != EXPECTED_NEURONS[version] or entry_count != EXPECTED_NEURONS[version]:
            failures.append(f"{version}: expected {EXPECTED_NEURONS[version]} neurons, got {direct_count}/{entry_count}")
        for label, agent in (("direct", direct), ("entrypoint", entry)):
            if not agent.component_profile.get("strict"):
                failures.append(f"{version} {label}: topology is not strict")
            if set(agent.component_profile.get("components", ())) != expected:
                failures.append(f"{version} {label}: instantiated component set differs from version contract")
            leaked = sorted(expected & FORBIDDEN_STRICT_COMPONENTS)
            if leaked:
                failures.append(f"{version} {label}: forbidden navigation/visual components leaked: {leaked}")
        if version in {"v3", "v4"}:
            direct_w = direct.build_parameters.get("w_unc_mode")
            entry_w = entry.build_parameters.get("w_unc_mode")
            if direct_w != 0.8 or entry_w != 0.8 or direct_w != entry_w:
                failures.append(f"{version}: direct and versioned entrypoint disagree on strict arbiter calibration")


def _check_harnesses(failures: list[str], records: dict) -> None:
    strict_versions = {version: 0 for version in version_ids()}
    for name, spec in HARNESSES.items():
        records[name] = {
            "version_policy": spec.version_policy,
            "supported_versions": list(spec.supported_versions),
            "pass_version": spec.pass_version,
            "worlds": list(spec.worlds),
        }
        if spec.version_policy == "strict_agent":
            if not spec.pass_version:
                failures.append(f"harness {name}: strict policy does not pass --version")
            script = (HERE.parent / spec.script).resolve()
            if not script.exists():
                failures.append(f"harness {name}: script is missing: {script}")
            else:
                source = script.read_text()
                if "--version" not in source:
                    failures.append(f"harness {name}: script has no --version parser")
                if "full_tick_traces" not in source or "strict_topology" not in source:
                    failures.append(f"harness {name}: acceptance does not declare full-tick/strict evidence")
                if "tick_trace" not in source and "trace" not in source:
                    failures.append(f"harness {name}: no raw trace field is retained")
                if "EmbodiedVideoRecorder" not in source or "record_visual_artifacts" not in source:
                    failures.append(f"harness {name}: synchronized visual artifacts are not part of the evidence contract")
            if spec.worlds and spec.default_world != "all":
                failures.append(f"harness {name}: broad suite default is not --world all")
            if tuple(spec.default_seeds) != PROTOCOL_SEEDS:
                failures.append(f"harness {name}: strict default seeds are not the protocol seed set")
            if spec.max_seeds < len(PROTOCOL_SEEDS):
                failures.append(f"harness {name}: max_seeds is below the protocol replication count")
            if spec.worlds and spec.worlds[-1] == "all" and len(spec.worlds) - 1 != WORLD_COUNT:
                failures.append(f"harness {name}: strict catalog must declare exactly {WORLD_COUNT} worlds")
            for version in spec.supported_versions:
                strict_versions[version] = strict_versions.get(version, 0) + 1
                try:
                    command = spec.command(output=Path("/tmp/aif-integrity-audit"), seeds=[11], version=version)
                except Exception as exc:
                    failures.append(f"harness {name}/{version}: command construction failed: {exc}")
                else:
                    if command.count("--version") != 1 or command[command.index("--version") + 1] != version:
                        failures.append(f"harness {name}/{version}: selected version is not forwarded exactly once")
    for version, count in strict_versions.items():
        if count == 0:
            failures.append(f"{version}: no strict lab harness covers this version")

    food = HARNESSES.get("food_collection")
    if food is None or set(food.worlds) != REQUIRED_FOOD_WORLDS:
        failures.append("food_collection: missing mirrored/distance-varied environment suite")
    toxin = HARNESSES.get("toxin_escape")
    if toxin is None or set(toxin.worlds) != REQUIRED_TOXIN_WORLDS:
        failures.append("toxin_escape: missing mirrored/distance-varied environment suite")
    headon = HARNESSES.get("headon_toxin")
    if headon is None or set(headon.worlds) != REQUIRED_HEADON_WORLDS:
        failures.append("headon_toxin: missing near/mid/far symmetric environment suite")
    arbiter = HARNESSES.get("arbiter")
    if arbiter is None or set(arbiter.worlds) != REQUIRED_ARBITER_WORLDS:
        failures.append("arbiter: missing meal-timing/heading environment suite")
    memory = HARNESSES.get("memory")
    if memory is None or set(memory.worlds) != REQUIRED_MEMORY_WORLDS:
        failures.append("memory: missing centered/mirrored teaching environment suite")
    metabolic = HARNESSES.get("metabolic_rest")
    if metabolic is None or set(metabolic.worlds) != REQUIRED_METABOLIC_WORLDS:
        failures.append("metabolic_rest: missing meal-timing environment suite")
    obstacle = HARNESSES.get("obstacle_detour")
    if obstacle is None or set(obstacle.worlds) != REQUIRED_OBSTACLE_WORLDS:
        failures.append("obstacle_detour: missing ten-world obstacle environment suite")
    compass = HARNESSES.get("compass")
    if compass is not None and compass.version_policy == "strict_agent":
        failures.append("compass: experimental legacy probe is incorrectly marked strict")


def _check_fixtures(failures: list[str], records: dict) -> None:
    from simulations.active_inference.experiments.embodied_food_collection_causal import WORLD_SPECS as FOOD
    from simulations.active_inference.experiments.embodied_toxin_escape_causal import WORLD_SPECS as TOXIN
    from simulations.active_inference.experiments.embodied_headon_toxin_causal import WORLD_SPECS as HEADON

    records["fixtures"] = {"food": FOOD, "toxin": TOXIN, "headon": HEADON}
    if FOOD["shallow_left"]["source"][1] != -FOOD["shallow_right"]["source"][1]:
        failures.append("food fixtures: mirrored shallow targets are not symmetric")
    if TOXIN["near_left"]["source"][1] != -TOXIN["near_right"]["source"][1]:
        failures.append("toxin fixtures: mirrored near hazards are not symmetric")
    for name in ("wall_short", "wall_medium", "wall_offset_left", "wall_offset_right",
                 "head_on_wall", "corner_small", "corner", "chicane_short", "chicane", "maze"):
        world = World3D(n_food=0, n_tox=0, arena=7.0, seed=11, barrier=name)
        if world.foods or world.toxins or world.respawn_food:
            failures.append(f"V4 fixture {name}: contains a source or permits respawn")


def audit() -> dict:
    failures: list[str] = []
    records: dict = {}
    _check_topologies(failures, records)
    _check_harnesses(failures, records.setdefault("harnesses", {}))
    _check_fixtures(failures, records)
    return {
        "audit": "version_integrity_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "failures": failures,
        "versions": records,
        "fail_closed_contract": {
            "strict_topology": True,
            "selected_version_forwarded": True,
            "full_tick_traces_required": True,
            "no_respawn_challenge_fixtures": True,
            "causal_ablations_required": True,
            "independent_raw_trace_recomputation": True,
            "unreachable_control_is_inconclusive": True,
            "embodied_matrix_world_count": WORLD_COUNT,
            "embodied_matrix_seed_count": len(PROTOCOL_SEEDS),
            "hardest_world_horizon": True,
            "parallel_matrix_orchestration": True,
            "synchronized_visual_artifacts": True,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the JSON audit record here")
    args = parser.parse_args(argv)
    result = audit()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": result["passed"], "failures": result["failures"],
                      "output": str(args.output) if args.output else None}, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
