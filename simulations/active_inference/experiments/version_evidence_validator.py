"""Independent fail-closed validator for a V1--V4 evidence directory.

The experiment process writes the behavioural acceptance result.  This module
is a second, intentionally boring reader: it checks that the result is backed
by the declared records, that every requested seed/world/condition exists,
that the raw traces have the advertised length and contiguous neural clock,
and that the instantiated PAULA topology is the selected strict profile.  It
does not turn a behavioural failure into a pass.  A directory is only valid
when its evidence is complete *and* its own acceptance record passes.

This catches endpoint-only reports, stale ``--resume`` directories, silently
substituted full brains, missing ablations, truncated traces, and mutated
fixtures.  The matrix runner also records SHA-256 digests of every validated
file so a later edit is detectable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from simulations.active_inference.live.versions import get_version
from simulations.active_inference.experiments.matrix_protocol import (
    PROTOCOL_SEEDS,
    PROTOCOL_VERSION,
    WORLD_COUNT,
    hardest_completion_steps,
)


EXPECTED_NEURONS = {"v1": 95, "v2": 299, "v3": 345, "v4": 373}
EXPECTED_COMPONENTS = {version: set(get_version(version).components) for version in EXPECTED_NEURONS}


def _case_name(record: dict) -> str | None:
    return record.get("condition") or record.get("case")


def _trace_groups(experiment: str, record: dict) -> list[tuple[str, list[dict], int]]:
    """Return (label, rows, expected length) for every raw phase trace."""
    if experiment == "embodied_mb_valence_causal":
        return [
            ("training", record.get("training_trace", []),
             int(record.get("trials", 0) * record.get("train_steps", 0) * record.get("substeps", 0))),
            ("food_probe", record.get("food_probe_trace", []),
             int(record.get("probe_steps", 0) * record.get("substeps", 0))),
            ("toxin_probe", record.get("toxin_probe_trace", []),
             int(record.get("probe_steps", 0) * record.get("substeps", 0))),
        ]
    key = "trace"
    if experiment in {
        "embodied_food_collection_causal", "embodied_toxin_escape_causal",
        "embodied_headon_toxin_causal", "embodied_arbiter_explore_causal",
    }:
        key = "tick_trace"
    rows = record.get(key, [])
    expected = int(record.get("steps", 0) * record.get("substeps", record.get("neural_substeps_per_body_step", 0)))
    return [(key, rows, expected)]


def _expected_world_dir(root: Path, worlds: list[str], world: str) -> Path:
    return root / world if len(worlds) > 1 else root


def _sum_sensor(rows: list[dict], key: str) -> int:
    return int(sum(int(row.get(key, {}).get("left", 0)) + int(row.get(key, {}).get("right", 0)) for row in rows))


def _max_actuator(rows: list[dict], field: str = "actuator_ctrl") -> float:
    return float(max((abs(float(value)) for row in rows for value in row.get(field, {}).values()), default=0.0))


def _recompute_behavior(experiment: str, manifest: dict, records: dict[tuple[str, int, str], dict]) -> list[str]:
    """Recompute the causal predicates from raw records, not summary/acceptance JSON."""
    failures: list[str] = []
    worlds = list(manifest.get("worlds", []))
    seeds = [int(seed) for seed in manifest.get("seeds", [])]
    protocol = manifest.get("acceptance_protocol", {})
    cases_value = manifest.get("cases", manifest.get("conditions", {}))
    cases = list(cases_value) if isinstance(cases_value, dict) else list(cases_value)

    def get(world: str, seed: int, case: str) -> dict | None:
        return records.get((world, seed, case))

    if experiment == "embodied_food_collection_causal":
        for world in worlds:
            for seed in seeds:
                full, off = get(world, seed, "food_collection"), get(world, seed, "food_sensor_to_turn_ablation")
                if not full or not off:
                    continue
                for label, rec in (("intact", full), ("ablation", off)):
                    trace = rec["tick_trace"]
                    if _sum_sensor(trace, "food_sensor_spikes") < 500:
                        failures.append(f"{world} seed {seed} {label}: food transducer did not drive both sides")
                    if _max_actuator(trace) <= 0:
                        failures.append(f"{world} seed {seed} {label}: no ordinary actuator activity")
                if full.get("food_eaten") != 1:
                    failures.append(f"{world} seed {seed}: intact route did not eat exactly one target")
                if off.get("food_eaten") != 0:
                    failures.append(f"{world} seed {seed}: food-turn ablation ate the target")
    elif experiment == "embodied_toxin_escape_causal":
        for world in worlds:
            for seed in seeds:
                full, off = get(world, seed, "toxin_avoidance"), get(world, seed, "toxin_turn_ablation")
                if not full or not off:
                    continue
                for label, rec in (("intact", full), ("ablation", off)):
                    if _sum_sensor(rec["tick_trace"], "toxin_sensor_spikes") < 100:
                        failures.append(f"{world} seed {seed} {label}: toxin transducer was silent")
                    if _max_actuator(rec["tick_trace"]) <= 0:
                        failures.append(f"{world} seed {seed} {label}: no ordinary actuator activity")
                if full.get("toxin_hits") != 0:
                    failures.append(f"{world} seed {seed}: intact toxin route entered the hazard")
                if off.get("toxin_hits", 0) < 1 or min(row["source_distance"] for row in off["tick_trace"]) >= 0.55:
                    failures.append(f"{world} seed {seed}: toxin-turn ablation did not enter the contact zone")
    elif experiment == "embodied_headon_toxin_causal":
        for world in worlds:
            for seed in seeds:
                full, off = get(world, seed, "headon_trise_full"), get(world, seed, "trise_to_steer_ablation")
                if not full or not off:
                    continue
                for label, rec in (("intact", full), ("ablation", off)):
                    trace = rec["tick_trace"]
                    if _sum_sensor(trace, "toxin_sensor_spikes") < 6000:
                        failures.append(f"{world} seed {seed} {label}: head-on toxin sensors were not sufficiently driven")
                    if sum(row.get("trise_spikes", 0) for row in trace) < 100:
                        failures.append(f"{world} seed {seed} {label}: TRISE did not detect the rise")
                    if _max_actuator(trace) <= 0:
                        failures.append(f"{world} seed {seed} {label}: no ordinary actuator activity")
                if full.get("toxin_hits") != 0 or min(row["source_distance"] for row in full["tick_trace"]) <= 0.55:
                    failures.append(f"{world} seed {seed}: intact TRISE route entered the contact zone")
                if off.get("toxin_hits", 0) < 1 or min(row["source_distance"] for row in off["tick_trace"]) >= 0.55:
                    failures.append(f"{world} seed {seed}: TRISE ablation did not enter the contact zone")
                if sum(row.get("steer_spike", 0) for row in full["tick_trace"]) < sum(row.get("steer_spike", 0) for row in off["tick_trace"]) + 20:
                    failures.append(f"{world} seed {seed}: TRISE did not add escape drive")
    elif experiment == "embodied_mb_valence_causal":
        trials = int(manifest.get("trials", 0))
        expected = {"v2": 299, "v3": 345}.get(str(manifest.get("version")))
        for world in worlds:
            for seed in seeds:
                trained, off = get(world, seed, "trained"), get(world, seed, "sting_trigger_ablation")
                if not trained or not off:
                    continue
                for label, rec in (("trained", trained), ("ablation", off)):
                    if max((row.get("tox_hits", 0) for row in rec["training_trace"]), default=0) < trials:
                        failures.append(f"{world} seed {seed} {label}: teaching contacts were not delivered")
                trained_stg = sum(row.get("stg_t", 0) for row in trained["training_trace"])
                off_stg = sum(row.get("stg_t", 0) for row in off["training_trace"])
                if trained_stg == 0 or off_stg != 0:
                    failures.append(f"{world} seed {seed}: sting teaching ablation did not isolate STG_T")
                if sum(row.get("stg_t", 0) for row in trained["toxin_probe_trace"]) != 0:
                    failures.append(f"{world} seed {seed}: toxin probe was contaminated by teaching")
                trained_toxin_mbon = sum(row.get("mbon_spikes", 0) for row in trained["toxin_probe_trace"])
                trained_food_mbon = sum(row.get("mbon_spikes", 0) for row in trained["food_probe_trace"])
                trained_avoid = sum(row.get("avoid_spike", 0) for row in trained["toxin_probe_trace"])
                off_toxin_mbon = sum(row.get("mbon_spikes", 0) for row in off["toxin_probe_trace"])
                off_avoid = sum(row.get("avoid_spike", 0) for row in off["toxin_probe_trace"])
                if trained_toxin_mbon <= trained_food_mbon or trained_avoid == 0:
                    failures.append(f"{world} seed {seed}: trained MB did not produce toxin-selective output")
                if off_toxin_mbon != 0 or off_avoid != 0:
                    failures.append(f"{world} seed {seed}: MB teaching ablation retained learned toxin output")
    elif experiment == "embodied_arbiter_explore_causal":
        for world in worlds:
            for seed in seeds:
                intact, off = get(world, seed, "mode_gated_exploration"), get(world, seed, "forage_search_output_ablation")
                if not intact or not off:
                    continue
                def metrics(rec):
                    trace = rec["tick_trace"]
                    first = next((i for i, row in enumerate(trace) if row.get("food_eaten_total", 0) > 0), None)
                    if first is None:
                        return None
                    pre = trace[40:max(40, first - 20)]
                    post = trace[min(len(trace), first + 160):]
                    def mode(rows, key): return sum(row.get("mode_spikes", {}).get(key, 0) for row in rows)
                    def asym(rows):
                        return sum(abs(row["actuator_ctrl"]["plp"] + row["actuator_ctrl"]["plr"]
                                       - row["actuator_ctrl"]["prp"] - row["actuator_ctrl"]["prr"]) for row in rows)
                    return {"pre": pre, "post": post,
                            "pre_hunger": sum(row.get("hunger_spikes", 0) for row in pre) / max(1, len(pre)),
                            "post_hunger": sum(row.get("hunger_spikes", 0) for row in post) / max(1, len(post)),
                            "pre_forage": mode(pre, "FORAGE"), "pre_explore": mode(pre, "EXPLORE"),
                            "post_forage": mode(post, "FORAGE"), "post_explore": mode(post, "EXPLORE"),
                            "pre_search": sum(row.get("search_spike", 0) for row in pre),
                            "post_search": sum(row.get("search_spike", 0) for row in post),
                            "pre_asym": asym(pre), "post_asym": asym(post),
                            "max_act": _max_actuator(trace)}
                im, om = metrics(intact), metrics(off)
                if im is None or om is None:
                    failures.append(f"{world} seed {seed}: scheduled meal was not consumed")
                    continue
                for label, m in (("intact", im), ("ablation", om)):
                    if m["pre_hunger"] <= 1.0 or m["post_hunger"] >= 0.25:
                        failures.append(f"{world} seed {seed} {label}: meal did not switch hunger")
                    if m["pre_forage"] <= m["pre_explore"] or m["post_explore"] <= m["post_forage"]:
                        failures.append(f"{world} seed {seed} {label}: mode winner did not switch")
                    if m["max_act"] <= 0:
                        failures.append(f"{world} seed {seed} {label}: no ordinary actuator activity")
                if im["pre_search"] > 6 or om["pre_search"] <= im["pre_search"] + 12 or om["pre_asym"] <= im["pre_asym"] + 30:
                    failures.append(f"{world} seed {seed}: FORAGE output ablation did not restore search/force asymmetry")
                if im["post_search"] <= 40:
                    failures.append(f"{world} seed {seed}: EXPLORE did not release search")
    elif experiment == "embodied_metabolic_rest_causal":
        for world in worlds:
            for seed in seeds:
                rows = {case: get(world, seed, case) for case in cases}
                if any(value is None for value in rows.values()):
                    continue
                metrics = {}
                for case, rec in rows.items():
                    trace = rec["trace"]
                    meal = next((i for i, row in enumerate(trace) if row.get("food_eaten", 0) > 0), None)
                    if meal is None:
                        failures.append(f"{world} seed {seed} {case}: scheduled meal was not consumed")
                        continue
                    post = trace[meal + 12:]
                    metrics[case] = {
                        "sleep": sum(row.get("mode_spikes", {}).get("SLEEP", 0) for row in post),
                        "act": sum(row.get("actuator_abs_sum", 0.0) for row in post),
                        "energy": float(rec["final_metabolic"]["energy_store"]),
                    }
                if len(metrics) != len(rows):
                    continue
                if metrics["v3_intact"]["sleep"] < 30:
                    failures.append(f"{world} seed {seed}: V3 did not sustain SLEEP")
                if metrics["v3_intact"]["energy"] <= metrics["v2_prior"]["energy"] + 0.05:
                    failures.append(f"{world} seed {seed}: V3 did not preserve energy over V2")
                if metrics["v3_intact"]["energy"] <= metrics["v3_sleep_output_ablation"]["energy"] + 0.05:
                    failures.append(f"{world} seed {seed}: SLEEP ablation did not reduce energy")
                if metrics["v3_intact"]["energy"] <= metrics["v3_metabolic_afferent_ablation"]["energy"] + 0.05:
                    failures.append(f"{world} seed {seed}: metabolic afferent ablation did not reduce energy")
                if metrics["v3_sleep_output_ablation"]["act"] <= metrics["v3_intact"]["act"] + 20.0:
                    failures.append(f"{world} seed {seed}: SLEEP ablation did not restore motor activity")
    elif experiment == "embodied_obstacle_detour_causal":
        for world in worlds:
            for seed in seeds:
                prior = get(world, seed, "v3_prior")
                intact = get(world, seed, "v4_intact")
                body_off = get(world, seed, "v4_body_afferent_ablation")
                reflex_off = get(world, seed, "v4_reflex_ablation")
                if not all((prior, intact, body_off, reflex_off)):
                    continue
                def m(rec):
                    trace = rec["trace"]
                    ix = rec["initial_pose"]["x"]
                    iy = rec["initial_pose"]["y"]
                    xs = [row["pose"]["x"] for row in trace]
                    ys = [row["pose"]["y"] for row in trace]
                    return {"contact": max((row.get("contact_count", 0) for row in trace), default=0),
                            "deflect": max((abs(y - iy) for y in ys), default=0.0),
                            "progress": max(0.0, ix - min(xs, default=ix)),
                            "obs": sum(sum(row.get("obstacle_spikes", {}).values()) for row in trace),
                            "reflex": sum(sum(row.get("reflex_spikes", {}).values()) for row in trace),
                            "act": sum(row.get("actuator_abs_sum", 0.0) for row in trace),
                            "food": rec.get("food_eaten", 0)}
                pm, im, bm, rm = map(m, (prior, intact, body_off, reflex_off))
                if any(value["food"] != 0 for value in (pm, im, bm, rm)):
                    failures.append(f"{world} seed {seed}: obstacle fixture contained food")
                if pm["contact"] < 1 or im["contact"] != 0 or bm["contact"] < 1 or rm["contact"] < 1:
                    failures.append(f"{world} seed {seed}: obstacle contact ablation contract failed")
                if im["progress"] < 0.6 or im["deflect"] < 0.25 or im["obs"] < 20 or im["reflex"] < 10:
                    failures.append(f"{world} seed {seed}: intact V4 did not detect/deflect the geometry")
                if im["act"] <= bm["act"] * 0.05:
                    failures.append(f"{world} seed {seed}: body-afferent ablation changed the motor substrate unexpectedly")
    return failures


def validate(evidence: Path) -> dict:
    failures: list[str] = []
    manifest_path = evidence / "manifest.json"
    acceptance_path = evidence / "acceptance.json"
    summary_path = evidence / "summary.json"
    if not manifest_path.exists():
        return {"valid": False, "behavior_passed": False, "failures": [f"missing {manifest_path}"]}
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "behavior_passed": False, "failures": [f"invalid manifest: {exc}"]}
    experiment = str(manifest.get("experiment", ""))
    version = str(manifest.get("version", ""))
    worlds = list(manifest.get("worlds", []))
    seeds = [int(seed) for seed in manifest.get("seeds", [])]
    protocol = manifest.get("acceptance_protocol", {})
    cases_value = manifest.get("cases", manifest.get("conditions", {}))
    cases = list(cases_value) if isinstance(cases_value, dict) else list(cases_value)
    if not experiment or not worlds or not seeds or not cases:
        failures.append("manifest does not declare experiment, worlds, seeds, and cases")
    if protocol.get("required"):
        if protocol.get("name") != "embodied_matrix" or protocol.get("version") != PROTOCOL_VERSION:
            failures.append("required acceptance protocol has an unknown name/version")
        if len(worlds) != WORLD_COUNT:
            failures.append(f"required protocol has {len(worlds)} worlds, expected {WORLD_COUNT}")
        catalog = manifest.get("world_catalog", {})
        catalog_names = list(catalog) if isinstance(catalog, dict) else []
        # JSON manifests are written with sorted object keys for deterministic
        # diffs, so object-key order is not the scientific catalog order.  The
        # protocol stores that order explicitly and the validator checks both
        # the ordered list and the catalog membership.
        if list(protocol.get("worlds", [])) != worlds:
            failures.append("required protocol worlds do not match protocol metadata order")
        if set(worlds) != set(catalog_names):
            failures.append("required protocol worlds do not match the declared catalog members")
        if tuple(seeds) != PROTOCOL_SEEDS:
            failures.append(f"required protocol seeds must be exactly {list(PROTOCOL_SEEDS)}")
        if protocol.get("world_count") != WORLD_COUNT or protocol.get("seed_count") != len(PROTOCOL_SEEDS):
            failures.append("required protocol world/seed counts are incorrect")
        try:
            hardest = hardest_completion_steps(catalog, worlds)
            if int(manifest.get("steps", manifest.get("trials", 0))) < hardest:
                failures.append("required protocol horizon is shorter than the hardest world")
            if int(protocol.get("hardest_completion_steps", -1)) != hardest:
                failures.append("required protocol hardest-world horizon is not reproducible from the catalog")
            if int(protocol.get("horizon_steps", -1)) != int(manifest.get("steps", manifest.get("trials", 0))):
                failures.append("required protocol horizon metadata differs from manifest")
        except (TypeError, ValueError) as exc:
            failures.append(f"required protocol world horizon is invalid: {exc}")
        if not protocol.get("compliant"):
            failures.append("manifest marks the required embodied matrix as non-compliant")
    if version not in EXPECTED_NEURONS and experiment != "embodied_obstacle_detour_causal":
        failures.append(f"manifest has unknown strict version {version!r}")
    if not acceptance_path.exists():
        failures.append("missing acceptance.json")
        acceptance = {"passed": False, "failures": []}
    else:
        try:
            acceptance = json.loads(acceptance_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            acceptance = {"passed": False, "failures": []}
            failures.append(f"invalid acceptance.json: {exc}")
    if not summary_path.exists():
        if experiment == "embodied_obstacle_detour_causal":
            for world in worlds:
                if not (_expected_world_dir(evidence, worlds, world) / "summary.json").exists():
                    failures.append(f"missing summary.json for {world}")
        else:
            failures.append("missing summary.json")
    if acceptance.get("passed") and acceptance.get("failures"):
        failures.append("acceptance claims pass while listing failures")
    if not acceptance.get("passed") and not isinstance(acceptance.get("failures", []), list):
        failures.append("failed acceptance does not contain a failure list")

    record_count = 0
    loaded_records: dict[tuple[str, int, str], dict] = {}
    expected_record_count = len(worlds) * len(seeds) * len(cases)
    for world in worlds:
        world_dir = _expected_world_dir(evidence, worlds, world)
        for seed in seeds:
            for case in cases:
                path = world_dir / f"{case}_seed{seed}.json"
                if not path.exists():
                    failures.append(f"missing record {world}/{case}/seed{seed}")
                    continue
                record_count += 1
                try:
                    record = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    failures.append(f"invalid record {path}: {exc}")
                    continue
                loaded_records[(world, seed, case)] = record
                if _case_name(record) != case or record.get("world") != world or int(record.get("seed", seed)) != seed:
                    failures.append(f"record identity mismatch: {path}")
                inferred_obstacle_version = "v3" if case == "v3_prior" else "v4"
                if experiment == "embodied_obstacle_detour_causal":
                    inferred_version = inferred_obstacle_version
                elif experiment == "embodied_metabolic_rest_causal":
                    inferred_version = "v2" if case == "v2_prior" else "v3"
                else:
                    inferred_version = version
                record_version = str(record.get("version") or inferred_version)
                expected_version = "v3" if experiment == "embodied_arbiter_explore_causal" else record_version
                if experiment == "embodied_obstacle_detour_causal":
                    expected_version = "v3" if case == "v3_prior" else "v4"
                if record_version != expected_version:
                    failures.append(f"{path}: record version {record_version!r} does not match {expected_version!r}")
                profile = record.get("component_profile", {})
                if not profile.get("strict"):
                    failures.append(f"{path}: non-strict topology")
                if (manifest.get("effective_build_parameters")
                        and (not isinstance(record.get("build_parameters"), dict) or not record["build_parameters"])):
                    failures.append(f"{path}: effective PAULA build parameters were not recorded")
                if manifest.get("record_visual_artifacts"):
                    visual = record.get("video_artifacts", {})
                    artifact_dir = path.parent / "artifacts" / f"{case}_seed{seed}"
                    if visual.get("status") != "complete":
                        failures.append(f"{path}: visual artifact capture is not complete")
                    frame_ticks = visual.get("frame_neural_ticks", [])
                    if (not isinstance(frame_ticks, list)
                            or len(frame_ticks) != int(visual.get("frame_count", -1))
                            or any(b <= a for a, b in zip(frame_ticks, frame_ticks[1:]))):
                        failures.append(f"{path}: visual frame clock is missing or non-monotonic")
                    for view in ("pov", "third_person", "top_down"):
                        filename = visual.get("views", {}).get(view)
                        if not filename or not (artifact_dir / filename).is_file():
                            failures.append(f"{path}: missing {view} video artifact")
                if record_version in EXPECTED_NEURONS:
                    if int(record.get("neuron_count", -1)) != EXPECTED_NEURONS[record_version]:
                        failures.append(f"{path}: wrong neuron count")
                    if set(profile.get("components", ())) != EXPECTED_COMPONENTS[record_version]:
                        failures.append(f"{path}: component profile differs from {record_version}")
                fixture = record.get("world_fixture", {})
                if fixture and fixture.get("respawn_food") is not False:
                    failures.append(f"{path}: fixture permits food respawn")
                if fixture and experiment == "embodied_obstacle_detour_causal":
                    if fixture.get("food_count") != 0 or fixture.get("toxin_count") != 0:
                        failures.append(f"{path}: obstacle fixture contains a source")
                if fixture and experiment != "embodied_obstacle_detour_causal":
                    if fixture.get("food_count") != 7 or fixture.get("toxin_count") != 6:
                        failures.append(f"{path}: source fixture counts changed")
                for label, rows, expected in _trace_groups(experiment, record):
                    if not isinstance(rows, list) or len(rows) != expected or expected <= 0:
                        failures.append(f"{path}: {label} trace has {len(rows) if isinstance(rows, list) else 'invalid'} ticks, expected {expected}")
                        continue
                    ticks = [row.get("neural_tick") for row in rows if isinstance(row, dict)]
                    if len(ticks) != len(rows) or any(not isinstance(tick, (int, float)) for tick in ticks):
                        failures.append(f"{path}: {label} has a malformed neural clock")
                    elif any(b != a + 1 for a, b in zip(ticks, ticks[1:])):
                        failures.append(f"{path}: {label} neural clock is not contiguous")
    if record_count != expected_record_count:
        failures.append(f"record count {record_count}/{expected_record_count}")
    # Causal controls must share the same physical fixture and initial state.
    # A different source placement, respawn flag, body pose, or horizon would
    # make an apparent ablation effect scientifically uninterpretable.
    for world in worlds:
        for seed in seeds:
            group = [loaded_records[(world, seed, case)] for case in cases if (world, seed, case) in loaded_records]
            if len(group) > 1:
                def fixture_signature(record: dict) -> str:
                    value = {
                        "world_fixture": record.get("world_fixture"),
                        "source": record.get("source"),
                        "barriers": record.get("barriers"),
                        "initial_pose": record.get("initial_pose"),
                        "steps": record.get("steps"),
                        "substeps": record.get("substeps", record.get("neural_substeps_per_body_step")),
                        "meal_body_step": record.get("meal_body_step"),
                        "scenario": record.get("scenario"),
                    }
                    return json.dumps(value, sort_keys=True, separators=(",", ":"))
                signatures = {fixture_signature(record) for record in group}
                if len(signatures) != 1:
                    failures.append(f"{world} seed {seed}: causal conditions do not share an identical fixture/initial state")
    behavior_failures = _recompute_behavior(experiment, manifest, loaded_records)
    acceptance_claim = bool(acceptance.get("passed", False))
    independent_behavior_passed = not behavior_failures
    if acceptance_claim != independent_behavior_passed:
        failures.append(
            "acceptance claim does not match independent raw-trace recomputation "
            f"(claim={acceptance_claim}, recomputed={independent_behavior_passed})"
        )
    return {
        "valid": not failures,
        "behavior_passed": independent_behavior_passed,
        "acceptance_claim": acceptance_claim,
        "behavior_failures": behavior_failures,
        "experiment": experiment,
        "version": version,
        "records": record_count,
        "expected_records": expected_record_count,
        "failures": failures,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, help="write the validator record here")
    args = parser.parse_args(argv)
    result = validate(args.evidence)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["valid"] and result["behavior_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
