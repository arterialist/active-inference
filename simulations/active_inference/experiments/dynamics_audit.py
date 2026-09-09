"""Bounded embodied clock audit with streaming, agent-readable evidence.

Run from active-inference with ``python -m
simulations.active_inference.experiments.dynamics_audit --output PATH``.
This diagnostic compares the same organism under different caller batching.
It is deliberately not a behavioural release gate or a parameter optimizer.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import gzip
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import platform
import random
import sys
import time

import mujoco
import numpy as np

from simulations.active_inference import aif_agent3d as ag
from simulations.active_inference.live.versions import get_version
from simulations.active_inference.experiments.embodied_recording import EmbodiedVideoRecorder


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def encoded(value):
    return json.dumps(value, default=plain, sort_keys=True, allow_nan=False, separators=(",", ":"))


def source_fingerprint():
    root = Path(__file__).resolve().parents[3]
    paths = list((root / "simulations/active_inference").rglob("*.py"))
    paths += list((root.parent / "neuron-model/neuron").rglob("*.py"))
    paths.append(root.parent / "neuron-model/paula_agent/ckit.py")
    return {str(p.relative_to(root.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def build_manifest(agent):
    """Resolved parameters, signed synapses, delays and classes, not defaults."""
    return {str(nid): {
        "class": type(n).__module__ + "." + type(n).__name__,
        "parameters": asdict(n.params), "metadata": n.metadata,
        "t_ref_bounds": [n.lower_t_ref_bound, n.upper_t_ref_bound],
        "synapses": {str(sid): {
            "source": n.synapse_sources.get(sid),
            "distance_ticks": n.distances[sid],
            "info_weight": p.u_i.info, "plast_weight": p.u_i.plast,
            "adapt": p.u_i.adapt,
            "effective_gain": (p.u_i.info + p.u_i.plast) * n.params.delta_decay**n.distances[sid],
        } for sid, p in n.postsynaptic_points.items()},
    } for nid, n in sorted(agent.nb.items())}


def neural_state(agent):
    return [[nid, float(n.S), float(n.O), float(n.r), float(n.b),
             float(n.t_ref), n.M_vector.tolist(), float(n.F_avg),
             None if n.t_last_fire == -np.inf else float(n.t_last_fire)]
            for nid, n in sorted(agent.nb.items())]


def hidden_state(agent):
    """Delayed signals and weights are state, not incidental implementation.

    This is an observational snapshot, not a restorable simulator checkpoint.
    Dendritic rows retain source synapse IDs and already-weighted potentials;
    a later weight change does not retroactively change an in-flight signal.
    """
    def wheel_state(wheel):
        return [[signal.arrival_tick,
                 {"type": type(signal.event).__name__, "value": asdict(signal.event)}
                 if is_dataclass(signal.event) else {"type": "release", "value": signal.event}]
                for slot in wheel for signal in slot]

    return {
        "presynaptic_wheel": wheel_state(agent.net.presynaptic_wheel),
        "retrograde_wheel": wheel_state(agent.net.retrograde_wheel),
        "dendritic_queues": {str(nid): sorted(n.propagation_queue)
                             for nid, n in agent.nb.items() if n.propagation_queue},
        "synapses": [[nid, sid, float(p.u_i.info), float(p.u_i.plast),
                      p.u_i.adapt.tolist(), float(p.potential)]
                     for nid, n in sorted(agent.nb.items())
                     for sid, p in sorted(n.postsynaptic_points.items())],
        "terminals": [[nid, sid, float(p.u_o.info), p.u_o.mod.tolist()]
                      for nid, n in sorted(agent.nb.items())
                      for sid, p in sorted(n.presynaptic_points.items())],
    }


def replay_record(path):
    """Replay actual afferents through a whole brain with physics disconnected.

    Exact replay tests neural reproduction under the embodied input history,
    NOT closed-loop competence. No recorded output is used to drive a neuron.
    """
    record = json.loads((path / "record.json").read_text())
    random.seed(record["seed"])
    np.random.seed(record["seed"])
    agent = ag.AIFAgent3D(seed=record["seed"], components=get_version(record["version"]).components,
                        config=ag.DEFAULT_EMBODIED_CONFIG.with_overrides(**record["build_overrides"]))
    agent.birth()
    if encoded(build_manifest(agent)) != encoded(json.loads((path / "build.json").read_text())):
        raise ValueError("resolved birth topology/parameters differ; this is not an exact replay")
    first = {}
    count = 0
    with gzip.open(path / "ticks.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if "external_calls" not in row:
                raise ValueError("record predates afferent replay; do not infer missing inputs")
            for nid, sid, info, mod in row["external_calls"]:
                agent.net.set_external_input(nid, sid, info, None if mod is None else np.asarray(mod))
            agent.core.do_tick()
            count += 1
            if row["tick"] != count:
                first.setdefault("tick_sequence", count)
            for field, actual in (("neurons", neural_state(agent)),
                                  ("hidden_state", hidden_state(agent))):
                if field == "hidden_state":
                    # Compare only channels that this schema actually captured.
                    # Adding an observer channel must not masquerade as a neural divergence.
                    actual = {key: actual[key] for key in row[field]}
                if field not in first and encoded(actual) != encoded(row[field]):
                    first[field] = row["tick"]
    if count != record["ticks"] or count == 0:
        first["record_length"] = count
    return {"record": str(path), "ticks_compared": count, "identical": not first,
            "first_divergence_tick": first, "physics_advanced": False,
            "release_evidence": False}


def inspect_tick(path, tick, neuron_ids):
    """Address a causal witness by tick and actual neuron ID, without a video."""
    build = json.loads((path / "build.json").read_text())
    previous = None
    with gzip.open(path / "ticks.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["tick"] == tick:
                selected = {str(nid): build[str(nid)] for nid in neuron_ids}
                prior_queues = (previous or {}).get("hidden_state", {}).get("dendritic_queues", {})
                arrivals = {}
                for nid in selected:
                    cell = selected[nid]
                    arrivals[nid] = [{
                        "synapse": sid, "source": cell["synapses"][str(sid)]["source"],
                        "scheduled_network_tick": arrival, "local_potential_at_emission": potential,
                        "attenuated_current": potential * cell["parameters"]["delta_decay"] ** cell["synapses"][str(sid)]["distance_ticks"],
                    } for arrival, target, potential, sid in prior_queues.get(nid, [])
                      if arrival <= row["network_tick"] - 1]
                return {
                    "tick": tick, "network_tick_after": row["network_tick"],
                    "physical": row["physical"], "sensory_current": row["sensor_current"],
                    "neurons": [n for n in row["neurons"] if str(n[0]) in selected],
                    "previous_neurons": [n for n in (previous or {}).get("neurons", []) if str(n[0]) in selected],
                    "resolved_cells": selected, "arrivals_from_prior_queue": arrivals,
                    "limitations": ["Queue ledger excludes newly scheduled zero-distance inputs.",
                                    "A current contribution is not an intervention-based causal effect."],
                }
            previous = row
    raise KeyError(f"tick {tick} is absent")


def run_record(output, version, ticks, sub, calls, seed, fixture, variant="intact"):
    """Observer never writes neural or physical state after birth."""
    output.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    overrides = {"intact": {}, "lh_output_ablated": {"w_lh_turn": 0.0},
                 "lh_subthreshold": {"w_lh_sensor": 0.04}}[variant]
    agent = ag.AIFAgent3D(seed=seed, components=get_version(version).components,
                          config=ag.DEFAULT_EMBODIED_CONFIG.with_overrides(**overrides))
    # A deliberately immediate meal isolates contact delivery from locomotion.
    # The lateral fixture tests moving bilateral samples without a forced meal.
    source = {"contact": [-0.6, 0.0], "lateral": [-1.2, -1.0],
              "mirror": [-1.2, 1.0]}[fixture]
    agent.world.foods = [source] + [[100.0, 100.0] for _ in range(6)]
    agent.world.toxins = [[100.0, 100.0] for _ in range(6)]
    agent.world.respawn_food = False
    agent.world._sync_mocap()
    agent.birth()
    birth_tick = agent.net.current_tick
    (output / "build.json").write_text(encoded(build_manifest(agent)) + "\n")
    recorder = EmbodiedVideoRecorder(agent, output / "artifacts", total_ticks=ticks,
                                      substeps=sub, max_frames=24)
    before = {}
    first_meal = first_us = None
    angular_sum = 0.0
    previous_yaw = float(agent.world.pose()[2])
    start = time.monotonic()
    neural_hash = hashlib.sha256()
    physical_hash = hashlib.sha256()
    minimum_distance = float("inf")
    minimum_tick = None
    path_length = 0.0
    previous_xy = np.asarray(agent.world.pose()[:2])
    spike_counts = {str(nid): 0 for nid in agent.nb}

    def pre(current):
        before.clear()
        before.update({"physics_time_s": float(current.world.data.time),
                       "pose": list(current.world.pose()),
                       "odour": current.world.odour(),
                       "pending_event": current.world.pending_event})
        # US is recorded directly by intercepting the transducer below, before
        # Network.run_tick clears its external input buffer.

    delivered = {}
    external_calls = []
    set_input = agent.net.set_external_input

    def observe_input(nid, sid, info, mod=None):
        external_calls.append([nid, sid, float(info), None if mod is None else list(mod)])
        if nid in (ag.STG_F, ag.STG_T) and sid == 0:
            delivered[str(nid)] = float(info)
        return set_input(nid, sid, info, mod)

    agent.net.set_external_input = observe_input
    with gzip.open(output / "ticks.jsonl.gz", "wt") as stream:
        def post(current):
            nonlocal first_meal, first_us, angular_sum, previous_yaw
            nonlocal minimum_distance, minimum_tick, path_length, previous_xy
            world = current.world
            states = neural_state(current)
            yaw = float(world.pose()[2])
            angular_sum += float(np.arctan2(np.sin(yaw-previous_yaw), np.cos(yaw-previous_yaw)))
            previous_yaw = yaw
            xy = np.asarray(world.pose()[:2])
            path_length += float(np.linalg.norm(xy - previous_xy))
            previous_xy = xy
            distance = float(np.linalg.norm(xy - source))
            if first_meal is None and distance < minimum_distance:
                minimum_distance, minimum_tick = distance, current.t
            for nid, n in current.nb.items():
                spike_counts[str(nid)] += int(n.O > 0)
            physical = {"qpos": world.data.qpos.tolist(), "qvel": world.data.qvel.tolist(),
                        "ctrl": world.data.ctrl.tolist(), "time_s": float(world.data.time),
                        "food_eaten": world.eaten, "toxin_entries": world.tox_hits,
                        "gut": world.gut_load, "energy": world.energy_store,
                        "actuator_force": world.data.actuator_force.tolist()}
            row = {"tick": current.t, "network_tick": current.net.current_tick,
                   "before_physics": dict(before), "physical": physical,
                   "sensor_current": dict(current.last_sensor_drives),
                   "contact_us": dict(delivered), "neurons": states,
                   "external_calls": list(external_calls), "hidden_state": hidden_state(current),
                   "yaw_displacement_rad": angular_sum,
                   "target_distance": distance,
                   "target_bearing_error_rad": float(np.arctan2(
                       np.sin(np.arctan2(source[1]-xy[1], source[0]-xy[0])-yaw-np.pi),
                       np.cos(np.arctan2(source[1]-xy[1], source[0]-xy[0])-yaw-np.pi)))}
            stream.write(encoded(row) + "\n")
            external_calls.clear()
            neural_hash.update(encoded(states).encode())
            physical_hash.update(encoded(physical).encode())
            if world.eaten and first_meal is None:
                first_meal = current.t
            if delivered.get(str(ag.STG_F), 0) > 0 and first_us is None:
                first_us = current.t
            recorder.observe(current)

        if calls == "single":
            ag.run_episode(agent, steps=ticks//sub, sub=sub, vision=False,
                           log_every=10**9, tick_hook=post, neural_input_hook=pre)
        else:
            for _ in range(ticks//sub):
                ag.run_episode(agent, steps=1, sub=sub, vision=False,
                               log_every=10**9, tick_hook=post, neural_input_hook=pre)
    summary = {"version": version, "seed": seed, "fixture": fixture,
               "source": source, "ticks": ticks, "sub": sub, "calls": calls,
               "variant": variant, "build_overrides": overrides,
               "birth_network_ticks": birth_tick,
               "neural_digest": neural_hash.hexdigest(), "physical_digest": physical_hash.hexdigest(),
               "first_meal_tick": first_meal, "first_food_us_tick": first_us,
               "contact_to_us_ticks": None if first_meal is None or first_us is None else first_us-first_meal,
               "food_eaten": agent.world.eaten, "final_pose": agent.world.pose(),
               "minimum_precollection_distance": minimum_distance,
               "minimum_precollection_tick": minimum_tick,
               "path_length": path_length, "positive_output_ticks": spike_counts,
               "wall_seconds": time.monotonic()-start, "config": agent.effective_config_manifest(),
               "video_artifacts": recorder.finalize(), "release_evidence": False}
    agent.net.set_external_input = set_input
    summary["physics_dt_s"] = float(agent.world.model.opt.timestep)
    (output / "record.json").write_text(encoded(summary) + "\n")
    return summary


def compare_records(a, b):
    """First divergence by channel, in physical units; no mechanistic guess."""
    first = {}
    count = 0
    with gzip.open(a / "ticks.jsonl.gz", "rt") as af, gzip.open(b / "ticks.jsonl.gz", "rt") as bf:
        for i, (aa, bb) in enumerate(zip_longest(af, bf), 1):
            count = i
            if aa is None or bb is None:
                first.setdefault("missing_tick", i)
                continue
            aa, bb = json.loads(aa), json.loads(bb)
            if aa["tick"] != i or bb["tick"] != i:
                first.setdefault("tick_sequence", i)
            for field in ("sensor_current", "contact_us", "neurons", "physical", "hidden_state"):
                if field not in aa and field not in bb:
                    continue
                if aa.get(field) != bb.get(field):
                    first.setdefault(field, i)
    if count == 0:
        first["empty_trace"] = 0
    return {"a": str(a), "b": str(b), "ticks_compared": count,
            "identical": not first, "first_divergence_tick": first}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--replay", type=Path)
    destination.add_argument("--inspect", type=Path)
    parser.add_argument("--tick", type=int, default=1)
    parser.add_argument("--neurons", type=int, nargs="+", default=[ag.TL, ag.TR])
    parser.add_argument("--versions", nargs="+", choices=("v1","v2","v3","v4"), default=["v1","v2","v3","v4"])
    parser.add_argument("--ticks", type=int, default=160)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--fixture", choices=("contact", "lateral", "mirror"), default="contact")
    parser.add_argument("--experiment", choices=("clock", "coupling"), default="clock")
    parser.add_argument("--variants", nargs="+", choices=("intact", "lh_output_ablated", "lh_subthreshold"),
                        default=["intact", "lh_output_ablated"])
    args = parser.parse_args()
    if args.inspect is not None:
        print(encoded(inspect_tick(args.inspect, args.tick, args.neurons)))
        return 0
    if args.replay is not None:
        result = replay_record(args.replay)
        print(encoded(result))
        return 0 if result["identical"] else 1
    if args.ticks <= 0 or args.ticks % 16:
        parser.error("ticks must be a positive multiple of 16")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "release_evidence": False,
                "schema": "aif-dynamics-audit/2",
                "neuron_fields": ["id", "S", "O", "r", "b", "t_ref", "M", "F_avg", "last_spike_tick"],
                "experiment": args.experiment, "fixture": args.fixture,
                "python": sys.version, "mujoco": mujoco.__version__, "numpy": np.__version__,
                "platform": platform.platform(), "sources": source_fingerprint(),
                "records": [], "comparisons": []}
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(encoded(manifest) + "\n")
    if args.experiment == "coupling":
        for version in args.versions:
            variants = args.variants
            for variant in variants:
                target = args.output / version / variant
                result = run_record(target, version, args.ticks, 16, "single", args.seed,
                                    args.fixture, variant)
                manifest["records"].append(result)
                print(encoded({k: result[k] for k in (
                    "version", "variant", "food_eaten", "minimum_precollection_distance",
                    "minimum_precollection_tick", "path_length", "wall_seconds")}), flush=True)
                manifest_path.write_text(encoded(manifest) + "\n")
        manifest["status"] = "complete"
        manifest_path.write_text(encoded(manifest) + "\n")
        return 0
    for version in args.versions:
        for sub, calls in ((16, "single"), (16, "repeated"), (1, "single")):
            target = args.output / version / f"sub{sub}_{calls}"
            result = run_record(target, version, args.ticks, sub, calls, args.seed, args.fixture)
            manifest["records"].append(result)
            print(encoded({k: result[k] for k in ("version", "sub", "calls", "contact_to_us_ticks", "wall_seconds")}), flush=True)
        base = args.output / version / "sub16_single"
        for other in ("sub16_repeated", "sub1_single"):
            report = compare_records(base, args.output / version / other)
            manifest["comparisons"].append(report)
            print(encoded(report), flush=True)
        manifest_path.write_text(encoded(manifest) + "\n")
    manifest["status"] = "complete"
    manifest["clock_invariant"] = all(c["identical"] for c in manifest["comparisons"])
    manifest_path.write_text(encoded(manifest) + "\n")
    return 0 if manifest["clock_invariant"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
