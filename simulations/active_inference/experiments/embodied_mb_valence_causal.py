"""Physical-contact causal test for mushroom-body odour valence learning."""

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
BUILD = {"w_km0": 0.4, "w_kc_max": 2.0, "w_av_mbon": 6.0}
WORLD_SPECS = {
    "center": {"train_toxin": [0.0, 0.0], "probe": [-2.0, 0.0], "label": "centered teaching / probe", "completion_steps": 8},
    "left_015": {"train_toxin": [0.0, 0.15], "probe": [-2.0, 0.15], "label": "left 0.15 teaching / probe", "completion_steps": 8},
    "right_015": {"train_toxin": [0.0, -0.15], "probe": [-2.0, -0.15], "label": "right 0.15 teaching / probe", "completion_steps": 8},
    "left_025": {"train_toxin": [0.0, 0.25], "probe": [-2.0, 0.25], "label": "left 0.25 teaching / probe", "completion_steps": 8},
    "right_025": {"train_toxin": [0.0, -0.25], "probe": [-2.0, -0.25], "label": "right 0.25 teaching / probe", "completion_steps": 8},
    "left_035": {"train_toxin": [0.0, 0.35], "probe": [-2.0, 0.35], "label": "left 0.35 teaching / probe", "completion_steps": 8},
    "right_035": {"train_toxin": [0.0, -0.35], "probe": [-2.0, -0.35], "label": "right 0.35 teaching / probe", "completion_steps": 8},
    "left_045": {"train_toxin": [0.0, 0.45], "probe": [-2.0, 0.45], "label": "left 0.45 teaching / probe", "completion_steps": 8},
    "right_045": {"train_toxin": [0.0, -0.45], "probe": [-2.0, -0.45], "label": "right 0.45 teaching / probe", "completion_steps": 8},
    "cross_050": {"train_toxin": [0.0, 0.50], "probe": [-2.0, 0.50], "label": "cross-body 0.50 teaching / probe", "completion_steps": 8},
}
WORLD_NAMES = tuple(WORLD_SPECS)
CASES = {"trained": BUILD, "sting_trigger_ablation": {**BUILD, "w_trig": 0.0}}


def _write(path, data): path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def _revision(path):
    try: return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError): return None


def _reset_body(world):
    world.data.qpos[world.jx] = world.data.qpos[world.jy] = world.data.qpos[world.jyaw] = 0.0
    world._in = [False] * len(world._in)
    mujoco.mj_forward(world.model, world.data)


def _weights(agent):
    return float(sum(float(pp[i].u_i.info) for pp in agent._mb_pp for i in range(ag.NKC)))


def _contact_train(agent, trials, steps, sub, world_name, recorder=None):
    world = agent.world
    world.respawn_food = False
    world.foods = [FAR.copy() for _ in range(7)]
    world.toxins = [list(WORLD_SPECS[world_name]["train_toxin"])] + [FAR.copy() for _ in range(5)]
    world._sync_mocap()
    trace = []
    for trial in range(trials):
        _reset_body(world)
        def capture(current):
            trace.append({"phase": "training", "trial": trial, "neural_tick": current.t,
                          "stg_t": int(current.nb[ag.STG_T].O > 0),
                          "kc_spikes": sum(int(current.nb[n].O > 0) for n in ag.KC),
                          "mbon_spikes": sum(int(current.nb[n].O > 0) for n in ag.MBONP),
                          "avoid_spike": int(current.nb[ag.AVOID].O > 0),
                          "weight_sum": _weights(current),
                          "tox_hits": current.world.tox_hits})
            if recorder is not None:
                recorder.observe(current)
        ag.run_episode(agent, steps=steps, sub=sub, render_every=10**9, render_ticks=0,
                       vision=False, log_every=10**9, tick_hook=capture)
    return trace


def _probe(agent, kind, steps, sub, world_name, recorder=None):
    world = agent.world
    world.respawn_food = False
    probe = list(WORLD_SPECS[world_name]["probe"])
    world.foods = ([probe] + [FAR.copy() for _ in range(6)]) if kind == "food" else [FAR.copy() for _ in range(7)]
    world.toxins = ([probe] + [FAR.copy() for _ in range(5)]) if kind == "toxin" else [FAR.copy() for _ in range(6)]
    world._sync_mocap(); _reset_body(world)
    rows = []
    def capture(current):
        rows.append({"phase": f"probe_{kind}", "neural_tick": current.t,
                     "stg_t": int(current.nb[ag.STG_T].O > 0),
                     "kc_spikes": sum(int(current.nb[n].O > 0) for n in ag.KC),
                     "mbon_spikes": sum(int(current.nb[n].O > 0) for n in ag.MBONP),
                     "avoid_spike": int(current.nb[ag.AVOID].O > 0),
                     "weight_sum": _weights(current), "tox_hits": current.world.tox_hits})
        if recorder is not None:
            recorder.observe(current)
    ag.run_episode(agent, steps=steps, sub=sub, render_every=10**9, render_ticks=0,
                   vision=False, log_every=10**9, tick_hook=capture)
    return rows


def run_case(seed, name, build, trials, train_steps, probe_steps, sub, version, world_name,
             artifact_root: Path | None = None):
    np.random.seed(seed)
    profile = get_version(version)
    agent = ag.AIFAgent3D(seed=seed, components=profile.components, **build)
    agent.birth()
    before = _weights(agent)
    recorder = EmbodiedVideoRecorder(
        agent, artifact_directory(artifact_root, name, seed),
        total_ticks=(trials * train_steps + 2 * probe_steps) * sub,
        substeps=sub,
    )
    training = _contact_train(agent, trials, train_steps, sub, world_name, recorder)
    food = _probe(agent, "food", probe_steps, sub, world_name, recorder)
    toxin = _probe(agent, "toxin", probe_steps, sub, world_name, recorder)
    video_artifacts = recorder.finalize()
    return {"condition": name, "seed": seed, "version": profile.id, "world": world_name,
            "component_profile": dict(agent.component_profile), "neuron_count": len(agent.nb),
            "build_parameters": dict(agent.build_parameters),
            "build": build, "trials": trials, "train_steps": train_steps,
            "probe_steps": probe_steps, "substeps": sub,
            "initial_weight_sum": before, "final_weight_sum": _weights(agent),
            "world_fixture": {"food_count": len(agent.world.foods), "toxin_count": len(agent.world.toxins),
                               "respawn_food": bool(agent.world.respawn_food),
                               "training_toxin": list(WORLD_SPECS[world_name]["train_toxin"]),
                               "probe": list(WORLD_SPECS[world_name]["probe"])},
            "training_trace": training, "food_probe_trace": food, "toxin_probe_trace": toxin,
            "video_artifacts": video_artifacts}


def _summary(record):
    train, food, toxin = record["training_trace"], record["food_probe_trace"], record["toxin_probe_trace"]
    total = lambda rows, key: int(sum(row[key] for row in rows))
    expected_training = int(record["trials"] * record["train_steps"] * record["substeps"])
    expected_probe = int(record["probe_steps"] * record["substeps"])
    return {"world": record["world"],
            "physical_toxin_entries": max((row["tox_hits"] for row in train), default=0),
            "training_stg_t_spikes": total(train, "stg_t"),
            "weight_delta": record["final_weight_sum"] - record["initial_weight_sum"],
            "food_probe_mbon": total(food, "mbon_spikes"), "toxin_probe_mbon": total(toxin, "mbon_spikes"),
            "food_probe_avoid": total(food, "avoid_spike"), "toxin_probe_avoid": total(toxin, "avoid_spike"),
            "toxin_probe_stg_t": total(toxin, "stg_t"),
            "training_ticks": len(train),
            "food_probe_ticks": len(food),
            "toxin_probe_ticks": len(toxin),
            "expected_training_ticks": expected_training,
            "expected_probe_ticks": expected_probe,
            "trace_ticks": len(train) + len(food) + len(toxin),
            "expected_trace_ticks": expected_training + (2 * expected_probe),
            "strict_topology": bool(record["component_profile"].get("strict")),
            "component_names": list(record["component_profile"].get("components", ())),
            "neuron_count": int(record["neuron_count"]),
            "fixture_no_respawn": record["world_fixture"].get("respawn_food") is False,
            "fixture_food_count": int(record["world_fixture"].get("food_count", -1)),
            "fixture_toxin_count": int(record["world_fixture"].get("toxin_count", -1))}


EXPECTED_NEURONS = {"v2": 299, "v3": 345}


def _accept(rows, trials, version):
    failures = []
    expected_components = set(get_version(version).components)
    for world, by_seed in rows.items():
        for seed, by_case in by_seed.items():
            trained, ablated = by_case["trained"], by_case["sting_trigger_ablation"]
            prefix = f"{world} seed {seed}"
            for label, row in by_case.items():
                if (not row["strict_topology"] or row["neuron_count"] != EXPECTED_NEURONS[version]
                        or set(row["component_names"]) != expected_components):
                    failures.append(f"{prefix} {label}: non-strict or wrong-size topology ({row['neuron_count']} neurons)")
                if not row["fixture_no_respawn"] or row["fixture_food_count"] != 7 or row["fixture_toxin_count"] != 6:
                    failures.append(f"{prefix} {label}: memory fixture was changed or permits respawn")
                if (row["training_ticks"] != row["expected_training_ticks"]
                        or row["food_probe_ticks"] != row["expected_probe_ticks"]
                        or row["toxin_probe_ticks"] != row["expected_probe_ticks"]
                        or row["trace_ticks"] != row["expected_trace_ticks"]):
                    failures.append(
                        f"{prefix} {label}: incomplete phase trace "
                        f"(train {row['training_ticks']}/{row['expected_training_ticks']}, "
                        f"food {row['food_probe_ticks']}/{row['expected_probe_ticks']}, "
                        f"toxin {row['toxin_probe_ticks']}/{row['expected_probe_ticks']})"
                    )
            if trained["physical_toxin_entries"] < trials or ablated["physical_toxin_entries"] < trials:
                failures.append(f"{prefix}: physical teaching contacts were not delivered")
            if trained["training_stg_t_spikes"] == 0 or ablated["training_stg_t_spikes"] != 0:
                failures.append(f"{prefix}: sting-trigger control did not isolate the neural teaching event")
            if trained["toxin_probe_stg_t"] != 0:
                failures.append(f"{prefix}: post-training toxin probe contaminated by contact teaching")
            if trained["toxin_probe_mbon"] <= trained["food_probe_mbon"] or trained["toxin_probe_avoid"] == 0:
                failures.append(f"{prefix}: teaching did not produce toxin-selective MBON/AVOID output")
            if ablated["toxin_probe_mbon"] != 0 or ablated["toxin_probe_avoid"] != 0:
                failures.append(f"{prefix}: teaching ablation retained learned toxin output")
    return failures


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=None,
                   help="training trials; defaults to the hardest selected world")
    p.add_argument("--train-steps", type=int, default=4)
    p.add_argument("--probe-steps", type=int, default=10); p.add_argument("--substeps", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 44, 77, 101]); p.add_argument("--output", type=Path)
    p.add_argument("--version", choices=("v2", "v3"), default="v2",
                    help="strict memory-capable agent profile under test")
    p.add_argument("--world", choices=("all", *WORLD_NAMES), default="all",
                    help="teaching/probe geometry; 'all' runs centered and mirrored offsets")
    p.add_argument("--protocol", action="store_true",
                   help="require the ten-world/five-seed embodied matrix contract")
    p.add_argument("--resume", type=Path, help="complete missing seed traces in an interrupted output directory")
    return p.parse_args()


def main():
    args = parse_args()
    worlds = selected_worlds(WORLD_SPECS, args.world)
    args.trials = hardest_completion_steps(WORLD_SPECS, worlds) if args.trials is None else int(args.trials)
    if args.protocol:
        try:
            require_protocol(world_specs=WORLD_SPECS, worlds=worlds, seeds=args.seeds,
                             steps=args.trials, requested_world=args.world)
        except ValueError as exc:
            raise SystemExit(str(exc))
    if args.resume and args.output: raise SystemExit("use either --output or --resume, not both")
    output = args.resume or args.output or HERE / "results" / f"embodied_mb_valence_causal_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if args.resume:
        manifest = output / "manifest.json"
        existing = json.loads(manifest.read_text()) if manifest.exists() else {}
        if existing.get("experiment") != "embodied_mb_valence_causal" or existing.get("version") != args.version or existing.get("world") != args.world:
            raise SystemExit("--resume requires this experiment's matching version/world manifest")
        expected_manifest = {
            "seeds": args.seeds, "trials": args.trials, "train_steps": args.train_steps,
            "probe_steps": args.probe_steps, "substeps": args.substeps,
        }
        if any(existing.get(key) != value for key, value in expected_manifest.items()):
            raise SystemExit("--resume requires the manifest's exact seeds and phase lengths")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write(output / "manifest.json", {"experiment": "embodied_mb_valence_causal", "created_utc": datetime.now(timezone.utc).isoformat(),
      "source_revision": {"active_inference": _revision(ROOT), "neuron_model": _revision(NEURON_MODEL)},
      "source_fingerprints": {"aif_agent3d.py": _sha(HERE.parent / "aif_agent3d.py"), "embodied_mb_valence_causal.py": _sha(Path(__file__).resolve())},
      "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "mujoco": mujoco.__version__},
      "version": args.version, "components": list(get_version(args.version).components),
      "effective_build_parameters": True,
      "record_visual_artifacts": True,
      "world": args.world, "worlds": worlds,
      "world_catalog": WORLD_SPECS,
      "build": BUILD, "conditions": CASES, "seeds": args.seeds, "trials": args.trials, "train_steps": args.train_steps, "probe_steps": args.probe_steps, "substeps": args.substeps,
      "acceptance_protocol": protocol_metadata(world_specs=WORLD_SPECS, worlds=worlds,
                                                 seeds=args.seeds, steps=args.trials,
                                                 required=args.protocol, requested_world=args.world),
      "scope": "physical toxin-contact teaching -> PAULA sting/DAN and KC-MBON plasticity -> later non-contact physical odour probe; not long-horizon navigation", "primary_evidence": "per-condition-per-seed training and probe traces"})
    summaries = {}
    for world in worlds:
        world_output = output / world if len(worlds) > 1 else output
        world_output.mkdir(parents=True, exist_ok=True)
        summaries[world] = {}
        for seed in args.seeds:
            summaries[world][str(seed)] = {}
            for name, build in CASES.items():
                record_path = world_output / f"{name}_seed{seed}.json"
                if args.resume and record_path.exists(): record = json.loads(record_path.read_text())
                else:
                    record = run_case(seed, name, build, args.trials, args.train_steps, args.probe_steps,
                                      args.substeps, args.version, world, artifact_root=world_output)
                    _write(record_path, record)
                summaries[world][str(seed)][name] = _summary(record)
    _write(output / "summary.json", summaries); failures = _accept(summaries, args.trials, args.version)
    protocol = json.loads((output / "manifest.json").read_text()).get("acceptance_protocol", {})
    protocol_failures = list(protocol.get("failures", [])) if protocol.get("required") else []
    _write(output / "acceptance.json", {"passed": not failures and not protocol_failures, "failures": failures + protocol_failures,
                                          "version": args.version, "worlds": worlds,
                                          "protocol_compliant": bool(protocol.get("compliant", False)),
                                          "full_tick_traces": True, "strict_topology": True})
    print(f"Wrote embodied MB valence evidence to {output}")
    for world, by_seed in summaries.items():
        for seed, rows in by_seed.items():
            print(f"world={world} seed={seed}: " + "; ".join(f"{name} STG={r['training_stg_t_spikes']} MBON food/tox={r['food_probe_mbon']}/{r['toxin_probe_mbon']} AVOID={r['toxin_probe_avoid']}" for name,r in rows.items()))
    if failures or protocol_failures: print("FAIL: " + " | ".join(failures + protocol_failures), file=sys.stderr); return 1
    print("PASS: physical toxin contact causally creates toxin-selective PAULA MBON/AVOID response"); return 0

if __name__ == "__main__": raise SystemExit(main())
