"""Trace the configuration control required before embodied circuit tuning.

This creates both required contexts for the configuration component:

* an isolated, stationary ring-maintenance trace; and
* a MuJoCo closed-loop trace using the same agent configuration.

It compares the canonical configuration (which explicitly preserves the old
``run_episode`` tonic of 1.0) against the old direct/live default of zero
tonic.  The traces, not the summaries, are the primary evidence.  No result
from this runner establishes compass or navigation performance.

Run from the active-inference root:

    uv run python -m simulations.active_inference.experiments.config_reconciliation
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NEURON_MODEL = ROOT.parent / "neuron-model"


def _load_agent_module():
    spec = importlib.util.spec_from_file_location(
        "aif_agent3d_config_experiment", HERE.parent / "aif_agent3d.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _ring_spikes(agent, ag) -> int:
    return sum(int(agent.nb[nid].O > 0) for nid in ag.cc.RING)


def isolated_trace(ag, config, seed: int, ticks: int):
    """Ring-only context: no body motion, visual input, or physical update."""
    np.random.seed(seed)
    agent = ag.AIFAgent3D(seed=seed, config=config)
    agent.birth()
    trace = []
    for t in range(ticks):
        tonic = agent.ring_tonic_current(0.0, 0.0)
        agent.tick(ccw=0.0, cw=0.0, speed=0.0, vision=False)
        trace.append({
            "t": t,
            "ccw_drive": 0.0,
            "cw_drive": 0.0,
            "speed_drive": 0.0,
            "tonic_current": tonic,
            "ring_spikes": _ring_spikes(agent, ag),
        })
    return trace


def embodied_trace(ag, config, seed: int, ticks: int):
    """Closed-loop context, sampled once on every neural/physics tick."""
    np.random.seed(seed)
    agent = ag.AIFAgent3D(seed=seed, config=config)
    agent.birth()
    trace = []
    tonic = {"ccw": 0.0, "cw": 0.0, "current": 0.0}
    drive = agent.drive_ring_tonic

    def capture_tonic(ccw, cw):
        current = drive(ccw, cw)
        tonic.update(ccw=float(ccw), cw=float(cw), current=float(current))
        return current

    def capture_tick(current_agent):
        x, y, yaw = current_agent.world.pose()
        trace.append({
            "t": current_agent.t,
            "ccw_drive": tonic["ccw"],
            "cw_drive": tonic["cw"],
            "tonic_current": tonic["current"],
            "ring_spikes": _ring_spikes(current_agent, ag),
            "yaw": float(yaw),
            "speed": float(current_agent.world.speed()),
            "x": float(x),
            "y": float(y),
            "home_distance": float(current_agent.world.dist_home()),
            "arena_distance": float(np.hypot(x, y)),
        })

    agent.drive_ring_tonic = capture_tonic
    ag.run_episode(
        agent,
        steps=ticks,
        sub=1,
        render_every=10**9,
        render_ticks=0,
        log_every=10**9,
        tick_hook=capture_tick,
    )
    return trace


def _summary(trace):
    # A ring cell is refractory after a spike, so requiring one or more ring
    # spikes on *every individual tick* mistakes ordinary sparse dynamics for
    # a dead attractor.  Use the lab-standard short temporal window for this
    # secondary status field; the raw per-tick trace remains primary evidence.
    active = [row["ring_spikes"] > 0 for row in trace]
    window = 8
    windowed = [any(active[max(0, i - window + 1):i + 1]) for i in range(len(active))]
    longest_silent = 0
    silent = 0
    for is_active in active:
        silent = 0 if is_active else silent + 1
        longest_silent = max(longest_silent, silent)
    return {
        "ticks": len(trace),
        "ring_active_window": window,
        "ring_windowed_live_fraction": float(np.mean(windowed)) if windowed else 0.0,
        "first_full_silent_window_tick": next(
            (trace[i]["t"] for i, is_live in enumerate(windowed) if i >= window - 1 and not is_live),
            None,
        ),
        "longest_silent_run": longest_silent,
        "max_arena_distance": max((row.get("arena_distance", 0.0) for row in trace), default=0.0),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=320, help="neural ticks per context and seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 44, 77])
    parser.add_argument("--output", type=Path, help="new directory for trace and manifest files")
    parser.add_argument(
        "--summarize-existing", type=Path,
        help="write summary_windowed.json from an existing trace directory without rerunning it",
    )
    return parser.parse_args()


def summarize_existing(output: Path) -> int:
    """Create the corrected secondary summary without changing primary traces."""
    summary_path = output / "summary_windowed.json"
    if summary_path.exists():
        raise SystemExit(f"Refusing to overwrite existing {summary_path}")
    summaries = {}
    for trace_path in sorted(output.glob("isolated_*_seed*.json")) + sorted(output.glob("embodied_*_seed*.json")):
        context, remainder = trace_path.stem.split("_", 1)
        label, seed = remainder.rsplit("_seed", 1)
        rows = json.loads(trace_path.read_text())
        summaries.setdefault(f"{label}/seed{seed}", {})[context] = _summary(rows)
    if not summaries:
        raise SystemExit(f"No configuration trace files found in {output}")
    _write_json(summary_path, summaries)
    print(f"Wrote corrected windowed summary to {summary_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.summarize_existing:
        return summarize_existing(args.summarize_existing)
    if args.ticks <= 0:
        raise SystemExit("--ticks must be positive")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or HERE / "results" / f"config_reconciliation_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    ag = _load_agent_module()
    conditions = {
        "canonical": ag.DEFAULT_EMBODIED_CONFIG,
        "legacy_direct_tick": ag.LEGACY_DIRECT_TICK_CONFIG,
    }
    manifest = {
        "experiment": "embodied_configuration_reconciliation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": {"active_inference": _revision(ROOT), "neuron_model": _revision(NEURON_MODEL)},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
        },
        "seeds": args.seeds,
        "ticks_per_context": args.ticks,
        "conditions": {name: config.manifest() for name, config in conditions.items()},
        "primary_evidence": "one JSON row per neural tick in each trace file",
        "not_established": [
            "compass accuracy", "path integration", "homing", "value learning", "action selection"
        ],
    }
    _write_json(output / "manifest.json", manifest)
    summaries = {}
    for label, config in conditions.items():
        for seed in args.seeds:
            isolated = isolated_trace(ag, config, seed, args.ticks)
            embodied = embodied_trace(ag, config, seed, args.ticks)
            _write_json(output / f"isolated_{label}_seed{seed}.json", isolated)
            _write_json(output / f"embodied_{label}_seed{seed}.json", embodied)
            summaries[f"{label}/seed{seed}"] = {
                "isolated": _summary(isolated),
                "embodied": _summary(embodied),
            }
    _write_json(output / "summary.json", summaries)
    print(f"Wrote configuration traces to {output}")
    for name, result in summaries.items():
        print(f"{name}: isolated windowed-live={result['isolated']['ring_windowed_live_fraction']:.3f}; "
              f"embodied windowed-live={result['embodied']['ring_windowed_live_fraction']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
