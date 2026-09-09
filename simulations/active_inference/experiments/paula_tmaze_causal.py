"""Causal acceptance experiment for the PAULA T-maze neural circuit.

It tests the full PAULA brain against two pathway ablations while recording
every neural tick:

* ``cue_path_ablation`` removes the uncertainty -> cue-action synapse;
* ``evidence_ablation`` removes cue current from the belief accumulators.

The host-language world only translates location to sensory evidence and the
winning action population to a location transition.  If all action populations
are silent it remains still; Python never selects an action on their behalf.
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

import numpy as np

from simulations.active_inference import paula_aif as p


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NEURON_MODEL = ROOT.parent / "neuron-model"
CASES = {
    "full": {},
    "cue_path_ablation": {"w_epi": 0.0},
    "evidence_ablation": {"w_ev": 0.0},
}


def _revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(seed: int, name: str, build_kwargs: dict, episodes: int):
    """Run balanced contexts with a persistent PAULA brain and raw tick evidence."""
    np.random.seed(seed)
    brain = p.Brain(**build_kwargs)
    traces = []
    episode_rows = []
    for episode in range(episodes):
        context = p.RL if episode % 2 == 0 else p.RR
        steps = []
        ticks = []
        result = p.run_episode(context, brain, log=steps, tick_log=ticks)
        for row in ticks:
            row["episode"] = episode
            traces.append(row)
        episode_rows.append({
            "episode": episode,
            "context": context,
            "visited_cue": result["visited_cue"],
            "reward": result["reward"],
            "steps": steps,
        })
    return {
        "condition": name,
        "seed": seed,
        "build_kwargs": build_kwargs,
        "episodes": episode_rows,
        "tick_trace": traces,
    }


def _summary(record):
    episodes = record["episodes"]
    return {
        "episodes": len(episodes),
        "cue_visits": sum(row["visited_cue"] for row in episodes),
        "rewards": sum(row["reward"] for row in episodes),
        "neural_ticks": len(record["tick_trace"]),
    }


def _accept(summaries, episodes: int) -> list[str]:
    """Expected causal pattern; primary diagnosis remains the raw tick trace."""
    failures = []
    for seed, rows in summaries.items():
        if rows["full"]["cue_visits"] != episodes or rows["full"]["rewards"] != episodes:
            failures.append(f"seed {seed}: full circuit did not complete every cue/reward route")
        if rows["cue_path_ablation"]["cue_visits"] != 0 or rows["cue_path_ablation"]["rewards"] != 0:
            failures.append(f"seed {seed}: cue-path ablation still caused cue/reward behaviour")
        if rows["evidence_ablation"]["cue_visits"] != episodes or rows["evidence_ablation"]["rewards"] != 0:
            failures.append(f"seed {seed}: evidence ablation did not separate cue from reward")
    return failures


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=40, help="balanced RL/RR episodes per seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 44, 77])
    parser.add_argument("--output", type=Path, help="new directory for manifest and raw traces")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes <= 0 or args.episodes % 2:
        raise SystemExit("--episodes must be a positive even number for balanced contexts")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or HERE / "results" / f"paula_tmaze_causal_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "manifest.json", {
        "experiment": "paula_tmaze_causal",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": {"active_inference": _revision(ROOT), "neuron_model": _revision(NEURON_MODEL)},
        "source_fingerprints": {
            "paula_aif.py": _sha256(HERE.parent / "paula_aif.py"),
            "paula_tmaze_causal.py": _sha256(Path(__file__).resolve()),
        },
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "seeds": args.seeds,
        "episodes_per_seed": args.episodes,
        "contexts": {"0": "reward_left", "1": "reward_right"},
        "conditions": CASES,
        "primary_evidence": "tick_trace in every per-condition-per-seed JSON file",
        "acceptance": {
            "full": "cue then the context-correct arm in every balanced episode",
            "cue_path_ablation": "no cue visit and no reward",
            "evidence_ablation": "cue visit but no reward",
        },
    })
    summaries = {}
    for seed in args.seeds:
        summaries[str(seed)] = {}
        for name, kwargs in CASES.items():
            record = run_case(seed, name, kwargs, args.episodes)
            _write_json(output / f"{name}_seed{seed}.json", record)
            summaries[str(seed)][name] = _summary(record)
    _write_json(output / "summary.json", summaries)
    failures = _accept(summaries, args.episodes)
    _write_json(output / "acceptance.json", {"passed": not failures, "failures": failures})
    print(f"Wrote PAULA T-maze causal evidence to {output}")
    for seed, rows in summaries.items():
        print(f"seed={seed}: " + "; ".join(
            f"{name} cue={row['cue_visits']}/{args.episodes} reward={row['rewards']}/{args.episodes}"
            for name, row in rows.items()
        ))
    if failures:
        print("FAIL: " + " | ".join(failures), file=sys.stderr)
        return 1
    print("PASS: full circuit and both neural-pathway controls show the expected causal pattern")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
