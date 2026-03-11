#!/usr/bin/env python3
"""
Analyze whether neuromodulation (M0/M1) drove the worm toward food despite causing issues.

Metrics:
  - Distance to food over time (head position vs food position)
  - Chemical concentration trend (NaCl + butanone as attractant proxy)
  - dC/dt: positive = moving toward food, negative = moving away
  - Fraction of steps with dC/dt > 0 (approach) vs < 0 (retreat)

Usage:
  cd active-inference/
  uv run python scripts/analyze_neuromod_food_seeking.py [--run-dir PATH] [--compare]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_log_data(log_dir: Path) -> tuple[dict, dict] | None:
    """Load config and data from a run log. Handles both npz and CSV formats."""
    log_dir = Path(log_dir)
    config_path = log_dir / "config.json"
    if not config_path.exists():
        return None

    with open(config_path) as f:
        config = json.load(f)

    # Try npz first
    npz_path = log_dir / "data.npz"
    if npz_path.exists():
        npz = np.load(npz_path, allow_pickle=False)
        data = {
            "tick": npz["tick"],
            "position": npz["position"],
            "head_position": npz.get("head_position", npz["position"]),
            "chemical": npz.get("chemical", None),
        }
        return config, data

    # Fall back to CSV
    pos_path = log_dir / "positions.csv"
    chem_path = log_dir / "chemical.csv"
    if not pos_path.exists():
        return None

    positions = np.loadtxt(pos_path, delimiter=",", skiprows=1)
    ticks = positions[:, 0].astype(int)
    pos = positions[:, 1:4]  # x, y, z

    head_path = log_dir / "head_position.csv"
    if head_path.exists():
        head_data = np.loadtxt(head_path, delimiter=",", skiprows=1)
        head_pos = head_data[:, 1:4]
    else:
        head_pos = pos

    chemicals = None
    chem_names = None
    if chem_path.exists():
        chem_data = np.loadtxt(chem_path, delimiter=",", skiprows=1)
        chemicals = chem_data[:, 1:]  # skip tick
        with open(chem_path) as f:
            chem_names = f.readline().strip().split(",")[1:]
        config.setdefault("data_keys", {})["chem_names"] = chem_names

    data = {
        "tick": ticks,
        "position": pos,
        "head_position": head_pos,
        "chemical": chemicals,
    }
    return config, data


def get_food_position(config: dict) -> np.ndarray | None:
    """Extract first food position from config. Returns None if no food."""
    cfg = config.get("config", config)
    fp = cfg.get("food_positions") or cfg.get("food_position")
    if fp is None or len(fp) == 0:
        return None
    arr = np.array(fp[0] if isinstance(fp[0], (list, tuple)) else fp)
    return arr.reshape(3)


def _neuromod_status(config: dict) -> str | bool | None:
    """Extract neuromodulation status; supports enable_m0/enable_m1 and legacy neuromodulation."""
    cfg = config.get("config", config)
    if "enable_m0" in cfg or "enable_m1" in cfg:
        m0 = cfg.get("enable_m0", True)
        m1 = cfg.get("enable_m1", True)
        return m0 and m1
    return cfg.get("neuromodulation")


def analyze_food_seeking(config: dict, data: dict) -> dict:
    """Compute food-seeking metrics from run data."""
    food = get_food_position(config)
    head = data["head_position"]
    n = len(head)

    if food is None:
        dist = np.zeros(n)  # No food to measure distance to
    else:
        dist = np.linalg.norm(head - food, axis=1)

    # Chemical: use NaCl + butanone (attractants). data_keys has column order.
    chem = data.get("chemical")
    d_keys = config.get("data_keys", {})
    chem_names = d_keys.get("chem_names", ["2-nonanone", "NaCl", "butanone"])

    # Attractant = NaCl + butanone (indices 1 and 2 typically)
    attractant = None
    if chem is not None and len(chem_names) >= 3:
        nacl_idx = next((i for i, n in enumerate(chem_names) if n == "NaCl"), 1)
        but_idx = next((i for i, n in enumerate(chem_names) if n == "butanone"), 2)
        attractant = chem[:, nacl_idx] + chem[:, but_idx]

    result = {
        "n_steps": n,
        "start_dist_m": float(dist[0]),
        "end_dist_m": float(dist[-1]),
        "min_dist_m": float(np.min(dist)),
        "dist_improvement": float(dist[0] - dist[-1]) if food is not None else 0.0,
        "neuromodulation": _neuromod_status(config),
    }

    if attractant is not None:
        dC = np.diff(attractant)
        steps_toward = np.sum(dC > 0)
        steps_away = np.sum(dC < 0)
        result["steps_toward_food"] = int(steps_toward)
        result["steps_away_food"] = int(steps_away)
        result["frac_toward"] = float(steps_toward / (len(dC) or 1))
        result["chem_start"] = float(attractant[0])
        result["chem_end"] = float(attractant[-1])
        result["chem_change"] = float(attractant[-1] - attractant[0])

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze neuromodulation effect on food-seeking"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Single run directory to analyze",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare neuromod vs no-neuromod from logs/food_reach_*",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available log dirs with neuromodulation info",
    )
    args = parser.parse_args()

    logs_dir = Path(__file__).resolve().parents[1] / "logs"

    if args.list:
        dirs = sorted(logs_dir.iterdir())
        print("Available log directories:\n")
        for d in dirs:
            if not d.is_dir():
                continue
            cfg_path = d / "config.json"
            if not cfg_path.exists():
                continue
            with open(cfg_path) as f:
                cfg = json.load(f)
            neuromod = _neuromod_status(cfg) if "config" in cfg else "?"
            summary = cfg.get("summary", {})
            start = summary.get("start_position_m", [0, 0, 0])
            end = summary.get("end_position_m", [0, 0, 0])
            food_cfg = cfg.get("config", cfg).get("food_positions") or cfg.get("config", cfg).get("food_position")
            fp = food_cfg[0] if food_cfg and len(food_cfg) > 0 else []
            fp = fp if isinstance(fp, (list, tuple)) else [fp]
            has_data = (d / "data.npz").exists() or (d / "positions.csv").exists()
            print(f"  {d.name}: neuromod={neuromod}, data={has_data}, food={fp}")
        return

    if args.compare:
        # Compare food_reach_neuromod vs food_reach_no_neuromod (summary only - they lack full data)
        neuromod_dir = logs_dir / "food_reach_neuromod"
        no_neuromod_dir = logs_dir / "food_reach_no_neuromod"
        for name, p in [("neuromod", neuromod_dir), ("no_neuromod", no_neuromod_dir)]:
            if not (p / "config.json").exists():
                print(f"Missing {p}")
                continue
            with open(p / "config.json") as f:
                cfg = json.load(f)
            food_cfg = cfg["config"].get("food_positions") or cfg["config"].get("food_position")
            if not food_cfg or len(food_cfg) == 0:
                continue
            food = np.array(food_cfg[0] if isinstance(food_cfg[0], (list, tuple)) else food_cfg)
            start = np.array(cfg["summary"]["start_position_m"])
            end = np.array(cfg["summary"]["end_position_m"])
            d_start = np.linalg.norm(start - food)
            d_end = np.linalg.norm(end - food)
            improvement = d_start - d_end
            print(f"\n{name}:")
            print(f"  Food: {food}")
            print(f"  Start dist: {d_start:.6f} m")
            print(f"  End dist:   {d_end:.6f} m")
            print(f"  Improvement: {improvement:.6f} m (positive = got closer)")

        # Also analyze runs with full data
        for run_name in ["final_close_food", "my_test", "alerm_fixed"]:
            p = logs_dir / run_name
            loaded = load_log_data(p)
            if loaded is None:
                continue
            config, data = loaded
            metrics = analyze_food_seeking(config, data)
            print(f"\n{run_name} (full data):")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
        return

    if args.run_dir:
        loaded = load_log_data(args.run_dir)
        if loaded is None:
            print(f"Could not load {args.run_dir}")
            sys.exit(1)
        config, data = loaded
        metrics = analyze_food_seeking(config, data)
        print("\nFood-seeking analysis:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        return

    # Default: analyze all runs with full data
    print("Analyzing runs with full position/chemical data...\n")
    for d in sorted(logs_dir.iterdir()):
        if not d.is_dir():
            continue
        loaded = load_log_data(d)
        if loaded is None:
            continue
        config, data = loaded
        if data.get("chemical") is None:
            continue
        metrics = analyze_food_seeking(config, data)
        neuromod = "ON" if metrics.get("neuromodulation") else "OFF" if metrics.get("neuromodulation") is False else "?"
        print(f"=== {d.name} (neuromod={neuromod}) ===")
        print(f"  Steps: {metrics['n_steps']}")
        print(f"  Distance: start={metrics['start_dist_m']:.6f} m -> end={metrics['end_dist_m']:.6f} m")
        print(f"  Improvement: {metrics['dist_improvement']:.6f} m")
        if "frac_toward" in metrics:
            print(f"  Steps toward food: {metrics['steps_toward_food']}/{metrics['n_steps']-1} ({100*metrics['frac_toward']:.1f}%)")
            print(f"  Chem: {metrics['chem_start']:.4f} -> {metrics['chem_end']:.4f} (Δ={metrics['chem_change']:+.4f})")
        print()


if __name__ == "__main__":
    main()
