#!/usr/bin/env python3
"""Run the larval zebrafish active-inference simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulations.zebrafish.simulation import build_zebrafish_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Larval zebrafish active inference simulation")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-neural-states", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine, loop = build_zebrafish_simulation(
        record_neural_states=not args.no_neural_states,
        suppress_connectome_summary=False,
        seed=args.seed,
    )
    loop.reset()
    positions = np.zeros((args.steps, 3))
    speeds = np.zeros(args.steps)
    drives = np.zeros(args.steps)
    turns = np.zeros(args.steps)
    for i in range(args.steps):
        step = engine.step()
        positions[i] = step.body_state.position
        speeds[i] = float(step.body_state.extra.get("speed_m_s", 0.0))
        drives[i] = float(step.body_state.extra.get("swim_drive", 0.0))
        turns[i] = float(step.body_state.extra.get("turn_bias", 0.0))

    displacement = float(np.linalg.norm(positions[-1, :2] - positions[0, :2]))
    path = float(np.sum(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)))
    print("Zebrafish simulation complete")
    print(f"  steps: {args.steps}")
    print(f"  displacement_m: {displacement:.5f}")
    print(f"  path_length_m: {path:.5f}")
    print(f"  mean_speed_m_s: {float(np.mean(speeds)):.5f}")
    print(f"  max_speed_m_s: {float(np.max(speeds)):.5f}")
    print(f"  mean_drive: {float(np.mean(drives)):.3f}")
    print(f"  mean_abs_turn: {float(np.mean(np.abs(turns))):.3f}")


if __name__ == "__main__":
    main()

