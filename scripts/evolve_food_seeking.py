#!/usr/bin/env python3
"""
Evolutionary optimization of neuromodulation and neuron params for food-seeking.

Robustness protocol: each genome is tested against 4 distinct food positions
to prevent directional overfitting ("lucky torpedo"). Fitness = average of
min_distance achieved per environment (or worst-case with --robust).
Uses min_distance over the run, not final tick, to avoid "final tick trap".

Usage:
  cd active-inference/
  uv run python scripts/evolve_food_seeking.py [--generations N] [--population M]
  uv run python scripts/evolve_food_seeking.py --low-memory   # Raspberry Pi / 8GB RAM

Checkpointing:
  Best config is saved every N generations (--checkpoint-every) and on Ctrl+C.
  Use the checkpoint for simulation at any time (even while evolution runs):
    uv run python scripts/run_c_elegans.py --evol-config evolved_food_seeking_checkpoint.json --viewer
  On crash, recover with: uv run python scripts/apply_evolved_config.py --config evolved_food_seeking_checkpoint.json
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import signal
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Suppress simulation logging during evolution
import loguru
loguru.logger.remove()
loguru.logger.add(sys.stderr, level="ERROR")

N_TICKS = 50_000
EAT_RADIUS_M = 0.0001  # Early exit if worm gets this close (0.1 mm)


def _rss_mb() -> float:
    """Peak resident set size in MB (Unix). macOS: bytes, Linux: KB."""
    r = resource.getrusage(resource.RUSAGE_SELF)
    rss = r.ru_maxrss
    return (rss / 1024 / 1024) if sys.platform == "darwin" else (rss / 1024)

# 4 distinct targets to prevent directional overfitting ("lucky torpedo")
TEST_ENVIRONMENTS = [
    (0.002, 0.002, 0.0),   # Top-right
    (-0.002, 0.002, 0.0),  # Top-left
    (0.002, -0.002, 0.0),  # Bottom-right
    (0.0, 0.003, 0.0),     # Straight ahead
]


def x_to_config(x: np.ndarray) -> dict:
    """Map normalized [0,1]^d vector to evol_config."""
    # Bounds for each parameter
    # 0: K_STRESS_SYN [1000, 8000]
    # 1: K_REWARD_SYN [1000, 8000]
    # 2: K_VOL_STRESS [500, 4000]
    # 3: K_VOL_REWARD [500, 4000]
    # 4: STRESS_DEADZONE log [1e-6, 0.01]
    # 5: CHEM_EMA_ALPHA [0.01, 0.99] — temporal filter, critical for phase lag
    # 6: TONIC_FWD_CMD [0.1, 0.5]
    # 7: TONIC_FWD_MOTOR [0.05, 0.2]
    # 8: motor w_tref M0 [15, 45]
    # 9: motor w_tref M1 [-25, -5]
    # 10: sensory w_tref M0 [8, 25]
    # 11: sensory w_tref M1 [-12, -3]
    cfg: dict = {
        "K_STRESS_SYN": 1000 + 7000 * x[0],
        "K_REWARD_SYN": 1000 + 7000 * x[1],
        "K_VOL_STRESS": 500 + 3500 * x[2],
        "K_VOL_REWARD": 500 + 3500 * x[3],
        "STRESS_DEADZONE": 1e-6 * (0.01 / 1e-6) ** x[4],
        "CHEM_EMA_ALPHA": 0.01 + 0.98 * x[5],
        "TONIC_FWD_CMD": 0.1 + 0.4 * x[6],
        "TONIC_FWD_MOTOR": 0.05 + 0.15 * x[7],
    }
    cfg["neuron_params"] = {
        "motor": {"w_tref": [15 + 30 * x[8], -5 - 20 * x[9]]},
        "sensory": {"w_tref": [8 + 17 * x[10], -3 - 9 * x[11]]},
    }
    return cfg


def _save_checkpoint(
    best_x: np.ndarray | None,
    best_dist: float,
    gen: int,
    n_evals: int,
    elapsed: float,
    out_path: Path,
    evol_args: dict | None = None,
) -> None:
    """Atomically save best config + metadata. Safe to call on crash/signal.

    Saves full simulation-ready config so you can run with --evol-config at any time:
      uv run python scripts/run_c_elegans.py --evol-config evolved_food_seeking_checkpoint.json --viewer
    """
    if best_x is None:
        return
    best_config = x_to_config(np.clip(best_x, 0, 1))
    payload = {
        # Simulation-ready: use this file directly with --evol-config
        "config": best_config,
        "best_config": best_config,
        "best_x": best_x.tolist(),
        "best_dist_mm": float(best_dist * 1000),
        "generation": gen,
        "n_evals": n_evals,
        "elapsed_min": elapsed / 60,
    }
    if evol_args is not None:
        payload["evol_args"] = evol_args
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.rename(out_path)


def evaluate(
    x: np.ndarray,
    n_ticks: int = N_TICKS,
    eval_counter: list[int] | None = None,
    show_sim_progress: bool = False,
    use_worst: bool = False,
    low_memory: bool = False,
) -> float:
    """
    Run simulation against 4 distinct food positions. Return average (or worst)
    of min_distance achieved in each run. DE minimizes this (lower = better).
    """
    from simulations.c_elegans.simulation import build_c_elegans_simulation

    evol_config = x_to_config(np.clip(x, 0, 1))
    min_distances: list[float] = []

    for env_idx, food_pos in enumerate(TEST_ENVIRONMENTS):
        try:
            engine, loop = build_c_elegans_simulation(
                use_connectome_cache=True,
                food_positions=[food_pos],
                log_level="ERROR",
                record_neural_states=False,
                evol_config=evol_config,
                max_history=1 if low_memory else 200,
                suppress_connectome_summary=low_memory,
            )
        except Exception:
            return 1e6  # Penalize failed builds (large distance)

        loop.reset()
        food = np.array(food_pos)
        min_dist = float("inf")

        tick_range = range(n_ticks - 1)
        if show_sim_progress:
            tick_range = tqdm(
                tick_range,
                unit="tick",
                desc=f"Sim env{env_idx+1}",
                position=1,
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
        for _ in tick_range:
            step = engine.step()
            head = step.body_state.head_position
            d = float(np.linalg.norm(head - food))
            min_dist = min(min_dist, d)
            if min_dist < EAT_RADIUS_M:
                break  # Successfully reached food
        min_distances.append(min_dist)

        if low_memory:
            del engine, loop
            gc.collect()

    if eval_counter is not None:
        eval_counter[0] += 1

    if low_memory:
        gc.collect()

    # Average across environments (or worst-case for robustness)
    return float(np.max(min_distances) if use_worst else np.mean(min_distances))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolve neuromod params for food-seeking")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--ticks", type=int, default=N_TICKS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", help="Scipy DE verbose output")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logging")
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Score on worst environment (max min_dist) instead of average",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="evolved_food_seeking_checkpoint.json",
        help="Path for periodic checkpoints (default: evolved_food_seeking_checkpoint.json)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save checkpoint every N generations (default: 1)",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Keep RAM flat for Raspberry Pi / 8GB: no sim progress bars, gc after each eval, min history",
    )
    parser.add_argument(
        "--measure-memory",
        action="store_true",
        help="Print peak RSS (MB) each generation",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).parent.parent / checkpoint_path

    np.random.seed(args.seed)
    n_dim = 12
    eval_counter: list[int] = [0]
    gen_counter: list[int] = [0]
    start_time = time.perf_counter()

    if not args.quiet:
        print("=" * 60)
        print("Evolution: food-seeking optimization (robustness protocol)")
        print("=" * 60)
        print(f"  generations={args.generations}  population={args.population}  ticks={args.ticks}")
        print(f"  {len(TEST_ENVIRONMENTS)} targets: avg min_dist" + (" (worst)" if args.robust else ""))
        print(f"  checkpoint: {checkpoint_path} (every {args.checkpoint_every} gen)")
        if args.low_memory:
            print("  low-memory: enabled (flat RAM for Pi/8GB)")
        print("-" * 60)

    # Approximate total evals: DE does ~popsize + generations*popsize, use 2x buffer
    total_evals = args.population * (args.generations + 1) * 2
    pbar = tqdm(
        total=total_evals,
        disable=args.quiet,
        unit="eval",
        desc="Evolution",
        position=0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    # Wrap to capture best (lowest) distance; DE minimizes directly
    class EvalWrapper:
        def __init__(self):
            self.best_dist = np.inf
            self.best_x = None

        def __call__(self, x):
            dist = evaluate(
                x,
                n_ticks=args.ticks,
                eval_counter=eval_counter,
                show_sim_progress=not args.quiet and not args.low_memory,
                use_worst=args.robust,
                low_memory=args.low_memory,
            )
            if dist < self.best_dist:
                self.best_dist = dist
                self.best_x = x.copy()
            if not args.quiet:
                pbar.update(1)
                postfix = {"best_dist": f"{self.best_dist*1000:.2f}mm"}
                if args.measure_memory:
                    postfix["RSS"] = f"{_rss_mb():.0f}MB"
                pbar.set_postfix(postfix)
            return dist  # DE minimizes distance directly

    wrapper = EvalWrapper()
    _abort_requested = [False]  # Mutable for signal handler
    evol_args = {
        "generations": args.generations,
        "population": args.population,
        "ticks": args.ticks,
        "robust": args.robust,
        "seed": args.seed,
    }

    def _on_sigint(signum, frame):
        _abort_requested[0] = True
        if wrapper.best_x is not None:
            elapsed = time.perf_counter() - start_time
            _save_checkpoint(
                wrapper.best_x,
                wrapper.best_dist,
                gen_counter[0],
                eval_counter[0],
                elapsed,
                checkpoint_path,
                evol_args=evol_args,
            )
            tqdm.write(f"\n[Ctrl+C] Checkpoint saved to {checkpoint_path}")
        sys.exit(130)

    signal.signal(signal.SIGINT, _on_sigint)

    def _progress_callback(xk, convergence):
        gen_counter[0] += 1
        # Checkpoint every N generations
        if wrapper.best_x is not None and gen_counter[0] % args.checkpoint_every == 0:
            elapsed = time.perf_counter() - start_time
            _save_checkpoint(
                wrapper.best_x,
                wrapper.best_dist,
                gen_counter[0],
                eval_counter[0],
                elapsed,
                checkpoint_path,
                evol_args=evol_args,
            )
        if args.quiet:
            return
        elapsed = time.perf_counter() - start_time
        n_evals = eval_counter[0]
        best_d = wrapper.best_dist if wrapper.best_x is not None else float("nan")
        evals_per_sec = n_evals / elapsed if elapsed > 0 else 0
        mem = f" | peak {_rss_mb():.0f}MB" if args.measure_memory else ""
        tqdm.write(
            f"  gen {gen_counter[0]:3d}/{args.generations} | "
            f"evals {n_evals:5d} | best {best_d*1000:.2f}mm | "
            f"conv {convergence:.3f} | {elapsed/60:.1f}m | {evals_per_sec:.1f} ev/s{mem}"
        )

    # Differential evolution (scipy)
    from scipy.optimize import differential_evolution

    bounds = [(0.0, 1.0)] * n_dim
    result = differential_evolution(
        wrapper,
        bounds,
        strategy="best1bin",
        maxiter=args.generations,
        popsize=args.population,
        seed=args.seed,
        disp=args.verbose,
        polish=False,
        atol=0,
        tol=0,
        callback=_progress_callback,
    )

    if not args.quiet:
        pbar.close()

    best_x = np.clip(result.x, 0, 1)
    best_config = x_to_config(best_x)
    best_distance = result.fun  # DE minimized distance
    total_time = time.perf_counter() - start_time

    print("\n" + "=" * 60)
    print("Evolution complete")
    print("=" * 60)
    print(f"Best avg min_dist: {best_distance*1000:.2f} mm")
    print(f"Total evals: {eval_counter[0]}, time: {total_time/60:.1f} min")
    if args.measure_memory:
        print(f"Peak RSS: {_rss_mb():.0f} MB")
    print("\nBest neuromod config:")
    for k, v in best_config.items():
        if k != "neuron_params":
            print(f"  {k}: {v}")
    print("\nBest neuron_params:")
    for k, v in best_config.get("neuron_params", {}).items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Write best config to file for easy application
    out_path = Path(__file__).parent.parent / "evolved_food_seeking_config.json"
    with open(out_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\nConfig saved to {out_path}")


if __name__ == "__main__":
    main()
