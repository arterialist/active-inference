#!/usr/bin/env python3
"""
C. elegans active inference simulation – entry point.

Usage
-----
    cd active-inference/
    python scripts/run_c_elegans.py [--steps N] [--save-plot]

The script:
  1. Loads the Cook et al. 2019 connectome (cached after first run)
  2. Builds 302 PAULA neurons wired by the connectome
  3. Initialises a MuJoCo worm body on an agar plate
  4. Runs the active inference sensorimotor loop
  5. Prints a summary and optionally saves a locomotion plot
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from simulations.c_elegans.simulation import build_c_elegans_simulation
from simulations.c_elegans.muscles import NeuromuscularJunction


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C. elegans active inference simulation")
    p.add_argument("--steps", type=int, default=500,
                   help="Number of physics steps (default: 500)")
    p.add_argument("--food-x", type=float, default=0.03,
                   help="Food source x position in metres (default: 0.03)")
    p.add_argument("--save-plot", action="store_true",
                   help="Save locomotion trajectory and motor pattern plot")
    p.add_argument("--no-cache", action="store_true",
                   help="Force fresh connectome download (ignore cache)")
    p.add_argument("--verbose", action="store_true",
                   help="Enable PAULA neuron INFO logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_level = "INFO" if args.verbose else "WARNING"

    logger.info(f"Starting C. elegans simulation: {args.steps} steps")

    # ---- Build simulation ----
    engine, loop = build_c_elegans_simulation(
        use_connectome_cache=not args.no_cache,
        food_position=(args.food_x, 0.0, 0.0),
        log_level=log_level,
    )

    # ---- Reset ----
    loop.reset()
    engine.reset()

    # ---- Run ----
    logger.info("Running sensorimotor loop …")
    results = loop.run(n_steps=args.steps, progress=True)

    # ---- Summary ----
    positions = np.array([s.body_state.position for s in results])
    start_pos = positions[0] if len(positions) > 0 else np.zeros(3)
    end_pos = positions[-1] if len(positions) > 0 else np.zeros(3)
    displacement = float(np.linalg.norm(end_pos - start_pos))

    motor_activations = [
        NeuromuscularJunction.mean_activation(s.motor_outputs) for s in results
    ]
    mean_motor = float(np.mean(motor_activations)) if motor_activations else 0.0

    print("\n" + "=" * 60)
    print("Simulation complete")
    print("=" * 60)
    print(f"Steps run         : {len(results)}")
    print(f"Start position    : ({start_pos[0]*1000:.2f}, {start_pos[1]*1000:.2f}) mm")
    print(f"End position      : ({end_pos[0]*1000:.2f}, {end_pos[1]*1000:.2f}) mm")
    print(f"Total displacement: {displacement*1000:.3f} mm")
    print(f"Mean motor act.   : {mean_motor:.4f}")
    print(
        f"Mean free energy  : "
        f"{loop.free_energy_trace.mean_prediction_error:.6f}"
    )
    print("=" * 60)

    # ---- Optional plot ----
    if args.save_plot:
        _save_plot(results, positions, motor_activations)


def _save_plot(results, positions, motor_activations) -> None:
    """Save a 2-panel plot: trajectory + motor wave."""
    try:
        import matplotlib.pyplot as plt
        from simulations.c_elegans.muscles import NeuromuscularJunction
        from simulations.c_elegans.config import N_BODY_SEGMENTS

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Trajectory
        ax = axes[0]
        xs = positions[:, 0] * 1000  # mm
        ys = positions[:, 1] * 1000  # mm
        ax.plot(xs, ys, "b-", linewidth=1.5, alpha=0.7)
        ax.plot(xs[0], ys[0], "go", markersize=8, label="Start")
        ax.plot(xs[-1], ys[-1], "r^", markersize=8, label="End")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Worm trajectory (head position)")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Panel 2: Motor wave (dorsal - ventral activation per segment over time)
        ax = axes[1]
        n_show = min(len(results), 200)
        wave = np.zeros((n_show, N_BODY_SEGMENTS))
        for t, step in enumerate(results[-n_show:]):
            wave[t] = NeuromuscularJunction.dorsal_minus_ventral(step.motor_outputs)

        im = ax.imshow(
            wave.T,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            origin="lower",
        )
        plt.colorbar(im, ax=ax, label="Dorsal − Ventral")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Body segment")
        ax.set_title("Motor wave (D−V activation)")

        plt.tight_layout()
        out_path = Path("c_elegans_run.png")
        plt.savefig(out_path, dpi=120)
        print(f"\nPlot saved to {out_path.resolve()}")
        plt.close()

    except ImportError:
        print("matplotlib not available — skipping plot")
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == "__main__":
    main()
