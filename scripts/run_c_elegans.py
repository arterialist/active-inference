#!/usr/bin/env python3
"""
C. elegans active inference simulation – entry point.

Usage
-----
    cd active-inference/
    uv run python scripts/run_c_elegans.py [--steps N] [--save-plot]

Logging
-------
    --save-log       Save full run log (config.json + data.npz) for post-hoc analysis
    --log-dir PATH   Output directory (default: logs/run_<timestamp>)
    Load with:  from simulations.run_log import load_run_log

Visualization
------------
  2D:  --save-plot  saves a trajectory plot + motor wave heatmap to c_elegans_run.png
  3D:  --viewer     launches interactive MuJoCo viewer (requires display)
       On macOS:  uv run mjpython scripts/run_c_elegans.py --viewer --steps 500

The script:
  1. Loads the Cook et al. 2019 connectome (cached after first run)
  2. Builds 302 PAULA neurons wired by the connectome
  3. Initialises a MuJoCo worm body on an agar plate
  4. Runs the active inference sensorimotor loop
  5. Prints a summary and optionally saves a locomotion plot
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on path when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from simulations.c_elegans.simulation import build_c_elegans_simulation
from simulations.c_elegans.muscles import NeuromuscularJunction
from simulations.run_log import RunConfig, RunSummary, default_log_dir, save_run_log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C. elegans active inference simulation")
    p.add_argument("--steps", type=int, default=500,
                   help="Number of physics steps (default: 500)")
    p.add_argument("--food-x", type=float, default=0.005,
                   help="Food source x position in metres (default: 0.005)")
    p.add_argument("--save-plot", action="store_true",
                   help="Save locomotion trajectory and motor pattern plot")
    p.add_argument("--save-log", action="store_true",
                   help="Save full simulation log for post-hoc analysis")
    p.add_argument("--log-dir", type=str, default=None,
                   help="Log directory when using --save-log (default: logs/run_<timestamp>)")
    p.add_argument("--viewer", action="store_true",
                   help="Launch interactive 3D MuJoCo viewer (requires display)")
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
    loop.reset()  # also resets engine internally

    # ---- Run ----
    if args.viewer:
        results = _run_with_viewer(engine, loop, args.steps)
    else:
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

    # ---- Optional log ----
    if args.save_log:
        log_dir = Path(args.log_dir) if args.log_dir else default_log_dir()
        config = RunConfig(
            steps=args.steps,
            food_position=(args.food_x, 0.0, 0.0),
            use_connectome_cache=not args.no_cache,
            log_level="INFO" if args.verbose else "WARNING",
        )
        summary = RunSummary(
            steps_run=len(results),
            start_position_m=start_pos.tolist(),
            end_position_m=end_pos.tolist(),
            displacement_m=displacement,
            mean_motor_activation=mean_motor,
            mean_free_energy=loop.free_energy_trace.mean_prediction_error,
        )
        save_run_log(results, loop, config, summary, log_dir)
        print(f"\nLog saved to {log_dir.resolve()}")


def _run_with_viewer(engine, loop, n_steps: int) -> list:
    """Run simulation with interactive 3D MuJoCo viewer."""
    import mujoco.viewer

    body = engine.body
    model, data = body.model, body.data
    dt = body.dt

    results: list = []
    try:
        viewer_ctx = mujoco.viewer.launch_passive(model, data)
    except RuntimeError as e:
        if "mjpython" in str(e).lower():
            print(
                "\nOn macOS, the interactive viewer requires mjpython:\n"
                "  uv run mjpython scripts/run_c_elegans.py --viewer --steps 500\n"
                "(mjpython is installed with the mujoco package)\n"
            )
        raise
    with viewer_ctx as viewer:
        logger.info(f"Viewer launched. Running {n_steps} steps (close window to stop early).")
        for _ in range(n_steps):
            if not viewer.is_running():
                break
            step = engine.step()
            results.append(step)
            viewer.sync()
            time.sleep(dt)
    return results


def _save_plot(results, positions, motor_activations) -> None:
    """Save a 2-panel plot: trajectory + motor wave."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend for reliable saving
        import matplotlib.pyplot as plt
        from simulations.c_elegans.muscles import NeuromuscularJunction
        from simulations.c_elegans.config import N_BODY_SEGMENTS

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Trajectory (positions in biological metres)
        ax = axes[0]
        xs = positions[:, 0] * 1000  # mm
        ys = positions[:, 1] * 1000  # mm
        ax.plot(xs, ys, "b-", linewidth=2, alpha=0.8)
        ax.plot(xs[0], ys[0], "go", markersize=10, label="Start")
        ax.plot(xs[-1], ys[-1], "r^", markersize=10, label="End")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Worm trajectory (head position)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Ensure visible axis even when trajectory is tiny
        x_range = float(np.ptp(xs)) if len(xs) > 1 else 0
        y_range = float(np.ptp(ys)) if len(ys) > 1 else 0
        min_span = 0.5  # mm
        if x_range < min_span or y_range < min_span:
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            span = max(min_span, x_range, y_range)
            ax.set_xlim(cx - span / 2, cx + span / 2)
            ax.set_ylim(cy - span / 2, cy + span / 2)
        ax.set_aspect("equal")

        # Panel 2: Motor wave (dorsal - ventral activation per segment over time)
        ax = axes[1]
        n_show = min(len(results), 500)
        wave = np.zeros((n_show, N_BODY_SEGMENTS))
        for t, step in enumerate(results[-n_show:]):
            wave[t] = NeuromuscularJunction.dorsal_minus_ventral(step.motor_outputs)

        w_min, w_max = float(np.min(wave)), float(np.max(wave))
        v_abs = max(abs(w_min), abs(w_max), 0.01)  # avoid 0 range
        im = ax.imshow(
            wave.T,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-v_abs,
            vmax=v_abs,
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
