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
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from simulations.c_elegans.simulation import build_c_elegans_simulation
from simulations.c_elegans.muscles import NeuromuscularJunction
from simulations.c_elegans.config import N_BODY_SEGMENTS
from simulations.run_log import RunConfig, RunSummary, default_log_dir, save_run_log
from simulations.engine import SimulationStep


class StreamingCollector:
    """Pre-allocates numpy arrays and fills them step-by-step.

    Avoids holding thousands of heavyweight SimulationStep dicts in memory.
    After the run, provides the same data that save_run_log needs.
    """

    def __init__(self, n_steps: int, record_neural: bool = False):
        self._n = n_steps
        self._i = 0
        self._record_neural = record_neural

        self.positions = np.zeros((n_steps, 3))
        self.head_positions = np.zeros((n_steps, 3))
        self.elapsed_ms = np.zeros(n_steps)
        self.ticks = np.zeros(n_steps, dtype=np.int64)

        self._joint_names: list[str] | None = None
        self._motor_names: list[str] | None = None
        self._chem_names: list[str] | None = None
        self._neuron_names: list[str] | None = None

        self.joint_angles: np.ndarray | None = None
        self.motor_outputs: np.ndarray | None = None
        self.chemicals: np.ndarray | None = None
        self.neural_S: np.ndarray | None = None
        self.neural_fired: np.ndarray | None = None

    def _init_from_step(self, step: SimulationStep) -> None:
        n = self._n
        self._joint_names = sorted(step.body_state.joint_angles.keys())
        self.joint_angles = np.zeros((n, len(self._joint_names)))

        self._motor_names = sorted(step.motor_outputs.keys())
        self.motor_outputs = np.zeros((n, len(self._motor_names)))

        self._chem_names = sorted(step.observation.chemicals.keys())
        self.chemicals = np.zeros((n, len(self._chem_names)))

        if self._record_neural and step.neural_states:
            self._neuron_names = sorted(
                k[:-2] for k in step.neural_states if k.endswith("_S")
            )
            self.neural_S = np.zeros((n, len(self._neuron_names)))
            self.neural_fired = np.zeros((n, len(self._neuron_names)))

    def record(self, step: SimulationStep) -> None:
        i = self._i
        if i >= self._n:
            return

        if self._joint_names is None:
            self._init_from_step(step)

        self.positions[i] = step.body_state.position
        self.head_positions[i] = step.body_state.head_position
        self.elapsed_ms[i] = step.elapsed_ms
        self.ticks[i] = step.tick

        for j, name in enumerate(self._joint_names):
            self.joint_angles[i, j] = step.body_state.joint_angles.get(name, 0.0)

        for j, name in enumerate(self._motor_names):
            self.motor_outputs[i, j] = step.motor_outputs.get(name, 0.0)

        for j, name in enumerate(self._chem_names):
            self.chemicals[i, j] = step.observation.chemicals.get(name, 0.0)

        if self._record_neural and self._neuron_names and step.neural_states:
            for j, name in enumerate(self._neuron_names):
                self.neural_S[i, j] = step.neural_states.get(f"{name}_S", 0.0)
                self.neural_fired[i, j] = step.neural_states.get(f"{name}_fired", 0.0)

        self._i += 1

    @property
    def n_recorded(self) -> int:
        return self._i

    def trim(self) -> None:
        """Trim arrays to actual recorded length (handles early stop)."""
        n = self._i
        self.positions = self.positions[:n]
        self.head_positions = self.head_positions[:n]
        self.elapsed_ms = self.elapsed_ms[:n]
        self.ticks = self.ticks[:n]
        if self.joint_angles is not None:
            self.joint_angles = self.joint_angles[:n]
        if self.motor_outputs is not None:
            self.motor_outputs = self.motor_outputs[:n]
        if self.chemicals is not None:
            self.chemicals = self.chemicals[:n]
        if self.neural_S is not None:
            self.neural_S = self.neural_S[:n]
        if self.neural_fired is not None:
            self.neural_fired = self.neural_fired[:n]

    def dorsal_minus_ventral_wave(self, last_n: int = 500) -> np.ndarray:
        """Compute D-V wave from raw motor output names."""
        if self.motor_outputs is None or self._motor_names is None:
            return np.zeros((0, N_BODY_SEGMENTS))
        n = min(self._i, last_n)
        start = max(0, self._i - last_n)
        wave = np.zeros((n, N_BODY_SEGMENTS))
        motor_dict: dict[str, float] = {}
        for t_idx in range(n):
            motor_dict.clear()
            row = self.motor_outputs[start + t_idx]
            for j, name in enumerate(self._motor_names):
                motor_dict[name] = float(row[j])
            wave[t_idx] = NeuromuscularJunction.dorsal_minus_ventral(motor_dict)
        return wave


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C. elegans active inference simulation")
    p.add_argument("--steps", type=int, default=500,
                   help="Number of physics steps (default: 500)")
    p.add_argument("--food-x", type=float, default=0.0005,
                   help="Food source x position in metres (default: 0.0005 = 0.5mm)")
    p.add_argument("--food-z", type=float, default=0.0,
                   help="Food source z position in metres (default: 0)")
    p.add_argument("--food-positions", type=str, default=None,
                   help='Multiple food sources as "x1,y1 x2,y2 ..." in mm (e.g. "0.5,0 1.0,0.3 1.5,-0.2")')
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
    p.add_argument("--no-neuromod", action="store_true",
                   help="Disable neuromodulation (M0/M1) for isolation experiments")
    p.add_argument("--verbose", action="store_true",
                   help="Enable PAULA neuron INFO logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_level = "INFO" if args.verbose else "WARNING"

    logger.info(f"Starting C. elegans simulation: {args.steps} steps")
    if args.no_neuromod:
        logger.info("Neuromodulation DISABLED (M0/M1 = 0)")

    if args.food_positions:
        food_positions = []
        for pair in args.food_positions.split():
            parts = pair.split(",")
            x_mm = float(parts[0])
            y_mm = float(parts[1]) if len(parts) > 1 else 0.0
            food_positions.append((x_mm / 1000.0, y_mm / 1000.0, 0.0))
        logger.info(f"Food sources: {len(food_positions)} at {[(p[0]*1000, p[1]*1000) for p in food_positions]}")
    else:
        food_positions = None

    record_neural = args.save_log
    collector = StreamingCollector(args.steps, record_neural=record_neural)

    engine, loop = build_c_elegans_simulation(
        use_connectome_cache=not args.no_cache,
        food_position=(args.food_x, 0.0, args.food_z),
        food_positions=food_positions,
        log_level=log_level,
        record_neural_states=record_neural,
        neuromodulation=not args.no_neuromod,
    )

    loop.reset()

    def _on_step(step: SimulationStep, _loop: Any) -> None:
        collector.record(step)

    if args.viewer:
        _run_with_viewer(engine, loop, args.steps, collector)
    else:
        logger.info("Running sensorimotor loop …")
        loop.run(
            n_steps=args.steps,
            progress=True,
            keep_results=False,
            on_step_raw=_on_step,
        )

    collector.trim()

    start_pos = collector.positions[0] if collector.n_recorded > 0 else np.zeros(3)
    end_pos = collector.positions[-1] if collector.n_recorded > 0 else np.zeros(3)
    displacement = float(np.linalg.norm(end_pos - start_pos))

    if collector.motor_outputs is not None and collector._motor_names:
        motor_means = np.mean(np.abs(collector.motor_outputs[:collector.n_recorded]), axis=1)
        mean_motor = float(np.mean(motor_means))
    else:
        mean_motor = 0.0

    print("\n" + "=" * 60)
    print("Simulation complete")
    print("=" * 60)
    print(f"Steps run         : {collector.n_recorded}")
    print(f"Start position    : ({start_pos[0]*1000:.2f}, {start_pos[1]*1000:.2f}) mm (x, y)")
    print(f"End position      : ({end_pos[0]*1000:.2f}, {end_pos[1]*1000:.2f}) mm (x, y)")
    print(f"Total displacement: {displacement*1000:.3f} mm")
    print(f"Mean motor act.   : {mean_motor:.4f}")
    print(
        f"Mean free energy  : "
        f"{loop.free_energy_trace.mean_prediction_error:.6f}"
    )
    if args.no_neuromod:
        print("Neuromodulation   : DISABLED")
    print("=" * 60)

    if args.save_plot:
        food_pos = (
            food_positions
            if food_positions is not None
            else [(args.food_x, 0.0, args.food_z)]
        )
        _save_plot(collector, food_positions=food_pos)

    if args.save_log:
        _save_streaming_log(collector, loop, args, food_positions=food_positions)


def _save_streaming_log(
    collector: StreamingCollector,
    loop: Any,
    args: argparse.Namespace,
    food_positions: list[tuple[float, float, float]] | None = None,
) -> None:
    """Save run log directly from the streaming collector's numpy arrays."""
    import json

    log_dir = Path(args.log_dir) if args.log_dir else default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    trace = loop.free_energy_trace
    food_cfg = (
        [[p[0], p[1], p[2]] for p in food_positions]
        if food_positions
        else [[args.food_x, 0.0, args.food_z]]
    )
    config_data: dict[str, Any] = {
        "config": {
            "steps": args.steps,
            "food_position": food_cfg[0],
            "food_positions": food_cfg,
            "use_connectome_cache": not args.no_cache,
            "log_level": "INFO" if args.verbose else "WARNING",
            "neuromodulation": not args.no_neuromod,
        },
        "summary": {
            "steps_run": collector.n_recorded,
            "start_position_m": collector.positions[0].tolist(),
            "end_position_m": collector.positions[-1].tolist(),
            "displacement_m": float(np.linalg.norm(
                collector.positions[-1] - collector.positions[0]
            )),
            "mean_motor_activation": float(
                np.mean(np.abs(collector.motor_outputs)) if collector.motor_outputs is not None else 0.0
            ),
            "mean_free_energy": trace.mean_prediction_error,
        },
        "free_energy_trace": {
            "ticks": list(trace.ticks),
            "prediction_error": list(trace.prediction_error),
            "motor_entropy": list(trace.motor_entropy),
        },
        "data_keys": {
            "joint_names": collector._joint_names or [],
            "motor_names": collector._motor_names or [],
            "chem_names": collector._chem_names or [],
            "neuron_names": collector._neuron_names or [],
        },
    }

    with open(log_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=2, default=_json_default)

    arrays: dict[str, np.ndarray] = {
        "tick": collector.ticks,
        "elapsed_ms": collector.elapsed_ms,
        "position": collector.positions,
        "head_position": collector.head_positions,
    }
    if collector.joint_angles is not None:
        arrays["joint_angle"] = collector.joint_angles
    if collector.motor_outputs is not None:
        arrays["motor_output"] = collector.motor_outputs
    if collector.chemicals is not None:
        arrays["chemical"] = collector.chemicals
    if collector.neural_S is not None:
        arrays["neural_S"] = collector.neural_S
    if collector.neural_fired is not None:
        arrays["neural_fired"] = collector.neural_fired

    np.savez_compressed(log_dir / "data.npz", **arrays)
    print(f"\nLog saved to {log_dir.resolve()}")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    return str(obj)


def _run_with_viewer(engine, loop, n_steps: int, collector: StreamingCollector) -> None:
    """Run simulation with interactive 3D MuJoCo viewer."""
    import mujoco.viewer

    body = engine.body
    model, data = body.model, body.data
    dt = body.dt

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
            collector.record(step)
            viewer.sync()
            time.sleep(dt)


def _save_plot(
    collector: StreamingCollector,
    food_positions: list[tuple[float, float, float]] | None = None,
) -> None:
    """Save a 2-panel plot: trajectory + motor wave."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        n = collector.n_recorded

        xs = collector.positions[:n, 0] * 1000
        ys = collector.positions[:n, 1] * 1000

        ax = axes[0]
        ax.plot(xs, ys, "b-", linewidth=2, alpha=0.8)
        ax.plot(xs[0], ys[0], "go", markersize=10, label="Start")
        ax.plot(xs[-1], ys[-1], "r^", markersize=10, label="End")

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Worm trajectory (head, floor plane)")
        ax.grid(True, alpha=0.3)

        all_x = xs.copy()
        all_y = ys.copy()
        if food_positions:
            for i, pos in enumerate(food_positions):
                fx, fy = pos[0] * 1000, pos[1] * 1000
                all_x = np.append(all_x, fx)
                all_y = np.append(all_y, fy)
                ax.plot(fx, fy, "m*", markersize=18, markeredgecolor="k",
                        markeredgewidth=0.5, label="Food" if i == 0 else None, zorder=5)

        x_lo, x_hi = float(np.min(all_x)), float(np.max(all_x))
        y_lo, y_hi = float(np.min(all_y)), float(np.max(all_y))
        margin = 0.15
        x_span = max(x_hi - x_lo, 0.2)
        y_span = max(y_hi - y_lo, 0.2)
        span = max(x_span, y_span)
        cx = (x_lo + x_hi) / 2
        cy = (y_lo + y_hi) / 2
        ax.set_xlim(cx - span / 2 - margin, cx + span / 2 + margin)
        ax.set_ylim(cy - span / 2 - margin, cy + span / 2 + margin)

        ax.legend()
        ax.set_aspect("equal")

        ax = axes[1]
        wave = collector.dorsal_minus_ventral_wave(last_n=collector.n_recorded)
        if wave.size > 0:
            w_min, w_max = float(np.min(wave)), float(np.max(wave))
            v_abs = max(abs(w_min), abs(w_max), 0.01)
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
