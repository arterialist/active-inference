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
  2D:  --save-plot       saves a trajectory plot + motor wave heatmap
       --save-animation  saves an animated GIF
       --plot-output     output path for plot (default: c_elegans_run.png)
       --animation-output  output path for GIF (default: c_elegans_run.gif)
  3D:  --viewer          launches interactive MuJoCo viewer (requires display)
       On macOS:  uv run mjpython scripts/run_c_elegans.py --viewer --steps 500

  Interactive:  --interactive  2D real-time viewer; click to add food, right-click to remove
                No logging or recording. Run until window closed.

Modulators
----------
  --no-M0       Disable M0 (stress, dC/dt<0)
  --no-M1       Disable M1 (reward, dC/dt>0)

Food
----
  No food by default. Use --food-positions "x1,y1 x2,y2" to add sources.

The script:
  1. Loads the Cook et al. 2019 connectome (cached after first run)
  2. Builds 302 PAULA neurons wired by the connectome
  3. Initialises a MuJoCo worm body on an agar plate
  4. Runs the active inference sensorimotor loop
  5. Prints a summary and optionally saves a locomotion plot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from simulations.c_elegans.simulation import build_c_elegans_simulation
from simulations.c_elegans.muscles import NeuromuscularJunction
from simulations.c_elegans.config import (
    N_BODY_SEGMENTS,
    FOOD_CONSUMPTION_RADIUS_M,
    BODY_RADIUS_M,
    BODY_LENGTH_M,
)
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
        self.prediction_error = np.zeros(n_steps)
        self.motor_entropy = np.zeros(n_steps)

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

    def record(self, step: SimulationStep, loop: Any = None) -> None:
        i = self._i
        if i >= self._n:
            return

        if self._joint_names is None:
            self._init_from_step(step)

        self.positions[i] = step.body_state.position
        self.head_positions[i] = step.body_state.head_position
        self.elapsed_ms[i] = step.elapsed_ms
        self.ticks[i] = step.tick

        if loop is not None:
            trace = loop.free_energy_trace
            if trace.prediction_error:
                self.prediction_error[i] = trace.prediction_error[-1]
            if trace.motor_entropy:
                self.motor_entropy[i] = trace.motor_entropy[-1]

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
        self.prediction_error = self.prediction_error[:n]
        self.motor_entropy = self.motor_entropy[:n]
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
                   help='Multiple food sources as "x1,y1 x2,y2 ..." (0.5=5cm, 1=10cm; e.g. "0.5,0 1,0.3 1.5,-0.2")')
    p.add_argument("--food-scale", type=float, default=1.0,
                   help="Scale factor for food positions (e.g. 0.5 = half distance, for closer food)")
    p.add_argument("--save-plot", action="store_true",
                   help="Save locomotion trajectory and motor pattern plot")
    p.add_argument("--save-animation", action="store_true",
                   help="Save animated trajectory + motor wave (GIF)")
    p.add_argument("--plot-output", type=str, default="c_elegans_run.png",
                   help="Output path for plot (default: c_elegans_run.png)")
    p.add_argument("--animation-output", type=str, default="c_elegans_run.gif",
                   help="Output path for animation GIF (default: c_elegans_run.gif)")
    p.add_argument("--animation-fps", type=int, default=15,
                   help="Animation frames per second (default: 15)")
    p.add_argument("--animation-frames", type=int, default=300,
                   help="Max animation frames; subsampled if steps > this (default: 300)")
    p.add_argument("--save-log", action="store_true",
                   help="Save full simulation log for post-hoc analysis")
    p.add_argument("--log-dir", type=str, default=None,
                   help="Log directory when using --save-log (default: logs/run_<timestamp>)")
    p.add_argument("--viewer", action="store_true",
                   help="Launch interactive 3D MuJoCo viewer (requires display)")
    p.add_argument("--interactive", action="store_true",
                   help="2D real-time viewer; click to add/remove food. No logging.")
    p.add_argument("--no-cache", action="store_true",
                   help="Force fresh connectome download (ignore cache)")
    p.add_argument("--no-M0", action="store_true",
                   help="Disable M0 modulator (stress, dC/dt<0)")
    p.add_argument("--no-M1", action="store_true",
                   help="Disable M1 modulator (reward, dC/dt>0)")
    p.add_argument("--verbose", action="store_true",
                   help="Enable PAULA neuron INFO logging")
    p.add_argument("--evol-config", type=str, default=None,
                   help="Path to evolved config JSON (from evolve_food_seeking.py)")
    return p.parse_args()


def _load_evol_config(path: str | None) -> dict | None:
    """Load evolved config from JSON file. Supports checkpoint format (config/best_config)."""
    if not path:
        return None
    with open(path) as f:
        data = json.load(f)
    # Checkpoint format: use config or best_config; else treat as flat config
    return data.get("config", data.get("best_config", data))


def _run_interactive(args: argparse.Namespace, log_level: str) -> None:
    """Run interactive 2D viewer; no logging or recording."""
    from simulations.c_elegans.interactive_viewer import CElegansInteractiveViewer

    evol_config = _load_evol_config(args.evol_config)

    if args.food_positions:
        food_positions = []
        for pair in args.food_positions.split():
            parts = pair.split(",")
            x_val = float(parts[0])
            y_val = float(parts[1]) if len(parts) > 1 else 0.0
            scale = args.food_scale
            food_positions.append((x_val * 0.1 * scale, y_val * 0.1 * scale, 0.0))
    else:
        food_positions = []  # No food by default in interactive mode

    engine, loop = build_c_elegans_simulation(
        use_connectome_cache=not args.no_cache,
        food_position=(args.food_x, 0.0, args.food_z),
        food_positions=food_positions,
        log_level=log_level,
        record_neural_states=False,
        enable_m0=not args.no_M0,
        enable_m1=not args.no_M1,
        evol_config=evol_config,
    )

    viewer = CElegansInteractiveViewer()
    logger.info("Interactive viewer: left-click add food, right-click remove. Close window to exit.")
    viewer.run(engine, loop)


def main() -> None:
    args = parse_args()
    log_level = "INFO" if args.verbose else "WARNING"

    if args.interactive:
        _run_interactive(args, log_level)
        return

    logger.info(f"Starting C. elegans simulation: {args.steps} steps")
    if args.no_M0:
        logger.info("M0 modulator DISABLED")
    if args.no_M1:
        logger.info("M1 modulator DISABLED")

    if args.food_positions:
        food_positions = []
        for pair in args.food_positions.split():
            parts = pair.split(",")
            x_val = float(parts[0])
            y_val = float(parts[1]) if len(parts) > 1 else 0.0
            # 0.5 → 5cm, 1 → 10cm (input value × 10 = cm); apply food-scale
            scale = args.food_scale
            food_positions.append((x_val * 0.1 * scale, y_val * 0.1 * scale, 0.0))
        logger.info(f"Food sources: {len(food_positions)} at {[(p[0]*100, p[1]*100) for p in food_positions]} cm (scale={args.food_scale})")
    else:
        food_positions = []  # No food by default
        logger.warning("No food placed. Use --food-positions to add food sources.")

    record_neural = args.save_log
    collector = StreamingCollector(args.steps, record_neural=record_neural)
    evol_config = _load_evol_config(args.evol_config)

    engine, loop = build_c_elegans_simulation(
        use_connectome_cache=not args.no_cache,
        food_position=(args.food_x, 0.0, args.food_z),
        food_positions=food_positions,
        log_level=log_level,
        record_neural_states=record_neural,
        enable_m0=not args.no_M0,
        enable_m1=not args.no_M1,
        evol_config=evol_config,
    )

    loop.reset()

    def _on_step(step: SimulationStep, _loop: Any) -> None:
        collector.record(step, loop=_loop)

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
    if args.no_M0 or args.no_M1:
        mods = []
        if args.no_M0:
            mods.append("M0")
        if args.no_M1:
            mods.append("M1")
        print(f"Modulators disabled: {', '.join(mods)}")
    print("=" * 60)

    if args.save_plot:
        initial_food = food_positions
        active_food = engine.environment.get_active_food_positions()
        _save_plot(
            collector,
            loop,
            initial_food_positions=initial_food,
            active_food_positions=active_food,
            output_path=args.plot_output,
        )

    if args.save_animation:
        initial_food = food_positions
        active_food = engine.environment.get_active_food_positions()
        _save_animation(
            collector,
            loop,
            initial_food_positions=initial_food,
            active_food_positions=active_food,
            fps=args.animation_fps,
            max_frames=args.animation_frames,
            output_path=args.animation_output,
        )

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
    food_cfg = [[p[0], p[1], p[2]] for p in food_positions]
    config_data: dict[str, Any] = {
        "config": {
            "steps": args.steps,
            "food_positions": food_cfg,
            "use_connectome_cache": not args.no_cache,
            "log_level": "INFO" if args.verbose else "WARNING",
            "enable_m0": not args.no_M0,
            "enable_m1": not args.no_M1,
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

    # Human-readable CSVs (same format as save_run_log)
    import pandas as pd

    ticks = arrays["tick"]
    n = len(ticks)
    if n > 0:
        df = pd.DataFrame(np.column_stack([ticks, arrays["elapsed_ms"]]), columns=["tick", "elapsed_ms"])
        df.to_csv(log_dir / "elapsed_ms.csv", index=False, float_format="%.6g")
        df = pd.DataFrame(np.column_stack([ticks, arrays["position"]]), columns=["tick", "x_m", "y_m", "z_m"])
        df.to_csv(log_dir / "positions.csv", index=False, float_format="%.6g")
        df = pd.DataFrame(np.column_stack([ticks, arrays["head_position"]]), columns=["tick", "x_m", "y_m", "z_m"])
        df.to_csv(log_dir / "head_position.csv", index=False, float_format="%.6g")
        if "joint_angle" in arrays and collector._joint_names:
            cols = ["tick"] + collector._joint_names
            df = pd.DataFrame(np.column_stack([ticks, arrays["joint_angle"]]), columns=cols)
            df.to_csv(log_dir / "joint_angle.csv", index=False, float_format="%.6g")
        if "motor_output" in arrays and collector._motor_names:
            cols = ["tick"] + collector._motor_names
            df = pd.DataFrame(np.column_stack([ticks, arrays["motor_output"]]), columns=cols)
            df.to_csv(log_dir / "motor_output.csv", index=False, float_format="%.6g")
        if "chemical" in arrays and collector._chem_names:
            cols = ["tick"] + collector._chem_names
            df = pd.DataFrame(np.column_stack([ticks, arrays["chemical"]]), columns=cols)
            df.to_csv(log_dir / "chemical.csv", index=False, float_format="%.6g")
        if "neural_S" in arrays and collector._neuron_names:
            cols = ["tick"] + collector._neuron_names
            df = pd.DataFrame(np.column_stack([ticks, arrays["neural_S"]]), columns=cols)
            df.to_csv(log_dir / "neural_S.csv", index=False, float_format="%.6g")
            df = pd.DataFrame(np.column_stack([ticks, arrays["neural_fired"]]), columns=cols)
            df.to_csv(log_dir / "neural_fired.csv", index=False, float_format="%.6g")

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
    loop: Any,
    initial_food_positions: list[tuple[float, float, float]] | None = None,
    active_food_positions: list[tuple[float, float, float]] | None = None,
    output_path: str | Path = "c_elegans_run.png",
) -> None:
    """Save a 3-panel plot: trajectory + motor wave + prediction error & free energy.

    Food markers: purple star = uneaten, grey X = eaten.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        CONSUMPTION_RADIUS_MM = FOOD_CONSUMPTION_RADIUS_M * 1000

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        n = collector.n_recorded

        xs = collector.positions[:n, 0] * 1000
        ys = collector.positions[:n, 1] * 1000

        ax = axes[0]
        ax.plot(xs, ys, "b-", linewidth=2, alpha=0.8)
        ax.plot(xs[0], ys[0], "go", markersize=10, label="Start")
        head_circle = Circle((xs[-1], ys[-1]), CONSUMPTION_RADIUS_MM, color="red", fill=True, alpha=0.8, label="End", zorder=4)
        ax.add_patch(head_circle)

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Worm trajectory (head, floor plane)")
        ax.grid(True, alpha=0.3)

        all_x = xs.copy()
        all_y = ys.copy()
        food_positions = initial_food_positions or active_food_positions or []
        active_set = {
            (round(p[0], 6), round(p[1], 6))
            for p in (active_food_positions or [])
        }
        showed_food_label = False
        showed_eaten_label = False
        for pos in food_positions:
            fx, fy = pos[0] * 1000, pos[1] * 1000
            all_x = np.append(all_x, fx)
            all_y = np.append(all_y, fy)
            key = (round(pos[0], 6), round(pos[1], 6))
            eaten = key not in active_set
            if eaten:
                ax.plot(fx, fy, "X", color="0.5", markersize=12, markeredgewidth=2,
                        label="Eaten" if not showed_eaten_label else None, zorder=5)
                showed_eaten_label = True
            else:
                ax.plot(fx, fy, "m*", markersize=18, markeredgecolor="k",
                        markeredgewidth=0.5, label="Food" if not showed_food_label else None, zorder=5)
                showed_food_label = True

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

        # Scale bar: 1mm = worm body length (worm is 1mm long, 80µm diameter)
        bar_x = cx - span / 2 + 0.05 * span
        bar_y = cy - span / 2 + 0.05 * span
        ax.plot([bar_x, bar_x + 1], [bar_y, bar_y], "k-", linewidth=3)
        ax.text(bar_x + 0.5, bar_y - 0.08 * span, "1 mm (worm length)", ha="center", fontsize=8)

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

        ax = axes[2]
        n = collector.n_recorded
        ticks = collector.ticks[:n]
        pe = collector.prediction_error[:n]
        me = collector.motor_entropy[:n]
        if n > 0:
            ax.plot(ticks, pe, "b-", alpha=0.8, label="Prediction error")
            ax.plot(ticks, me, "orange", alpha=0.8, label="Motor entropy")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Magnitude")
        ax.set_title("Prediction error & free energy proxy")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = Path(output_path)
        plt.savefig(out_path, dpi=120)
        print(f"\nPlot saved to {out_path.resolve()}")
        plt.close()

    except ImportError:
        print("matplotlib not available — skipping plot")
    except Exception as e:
        print(f"Plot failed: {e}")


def _save_animation(
    collector: StreamingCollector,
    loop: Any,
    initial_food_positions: list[tuple[float, float, float]] | None = None,
    active_food_positions: list[tuple[float, float, float]] | None = None,
    fps: int = 15,
    max_frames: int = 300,
    output_path: str | Path = "c_elegans_run.gif",
) -> None:
    """Save animated trajectory + motor wave + prediction error as GIF.

    Uses same data as _save_plot. Food eaten status inferred per frame from
    head trajectory vs FOOD_CONSUMPTION_RADIUS.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.patches import Circle

        CONSUMPTION_RADIUS_MM = FOOD_CONSUMPTION_RADIUS_M * 1000

        n = collector.n_recorded
        if n == 0:
            print("No data to animate")
            return

        trace_ticks = collector.ticks[:n]
        trace_pe = collector.prediction_error[:n]
        trace_me = collector.motor_entropy[:n]

        # Subsample frame indices
        n_frames = min(n, max_frames)
        frame_indices = np.linspace(0, n - 1, n_frames, dtype=int)

        xs = collector.positions[:n, 0] * 1000
        ys = collector.positions[:n, 1] * 1000
        head_pos = collector.head_positions[:n]
        wave = collector.dorsal_minus_ventral_wave(last_n=n)

        food_positions = initial_food_positions or active_food_positions or []
        active_set = {
            (round(p[0], 6), round(p[1], 6))
            for p in (active_food_positions or [])
        }

        all_x = np.concatenate([xs, np.array([p[0] * 1000 for p in food_positions])])
        all_y = np.concatenate([ys, np.array([p[1] * 1000 for p in food_positions])])
        x_lo, x_hi = float(np.min(all_x)), float(np.max(all_x))
        y_lo, y_hi = float(np.min(all_y)), float(np.max(all_y))
        margin = 0.15
        x_span = max(x_hi - x_lo, 0.2)
        y_span = max(y_hi - y_lo, 0.2)
        span = max(x_span, y_span)
        cx = (x_lo + x_hi) / 2
        cy = (y_lo + y_hi) / 2
        xlim = (cx - span / 2 - margin, cx + span / 2 + margin)
        ylim = (cy - span / 2 - margin, cy + span / 2 + margin)

        w_min = float(np.min(wave)) if wave.size > 0 else -0.01
        w_max = float(np.max(wave)) if wave.size > 0 else 0.01
        v_abs = max(abs(w_min), abs(w_max), 0.01)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Left: trajectory
        ax_traj = axes[0]
        line_traj, = ax_traj.plot([], [], "b-", linewidth=2, alpha=0.8)
        pt_start = ax_traj.plot(xs[0], ys[0], "go", markersize=10, label="Start")[0]
        head_circle = Circle((xs[0], ys[0]), CONSUMPTION_RADIUS_MM, color="red", fill=True, alpha=0.8, label="Head", zorder=4)
        ax_traj.add_patch(head_circle)
        food_artists: list[Any] = []
        for _ in food_positions:
            food_artists.append(ax_traj.plot([], [], "m*", markersize=18, markeredgecolor="k",
                    markeredgewidth=0.5, zorder=5)[0])
            food_artists.append(ax_traj.plot([], [], "X", color="0.5", markersize=12,
                    markeredgewidth=2, zorder=5)[0])
        ax_traj.set_xlim(xlim)
        ax_traj.set_ylim(ylim)
        ax_traj.set_xlabel("x (mm)")
        ax_traj.set_ylabel("y (mm)")
        ax_traj.set_title("Worm trajectory (head, floor plane)")
        ax_traj.grid(True, alpha=0.3)
        ax_traj.set_aspect("equal")
        ax_traj.legend(loc="upper right")

        # Middle: motor wave
        ax_wave = axes[1]
        im = ax_wave.imshow(
            np.zeros((1, N_BODY_SEGMENTS)),
            aspect="auto", cmap="RdBu_r", vmin=-v_abs, vmax=v_abs, origin="lower",
        )
        ax_wave.set_xlabel("Time step")
        ax_wave.set_ylabel("Body segment")
        ax_wave.set_title("Motor wave (D−V activation)")

        # Right: prediction error & free energy
        ax_fe = axes[2]
        line_pe, = ax_fe.plot([], [], "b-", alpha=0.8, label="Prediction error")
        line_me, = ax_fe.plot([], [], "orange", alpha=0.8, label="Motor entropy")
        ax_fe.set_xlabel("Time step")
        ax_fe.set_ylabel("Magnitude")
        ax_fe.set_title("Prediction error & free energy proxy")
        ax_fe.legend()
        ax_fe.grid(True, alpha=0.3)
        if len(trace_ticks) > 0:
            ax_fe.set_xlim(0, max(n, float(np.max(trace_ticks))))
            pe_max = max(float(np.max(trace_pe)), 0.01) if len(trace_pe) > 0 else 0.01
            me_max = max(float(np.max(trace_me)), 0.01) if len(trace_me) > 0 else 0.01
            ax_fe.set_ylim(0, max(pe_max, me_max) * 1.1)

        def _eaten_up_to(food_pos: tuple[float, float, float], up_to_idx: int) -> bool:
            fp = np.array(food_pos)
            dists = np.linalg.norm(head_pos[: up_to_idx + 1, :3] - fp, axis=1)
            return bool(np.any(dists <= FOOD_CONSUMPTION_RADIUS_M))

        def _update(frame_idx: int) -> tuple[Any, ...]:
            idx = int(frame_idx)
            line_traj.set_data(xs[: idx + 1], ys[: idx + 1])
            head_circle.center = (xs[idx], ys[idx])

            for i, pos in enumerate(food_positions):
                fx, fy = pos[0] * 1000, pos[1] * 1000
                eaten = _eaten_up_to(pos, idx)
                star_artist = food_artists[i * 2]
                x_artist = food_artists[i * 2 + 1]
                if eaten:
                    star_artist.set_data([], [])
                    x_artist.set_data([fx], [fy])
                else:
                    star_artist.set_data([fx], [fy])
                    x_artist.set_data([], [])

            wave_slice = wave[: idx + 1]
            if wave_slice.size > 0:
                im.set_data(wave_slice.T)
                im.set_extent([0, idx + 1, 0, N_BODY_SEGMENTS])

            # Update PE / free energy: show trace up to current step
            if idx < len(trace_ticks):
                line_pe.set_data(trace_ticks[: idx + 1], trace_pe[: idx + 1])
                line_me.set_data(trace_ticks[: idx + 1], trace_me[: idx + 1])
            else:
                line_pe.set_data([], [])
                line_me.set_data([], [])

            return (line_traj, head_circle, im, line_pe, line_me) + tuple(food_artists)

        anim = FuncAnimation(
            fig, _update, frames=frame_indices, interval=1000 // fps, blit=False,
        )
        out_path = Path(output_path)
        writer = PillowWriter(fps=fps)
        anim.save(out_path, writer=writer)
        plt.close()
        print(f"\nAnimation saved to {out_path.resolve()}")

    except ImportError as e:
        print(f"Animation requires matplotlib and Pillow: {e}")
    except Exception as e:
        print(f"Animation failed: {e}")


if __name__ == "__main__":
    main()
