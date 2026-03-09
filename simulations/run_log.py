"""
Simulation run logging for post-hoc analysis.

Saves run configuration, summary statistics, and per-step time-series data
to a directory with JSON metadata and NPZ arrays.

Log format
----------
  config.json  - RunConfig, RunSummary, FreeEnergyTrace, data_keys (array labels)
  data.npz     - Binary arrays (compact, fast to load)
  *.csv       - Human-readable CSV per array (positions.csv, motor_output.csv, etc.)

Load with:  config, data = load_run_log(Path("logs/run_20250101_120000Z"))
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from simulations.engine import SimulationStep
from simulations.sensorimotor_loop import FreeEnergyTrace, SensorimotorLoop


@dataclass
class RunConfig:
    """Run parameters for reproducibility."""

    steps: int
    food_position: tuple[float, float, float]
    use_connectome_cache: bool
    log_level: str


@dataclass
class RunSummary:
    """Aggregate statistics from the run."""

    steps_run: int
    start_position_m: list[float]
    end_position_m: list[float]
    displacement_m: float
    mean_motor_activation: float
    mean_free_energy: float
    elapsed_steps: int = 0  # if early stop


def _serialize(obj: Any) -> Any:
    """Convert numpy types and nested structures for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


def save_run_log(
    results: list[SimulationStep],
    loop: SensorimotorLoop,
    config: RunConfig,
    summary: RunSummary,
    out_dir: Path,
) -> Path:
    """
    Save full simulation log to out_dir.

    Creates:
      - config.json: RunConfig, RunSummary, FreeEnergyTrace
      - data.npz: Per-step arrays (positions, motor, chemicals, neural, etc.)

    Returns:
        Path to the created directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- config.json (written after we have data_keys) ---
    trace = loop.free_energy_trace
    config_data = {
        "config": asdict(config),
        "summary": asdict(summary),
        "free_energy_trace": {
            "ticks": trace.ticks,
            "prediction_error": trace.prediction_error,
            "motor_entropy": trace.motor_entropy,
        },
    }

    # --- data.npz: build arrays from results ---
    n = len(results)
    if n == 0:
        np.savez_compressed(out_dir / "data.npz", tick=np.array([]))
        return out_dir

    # Positions (head / body CoM)
    positions = np.array([s.body_state.position for s in results])
    head_positions = np.array([s.body_state.head_position for s in results])

    # Joint angles: collect all joint names from first step
    joint_names = sorted(results[0].body_state.joint_angles.keys())
    joint_angles = np.zeros((n, len(joint_names)))
    for i, s in enumerate(results):
        for j, name in enumerate(joint_names):
            joint_angles[i, j] = s.body_state.joint_angles.get(name, 0.0)

    # Motor outputs: collect all muscle names
    motor_names = sorted(results[0].motor_outputs.keys())
    motor_outputs = np.zeros((n, len(motor_names)))
    for i, s in enumerate(results):
        for j, name in enumerate(motor_names):
            motor_outputs[i, j] = s.motor_outputs.get(name, 0.0)

    # Chemicals: collect molecule names from first observation
    chem_names = sorted(results[0].observation.chemicals.keys())
    chemicals = np.zeros((n, len(chem_names)))
    for i, s in enumerate(results):
        for j, name in enumerate(chem_names):
            chemicals[i, j] = s.observation.chemicals.get(name, 0.0)

    # Neural states: S (membrane potential) and fired
    ns0 = results[0].neural_states
    neuron_names = sorted(
        k[:-2] for k in ns0.keys() if k.endswith("_S")
    )  # strip "_S"
    neural_S = np.zeros((n, len(neuron_names)))
    neural_fired = np.zeros((n, len(neuron_names)))
    for i, s in enumerate(results):
        for j, name in enumerate(neuron_names):
            neural_S[i, j] = s.neural_states.get(f"{name}_S", 0.0)
            neural_fired[i, j] = s.neural_states.get(f"{name}_fired", 0.0)

    # Elapsed ms per step
    elapsed_ms = np.array([s.elapsed_ms for s in results])
    ticks = np.array([s.tick for s in results])

    # Store array dimension names in config for reconstruction
    config_data["data_keys"] = {
        "joint_names": joint_names,
        "motor_names": motor_names,
        "chem_names": chem_names,
        "neuron_names": neuron_names,
    }
    config_data = _serialize(config_data)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)

    np.savez_compressed(
        out_dir / "data.npz",
        tick=ticks,
        elapsed_ms=elapsed_ms,
        position=positions,
        head_position=head_positions,
        joint_angle=joint_angles,
        motor_output=motor_outputs,
        chemical=chemicals,
        neural_S=neural_S,
        neural_fired=neural_fired,
    )

    # Human-readable CSVs
    _save_csv(out_dir, "elapsed_ms", ticks, elapsed_ms, ["tick", "elapsed_ms"])
    _save_csv(
        out_dir, "positions",
        ticks, positions,
        ["tick", "x_m", "y_m", "z_m"],
    )
    _save_csv(
        out_dir, "head_position",
        ticks, head_positions,
        ["tick", "x_m", "y_m", "z_m"],
    )
    _save_csv(
        out_dir, "joint_angle",
        ticks, joint_angles,
        ["tick"] + joint_names,
    )
    _save_csv(
        out_dir, "motor_output",
        ticks, motor_outputs,
        ["tick"] + motor_names,
    )
    _save_csv(
        out_dir, "chemical",
        ticks, chemicals,
        ["tick"] + chem_names,
    )
    _save_csv(
        out_dir, "neural_S",
        ticks, neural_S,
        ["tick"] + neuron_names,
    )
    _save_csv(
        out_dir, "neural_fired",
        ticks, neural_fired,
        ["tick"] + neuron_names,
    )

    return out_dir


def _save_csv(
    out_dir: Path,
    name: str,
    ticks: np.ndarray,
    arr: np.ndarray,
    columns: list[str],
) -> None:
    """Write array to CSV with tick as first column."""
    df = pd.DataFrame(np.column_stack([ticks, arr]), columns=columns)
    df.to_csv(out_dir / f"{name}.csv", index=False, float_format="%.6g")


def default_log_dir() -> Path:
    """Return default log directory with timestamp."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return Path("logs") / f"run_{ts}"


def load_run_log(log_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """
    Load a saved run log for analysis.

    Args:
        log_dir: Path to the run directory (contains config.json and data.npz).

    Returns:
        (config, data) where config is the parsed JSON and data is a dict of
        numpy arrays with keys: tick, elapsed_ms, position, head_position,
        joint_angle, motor_output, chemical, neural_S, neural_fired.
        config["data_keys"] contains joint_names, motor_names, chem_names, neuron_names.
    """
    log_dir = Path(log_dir)
    with open(log_dir / "config.json") as f:
        config = json.load(f)

    npz = np.load(log_dir / "data.npz", allow_pickle=False)
    data = {
        "tick": npz["tick"],
        "elapsed_ms": npz["elapsed_ms"],
        "position": npz["position"],
        "head_position": npz["head_position"],
        "joint_angle": npz["joint_angle"],
        "motor_output": npz["motor_output"],
        "chemical": npz["chemical"],
        "neural_S": npz["neural_S"],
        "neural_fired": npz["neural_fired"],
    }
    return config, data
