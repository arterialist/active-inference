"""
Active inference sensorimotor loop.

The SensorimotorLoop class explicitly frames the organism's behaviour as
active inference: at each timestep the nervous system simultaneously

  1. **Perceives** — synaptic prediction errors (E_dir in PAULA) drive
     Hebbian weight updates, refining the generative model.
  2. **Acts**      — motor neuron spikes propagate to muscles, moving the
     body to reduce future sensory surprise.

The two processes share the same PAULA tick() call; this wrapper exists
to make the active inference framing transparent and to provide hooks for
logging free energy proxies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from simulations.engine import SimulationEngine, SimulationStep
from simulations.types import StepCallbackWithLoop


@dataclass
class FreeEnergyTrace:
    """
    Running proxy for variational free energy.

    We approximate free energy as the sum of:
      - Prediction error magnitude (surprise / accuracy term)
      - Negative entropy proxy via average firing rate deviation

    These are cheap to compute from PAULA's existing state variables.
    """

    ticks: list[int] = field(default_factory=list)
    prediction_error: list[float] = field(default_factory=list)
    motor_entropy: list[float] = field(default_factory=list)

    def record(
        self,
        tick: int,
        neural_states: dict[str, Any],
        motor_outputs: dict[str, float],
    ) -> None:
        """Extract proxy metrics from neural states and motor outputs."""
        self.ticks.append(tick)

        # Prediction error proxy: mean absolute weight-change signal across neurons
        errors = [
            abs(v) for k, v in neural_states.items() if k.endswith("_S")
        ]
        self.prediction_error.append(float(np.mean(errors)) if errors else 0.0)

        # Motor entropy proxy: variance of motor activations (high variance = uncertain action)
        vals = list(motor_outputs.values())
        self.motor_entropy.append(float(np.var(vals)) if vals else 0.0)

    @property
    def mean_prediction_error(self) -> float:
        return float(np.mean(self.prediction_error)) if self.prediction_error else 0.0


class SensorimotorLoop:
    """
    High-level active inference sensorimotor loop.

    Wraps SimulationEngine and adds:
      - Free energy tracing
      - Per-step hooks for analysis / visualisation
      - Convergence monitoring (optional early stop)

    Args:
        engine:           Configured SimulationEngine.
        log_free_energy:  Track free energy proxies per step.
        on_step:          Additional callback after every step.
    """

    def __init__(
        self,
        engine: SimulationEngine,
        log_free_energy: bool = True,
        on_step: StepCallbackWithLoop | None = None,
    ):
        self.engine = engine
        self.log_free_energy = log_free_energy
        self._on_step = on_step
        self.free_energy_trace = FreeEnergyTrace()

        # Wrap engine's on_step to add free energy tracing
        self._original_on_step = engine.on_step
        engine.on_step = self._handle_step

    def _handle_step(self, step: SimulationStep) -> None:
        if self.log_free_energy:
            self.free_energy_trace.record(
                step.tick, step.neural_states, step.motor_outputs
            )
        if self._original_on_step is not None:
            self._original_on_step(step)
        if self._on_step is not None:
            self._on_step(step, self)

    def reset(self) -> SimulationStep:
        self.free_energy_trace = FreeEnergyTrace()
        return self.engine.reset()

    def run(
        self,
        n_steps: int,
        progress: bool = True,
        converge_threshold: float | None = None,
        converge_window: int = 100,
    ) -> list[SimulationStep]:
        """
        Run the active inference loop for n_steps.

        Args:
            n_steps:            Number of physics steps.
            progress:           Print progress updates.
            converge_threshold: If set, stop early when mean prediction error
                                over the last converge_window steps falls below
                                this value.
            converge_window:    Window size for convergence check.

        Returns:
            List of SimulationStep records.
        """
        results: list[SimulationStep] = []

        for i in range(n_steps):
            step = self.engine.step()
            results.append(step)

            # Convergence check
            if (
                converge_threshold is not None
                and len(self.free_energy_trace.prediction_error) >= converge_window
            ):
                recent = self.free_energy_trace.prediction_error[-converge_window:]
                if float(np.mean(recent)) < converge_threshold:
                    logger.info(
                        f"Converged at step {step.tick} "
                        f"(mean PE={np.mean(recent):.4f} < {converge_threshold})"
                    )
                    break

        if progress:
            logger.info(
                f"Loop complete: {len(results)} steps | "
                f"mean PE={self.free_energy_trace.mean_prediction_error:.4f}"
            )

        return results

    @property
    def tick(self) -> int:
        return self.engine.tick
