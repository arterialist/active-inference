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

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from simulations.engine import SimulationEngine, SimulationStep
from simulations.types import StepCallbackWithLoop

_FE_MAXLEN = 50_000  # capture full 10k–30k runs for convergence checks


@dataclass
class FreeEnergyTrace:
    """
    Running proxy for variational free energy.

    We approximate free energy as the sum of:
      - Prediction error magnitude (surprise / accuracy term)
      - Negative entropy proxy via average firing rate deviation

    Uses bounded deques to avoid O(n) memory growth on long runs.
    """

    ticks: deque[int] = field(default_factory=lambda: deque(maxlen=_FE_MAXLEN))
    prediction_error: deque[float] = field(default_factory=lambda: deque(maxlen=_FE_MAXLEN))
    motor_entropy: deque[float] = field(default_factory=lambda: deque(maxlen=_FE_MAXLEN))
    _pe_sum: float = 0.0
    _pe_count: int = 0

    def record(
        self,
        tick: int,
        neural_states: dict[str, Any],
        motor_outputs: dict[str, float],
    ) -> None:
        """Extract proxy metrics from neural states and motor outputs."""
        self.ticks.append(tick)

        errors = [
            abs(v) for k, v in neural_states.items() if k.endswith("_S")
        ]
        if errors:
            pe = float(np.mean(errors))
        else:
            vals = list(motor_outputs.values())
            pe = float(np.mean(np.abs(vals))) if vals else 0.0
        self.prediction_error.append(pe)
        self._pe_sum += pe
        self._pe_count += 1

        vals = list(motor_outputs.values())
        self.motor_entropy.append(float(np.var(vals)) if vals else 0.0)

    @property
    def mean_prediction_error(self) -> float:
        return self._pe_sum / self._pe_count if self._pe_count > 0 else 0.0


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
        keep_results: bool = True,
        on_step_raw: StepCallbackWithLoop | None = None,
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
            keep_results:       Whether to accumulate all steps in-memory.
                                Set False for long runs when streaming to disk.
            on_step_raw:        Additional per-step callback for streaming.

        Returns:
            List of SimulationStep records (empty if keep_results=False).
        """
        results: list[SimulationStep] = []
        log_interval = max(1, n_steps // 20)

        for i in range(n_steps):
            step = self.engine.step()

            if keep_results:
                results.append(step)

            if on_step_raw is not None:
                on_step_raw(step, self)

            if progress and i % log_interval == 0:
                pos = step.body_state.position
                logger.info(
                    f"Step {i + 1}/{n_steps} | "
                    f"pos=({pos[0]:.4f}, {pos[1]:.4f}) | "
                    f"{step.elapsed_ms:.1f} ms/step"
                )

            if (
                converge_threshold is not None
                and len(self.free_energy_trace.prediction_error) >= converge_window
            ):
                recent = list(self.free_energy_trace.prediction_error)[-converge_window:]
                if float(np.mean(recent)) < converge_threshold:
                    logger.info(
                        f"Converged at step {step.tick} "
                        f"(mean PE={np.mean(recent):.4f} < {converge_threshold})"
                    )
                    break

        if progress:
            logger.info(
                f"Loop complete: {n_steps} steps | "
                f"mean PE={self.free_energy_trace.mean_prediction_error:.4f}"
            )

        return results

    @property
    def tick(self) -> int:
        return self.engine.tick
