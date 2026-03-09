"""
Core tick-based simulation engine.

SimulationEngine orchestrates the three subsystems (body, environment,
nervous system) and exposes a clean step / run interface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from simulations.base_body import BaseBody, BodyState
from simulations.base_environment import BaseEnvironment, EnvironmentObservation
from simulations.base_nervous_system import BaseNervousSystem
from simulations.types import StepCallback


@dataclass
class SimulationStep:
    """All data produced by a single simulation step."""

    tick: int
    body_state: BodyState
    observation: EnvironmentObservation
    motor_outputs: dict[str, float]
    neural_states: dict[str, Any]
    elapsed_ms: float = 0.0


class SimulationEngine:
    """
    Tick-based simulation engine.

    Wires together a BaseBody, BaseEnvironment, and BaseNervousSystem and
    drives the sensorimotor loop forward.

    Args:
        body:            Physics body (MuJoCo-backed).
        environment:     Environment providing sensory stimuli.
        nervous_system:  Neural network (PAULA-backed).
        neural_ticks_per_physics_step:
                         How many PAULA ticks to run per single physics step.
                         C. elegans neurons operate on ~ms timescales while
                         MuJoCo steps at ~2 ms; a ratio of 1-4 is sensible.
        on_step:         Optional callback invoked after every step.
    """

    def __init__(
        self,
        body: BaseBody,
        environment: BaseEnvironment,
        nervous_system: BaseNervousSystem,
        neural_ticks_per_physics_step: int = 2,
        on_step: StepCallback | None = None,
        record_neural_states: bool = True,
    ):
        self.body = body
        self.environment = environment
        self.nervous_system = nervous_system
        self.neural_ticks_per_physics_step = neural_ticks_per_physics_step
        self.on_step = on_step
        self.record_neural_states = record_neural_states

        self._tick: int = 0
        self._history: list[SimulationStep] = []
        self._max_history: int = 200

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> SimulationStep:
        """Reset all subsystems to initial state."""
        self._tick = 0
        self._history.clear()

        body_state = self.body.reset()
        observation = self.environment.reset()
        self.nervous_system.reset()

        step = SimulationStep(
            tick=0,
            body_state=body_state,
            observation=observation,
            motor_outputs={},
            neural_states=self.nervous_system.get_neuron_states(),
        )
        self._history.append(step)
        return step

    def step(self) -> SimulationStep:
        """Advance the simulation by one physics step."""
        t0 = time.perf_counter_ns()

        # --- read sensory inputs from last body state ---
        last_body = self.body.get_state()
        obs = self.environment.step(self._body_state_as_dict(last_body))

        # --- convert observation to named sensory inputs ---
        sensory_inputs = self._observation_to_sensory_inputs(obs, last_body)

        # --- run PAULA neural ticks ---
        motor_outputs: dict[str, float] = {}
        for sub_tick in range(self.neural_ticks_per_physics_step):
            motor_outputs = self.nervous_system.tick(
                sensory_inputs,
                current_tick=self._tick * self.neural_ticks_per_physics_step + sub_tick,
            )

        # --- apply muscle activations to physics body ---
        body_state = self.body.step(motor_outputs)

        self._tick += 1
        elapsed = (time.perf_counter_ns() - t0) / 1e6

        step = SimulationStep(
            tick=self._tick,
            body_state=body_state,
            observation=obs,
            motor_outputs=motor_outputs,
            neural_states=(self.nervous_system.get_neuron_states()
                           if self.record_neural_states else {}),
            elapsed_ms=elapsed,
        )
        self._history.append(step)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        if self.on_step is not None:
            self.on_step(step)

        return step

    def run(
        self,
        n_steps: int,
        progress: bool = True,
        max_history: int = 10_000,
    ) -> list[SimulationStep]:
        """
        Run the simulation for n_steps physics steps.

        Args:
            n_steps:     Number of physics steps.
            progress:    Log progress every 10 % of steps.
            max_history: Cap history to avoid unbounded memory use.

        Returns:
            List of SimulationStep records.
        """
        log_interval = max(1, n_steps // 10)
        results: list[SimulationStep] = []

        for i in range(n_steps):
            s = self.step()
            results.append(s)

            if progress and i % log_interval == 0:
                pos = s.body_state.position
                logger.info(
                    f"Step {self._tick}/{n_steps} | "
                    f"pos=({pos[0]:.3f}, {pos[1]:.3f}) | "
                    f"{s.elapsed_ms:.1f} ms/step"
                )

            # Trim history to avoid unbounded memory growth
            if len(self._history) > max_history:
                self._history = self._history[-max_history:]

        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def history(self) -> list[SimulationStep]:
        return self._history

    # ------------------------------------------------------------------
    # Helpers (overridable in subclasses)
    # ------------------------------------------------------------------

    def _body_state_as_dict(self, state: BodyState) -> dict[str, Any]:
        return {
            "position": state.position,
            "orientation": state.orientation,
            "joint_angles": state.joint_angles,
            "joint_velocities": state.joint_velocities,
            "contact_forces": state.contact_forces,
            "head_position": state.head_position,
        }

    def _observation_to_sensory_inputs(
        self,
        obs: EnvironmentObservation,
        body_state: BodyState,
    ) -> dict[str, float]:
        """
        Flatten a structured EnvironmentObservation into a flat dict of
        named scalar inputs for the nervous system.

        Subclasses or organism-specific engines typically override this
        with a properly typed SensorEncoder object.  The default
        implementation is a shallow merge of all observation fields.
        """
        inputs: dict[str, float] = {}
        inputs.update(obs.chemicals)
        for site, force in obs.contact_forces.items():
            inputs[f"touch_{site}"] = float(np.linalg.norm(force))
        inputs.update(obs.proprioception)
        return inputs
