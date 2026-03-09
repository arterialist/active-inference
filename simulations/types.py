"""
Type aliases and shared type definitions for the simulation framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol

from simulations.base_body import BodyState
from simulations.base_environment import EnvironmentObservation

if TYPE_CHECKING:
    from simulations.engine import SimulationStep

# Callback types (SimulationStep imported lazily to avoid circular import)
StepCallback = Callable[["SimulationStep"], None]
StepCallbackWithLoop = Callable[["SimulationStep", Any], None]

# Sensory and motor mappings
SensoryInputs = dict[str, float]
MuscleActivations = dict[str, float]


class SensorEncoderProtocol(Protocol):
    """Protocol for objects that encode observations into sensory inputs."""

    def encode(
        self,
        observation: EnvironmentObservation,
        body_state: BodyState,
    ) -> SensoryInputs:
        """Produce sensory input dict from observation and body state."""
        ...
