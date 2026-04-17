"""
Abstract base class for simulation environments.

An environment provides sensory stimuli to the organism and receives
the physical consequences of its actions (via the body).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EnvironmentObservation:
    """A snapshot of all sensory signals the environment delivers."""

    # Scalar chemical concentrations keyed by molecule name
    chemicals: dict[str, float] = field(default_factory=dict)
    # Contact/touch forces at named body sites; shape (3,) per site
    contact_forces: dict[str, np.ndarray] = field(default_factory=dict)
    # Proprioceptive signals (joint angles, angular velocities)
    proprioception: dict[str, float] = field(default_factory=dict)
    # Any extra modality
    extra: dict[str, Any] = field(default_factory=dict)


class BaseEnvironment(ABC):
    """
    Abstract environment.

    Subclasses define the physical world (surface, attractants, repellents)
    and implement the three lifecycle methods below.
    """

    @abstractmethod
    def reset(self) -> EnvironmentObservation:
        """Reset environment to initial conditions and return first observation."""

    @abstractmethod
    def step(self, body_state: dict[str, Any]) -> EnvironmentObservation:
        """
        Advance the environment one step given the organism's current body state.

        Args:
            body_state: Position, orientation, contact points, etc. from the body.

        Returns:
            Updated sensory observation for this time step.
        """

    @abstractmethod
    def render(self) -> np.ndarray | None:
        """
        Render a top-down view of the environment.

        Returns an (H, W, 3) uint8 RGB array or None if headless.
        """

    def post_body_step(self, body_state: dict[str, Any]) -> None:
        """
        Called after the body integrates motor outputs for this tick.

        Use for effects that depend on the end-of-step pose (for example food
        contact) while :meth:`step` still receives the pre-step body state for
        sensory observations.
        """

    def get_chemical_concentration(
        self, molecule: str, position: np.ndarray
    ) -> float:
        """Convenience: query concentration of a named molecule at a 3-D position."""
        return 0.0
