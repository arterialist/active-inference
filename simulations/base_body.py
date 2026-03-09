"""
Abstract base class for physics body wrappers (MuJoCo-backed).

A body receives motor commands from the nervous system and returns
a physical state dict that the environment and sensors can read.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BodyState:
    """
    Physical state of the organism body.

    Attributes:
        position:       Centre-of-mass 3-D position [x, y, z].
        orientation:    Unit quaternion [w, x, y, z].
        joint_angles:   Dict of joint name -> angle (radians).
        joint_velocities: Dict of joint name -> angular velocity.
        contact_forces: Dict of site name -> force vector (3,).
        head_position:  3-D position of the head/nose.
        extra:          Any additional physics quantities.
    """

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))
    joint_angles: dict[str, float] = field(default_factory=dict)
    joint_velocities: dict[str, float] = field(default_factory=dict)
    contact_forces: dict[str, np.ndarray] = field(default_factory=dict)
    head_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    extra: dict[str, Any] = field(default_factory=dict)


class BaseBody(ABC):
    """
    Abstract physics body.

    Subclasses wrap a MuJoCo model and expose a uniform interface for
    the sensorimotor loop.
    """

    @abstractmethod
    def reset(self) -> BodyState:
        """Reset the body to its neutral pose and return initial state."""

    @abstractmethod
    def step(self, muscle_activations: dict[str, float]) -> BodyState:
        """
        Apply muscle activations and advance physics by one time step.

        Args:
            muscle_activations: Mapping of muscle name -> activation in [0, 1].

        Returns:
            Updated body state after the physics step.
        """

    @abstractmethod
    def get_state(self) -> BodyState:
        """Return current body state without stepping the simulation."""

    @abstractmethod
    def render(self, camera: str = "top") -> np.ndarray | None:
        """
        Render the body from the named camera.

        Returns an (H, W, 3) uint8 RGB array or None if headless.
        """

    @property
    @abstractmethod
    def dt(self) -> float:
        """Physics timestep in seconds."""

    @property
    @abstractmethod
    def joint_names(self) -> list[str]:
        """Names of all controllable joints."""

    @property
    @abstractmethod
    def muscle_names(self) -> list[str]:
        """Names of all actuated muscles / tendons."""
