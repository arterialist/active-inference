"""
Abstract base for interactive real-time environment viewers.

Core logic: run simulation loop without recording, step-by-step,
with display and input handling delegated to organism-specific subclasses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulations.engine import SimulationEngine
    from simulations.sensorimotor_loop import SensorimotorLoop


class BaseInteractiveViewer(ABC):
    """
    Abstract interactive viewer for real-time simulation.

    Runs the simulation in real-time with no logging or recording.
    Subclasses implement organism-specific display and interaction
    (e.g. food placement for C. elegans).
    """

    @abstractmethod
    def run(
        self,
        engine: SimulationEngine,
        loop: SensorimotorLoop,
    ) -> None:
        """
        Run the interactive viewer until the user closes it.

        Steps the simulation in real-time, renders each frame, and
        processes user input. No logs or recordings are produced.
        """
        ...
