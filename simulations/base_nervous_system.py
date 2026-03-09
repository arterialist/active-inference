"""
Abstract base class for the nervous system (PAULA network wrapper).

A NervousSystem receives encoded sensory observations and produces
motor neuron activations.  Internally it wraps a NeuronNetwork of
PAULA units whose synaptic plasticity implements active inference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseNervousSystem(ABC):
    """
    Abstract nervous system wrapping a PAULA NeuronNetwork.

    Lifecycle:
        1. ``reset()``   — (re)initialise network state
        2. ``tick(obs)`` — one neural simulation step
        3. ``motor_outputs()`` — read motor neuron activations
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset all neuron states to resting potential."""

    @abstractmethod
    def tick(
        self,
        sensory_inputs: dict[str, float],
        current_tick: int,
    ) -> dict[str, float]:
        """
        Run one tick of the neural simulation.

        Args:
            sensory_inputs: Mapping of neuron_name -> normalised scalar
                            input intensity for this tick.
            current_tick:   Global simulation tick counter.

        Returns:
            Motor outputs: mapping of muscle_name -> activation in [0, 1].
        """

    @abstractmethod
    def get_neuron_states(self) -> dict[str, Any]:
        """Return a snapshot of all neuron membrane potentials and firing flags."""

    @property
    @abstractmethod
    def n_neurons(self) -> int:
        """Total number of neurons in the network."""
