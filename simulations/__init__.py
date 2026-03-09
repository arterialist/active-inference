"""
Active Inference simulation engine.

Common components live here; organism-specific implementations
live in sub-packages (e.g. simulations/c_elegans/).
"""

from simulations.engine import SimulationEngine, SimulationStep
from simulations.sensorimotor_loop import SensorimotorLoop
from simulations.connectome_loader import ConnectomeData, NeuronInfo, SynapticEdge
from simulations.types import MuscleActivations, SensoryInputs, StepCallback

__all__ = [
    "ConnectomeData",
    "MuscleActivations",
    "NeuronInfo",
    "SensorimotorLoop",
    "SensoryInputs",
    "SimulationEngine",
    "SimulationStep",
    "StepCallback",
    "SynapticEdge",
]
