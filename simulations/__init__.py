"""
Active Inference simulation engine.

Common components live here; organism-specific implementations
live in sub-packages (e.g. simulations/c_elegans/).
"""

from simulations.engine import SimulationEngine
from simulations.sensorimotor_loop import SensorimotorLoop

__all__ = ["SimulationEngine", "SensorimotorLoop"]
