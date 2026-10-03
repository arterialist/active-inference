"""History limits and the common pre/post-physics lifecycle."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from simulations.base_body import BodyState
from simulations.base_environment import EnvironmentObservation
from simulations.c_elegans.simulation import CElegansEngine
from simulations.engine import SimulationEngine


class Body:
    def __init__(self):
        self.position = 0.0

    def get_state(self):
        return BodyState(head_position=np.array([self.position, 0.0, 0.0]))

    def reset(self):
        self.position = 0.0
        return self.get_state()

    def step(self, motor_outputs):
        self.position += 1.0
        return self.get_state()


class Environment:
    def __init__(self):
        self.events = []

    def reset(self):
        self.events.clear()
        return EnvironmentObservation()

    def step(self, body_state):
        self.events.append(("before", float(body_state["head_position"][0])))
        return EnvironmentObservation()

    def post_body_step(self, body_state):
        self.events.append(("after", float(body_state["head_position"][0])))


class NervousSystem:
    def __init__(self):
        self.ticks = []

    def reset(self, *, rebuild_network=True):
        self.ticks.clear()

    def tick(self, sensory_inputs, current_tick):
        self.ticks.append(current_tick)
        return {"seg0_DL": 0.25}

    def get_neuron_states(self):
        return {}


class Encoder:
    def encode(self, observation, body_state):
        return {}


def engine(engine_type, *, max_history=2):
    options: dict[str, Any] = {
        "body": Body(),
        "environment": Environment(),
        "nervous_system": NervousSystem(),
        "max_history": max_history,
        "neural_ticks_per_physics_step": 2,
    }
    if engine_type is CElegansEngine:
        options["sensor_encoder"] = Encoder()
    return engine_type(**options)


@pytest.mark.parametrize("engine_type", [SimulationEngine, CElegansEngine])
@pytest.mark.parametrize("max_history", [0, 2])
def test_history_budget_also_applies_to_reset(engine_type, max_history):
    simulation = engine(engine_type, max_history=max_history)
    callbacks = []
    simulation.on_step = callbacks.append
    simulation.reset()
    assert len(simulation.history) == min(1, max_history)
    for _ in range(40):
        simulation.step()
        assert len(simulation.history) <= max_history
    assert simulation.tick == len(callbacks) == 40
    assert [step.tick for step in simulation.history] == (
        [39, 40] if max_history else []
    )
    simulation.reset()
    assert [step.tick for step in simulation.history] == ([0] if max_history else [])


@pytest.mark.parametrize("engine_type", [SimulationEngine, CElegansEngine])
def test_negative_history_is_rejected(engine_type):
    with pytest.raises(ValueError, match="max_history must be non-negative"):
        engine(engine_type, max_history=-1)


def test_run_zero_history_retains_explicit_returned_results():
    simulation = engine(SimulationEngine)
    simulation.reset()
    results = simulation.run(3, progress=False, max_history=0)
    assert [step.tick for step in results] == [1, 2, 3]
    assert simulation.history == []
    with pytest.raises(ValueError, match="max_history must be non-negative"):
        simulation.run(3, progress=False, max_history=-1)
    assert simulation.tick == 3


@pytest.mark.parametrize("engine_type", [SimulationEngine, CElegansEngine])
def test_environment_receives_pre_and_post_physics_pose(engine_type):
    simulation = engine(engine_type, max_history=0)
    simulation.reset()
    simulation.step()
    simulation.step()
    assert simulation.environment.events == [
        ("before", 0.0), ("after", 1.0),
        ("before", 1.0), ("after", 2.0),
    ]
    assert simulation.nervous_system.ticks == [0, 1, 2, 3]
