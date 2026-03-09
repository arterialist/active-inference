"""
C. elegans full simulation: wires all subsystems together.

CElegansEngine subclasses SimulationEngine and overrides
_observation_to_sensory_inputs() to use the biologically correct SensorEncoder.
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from simulations.engine import SimulationEngine, SimulationStep
from simulations.sensorimotor_loop import SensorimotorLoop
from simulations.base_body import BodyState
from simulations.base_environment import EnvironmentObservation
from simulations.types import SensorEncoderProtocol, StepCallback

from simulations.c_elegans.body import CElegansBody
from simulations.c_elegans.environment import AgarPlateEnvironment
from simulations.c_elegans.connectome import load_connectome, print_connectome_summary
from simulations.c_elegans.neuron_mapping import CElegansNervousSystem
from simulations.c_elegans.sensors import SensorEncoder
from simulations.c_elegans.muscles import NeuromuscularJunction
from simulations.c_elegans.config import NEURAL_TICKS_PER_PHYSICS_STEP


class CElegansEngine(SimulationEngine):
    """
    SimulationEngine specialised for C. elegans.

    Overrides _observation_to_sensory_inputs to use SensorEncoder,
    and intercepts step() to route muscle activations through the
    NeuromuscularJunction name mapper.
    """

    def __init__(
        self,
        body: CElegansBody,
        environment: AgarPlateEnvironment,
        nervous_system: CElegansNervousSystem,
        sensor_encoder: SensorEncoderProtocol,
        neural_ticks_per_physics_step: int = NEURAL_TICKS_PER_PHYSICS_STEP,
        on_step: StepCallback | None = None,
    ):
        super().__init__(
            body=body,
            environment=environment,
            nervous_system=nervous_system,
            neural_ticks_per_physics_step=neural_ticks_per_physics_step,
            on_step=on_step,
        )
        self._sensor_encoder = sensor_encoder

    def _observation_to_sensory_inputs(
        self,
        obs: EnvironmentObservation,
        body_state: BodyState,
    ) -> dict[str, float]:
        return self._sensor_encoder.encode(obs, body_state)

    def step(self) -> SimulationStep:
        """
        Override to translate nervous system outputs through
        the NeuromuscularJunction mapper before applying to body.
        """
        t0 = time.perf_counter_ns()

        last_body = self.body.get_state()
        obs = self.environment.step(self._body_state_as_dict(last_body))
        sensory_inputs = self._observation_to_sensory_inputs(obs, last_body)

        motor_outputs: dict[str, float] = {}
        for sub_tick in range(self.neural_ticks_per_physics_step):
            motor_outputs = self.nervous_system.tick(
                sensory_inputs,
                current_tick=(
                    self._tick * self.neural_ticks_per_physics_step + sub_tick
                ),
            )

        # Translate from nervous-system naming to MuJoCo actuator naming
        ctrl = NeuromuscularJunction.to_ctrl(motor_outputs)
        body_state = self.body.step(ctrl)

        self._tick += 1
        elapsed = (time.perf_counter_ns() - t0) / 1e6

        step = SimulationStep(
            tick=self._tick,
            body_state=body_state,
            observation=obs,
            motor_outputs=motor_outputs,
            neural_states=self.nervous_system.get_neuron_states(),
            elapsed_ms=elapsed,
        )
        self._history.append(step)

        if self.on_step is not None:
            self.on_step(step)

        return step


def build_c_elegans_simulation(
    use_connectome_cache: bool = True,
    food_position: tuple[float, float, float] = (0.03, 0.0, 0.0),
    log_level: str = "WARNING",
    on_step: StepCallback | None = None,
) -> tuple[CElegansEngine, SensorimotorLoop]:
    """
    Factory: load connectome, build all subsystems, return engine + loop.

    Args:
        use_connectome_cache: Use cached JSON connectome if available.
        food_position:        (x, y, z) food source in biological metres.
        log_level:            PAULA neuron log verbosity.
        on_step:              Optional per-step callback.

    Returns:
        (engine, sensorimotor_loop) ready to run.
    """
    # 1. Connectome
    logger.info("Loading C. elegans connectome …")
    connectome = load_connectome(use_cache=use_connectome_cache)
    print_connectome_summary(connectome)

    # 2. Nervous system (302 PAULA neurons)
    logger.info("Building PAULA nervous system …")
    nervous_system = CElegansNervousSystem(connectome, log_level=log_level)

    # 3. Body (MuJoCo)
    logger.info("Initialising MuJoCo body …")
    body = CElegansBody()

    # 4. Environment
    logger.info("Initialising agar plate environment …")
    environment = AgarPlateEnvironment(food_position=food_position)

    # 5. Sensor encoder
    sensor_encoder = SensorEncoder()

    # 6. Engine
    engine = CElegansEngine(
        body=body,
        environment=environment,
        nervous_system=nervous_system,
        sensor_encoder=sensor_encoder,
        on_step=on_step,
    )

    # 7. Active inference loop wrapper
    loop = SensorimotorLoop(engine, log_free_energy=True)

    logger.info("C. elegans simulation ready.")
    return engine, loop
