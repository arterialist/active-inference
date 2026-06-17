"""Wire the larval zebrafish body, environment, sensors, and PAULA circuit."""

from __future__ import annotations

from loguru import logger

from simulations.base_body import BodyState
from simulations.base_environment import EnvironmentObservation
from simulations.engine import SimulationEngine
from simulations.sensorimotor_loop import SensorimotorLoop
from simulations.types import SensorEncoderProtocol, StepCallback
from simulations.zebrafish import config as zfc
from simulations.zebrafish.body import ZebrafishBody
from simulations.zebrafish.connectome import load_connectome, print_connectome_summary
from simulations.zebrafish.environment import AquaticArenaEnvironment
from simulations.zebrafish.neuron_mapping import ZebrafishNervousSystem
from simulations.zebrafish.sensors import ZebrafishSensorEncoder


class ZebrafishEngine(SimulationEngine):
    """SimulationEngine specialised for the MuJoCo larval zebrafish model."""

    def __init__(
        self,
        body: ZebrafishBody,
        environment: AquaticArenaEnvironment,
        nervous_system: ZebrafishNervousSystem,
        sensor_encoder: SensorEncoderProtocol,
        neural_ticks_per_physics_step: int = zfc.NEURAL_TICKS_PER_PHYSICS_STEP,
        on_step: StepCallback | None = None,
        record_neural_states: bool = True,
        max_history: int = 200,
    ):
        super().__init__(
            body=body,
            environment=environment,
            nervous_system=nervous_system,
            neural_ticks_per_physics_step=neural_ticks_per_physics_step,
            on_step=on_step,
            record_neural_states=record_neural_states,
            max_history=max_history,
        )
        self._sensor_encoder = sensor_encoder

    def _observation_to_sensory_inputs(
        self,
        obs: EnvironmentObservation,
        body_state: BodyState,
    ) -> dict[str, float]:
        return self._sensor_encoder.encode(obs, body_state)

    def _body_state_as_dict(self, state: BodyState) -> dict[str, object]:
        payload = super()._body_state_as_dict(state)
        payload["extra"] = state.extra
        return payload


def build_zebrafish_simulation(
    food_positions: list[tuple[float, float, float]] | None = None,
    log_level: str = "WARNING",
    on_step: StepCallback | None = None,
    record_neural_states: bool = True,
    max_history: int = 200,
    suppress_connectome_summary: bool = False,
    seed: int | None = 7,
) -> tuple[ZebrafishEngine, SensorimotorLoop]:
    """
    Build a ready-to-run larval zebrafish simulation.

    The current connectome is a reduced source-annotated scaffold. It is wired
    into a MuJoCo body/environment stack so a higher-resolution public
    connectome can replace the scaffold without changing the embodied loop.
    """
    logger.info("Loading reduced larval zebrafish connectome scaffold")
    connectome = load_connectome()
    if not suppress_connectome_summary:
        print_connectome_summary(connectome)

    logger.info("Building PAULA zebrafish nervous system")
    nervous_system = ZebrafishNervousSystem(
        connectome,
        log_level=log_level,
        seed=seed,
    )
    body = ZebrafishBody(arena_radius_m=zfc.ARENA_RADIUS_M)
    environment = AquaticArenaEnvironment(food_positions=food_positions)
    sensor_encoder = ZebrafishSensorEncoder()

    engine = ZebrafishEngine(
        body=body,
        environment=environment,
        nervous_system=nervous_system,
        sensor_encoder=sensor_encoder,
        on_step=on_step,
        record_neural_states=record_neural_states,
        max_history=max_history,
    )
    loop = SensorimotorLoop(engine, log_free_energy=True)
    logger.info("Larval zebrafish simulation ready")
    return engine, loop
