# SimulationEngine

← [Index](../INDEX.md) | **File:** `simulations/engine.py`

Orchestrates one physics body, one environment, and one nervous system in a tick-based loop.

## Constructor

```python
class SimulationEngine:
    def __init__(
        body:                        BaseBody,
        environment:                 BaseEnvironment,
        nervous_system:              BaseNervousSystem,
        neural_ticks_per_physics_step: int = 2,
        on_step:                     StepCallback | None = None,
        record_neural_states:        bool = True,
        max_history:                 int = 200,
    )
```

## Key Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `reset()` | `SimulationStep` | Reset all three subsystems, return initial state |
| `step()` | `SimulationStep` | Advance one physics step (may run multiple neural ticks) |
| `run(n_steps, ...)` | `list[SimulationStep]` | Run *n* steps with optional progress bar |

## step() Sequence

1. Read body state and environment observation.
2. Flatten observation to `dict[str, float]` via `_observation_to_sensory_inputs`.
3. Run `neural_ticks_per_physics_step` neural ticks, collecting last motor output.
4. Step the body with motor output.
5. Step the environment with new body state.
6. Record timing and neural states into `SimulationStep`.

## SimulationStep Dataclass

| Field | Type | Description |
|-------|------|-------------|
| `tick` | `int` | Global tick counter |
| `body_state` | `BodyState` | Position, orientation, joint angles, contacts |
| `observation` | `EnvironmentObservation` | Chemicals, touch forces, proprioception |
| `motor_outputs` | `dict[str, float]` | Per-muscle activation |
| `neural_states` | `dict[str, Any]` | Membrane potentials, firing flags |
| `elapsed_ms` | `float` | Wall-clock time for this step |

## See Also

- [SensorimotorLoop](sensorimotor-loop.md) — wraps engine with free-energy instrumentation
- [Abstract Interfaces](interfaces.md) — BaseBody, BaseEnvironment, BaseNervousSystem
- [C. elegans Engine](../c-elegans/engine-factory.md) — concrete subclass
