# CElegansEngine and Factory

← [Index](../INDEX.md) | [C. elegans Overview](overview.md) | **File:** `simulations/c_elegans/simulation.py`

`CElegansEngine` subclasses `SimulationEngine` with two overrides:
1. `_observation_to_sensory_inputs` → uses `SensorEncoder.encode()`
2. `step()` → translates motor outputs through `NeuromuscularJunction.to_ctrl()`

## Factory Function

```python
def build_c_elegans_simulation(
    use_connectome_cache: bool = True,
    food_position:  tuple[float, float, float] = (0.03, 0.0, 0.0),
    food_positions: list[tuple[float, float, float]] | None = None,
    log_level:      str = "WARNING",
    enable_m0:      bool = True,
    enable_m1:      bool = True,
    evol_config:    dict[str, Any] | None = None,
    max_history:    int = 200,
    ...
) -> tuple[CElegansEngine, SensorimotorLoop]
```

## Assembly Sequence

1. Load connectome (cached) — [celegans-loader](../connectome/celegans-loader.md)
2. Build `CElegansNervousSystem` with 302 PAULA neurons
3. Initialise `CElegansBody` (MuJoCo) — [mujoco-body](mujoco-body.md)
4. Initialise `AgarPlateEnvironment` with food sources — [agar-plate-environment](agar-plate-environment.md)
5. Create `SensorEncoder` — [sensor-encoder](sensor-encoder.md)
6. Assemble `CElegansEngine`
7. Wrap in `SensorimotorLoop` — [sensorimotor-loop](../engine/sensorimotor-loop.md)

## See Also

- [SimulationEngine](../engine/simulation-engine.md) — parent class
- [Neuromodulation System](../neuromodulation/overview.md) — CElegansNervousSystem implements ALERM
- [Configuration Reference](../config/nervous-system-constants.md) — evol_config keys
