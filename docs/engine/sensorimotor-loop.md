# SensorimotorLoop

← [Index](../INDEX.md) | **File:** `simulations/sensorimotor_loop.py`

Wraps `SimulationEngine` with active-inference instrumentation.

## Constructor

```python
class SensorimotorLoop:
    def __init__(
        engine:           SimulationEngine,
        log_free_energy:  bool = True,
        on_step:          StepCallbackWithLoop | None = None,
    )
```

## FreeEnergyTrace

Records per-tick:

| Metric | Description |
|--------|-------------|
| **Prediction error** | Mean absolute membrane potential (`S`) across all neurons (from `neural_states`). This is the raw surprise proxy. |
| **Motor entropy** | Variance of the motor output distribution. Low variance = stereotyped behaviour; high variance = disorganised. |

## Convergence Monitoring

`run()` supports optional convergence monitoring: if the mean prediction error over a rolling window drops below a threshold, the simulation stops early.

## See Also

- [SimulationEngine](simulation-engine.md) — the wrapped engine
- [Active Inference Theory](../theory/active-inference.md)
- [Data Logging](../data-logging/run-logs.md) — FreeEnergyTrace is saved to logs
