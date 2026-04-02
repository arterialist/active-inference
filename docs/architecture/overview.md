# Architecture Overview

← [Index](../INDEX.md)

## High-Level Loop

```
Environment ──sensory stimuli──► Sensor Encoder
                                       │
                                  dict[str, float]
                                       │
                          ┌─────── Nervous System (PAULA) ───────┐
                          │  Sensory Neurons                     │
                          │       │                              │
                          │  Interneurons                        │
                          │       │                              │
                          │  Motor Neurons                       │
                          └──────────────────────────────────────┘
                                       │
                                 dict[str, float]
                                       │
                              Neuromuscular Junction
                                       │
                              MuJoCo Body Physics
                                       │
                                    BodyState
                                       │
                          ◄────── Environment Update
```

Each physics step:

1. **Environment** produces `EnvironmentObservation` (chemicals, contact forces, proprioception).
2. **Sensor encoder** maps observation → flat `dict[str, float]` keyed by neuron name.
3. **Nervous system** runs one or more PAULA ticks, injecting sensory inputs and reading motor outputs.
4. **Neuromuscular junction** translates motor activations → MuJoCo actuator commands.
5. **Body** advances one physics step, returns new `BodyState`.
6. **Environment** updates based on new body state (e.g. food consumption).

## See Also

- [Module Layout](module-layout.md)
- [Core Engine](../engine/simulation-engine.md)
- [C. elegans Implementation](../c-elegans/overview.md)
