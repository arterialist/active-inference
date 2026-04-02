# C. elegans Implementation

← [Index](../INDEX.md)

The first organism: *Caenorhabditis elegans* — 302 neurons, Cook et al. 2019 connectome, simulated in MuJoCo.

## Components

| Component | File | Docs |
|-----------|------|------|
| Biological constants | `c_elegans/config.py` | [biological-constants](biological-constants.md) |
| MuJoCo body | `c_elegans/body.py` + `body_model.xml` | [mujoco-body](mujoco-body.md) |
| Agar plate environment | `c_elegans/environment.py` | [agar-plate-environment](agar-plate-environment.md) |
| Sensor encoder | `c_elegans/sensors.py` | [sensor-encoder](sensor-encoder.md) |
| Neuromuscular junction | `c_elegans/muscles.py` | [neuromuscular-junction](neuromuscular-junction.md) |
| Engine & factory | `c_elegans/simulation.py` | [engine-factory](engine-factory.md) |
| Interactive viewer | `c_elegans/interactive_viewer.py` | [interactive-viewer](interactive-viewer.md) |

## Assembly Order

1. Load connectome (cached) → [celegans-loader](../connectome/celegans-loader.md)
2. Build `CElegansNervousSystem` (302 PAULA neurons) → [engine-factory](engine-factory.md)
3. Initialise `CElegansBody` (MuJoCo) → [mujoco-body](mujoco-body.md)
4. Initialise `AgarPlateEnvironment` → [agar-plate-environment](agar-plate-environment.md)
5. Create `SensorEncoder` → [sensor-encoder](sensor-encoder.md)
6. Assemble `CElegansEngine` → [engine-factory](engine-factory.md)
7. Wrap in `SensorimotorLoop` → [sensorimotor-loop](../engine/sensorimotor-loop.md)
