# MuJoCo Body

← [Index](../INDEX.md) | [C. elegans Overview](overview.md) | **File:** `simulations/c_elegans/body.py`

`CElegansBody` wraps a MuJoCo model defined in `body_model.xml`.

## Scaling

All lengths in the MJCF are **1000× biological**. Biological 1 mm → model 1.0 m. This keeps MuJoCo above its internal precision floor (`mjMINVAL ≈ 1e-15`). Dimensionless locomotion dynamics are unaffected.

## MJCF Model Structure

- 13 capsule segments (`seg0` = head at origin, `seg12` = tail at ~+1 m model space).
- Each inter-segment joint has two DOF: **pitch** (y-axis, dorsal-ventral) and **yaw** (z-axis, left-right).
- **48 actuators**: 4 muscle quadrants × 12 inter-segment joints. Named `muscle_seg{N}_{QUAD}`.
- Contact dynamics: low friction floor (`0.005`) models agar surface crawling.
- Physics: `implicitfast` integrator, gravity −5 m/s² (scaled), fluid density 4000, viscosity 0.1.

## Sensor Sites

| Site | Position | Purpose |
|------|----------|---------|
| `nose` | Front of seg0 | Head position tracking |
| `touch_anterior` | Near nose | Anterior touch detection |
| `touch_posterior` | Rear of seg12 | Posterior touch detection |
| `touch_ant_sensor` | Gyro at anterior | Contact force sensor |
| `touch_post_sensor` | Gyro at posterior | Contact force sensor |

## Key Methods

| Method | Description |
|--------|-------------|
| `reset()` | Resets to default pose, runs 2000 settle steps under gravity |
| `step(muscle_activations)` | Applies actuator controls, steps physics by `CONTROL_DECIMATION` substeps |
| `get_state()` | Reads position (biological metres via `_SCALE_MODEL_TO_BIO`), orientation, joint angles/velocities, contact forces, head position |
| `get_body_shape()` | Returns (13, 3) array of segment centres in biological metres |
| `render(camera)` | Returns RGB frame from named camera |

## See Also

- [Biological Constants](biological-constants.md) — N_BODY_SEGMENTS, MUSCLE_QUADRANTS, etc.
- [Neuromuscular Junction](neuromuscular-junction.md) — translates motor outputs to MuJoCo actuator commands
- [Abstract Interface](../engine/interfaces.md) — BaseBody
