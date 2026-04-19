# MuJoCo Body

← [Index](../INDEX.md) | [C. elegans Overview](overview.md) | **File:** `simulations/c_elegans/body.py`

`CElegansBody` wraps a MuJoCo model defined in `body_model.xml`.

## Scaling

All lengths in the MJCF are **1000× biological**. Biological 1 mm → model 1.0 m. This keeps MuJoCo above its internal precision floor (`mjMINVAL ≈ 1e-15`). Dimensionless locomotion dynamics are unaffected.

## MJCF Model Structure

- 13 capsule segments (`seg0` = head at origin, `seg12` = tail at ~+1 m model space).
- Each inter-segment joint has two DOF: **pitch** (y-axis, dorsal-ventral) and **yaw** (z-axis, left-right).
- **48 actuators**: 4 muscle quadrants × 12 inter-segment joints. Named `muscle_seg{N}_{QUAD}`.
- **Floor:** default geom friction on the plane is `0.005` (tangential) on the broad floor mesh; **segment–floor** contact uses explicit `<pair>` rows with **condim=6** and anisotropic friction `[0.01, 1.2, …]` (low along-body slip, higher transverse) for undulatory thrust.
- **Global `<option>` (aligned with lab crawling):** `implicitfast` integrator, gravity **(0, 0, −9.8)** m/s² in MJCF model units, **`density="0"`** (no fluid buoyancy — avoids lifting the worm off the substrate), **`viscosity="0.3"`** (light viscous drag). Defaults match `body_model.xml`; the virtual lab can override `mjOption` live via its API.
- **Actuators:** default `<general forcerange="±3.5">` (model units); lab may patch per-actuator limits.

## Sensor Sites and Sensors

| Name | Type | Position | Purpose |
|------|------|----------|---------|
| `nose` | Site | Front of seg0 | Head position tracking |
| `touch_anterior` | Site | Near nose | Anterior touch detection |
| `touch_posterior` | Site | Rear of seg12 | Posterior touch detection |
| `touch_ant_sensor` | Touch sensor | References `touch_anterior` | Contact force sensor |
| `touch_post_sensor` | Touch sensor | References `touch_posterior` | Contact force sensor |
| `touch_nose_sensor` | Touch sensor | References `nose` | Nose contact force sensor |

## Key Methods

| Method | Description |
|--------|-------------|
| `reset()` | Resets to default pose, runs 2000 settle steps under gravity |
| `step(muscle_activations)` | Applies actuator controls, steps physics once via `mujoco.mj_step` |
| `get_state()` | Reads **mass-weighted COM** as `BodyState.position` (biological metres via `_SCALE_MODEL_TO_BIO`), orientation, joint angles/velocities, contact forces, segment poses |
| `get_body_shape()` | Returns (13, 3) array of segment centres in biological metres |
| `render(camera)` | Returns RGB frame from named camera |

## See Also

- [Lab parity](lab-parity.md) — virtual lab vs demo server; when MJCF / config can diverge
- [Biological Constants](biological-constants.md) — N_BODY_SEGMENTS, MUSCLE_QUADRANTS, etc.
- [Neuromuscular Junction](neuromuscular-junction.md) — translates motor outputs to MuJoCo actuator commands
- [Abstract Interface](../engine/interfaces.md) — BaseBody
