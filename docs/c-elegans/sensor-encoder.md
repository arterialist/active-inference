# Sensor Encoder

← [Index](../INDEX.md) | [C. elegans Overview](overview.md) | **File:** `simulations/c_elegans/sensors.py`

`SensorEncoder` maps `EnvironmentObservation + BodyState` → `dict[str, float]` keyed by neuron name. All outputs normalised to `[0, 1]`.

## Chemosensory Encoding

| Neuron pair | Molecule |
|-------------|----------|
| ASEL, ASER | NaCl |
| AWCL, AWCR | butanone |
| AWBL, AWBR | 2-nonanone |
| AFDL, AFDR | temperature |
| ASHL, ASHR | nociceptive |
| ASJL, ASJR | ascaroside |
| AIZL, AIZR | NaCl |

## Mechanosensory Encoding

| Neuron | Body site | Encoding |
|--------|-----------|----------|
| PLML, PLMR | touch_post_sensor | `magnitude / (magnitude + 1e-9)` (saturating) |
| ALML, ALMR | touch_ant_sensor | `magnitude / (magnitude + 1e-9)` (saturating) |
| AVM | touch_ant_sensor | `magnitude / (magnitude + 1e-9)` (saturating) |
| PVM | touch_post_sensor | `magnitude / (magnitude + 1e-9)` (saturating) |

## Proprioceptive Encoding

Two sources:

1. **Stretch receptors** (PVDL, PVDR at segment 10; DVA at segment 6): `|angle| / JOINT_ANGLE_MAX_RAD`, clipped to [0, 1].
2. **Motor proprioception** (B-type motor neurons DB/VB): **signed** yaw curvature from a joint **several segments anterior** of the motor neuron’s mapped segment (`SensorEncoder._PROPRIO_ANT_OFFSET`, currently **4**), normalised with `tanh` against `JOINT_ANGLE_MAX_RAD`. Keys are `_mpr_{neuron_name}`. This non-local, alternating DB/VB drive matches the Wen et al. 2012-style coupling used in the virtual lab (avoids local positive-feedback locking).

## See Also

- [Agar Plate Environment](agar-plate-environment.md) — source of chemical concentrations
- [Neuromodulation](../neuromodulation/overview.md) — processes the chemosensory signal further
- [Biological Constants](biological-constants.md) — CHEMOSENSORY_NEURONS, TOUCH_NEURONS lists
