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
| PLML, PLMR | touch_post_sensor | ‖force‖ / 10 |
| ALML, ALMR | touch_ant_sensor | ‖force‖ / 10 |
| AVM | touch_ant_sensor | ‖force‖ / 10 |
| PVM | touch_post_sensor | ‖force‖ / 10 |

## Proprioceptive Encoding

Two sources:

1. **Stretch receptors** (PVDL, PVDR at segment 10; DVA at segment 6): `|angle| / JOINT_ANGLE_MAX_RAD`, clipped to [0, 1].
2. **Motor proprioception** (B-type motor neurons): segment curvature at neuron's anatomical position, keyed as `_mpr_{neuron_name}` (Wen et al. 2012).

## See Also

- [Agar Plate Environment](agar-plate-environment.md) — source of chemical concentrations
- [Neuromodulation](../neuromodulation/overview.md) — processes the chemosensory signal further
- [Biological Constants](biological-constants.md) — CHEMOSENSORY_NEURONS, TOUCH_NEURONS lists
