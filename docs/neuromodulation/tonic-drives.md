# Tonic Drives

← [Index](../INDEX.md) | [Neuromodulation Overview](overview.md)

Three tonic currents are injected each tick to maintain biologically realistic baseline activity.

## Forward Drive (`_inject_tonic_forward`)

Models AVB↔B-motor gap junction coupling.

| Target | Parameter | Default | Evolved | Description |
|--------|-----------|---------|---------|-------------|
| AVBL, AVBR | `_TONIC_FWD_CMD` | 0.25 | 0.262 | Depolarising current to command interneurons |
| DB*, VB* neurons | `_TONIC_FWD_MOTOR` | 0.0 | 0.098 | Depolarising current to B-type motor neurons |

The forward command interneurons (AVB) maintain a tonic depolarisation that biases the worm toward forward locomotion, reflecting *C. elegans*' ~80% forward bias.

## OFF-Cell Tonic (`_inject_off_cell_tonic`)

Models spontaneous firing of AWC/ASER in the absence of stimulus.

| Target | Parameter | Default |
|--------|-----------|---------|
| AWCL, AWCR, ASER | `_TONIC_OFF_CELL` | 0.15 |

OFF cells are tonically active at baseline and suppressed by stimulus presence (see [per-synapse modulation](per-synapse-modulation.md) for OFF-cell M1 suppression logic).

## Motor Proprioception (`_inject_motor_proprioception`)

B-type motor neurons receive curvature feedback from the sensor encoder (keys prefixed `_mpr_`).

| Parameter | Value |
|-----------|-------|
| `_PROPRIO_MOTOR_GAIN` | 0.08 |

Implements the Wen et al. 2012 proprioceptive feedback mechanism: body curvature at each motor neuron's anatomical position feeds back to that neuron, creating a distributed body-state signal without dedicated proprioceptor neurons.

## See Also

- [Sensor Encoder](../c-elegans/sensor-encoder.md) — generates `_mpr_` keys
- [Biological Constants](../c-elegans/biological-constants.md) — COMMAND_INTERNEURONS_FORWARD
- [Neuron Presets](../config/neuron-presets.md) — command interneuron parameter rationale
