# Motor Decoding Constants

← [Index](../INDEX.md) | [Motor Decoding](../motor-decoding/overview.md)

Constants used in `_decode_motor_outputs` within `CElegansNervousSystem`.

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `_S_NORM` | 0.25 | Graded output normalisation |
| `_INHIB_WEIGHT` | 0.3 | Inhibitory synapse gain |
| `_RECIP_INHIB` | 0.5 | Cross-inhibition (D↔V) factor |
| `_FWD_WEIGHT` | 1.0 | Forward motor neuron (DB/VB) contribution |
| `_BKW_WEIGHT` | 0.3 | Backward motor neuron (DA/VA) contribution |

## See Also

- [Motor Decoding Overview](../motor-decoding/overview.md) — how these are used
- [Neuromuscular Junction](../c-elegans/neuromuscular-junction.md) — downstream of motor decoding
