# Per-Synapse Modulation

← [Index](../INDEX.md) | [Neuromodulation Overview](overview.md)

Given `delta_c` from the [dual-EMA filter](dual-ema-filter.md), the system computes synapse-level neuromodulatory signals for each sensory neuron.

## Regular Sensory Neurons

```
if delta_c < −STRESS_DEADZONE:
    excess = |delta_c| − STRESS_DEADZONE
    m0 = tanh(excess × K_STRESS_SYN / 5.0) × 5.0
    m1 = 0
elif delta_c > 0:
    m0 = 0
    m1 = tanh(delta_c × K_REWARD_SYN / 5.0) × 5.0
else:
    m0 = 0, m1 = 0
```

## OFF-Cell Neurons (AWCL, AWCR, ASER)

Tonically active, suppressed by stimulus, burst on removal (Chalasani et al. 2007):

```
m1 = tanh(clamped × K_OFF_SUPPRESS / 5.0) × 5.0    # Absolute concentration → suppression
if delta_c < −STRESS_DEADZONE:
    m0 = tanh(excess × K_STRESS_SYN / 5.0) × 5.0    # Decrease → burst facilitation
```

## Non-Chemosensory Neurons

Use simple frame-to-frame difference: `delta_c = clamped − prev`.

## Soft Saturation Function

`tanh(x / ceiling) × ceiling` provides smooth soft-capping:

- Linear for small inputs (derivative = 1 at origin)
- Smoothly approaches ceiling for large inputs (no hard discontinuity)
- Ceiling is **5.0** for per-synapse signals

## Signal Injection

The `mod = [m0, m1]` array is injected via `set_external_input()`. Inside PAULA, it is multiplied by the synapse's `u_i.adapt` receptor sensitivity before updating `M_vector` (see [tick() Phase A](../paula/tick-method.md)).

## See Also

- [Dual-EMA Filter](dual-ema-filter.md) — computes delta_c
- [Volume Transmission](volume-transmission.md) — global broadcast after per-synapse injection
- [Gain Constants](overview.md#gain-constants)
