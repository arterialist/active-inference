# CElegansNervousSystem Constants

← [Index](../INDEX.md) | **File:** `simulations/c_elegans/neuron_mapping.py`

Complete list of tunable parameters on the `CElegansNervousSystem` class. All can be overridden at construction time via `evol_config` dict.

## Constants

| Attribute | Type | Default | Evolved | Description |
|-----------|------|---------|---------|-------------|
| `_K_STRESS_SYN` | float | 4000.0 | 2655.7 | Per-synapse stress (M0) gain |
| `_K_REWARD_SYN` | float | 4000.0 | 2678.0 | Per-synapse reward (M1) gain |
| `_K_VOL_STRESS` | float | 2000.0 | 3689.7 | Volume transmission stress gain |
| `_K_VOL_REWARD` | float | 2000.0 | 1205.3 | Volume transmission reward gain |
| `_STRESS_DEADZONE` | float | 0.00005 | 0.00228 | \|delta_c\| threshold for M0 |
| `_CHEM_EMA_ALPHA_FAST` | float | 0.2 | 0.2 | Fast EMA filter (head-sweep) |
| `_CHEM_EMA_ALPHA_SLOW` | float | 0.01 | 0.0255 | Slow EMA filter (gradient) |
| `_TONIC_FWD_CMD` | float | 0.30 | 0.262 | Tonic current to AVB |
| `_TONIC_FWD_MOTOR` | float | 0.0 | 0.098 | Tonic current to B-motors |
| `_K_OFF_SUPPRESS` | float | 5.0 | 5.0 | Concentration → M1 for OFF cells |
| `_TONIC_OFF_CELL` | float | 0.15 | 0.15 | Tonic baseline for AWC/ASER |
| `_PROPRIO_MOTOR_GAIN` | float | 0.10 | 0.08 | Motor proprioception gain (class default; lab-tuned) |
| `_PROPRIO_TAIL_DECAY` | float | 0.5 | — | Anterior→posterior taper on proprioceptive gain (0=uniform, 1=no tail proprioception) |
| `_HEAD_CPG_FREQ_HZ` | float | 0.4 | — | Head oscillator frequency (Hz) on DB1/VB1 |
| `_HEAD_CPG_AMP` | float | 0.08 | — | Head CPG amplitude (0 disables) |

## See Also

- [Neuromodulation Overview](../neuromodulation/overview.md) — what these constants do
- [Evolved Config Files](../data-logging/evolved-configs.md) — evolved values source
- [Evolutionary Parameter Space](../evolution/parameter-space.md) — search ranges
- [Lab parity](../c-elegans/lab-parity.md) — when lab vs demo can diverge
