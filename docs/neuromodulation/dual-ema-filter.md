# Dual-EMA Bandpass Filter

← [Index](../INDEX.md) | [Neuromodulation Overview](overview.md)

The four primary chemosensory neurons (ASEL, ASER, AWCL, AWCR) use a dual-EMA bandpass filter to extract the true navigational gradient while rejecting head-sweep oscillation noise.

## Algorithm

For each chemosensory neuron on each tick:

```
fast(t) = α_fast × clamped + (1 − α_fast) × fast(t−1)
slow(t) = α_slow × clamped + (1 − α_slow) × slow(t−1)
delta_c  = fast − slow
```

## Parameters

| Parameter | Default | Evolved | Description |
|-----------|---------|---------|-------------|
| `_CHEM_EMA_ALPHA_FAST` | 0.2 | 0.2 | Fast EMA: tracks head-sweep (~10 tick window) |
| `_CHEM_EMA_ALPHA_SLOW` | 0.01 | 0.0255 | Slow EMA: tracks environmental gradient (~40+ tick window) |

## Why Two EMAs?

- The **fast EMA** responds to rapid oscillations caused by the worm's sinusoidal head sweep.
- The **slow EMA** tracks the underlying spatial gradient.
- Their **difference** cancels the head-sweep frequency, leaving only the navigational signal dC/dt.

This implements the temporal derivative detection that drives ALERM's M0/M1 computation — the key input to [per-synapse modulation](per-synapse-modulation.md).

## See Also

- [Per-Synapse Modulation](per-synapse-modulation.md) — uses delta_c to compute m0/m1
- [Gain Constants](overview.md#gain-constants) — CHEM_EMA_ALPHA_SLOW is an evolved parameter
- [Evolutionary Optimisation §9.2](../evolution/parameter-space.md) — indices 5 and 6 in the search space
