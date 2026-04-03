# Neuromodulation System

← [Index](../INDEX.md) | **File:** `simulations/c_elegans/neuron_mapping.py` (class `CElegansNervousSystem`)

Implements ALERM Eq. 3–4. Two modulators (M0 and M1) are computed from the temporal derivative of chemosensory concentration and delivered through two pathways.

## Delivery Pathways

| Pathway | Mechanism | Description |
|---------|-----------|-------------|
| **Synaptic injection** | `set_external_input()` | Per-sensory-neuron m0/m1 values passed as `mod` array → goes through PAULA M_vector EMA |
| **Volume transmission** | Direct M_vector addition | Global average M0/M1 added to every neuron's M_vector after network tick |

## Contents

| Topic | Page |
|-------|------|
| Dual-EMA bandpass filter | [dual-ema-filter](dual-ema-filter.md) |
| Per-synapse modulation logic | [per-synapse-modulation](per-synapse-modulation.md) |
| Global volume transmission | [volume-transmission](volume-transmission.md) |
| Tonic drives | [tonic-drives](tonic-drives.md) |

## Gain Constants {#gain-constants}

All gains can be overridden at runtime via `evol_config`:

| Constant | Default | Evolved | Description |
|----------|---------|---------|-------------|
| `_K_STRESS_SYN` | 4000.0 | 2655.7 | Per-synapse stress (M0) gain |
| `_K_REWARD_SYN` | 4000.0 | 2678.0 | Per-synapse reward (M1) gain |
| `_K_VOL_STRESS` | 2000.0 | 3689.7 | Volume transmission stress gain |
| `_K_VOL_REWARD` | 2000.0 | 1205.3 | Volume transmission reward gain |
| `_STRESS_DEADZONE` | 0.00005 | 0.00228 | Minimum \|delta_c\| to trigger M0 |
| `_K_OFF_SUPPRESS` | 5.0 | 5.0 | Absolute concentration → M1 for OFF cells |

## See Also

- [ALERM Theory](../theory/alerm.md)
- [Configuration Reference](../config/nervous-system-constants.md) — full constant list
