# Global Volume Transmission

← [Index](../INDEX.md) | [Neuromodulation Overview](overview.md)

After all sensory neurons are processed, the average chemosensory delta drives a global broadcast to all neurons — modelling diffuse monoamine signaling in *C. elegans*.

## Computation

```
avg_delta = Σ(delta_c for chemosensory neurons) / n_chem

if avg_delta < −STRESS_DEADZONE:
    global_m0 = tanh(excess × K_VOL_STRESS / 2.0) × 2.0
    global_m1 = 0
elif avg_delta > 0:
    global_m0 = 0
    global_m1 = tanh(avg_delta × K_VOL_REWARD / 2.0) × 2.0
```

Volume ceiling is **2.0** (lower than per-synapse ceiling of 5.0).

## Broadcast (`_volume_broadcast`)

For every neuron in the network:

```
M_vector[k] += (1 − gamma[k]) × global_m[k]
```

This additive term, combined with the existing EMA decay `M_vector = gamma × M_vector + ...`, converges `M_vector[k]` toward `global_m[k]` at steady state.

## Biological Rationale

This pathway models diffuse octopamine/tyramine (M0) and dopamine/serotonin (M1) signaling that affects all neurons simultaneously — independent of direct synaptic connectivity. In the [ALERM framework](../theory/alerm.md), this is the global E↔L coupling: environmental energy state modulates network-wide learning dynamics.

## See Also

- [Per-Synapse Modulation](per-synapse-modulation.md) — the targeted synaptic pathway
- [ALERM Theory](../theory/alerm.md) — M_vector as the neuromodulatory state
- [Gain Constants](overview.md#gain-constants) — K_VOL_STRESS, K_VOL_REWARD
