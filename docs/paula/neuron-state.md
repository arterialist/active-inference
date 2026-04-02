# Neuron State Variables

← [Index](../INDEX.md) | [NeuronParameters](neuron-parameters.md)

These are the dynamic runtime variables of a PAULA neuron.

## State Variables

| Variable | Type | Initial | Description |
|----------|------|---------|-------------|
| `S` | `float` | 0.0 | Membrane potential at axon hillock |
| `O` | `float` | 0.0 | Output (p on spike, 0 otherwise) |
| `t_last_fire` | `float` | -∞ | Tick of last spike |
| `F_avg` | `float` | 0.0 | Long-term firing rate (EMA) |
| `M_vector` | `ndarray` | zeros | Neuromodulatory state (EMA) |
| `r` | `float` | r_base | Dynamic primary threshold |
| `b` | `float` | b_base | Dynamic post-cooldown threshold |
| `t_ref` | `float` | c × num_inputs | Dynamic learning window |

## t_ref Bounds

| Bound | Formula | Description |
|-------|---------|-------------|
| `upper_t_ref_bound` | `c × num_inputs` | Widest learning window |
| `lower_t_ref_bound` | `2 × c` | Narrowest learning window |

## See Also

- [tick() — Phase A: Neuromodulation](tick-method.md) — how M_vector, r, b, t_ref are updated
- [tick() — Phase D: Spike Generation](tick-method.md) — how S and O are updated
