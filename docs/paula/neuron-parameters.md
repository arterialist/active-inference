# NeuronParameters

← [Index](../INDEX.md) | [PAULA Theory](../theory/paula.md) | **Source:** `neuron-model/neuron/neuron.py`

Every PAULA neuron is configured with a `NeuronParameters` dataclass. Arrays are auto-resized to match `num_neuromodulators` in `__post_init__`.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `r_base` | `float` | 1.0 | Resting primary threshold |
| `b_base` | `float` | 1.2 | Post-cooldown threshold (higher than `r`, prevents immediate re-spiking) |
| `c` | `int` | 10 | Refractory period (ticks) |
| `lambda_param` | `float` | 20.0 | Membrane time constant (leaky integrator decay) |
| `p` | `float` | 1.0 | Spike output amplitude |
| `delta_decay` | `float` | 0.95 | Cable propagation decay per compartment |
| `eta_post` | `float` | 0.01 | Postsynaptic learning rate |
| `eta_retro` | `float` | 0.01 | Retrograde (presynaptic) learning rate |
| `beta_avg` | `float` | 0.999 | Firing rate EMA decay |
| `gamma` | `ndarray` | [0.99, 0.995] | M_vector EMA decay per modulator |
| `w_r` | `ndarray` | [-0.2, 0.05] | M_vector → primary threshold sensitivity |
| `w_b` | `ndarray` | [-0.2, 0.05] | M_vector → post-cooldown threshold sensitivity |
| `w_tref` | `ndarray` | [-20.0, 10.0] | M_vector → learning window sensitivity |
| `num_neuromodulators` | `int` | 2 | Dimensionality of M_vector |
| `num_inputs` | `int` | 10 | Number of postsynaptic points (synapses) |

## Global Bounds

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_SYNAPTIC_WEIGHT` | 2.0 | Excitatory ceiling for `u_i.info` and `u_o.info` |
| `MIN_SYNAPTIC_WEIGHT` | 0.01 | Floor (prevents zero-weight deadlock) |
| `MAX_MEMBRANE_POTENTIAL` | 20.0 | Clamp for S |
| `MIN_MEMBRANE_POTENTIAL` | -20.0 | Clamp for S |

## See Also

- [Synaptic Structures](synaptic-structures.md) — u_i / u_o
- [Neuron State Variables](neuron-state.md) — dynamic variables at runtime
- [Neuron Presets](../config/neuron-presets.md) — per-class parameter configurations
- [tick() Method](tick-method.md) — how parameters are used at each step
