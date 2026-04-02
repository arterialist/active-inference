# Synaptic Data Structures

← [Index](../INDEX.md) | [NeuronParameters](neuron-parameters.md)

## PostsynapticInputVector (`u_i`)

The receiving end of a synapse on the dendrite.

| Field | Type | Init | Description |
|-------|------|------|-------------|
| `info` | `float` | U(0.5, 1.5) | Synaptic efficacy for information |
| `plast` | `float` | U(0.5, 1.5) | Synaptic efficacy for plasticity signal |
| `adapt` | `ndarray` | U(0.1, 0.5) | Receptor sensitivity to each neuromodulator |

## PresynapticOutputVector (`u_o`)

The releasing end at the axon terminal.

| Field | Type | Init | Description |
|-------|------|------|-------------|
| `info` | `float` | p (1.0) | Spike amplitude transmitted |
| `mod` | `ndarray` | U(0.1, 0.5) | Neuromodulator release profile |

## Wrapper Types

**PostsynapticPoint:** Wraps `u_i` + a local potential `float`. One per input synapse.

**PresynapticPoint:** Wraps `u_o` + retrograde input susceptibility `u_i_retro`. One per axon terminal.

## See Also

- [tick() Phase E — Plasticity](tick-method.md#phase-e--plasticity) — weight update using u_i
- [Retrograde Signaling](tick-method.md#phase-e--plasticity) — u_o update via E_dir
