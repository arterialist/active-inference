# Synaptic Data Structures

← [Index](../INDEX.md) | [NeuronParameters](neuron-parameters.md)

## PostsynapticInputVector (`u_i`)

The receiving end of a synapse on the dendrite.

| Field | Type | Class default | Runtime init | Description |
|-------|------|---------------|--------------|-------------|
| `info` | `float` | U(0.5, 1.5) | U(0.5, 1.5) | Synaptic efficacy for information |
| `plast` | `float` | U(0.5, 1.5) | U(0.5, 1.5) | Synaptic efficacy for plasticity signal |
| `adapt` | `ndarray` | zeros(2) | U(0.1, 0.5) | Receptor sensitivity to each neuromodulator |

## PresynapticOutputVector (`u_o`)

The releasing end at the axon terminal.

| Field | Type | Class default | Runtime init | Description |
|-------|------|---------------|--------------|-------------|
| `info` | `float` | U(0.5, 1.5) | p (1.0) | Spike amplitude transmitted |
| `mod` | `ndarray` | zeros(2) | U(0.1, 0.5) | Neuromodulator release profile |

> **Note:** Class-level `default_factory` values differ from runtime initialization for `adapt`, `u_o.info`, and `u_o.mod`. The runtime constructors in `add_synapse()` and `add_axon_terminal()` override these defaults to the values shown in the "Runtime init" column.

## Wrapper Types

**PostsynapticPoint:** Wraps `u_i` + a local potential `float`. One per input synapse.

**PresynapticPoint:** Wraps `u_o` + retrograde input susceptibility `u_i_retro`. One per axon terminal.

## See Also

- [tick() Phase E — Plasticity](tick-method.md#phase-e--plasticity) — weight update using u_i
- [Retrograde Signaling](tick-method.md#phase-e--plasticity) — u_o update via E_dir
