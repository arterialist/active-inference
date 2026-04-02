# Neuron Parameter Presets

← [Index](../INDEX.md) | [NeuronParameters](../paula/neuron-parameters.md)

Eight preset functions define PAULA parameters for distinct neuron classes within the C. elegans nervous system.

## Presets

| Preset | Applies to | r_base | b_base | c | λ | w_tref |
|--------|-----------|--------|--------|---|---|--------|
| `_base_params` | Fallback | 0.9 | 1.1 | 8 | 15.0 | [15, -8] |
| `_sensory_params` | Sensory neurons | 0.5 | 0.7 | 4 | 6.0 | [12, -6] |
| `_off_cell_sensory_params` | AWCL, AWCR, ASER | 0.25 | 0.35 | 4 | 20.0 | [12, 0] |
| `_motor_params` | Excitatory motor (DB, VB, DA, VA, AS) | 0.2 | 0.3 | 4 | 6.0 | [30, -15] |
| `_motor_inhib_params` | Inhibitory motor (DD, VD) | 0.15 | 0.25 | 3 | 4.0 | [20, -10] |
| `_interneuron_params` | Local interneurons | 0.6 | 0.85 | 8 | 10.0 | [15, -8] |
| `_command_fwd_params` | AVBL, AVBR, PVCL, PVCR | 0.4 | 0.6 | 8 | 10.0 | [12, -6] |
| `_command_bkw_params` | AVAL, AVAR, AVDL, AVDR | 0.85 | 1.1 | 12 | 16.0 | [12, -6] |

## Design Rationale

**Sensory neurons** (`r_base=0.5`, `λ=6`): Low threshold, fast time constant — must respond quickly to environmental stimuli.

**OFF-cell neurons** (`r_base=0.25`, `w_b[1]=1.2`, fast `gamma[1]=0.90`): Extremely low threshold for tonic firing; strong M1-mediated suppression; threshold drops quickly on stimulus removal to generate the OFF burst (Chalasani et al. 2007).

**Excitatory motor neurons** (`r_base=0.2`, `w_tref=[30,-15]`): Low threshold, strong t_ref modulation — neuromodulators dramatically alter learning dynamics.

**Inhibitory motor neurons** (`λ=4`): Faster than excitatory — cross-inhibition must respond quickly for dorsal-ventral alternation.

**Forward command interneurons** (AVB, PVC, `r_base=0.4`): Lower thresholds than backward command neurons, modelling *C. elegans*' ~80% forward locomotion bias.

**Backward command interneurons** (AVA, AVD, `r_base=0.85`): Highest thresholds — only strong aversive stimuli should trigger reversal.

## ALERM Context

Preset differentiation is a direct implementation of ALERM's **Architecture** component: neuronal heterogeneity in temporal integration properties (`λ`, `c`) creates temporal dimensionality expansion — parallel, multi-variate representations of inputs across different timescales. The ALERM paper shows that homogeneous populations suffer 4.6× variance explosion and 25% performance collapse on temporal tasks.

## See Also

- [NeuronParameters](../paula/neuron-parameters.md) — parameter definitions
- [build_paula_network()](../connectome/build-paula-network.md) — where presets are selected
- [ALERM Theory](../theory/alerm.md) — functional heterogeneity rationale
