# Motor Decoding

← [Index](../INDEX.md) | **File:** `simulations/c_elegans/neuron_mapping.py` (method `_decode_motor_outputs`)

Translates PAULA motor neuron membrane potentials into per-segment muscle activations.

## Pipeline

```
Motor neuron S values
        │
   Graded output normalisation  (S / _S_NORM, clipped [0,1])
        │
   Segment map  (Gaussian anatomical spillover)
        │
   Excitation / Inhibition accumulation  (per segment, per D/V channel)
        │
   Reciprocal inhibition  (D↔V cross-suppression)
        │
   Low-pass muscle filter  (α = 0.3)
        │
   {seg{N}_{QUAD}: float}  → NeuromuscularJunction
```

## Segment Map

Built lazily via Gaussian spillover (σ = 0.8 body segments):

```python
dist = abs(frac_pos - seg_frac) * (N_BODY_SEGMENTS - 1)
w    = exp(-0.5 × (dist / sigma)²)   # zero if w < 0.01 or dist > 3.0
```

Each motor neuron contributes to nearby segments weighted by anatomical proximity.

## Graded Output

Motor neurons transmit via graded potentials (not binary spikes):

```
graded(S) = clip(S / _S_NORM, 0, 1)      # _S_NORM = 0.25
```

## Excitation and Inhibition

| Neuron prefix | Type | Target | Weight |
|---------------|------|--------|--------|
| DB | Excitatory forward | Dorsal | `_FWD_WEIGHT = 1.0` |
| DA | Excitatory backward | Dorsal | `_BKW_WEIGHT = 0.3` |
| AS | Excitatory (dorsal sublateral) | Dorsal | 0.5 |
| VB | Excitatory forward | Ventral | `_FWD_WEIGHT = 1.0` |
| VA | Excitatory backward | Ventral | `_BKW_WEIGHT = 0.3` |
| DD | Inhibitory | Dorsal | `_INHIB_WEIGHT = 0.3` |
| VD | Inhibitory | Ventral | `_INHIB_WEIGHT = 0.3` |

Net excitation per segment:

```
d_excit = d_exc − d_inh × _INHIB_WEIGHT
v_excit = v_exc − v_inh × _INHIB_WEIGHT
```

## Reciprocal Inhibition

Cross-inhibition between dorsal and ventral channels produces the alternating contraction pattern:

```
d_push = clip(d_excit − v_excit × _RECIP_INHIB, 0, ∞)
v_push = clip(v_excit − d_excit × _RECIP_INHIB, 0, ∞)
```

`_RECIP_INHIB = 0.5`

## Muscle Filtering

Final activations are low-pass filtered to smooth jitter:

```
activation = MUSCLE_FILTER_ALPHA × target + (1 − MUSCLE_FILTER_ALPHA) × prev
```

`MUSCLE_FILTER_ALPHA = 0.3` — 30% new signal, 70% previous. Outputs stored in `_muscle_activations`.

## Constants

See [Motor Decoding Constants](../config/motor-constants.md) for the full table.

## See Also

- [Neuromuscular Junction](../c-elegans/neuromuscular-junction.md) — downstream translator to MuJoCo
- [Biological Constants](../c-elegans/biological-constants.md) — MOTOR_NEURON_POSITIONS, VENTRAL_CORD_MOTOR_NEURONS
- [Neuron Presets](../config/neuron-presets.md) — motor neuron parameter rationale
