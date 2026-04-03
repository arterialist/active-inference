# tick() Method — Full State Transition

← [Index](../INDEX.md) | [NeuronParameters](neuron-parameters.md) | [Neuron State](neuron-state.md)

`tick(external_inputs, current_tick, dt=1.0)` executes five sequential phases.

---

## Phase A — Neuromodulation and Dynamic Parameters

**A.1 Aggregate modulation input:**

```
total_adapt_signal = Σ_synapses( O_ext["mod"] ⊙ u_i.adapt )
```

Element-wise product of incoming neuromodulator concentration vector and receptor sensitivity, summed across active synapses.

**A.2 Update M_vector (EMA):**

```
M(t+1) = γ ⊙ M(t) + (1 − γ) ⊙ total_adapt_signal
```

Per-element decay γ = [0.99, 0.995] — 1% and 0.5% of new signal mixes in per tick.

**A.3 Update firing rate (EMA):**

```
F_avg(t+1) = β_avg × F_avg(t) + (1 − β_avg) × O(t)
```

**A.4 Dynamic thresholds:**

```
r(t) = r_base + w_r · M(t)
b(t) = b_base + w_b · M(t)
```

**A.5 Dynamic learning window:**

```
normalised_F = clip(F_avg × c, 0, 1)
t_ref_homeo  = upper_bound − (upper_bound − lower_bound) × normalised_F
t_ref(t)     = clip(t_ref_homeo + w_tref · M(t), lower_bound, upper_bound)
```

High average firing rate → homeostatic narrowing. Neuromodulators shift on top of that.

---

## Phase B — Input Processing

For each synapse with non-zero input in `input_buffer`:

```
V_local = info_val × (u_i.info + u_i.plast)
arrival = current_tick + cable_distance[synapse_id]
```

Signal is pushed onto a min-heap with arrival time.

---

## Phase C — Signal Integration at Axon Hillock

Pop all signals that have arrived by `current_tick`:

```
I_t = Σ( V_initial × δ^distance )
```

`δ = delta_decay = 0.95`. Cable propagation attenuates signals exponentially with distance.

---

## Phase D — Spike Generation

Leaky integrator update:

```
S(t+1) = S(t) + (dt / λ) × (−S(t) + I_t)
```

Threshold selection:

```
threshold = b   if (current_tick − t_last_fire) ≤ c   (refractory period)
            r   otherwise
```

Spike condition:

```
if S ≥ threshold  and  (current_tick − t_last_fire) ≥ c:
    O = p        # emit spike
    S = 0        # reset
    t_last_fire = current_tick
    → emit tuple (source_id, terminal_id, info_value) for each axon terminal
```

Additional details:
- If `S < 0.005` (near-zero resting state), threshold resets to `r` regardless of refractory status.
- `S` is clamped to `[−20.0, 20.0]` each tick for numerical stability; NaN/infinity triggers a warning.
```

---

## Phase E — Plasticity

For each synapse with input this tick:

**E.1 Prediction error:**

```
E_dir = [ info_received − u_i.info,
          plast_received − u_i.plast,
          mod_0_received, mod_1_received, ... ]
```

**E.2 Temporal direction:**

```
direction = +1   if (current_tick − t_last_fire) ≤ t_ref    (causal window)
            −1   otherwise                                    (anti-causal)
```

**E.3 Postsynaptic weight update (Hebbian STDP):**

```
Δu_i.info = η_post × direction × ‖E_dir‖ × u_i.info
u_i.info  = clip(u_i.info + Δu_i.info, 0.01, 2.0)
```

**E.4 Retrograde signaling:**

```
terminal.u_o.info += η_retro × E_dir[0] × direction
terminal.u_o.mod  += η_retro × E_dir[2:] × direction
```

Both clipped to bounds. Creates a bi-directional learning loop: postsynaptic refines receptive field, then tells presynaptic to adjust output.

## See Also

- [NeuronNetwork](neuron-network.md) — how tick() is called at network level
- [Active Inference Theory](../theory/active-inference.md) — E_dir as free energy gradient
- [Neuromodulation](../neuromodulation/overview.md) — how M0/M1 flow into M_vector
