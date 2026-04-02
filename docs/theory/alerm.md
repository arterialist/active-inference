# ALERM Framework

← [Index](../INDEX.md) | [Theory](active-inference.md)

ALERM (**Architecture, Learning, Energy, Recall, Memory**) is a theoretical framework proposing that biological constraints — metabolic cost, localised plasticity, temporal delays — are not mere limitations but indispensable generative regularisers required for robust intelligence. It formalises the interdependencies between its five components to delineate the functional principles needed for self-organising, biologically plausible computing architectures.

| Component | Role |
|-----------|------|
| **A (Architecture)** | Physical capacity of the spatiotemporal graph G(V,E,d); defines the Markov Blanket boundary |
| **L (Learning)** | Continuous differential adaptation of topology; local-only, no global gradients |
| **E (Energy)** | Metabolic budget bounding the system; selects for sparse coding and temporal efficiency |
| **R (Recall)** | Temporal inference process — evidence accumulation until attractor threshold is crossed |
| **M (Memory)** | Emergent property of architecture: crystallised limit-cycle pathways (M ⊆ A) |

The system minimises **Variational Free Energy** at any hierarchical scale k:

```
F^(k)(s,o) = Σ_i ( D_KL[q(s_i) || p(s_i)] − E_q[log p(o_i|s_i)] )
             \_______________________/   \________________________/
                    Complexity                    Accuracy
```

## C. elegans Application: ALERM Neuromodulators

In the C. elegans simulation, ALERM's M_vector is driven by the chemosensory gradient. Two modulators implement the A↔L coupling:

## Modulators

| Modulator | Trigger | Biological analogue | Effect on PAULA |
|-----------|---------|---------------------|-----------------|
| **M0** (stress) | dC/dt < 0 (concentration decreasing) | Octopamine, tyramine | Broadens learning window (t_ref ↑), lowers threshold (r ↓) → pirouette, explore |
| **M1** (reward) | dC/dt > 0 (concentration increasing) | Dopamine, serotonin | Narrows learning window (t_ref ↓), raises threshold slightly → crystallise current run |

These global signals are broadcast to all neurons via **volume transmission** after each network tick, modelling the diffuse monoamine signaling in *C. elegans*.

See the upstream paper: <https://al.arteriali.st/blog/alerm-framework>

## See Also

- [Neuromodulation System](../neuromodulation/overview.md) — full implementation details
- [Volume Transmission](../neuromodulation/volume-transmission.md)
- [Gain Constants](../neuromodulation/overview.md#gain-constants)
