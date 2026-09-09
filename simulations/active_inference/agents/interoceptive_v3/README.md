# Interoceptive V3: metabolic arbitration and sleep

V3 is implemented as a one-PAULA composition and is covered by the
tick-level `embodied_v3_trajectory_causal.py` harness.  The older
`embodied_metabolic_rest_causal.py` fixture remains as a focused energy/SLEEP
ablation, but is not the sole acceptance criterion.

The prior V2 agent had an accepted PAULA hunger ladder and a PAULA
FORAGE/EXPLORE competition.  In that version, food contact drained hunger
immediately, so it had no separate digestion or usable-energy state.  That
limitation prevented a biologically meaningful decision between continuing to
search, exploring, or resting to digest; V3 supplies the missing body state and
its neural transduction.

V3’s single proposed environmental change is that contact with food loads a
gut-like store, while usable energy increases only over time and active motion
has a metabolic cost.  The smallest body delta has three state variables:

- **gut load:** rises on food contact and falls through digestion;
- **usable energy store:** rises from digestion and falls with basal/activity
  cost;
- **metabolic rate:** reports whether the animal is still, resting, or active.

These are body/interoceptive transducers, not a Python policy.  They provide
bounded afferent currents to PAULA populations.  In this strict V3 profile
there is no path-integration component, so the neural delta is a causal
FORAGE/EXPLORE/SLEEP WTA rather than an un-driven HOME population.  Usable
energy provides the missing exploratory-readiness afferent when the compass
uncertainty ladder is absent; low energy recruits FORAGE and gut load recruits
SLEEP.  SLEEP suppresses the search/motor drive through PAULA inhibition while
retaining basal body steps so digestion proceeds.

The trajectory acceptance is implemented in
`../../experiments/embodied_v3_trajectory_causal.py`; it records every neural
tick and checks physical approach, toxin clearance, exploration, hunger
transition, and post-meal motor suppression.  The focused energy acceptance
is implemented in `../../experiments/embodied_metabolic_rest_causal.py`.  With
seeds 11 and 23 (80 body steps × 4 neural ticks), intact V3 reaches final
energy `0.63931` versus unchanged V2 `0.41839`, emits 768 post-meal SLEEP
spikes, and reduces post-meal actuator drive to `2.37` in the delayed-meal
fixture.
Removing SLEEP output or the metabolic afferents removes the energy advantage.
The implementation does not decode a body state in Python and select a mode;
the competition and veto are PAULA synapses in the single whole-brain build.
