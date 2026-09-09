# Metabolic rest challenge

This challenge is the authorized catalyst for V3 and is implemented by
`experiments/embodied_metabolic_rest_causal.py`.

The environment changes one rule: food contact is delayed nutrition rather
than immediate satiation.  The body must digest food over time; locomotion and
exploration consume usable energy.  The prior V2 agent is evaluated unchanged
first.  Only if it fails the preregistered energy/survival condition do the
metabolic transducers and the PAULA SLEEP population become justified.

The two-seed full-tick run passes: intact V3 ends at energy `0.58834`, while
unchanged V2 ends at `0.36616`; removing either the SLEEP output or the
metabolic afferents removes the advantage.  The retained evidence bundle is
`experiments/results/embodied_metabolic_rest_causal_20260803T101248Z/`.
