"""True active inference — a planning agent with an explicit generative world model.

Unlike the reactive winner-take-all `organism`, this agent:
  - holds an explicit generative model (A: likelihoods, B: transitions, C: preferences, D: prior);
  - PERCEIVES by Bayesian belief-updating (variational free-energy minimisation);
  - PLANS by minimising EXPECTED free energy over a tree of future action/observation branches
    (sophisticated inference), balancing pragmatic value (reach preferred outcomes) against
    epistemic value (gather information).

`tmaze_agent.py` implements it on the canonical epistemic-foraging T-maze; `record.py` renders a
self-contained HTML visualization. See README.md.
"""
