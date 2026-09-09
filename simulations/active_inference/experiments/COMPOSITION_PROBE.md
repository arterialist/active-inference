# Relational-state composition probe

2026-09-08. Exploratory mechanism study. No embodied or consciousness claim.

Before the first run, the questions are:

1. Do two three-cell PAULA synfire rings retain two different phase relations
   after their one-time birth pulses, with two PAULA coincidence consumers
   distinguishing those relations? No trained decoder receives the state.
2. Does attaching the consumers affect the source dynamics when retrograde
   plasticity is enabled? Compare no attachment, a shared source terminal, and
   a separate terminal on the same source neuron.
3. If an effect appears, does selectively cutting consumer retrograde events
   remove it? Does freezing t_ref change the effect? These are causal follow-ups,
   not parameters to silently optimize until the experiment passes.

All cells use the exact base Neuron class and the legacy multiplicative rule.
Post and retrograde rates are independently 0 or .01. A .001 arm is available
to distinguish delayed deterioration from robustness. No external modulators
are injected. This does not activate every possible PAULA mechanism: the
constructor kit's beta=.9, gamma=.9 and neutral sensitivity vectors are retained,
and delta_decay is explicitly .95. Instantiated parameters are recorded.

Each source cell has recurrent, birth and perturbation ports. Three ports give
valid 12..18 tick t_ref bounds at cooldown 6. The consumers have cooldown 8,
lambda 2, threshold 1.4 and weights 2 on each of two inputs. These are designed
coincidence detectors, not a biological or consciousness claim. Five seeds
vary recurrent weights uniformly in [3.8,4.2]; seed labels need not yield
different spike trajectories in the frozen condition.

Birth at executed tick 3 activates source 1 and either source 4 or 5. Afterwards
there is no continuing external drive. Source 1 + source 4 feed consumer 7;
source 1 + source 5 feed consumer 8. Success cannot be inferred from average
activity or silence. Record ordered spikes and both consumers, including
responses by the wrong consumer and cessation of recurrence.

Every completed tick preserves cell state, synaptic and terminal vectors,
actual delivered input buffers, dendritic queues and network delivery wheels.
An observer wraps the base tick without changing its arguments or return;
an unobserved identical run must have the same full-state digest. RNG state,
birth drive, exact configuration and source hashes are included. This is an
observational record, not a claim of restorable arbitrary checkpoints.

The initial calibration uses 420 ticks and seed 11. Preserve failed settings.
Confirmation after calibration freezes the circuit and uses seeds 11,23,44,77,101
with a disclosed finite horizon. Seed 11 remains calibration, not held-out.
Windows are 105 ticks, five nominal cycles; all exact spike times remain saved.
Further durations or interventions get separate directories and explicit labels.

Only one worker runs. No live servers or large embodied suites start. This
preparation establishes prerequisites for a later supervisor experiment; it
does not yet contain a supervisor or show developmental emergence.

Starting motif: paula_agent/circuits_b.py, c11_cpg_ring. Unlike that smoke test,
this study preserves long per-tick traces, supplies valid input-count bounds,
compares native adaptation pathways, and measures neural use of phase relations.
Related conceptual work: Barnett & Seth, Physical Review E 108, 014304 (2023),
https://doi.org/10.1103/PhysRevE.108.014304. It does not validate this preparation.
