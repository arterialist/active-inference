# Recall can regulate learning, but its regulator must remain functional

This investigation follows `COMPOSITION_FINDINGS_2026-09-08.md`. It tests a small
PAULA realization of the memory, recall, and learning feedback proposed in
ALERM. It does not test consciousness. No live agent, neuron-model source, body,
or embodied acceptance suite was changed or started.

## The mechanism we can now test

The preparation contains eight spiking PAULA neurons. Two receive cues, two
receive outcomes, two predict outcomes, and two compare incoming outcomes with
predictions. A comparison neuron releases a pure neuromodulatory signal. That
signal changes the receiving prediction neurons' plasticity rates through the
previous experiment's explicit `PlasticityRateNeuron` extension. All membrane,
delay, refractory, timing-dependent learning, and retrograde dynamics otherwise
come from PAULA. No decoder, loss calculation, phase flag, or software learning
switch controls the network.

The final candidate also uses ordinary PAULA's existing `w_tref` response to
narrow its timing-dependent credit window during modulation. The two outcomes
inhibit the opposite prediction, and either comparator can modulate both
prediction cells. These connections express an explicit assumption that the
outcomes are mutually exclusive. They do not encode which cue predicts which
outcome. All four cue weights begin in the same small range. Swapping the
environment's associations or their acquisition order leaves the wiring intact.

The external preparation presents cue pulses and overlapping outcome pulses,
followed by cue-only probes. It teaches one association, teaches the second,
probes retention after 1,600 silent ticks, then reverses both associations.
There are no neural resets between these phases. Each standard reversal run
lasts 19,200 ticks. The 160-tick trial spacing is an external laboratory
protocol, not a neural clock. Units are simulator ticks, not calibrated
milliseconds. The circuit has no body, energy budget, omission detector, or
learned outcome vocabulary.

## The failed variants explain the successful one

| Intervention | Observation | Causal interpretation within this preparation |
| --- | --- | --- |
| Initial prediction inhibits its modulator | Two associations acquired; cue-only recall retained | Acquired recall can reduce the signal that amplified its own acquisition |
| Remove rate amplification, keep basal adaptation | No acquisition within the same exposure | Amplification is needed on this horizon; this is not a proof that basal learning can never learn |
| Separate cue and outcome timing, preserve pulse counts | No cue-only acquisition | Exposure alone does not explain the result |
| Replay recorded modulator packets at their actual times | Neuronal states reproduce the reference | Explicit causal transport control |
| Shift those packets by 60 ticks within each trial | Acquisition fails despite matched total rate exposure | The timing of modulation matters |
| Remove prediction feedback | Acquisition still works, with much greater modulation | Feedback reduces continuing amplification; it is not necessary for initial acquisition here |
| Reverse the associations in the initial circuit | Both predictions become active; the comparators eventually stop firing | Absence of positive mismatch does not establish correct memory |
| Add sensory competition and shared modulation | Reversal still fails | Silencing an old prediction does not immediately remove its temporal credit |
| Also narrow the learning window during modulation | Reversal succeeds in 17 of 20 first confirmation cases | Timing-window regulation changes the direction of learning, not merely its amount |
| Slow prediction-terminal adaptation to `1e-9` | Behavioral tests pass, but basal terminal updates round to zero | Rejected as a candidate with continuing effective basal adaptation |
| Use `1e-7` prediction-terminal adaptation | Passed 20 new held-out mapping/order cases and three earlier failed cases | The regulator can remain functional while still adapting; finite-horizon evidence only |

The resolved candidate also passed a 70,400-tick run with 100 training trials
per cue before reversal and 100 per cue after reversal. Its matched
window-regulation cuts failed reversal in all five new seeds. Across the 20
held-out cases, all 16,902 basal prediction-terminal update ticks and all 2,220
modulated update ticks changed the represented value. The equation audit's
largest membrane residual was below `4.2e-7`; both plasticity residuals were
below `1.7e-7`. This tolerance covers float32 arithmetic, not a behavioral error
allowance. Separate checks count updates lost to rounding.

Timing transfer failed. In seed 29, replacing the trained two-tick cue spacing
with three-, four-, or six-tick spacing, while retaining 18 cue pulses, produced
no prediction spikes in all 18 transfer probes. The original faster-retrograde
preparation tolerated some of these spacings, so the resolved variant should
not be called a uniform improvement across conditions. A regulator that shuts
learning down near the acquisition threshold can leave little recall margin.
This is a hypothesis about the transfer failure, not yet a demonstrated repair.

A subsequent `timing_recovery` experiment restored two-tick spacing after all
the slower probes, without outcomes, retraining, or state resets. All six
return probes recalled only the correct outcome. The memory remained available
under its learned temporal conditions. The trace explains the failure through
subthreshold integration rather than memory erasure. Cue 0's measured peak
before reset fell from about 0.611 in retention to 0.532 at period three; it
returned to about 0.609 at period two. The resting threshold is 0.6.

For a nonspiking leaky cell with fixed local parameters and one input every
`p` ticks, the periodic peak is `q*w*delta^d / (lambda*(1-a^p))`, where
`a=1-1/lambda`. Using the recorded weights and source gain predicts a period-three
peak about 0.53194, close to the reconstructed 0.53188. The approximation freezes
the small within-trial adaptation and does not describe the reset trajectory
once a neuron fires. `association_analysis` now exports actual per-probe
membrane and threshold margins alongside its outcome classifications.

The first confirmation used seeds 11, 23, 44, 77, and 101 with both mappings and
both learning orders. The `1e-9` diagnostic used new seeds 7, 19, 37, 61, and 89.
The resolved `1e-7` confirmation used new seeds 13, 29, 47, 71, and 97. These are
initial-weight perturbations in one designed preparation. They are not a sample
of organisms or environments. The mechanism was selected through disclosed
exploration before each fixed confirmation batch; failed configurations and
full traces remain available.

The final candidate keeps postsynaptic basal rates at `1e-5`, prediction-cell
retrograde basal rates at `1e-7`, and other retrograde basal rates at `1e-5`.
Both prediction adaptation paths still receive the same local multiplier.
None is disabled. In the targeted seed-44 resolved run, all 813 basal
prediction-terminal update ticks and all 114 modulated update ticks changed
the represented terminal value. The `1e-9` diagnostic's seed-19 run had 781
basal update ticks with zero represented changes. A positive parameter was
therefore insufficient evidence of ongoing adaptation.

## Two trace-level causes

In the seed-11 `corrective` run, the old prediction last fired at tick 11,865.
At tick 11,870, contradictory outcome input had silenced it, but its learning
window was still about 7.893 ticks. The five-tick-old spike still earned positive
credit. With a local rate multiplier about 4,737, the old association grew from
about 0.803835 to 0.827253. A strong learning signal strengthened the wrong
memory because its sign came from a stale causal window.

In the matched `corrective_window` run, the modulator shortened that window to
four ticks through PAULA's existing `w_tref`. The same five-tick age then earned
negative credit. At tick 11,870 the old weight fell from about 0.803835 to
0.776781. The active, outcome-supported prediction remained eligible for
potentiation. This is a local temporal distinction, not an externally supplied
instruction about which weight should decrease.

That correction exposed a second failure. In seed 44 of the first confirmation,
fast retrograde adaptation drove prediction export gains below zero at ticks
11,879 and 12,037. PAULA's information-input path processes positive inputs;
these negative exported values ceased to deliver the expected inhibition to
the comparator. More mismatch then maintained high plasticity. The correction
pathway itself was being changed too quickly. Slowing its adaptation recovered
the failed cases without modifying PAULA's equations. The numerical check then
set a lower limit on how slowly it could adapt in the current arithmetic.

## What this adds to the ALERM investigation

The useful result is a causal loop in which learned recall changes the local
conditions for subsequent learning. Recall, memory, and learning cannot be
assigned independent acceptance scores and assumed to compose. A change in
recall timing can reverse a synaptic update; that update can change the export
gain of the pathway regulating future updates.

The counterexample also sharpens the theory. A one-sided comparison of the
form `e_j = max(0, outcome_j - prediction_j)` has zero error whenever predictions
cover the outcomes. Predicting both mutually exclusive outcomes can therefore
silence both positive-error channels. This elementary reduction explains the
failure motif; it is not an exact scalar equation for the spiking circuit.
An ALERM realization cannot infer correct organization from a quiet stress
channel without showing what mismatches that channel can detect. Learning must
also preserve the pathways through which contradictory evidence can change it.

For hierarchical composition, I would test that condition at every proposed
boundary. Perturb the relationship a module supplies to its consumers, check
whether the regulator still detects the consequence, and then test recovery.
Monitor transmission gain and temporal credit along with the visible output.
The earlier ring experiment already showed that a stable spike pattern can
conceal shrinking transmission margins. This preparation shows that a quiet
regulator can conceal an incorrect prediction.

ALERM's formalization currently uses both learning cessation at stationarity
and a temperature/rate narrative whose signs need disambiguation. This
experiment follows the user's explicit requirement of continuing adaptation.
An appropriate next mathematical target is a bounded, revisable dynamical
regime under continuing plasticity. Neither the paper's proposed objective nor
this experiment establishes that PAULA's implemented update is a gradient
descent of a defined variational free energy. Energy and architectural change
remain untested here. These restrictions concern what this realization proves,
not what an extensible PAULA model could implement.

The next substantive experiment should present delayed or omitted outcomes
with varying cue timing. It should add neural temporal support and an explicit
way for an absent expected event to produce a mismatch. Merely adding another
positive-error cell would preserve the demonstrated blind spot. Successful
reversal with overlapping pulses is not evidence that the network can bridge a
behavioral delay or detect omission. Higher-level composition should follow
that test, with no external timing/error calculation supplied to the brain.
Repeated reversals and consumer attachment are also still required before
claiming a persistent compositional operating regime. One successful reversal,
even with longer exposure, does not establish that property.

## Biological grounding and limits

Expected-reward inhibition of dopamine neurons has causal experimental support
in [Eshel et al., 2015](https://www.nature.com/articles/nature14855). It motivates
the comparison motif, not these eight cells or their parameter values.
[Seol et al., 2007](https://pubmed.ncbi.nlm.nih.gov/17880895/) showed that
neuromodulatory receptor activation can change the polarity and gating of
cortical timing-dependent plasticity. It supports testing effects beyond a
scalar rate. PAULA's retrospective `t_ref` rule and the coefficient `-100` are
not a reconstruction of their molecular mechanism. This experiment implements
signed influences directly; a cell-type-faithful biological realization would
need explicit inhibitory intermediates and appropriate transmitter rules.

The evidence here supports a working adaptive neural mechanism and exposes
conditions under which it fails. It does not establish novel biological laws,
scale invariance of effective equations, or subjective experience.

## Reproduction and machine inspection

All paths below are relative to `active-inference/`. Each run saves its resolved
configuration, manifest, every completed tick, input delivery, dendritic and
network queues, synaptic and terminal values, plasticity rates, and probe
results. The audit imports no PAULA code. It reconstructs membrane updates,
spikes, modulator filtering, the learning window, and both adaptation rules.
`association_replay` separately executes the saved configuration and recorded
inputs, comparing complete recorded state at each tick. It never restores a
saved internal state during execution.

```sh
.venv/bin/python -m simulations.active_inference.experiments.adaptive_association --output .live/research/NEW_RUN --variant corrective_window_resolved_retro --challenge reversal --seed 44
.venv/bin/python -m simulations.active_inference.experiments.association_analysis .live/research/NEW_RUN --output .live/research/NEW_AUDIT.json
.venv/bin/python -m simulations.active_inference.experiments.association_replay .live/research/NEW_RUN
.venv/bin/python -m simulations.active_inference.experiments.association_batch --suite resolved --output .live/research/NEW_BATCH
uv run --no-sync --with pytest python -m pytest tests/test_adaptive_association.py -q
```

Principal evidence directories are `.live/research/20260908_association_confirmation1/`,
`.live/research/20260908_association_timescale_confirmation1/`, and
`.live/research/20260908_association_resolved_confirmation1/`. Each contains the
case list written before execution and the individual successful and failed
records. Earlier exploration is retained under the same dated
`20260908_association_` prefix. Runs are single-worker, with bounded streaming
recording; no large embodied matrix or video-rendering batch is involved.

The instrument regression suite passed 19 tests across
`test_adaptive_association.py` and `test_composition_probe.py`. It includes
same-time and time-shifted packet replay, full saved-config replay, the
credit-window intervention, numerical adaptation checks, and deliberately false
spike-count and success-flag records. These tests validate the instrument and
specific finite preparations, not the embodied agent versions.

There are 128 recorded exploratory and confirmation runs, totaling 2,472,320
ticks and about 652 MB of artifacts. These include failures and repeated causal
controls; the run count is not a count of independent successful replications.
Simulation plus recording took about 730 seconds in total, excluding audits,
replays, and tests. The final confirmation batch used one CPU worker and about
38 MB resident memory in the process sample. The selected seed-44 final
configuration replayed every state exactly for 19,200 ticks, recorded in
`.live/research/20260908_association_resolved_replay.json`.
