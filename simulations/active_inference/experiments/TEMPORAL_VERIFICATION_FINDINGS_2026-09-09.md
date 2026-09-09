# Delayed verification in a continuously learning embodied network

## Declared comparison

The prior pilot separated recency-dependent memory, unstable coupled correction
and teaching against old bodily evidence. This experiment tests delayed neural
verification during acquisition itself. Both alternatives have the same
590-cell network, real audiovisual stimuli, body, 64-tick somatic delay and
positive learning. One adds 64 ticks of dendritic travel to predictor-to-error
projections; the other does not. Motor projections retain their original delay.
Delay attenuation is compensated at construction, not on each tick. No host
prediction history, error, class decoder or teaching switch enters the brain.

Both alternatives use quarter-strength sensory input to the mixed populations,
half-strength comparison input and a 64-tick context trace in the existing
PredictiveReceptorNeuron. Context inhibition is unchanged. No neuron equation
is changed. These are provisional operating conditions, not a fitted optimum.
Their individual benefits cannot be separated by the two-condition comparison.
Only verification-path timing and its necessary attenuation compensation differ
within a pair.

The amplitude choice follows the prior trace rather than a parameter search.
The observed squared context norm reached 7.12 for recording 0 and 15.56 for
recording 1. Local rate times that norm reached 0.0343 and 0.0683 per tick in
the separate-error preparation. Coupled branches can both change a differential
prediction. Quartering context amplitude reduces this squared norm by sixteen
under a fixed positive activity pattern. Native adaptation and rectification
mean this scaling is only a conditional estimate. It is not a proof of global
stability. The longer trace retains information across feedback delay but also
mixes successive inputs; it is not exact delayed credit assignment.

There are eight bounded courses: matched and unmatched verification for seeds
11, 23, 44 and 77. Seeds 11 and 44 use presentation order 0; seeds 23 and 77 use
order 1. Thus each architectural comparison is paired within a seed and both
orders occur, but this is not a full order-by-seed factorial. Acquisition has
16 episodes of 364 ticks. All eight final context/recording/selected-weight-reset
probes also last 364 ticks and continue learning. The physical delay queue
continues through every acquisition episode and branches with the exact body
and neural state for probes. Initial zero history is consistent with the
initial resting, unloaded arm.

Primary evidence is the full trajectory, especially ticks 0–63 before current
bodily evidence, 64–127, 128–255, 256–299, and stimulus withdrawal at 300–363.
Compare acquired versus selected-reset memory on the same initial state, not
only matched versus unmatched networks whose bodies evolved differently.
Retain opposing predictions, actual neural teaching, weight changes and motor
motion. A small final angle, silence or apparent oscillation damping is not an
acceptance criterion by itself. Replication cannot convert a weak test into
evidence for semantics, latent context, reasoning or consciousness.

## Biological relation and scope

[Miall et al., 1993](https://www.tandfonline.com/doi/abs/10.1080/00222895.1993.9942050)
proposed separating a rapid predictive representation from a delayed copy used
for sensory comparison. The indexed abstract was available; direct publisher
access returned 403. This experiment borrows that temporal distinction, not a
verified cerebellar circuit or complete Smith-predictor controller. The current
prediction concerns an imposed load associated with sensory input, not a learned
forward model of the entire body.

[Shindou et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6585681/) report
temporally selective dopamine-dependent potentiation after prior synaptic
activity in mouse striatum. The indexed abstract was reviewed; direct PMC access
was blocked by a browser check. That result motivates retaining local past
activity, but does not validate this model's graded trace, 64-tick constant or
signed-error update. The existing extension remains phenomenological.

## Preflight

The impulse test verifies an additional 64 ticks on the comparison copy while
preserving its isolated arrival waveform to floating-point precision. The
motor projection configuration is identical. Checkpoint continuation includes
signals in flight and the physical afferent queue. The independent learning
auditor uses the declared 64-tick context recurrence and rejects both corrupted
weights and the old, incorrect 8-tick recurrence.

All eight acquisition/probe courses completed, followed by eight selected-weight
transfer courses. They executed 78,944 ticks in total. The previous recorded
producers and shared neuron-model sources remain unchanged. This experiment is
a step toward usable learned sensorimotor organization, not a replacement
objective for the full PAULA/ALERM embodied brain.

## Results and the body-state confound

The acquisition/probe courses recorded 69,888 ticks across four graph seeds.
Every recorded selected-learning update, MuJoCo trajectory, physical-afferent
history and neural-to-muscle command passed its independent check. Each of the
eight transfer courses additionally reproduced the original first 364-tick
acquisition episode exactly from the birth checkpoint, then ran eight 96-tick
probes. Those 9,056 executed ticks include 6,144 newly recorded probe ticks.

The final acquisition body can be displaced by about 1.4 radians. In a probe,
the joint-restoring reflex then contributes to motor output alongside learned
load compensation. Net torque along an imposed load can therefore accompany a
correct neural prediction while the arm returns from its prior displacement.
The first inspection confused these quantities. Looking at both opponent
predictor outputs corrected that interpretation. Body state belongs in the
causal analysis, not merely in the rendering.

To separate stored content from that displacement, the transfer assay copies
only the selected learned prediction weights into the original executable birth
brain. Every body starts resting at zero, with zero somatic delay history.
All other neural state starts at birth. Learning remains enabled. This is a
diagnostic intervention, not a reset routine for the agent.

Across all four seeds, both context signs and both recordings, the transferred
weights produce a load-specific response before fresh bodily feedback. This
does not mean every response is correct. In each architecture, 14 of the 16
learned cases have less displacement at 256 ms than their birth-weight control.
The full trajectories, not this count, determine the interpretation:

| Observation | Delayed verification | Unmatched verification |
| --- | --- | --- |
| Seeds 11, 23 and 44, recording 0 | Small wrong-sign output at zero-based ticks 6–7, magnitude below 0.00006; correct sign from tick 8 through tick 95 | Same temporal pattern |
| All four seeds, recording 1 | No wrong-sign prediction in the 96 recorded ticks; initially zero, then positive along the true load | Same sign pattern |
| Seed 77, context 0, recording 0 | Wrong sign at ticks 6–43 and 74–77; minimum −0.02573 release units | Wrong sign at ticks 6–38; minimum −0.01588 |
| Seed 77, context 1, recording 0 | Wrong sign at ticks 6–50 and 73–80; minimum −0.03683 | Wrong sign at ticks 6–40; minimum −0.01960 |
| Favorable cases, displacement reduction at 256 ms | 4.98–36.64 percent | 5.30–33.39 percent |
| Seed 77 recording 0, displacement increase at 256 ms | 0.90 and 1.51 percent in contexts 0 and 1 | 0.36 and 0.57 percent |

All index ranges above are inclusive and zero-based. Neural outputs are sampled
after the tick; body samples follow each 4 ms physical step. Fresh somatic input
first enters at index 64. The tiny two-tick onset reversal is reported, but is
not equated with the sustained seed-77 failure or made a perfection criterion.

In the delayed-verification transfer probes, selected weights do not change
during the first 64 ticks because no teaching signal has yet arrived. Basal
plasticity is still 0.00001 per tick. This is absence of eligible evidence,
not a frozen learning configuration. In the unmatched seed-77/context-0/
recording-0 probe, selected weights already change by up to 0.00001048 before
fresh somatic input; its local rate rises as high as 0.002223. Both architectures
nevertheless retain the same broad strengths and weakness. A correct prediction
in this assay cannot be attributed solely to delayed verification.

The transfer result establishes that selected weights can carry cue-dependent
information sufficient to alter neural predictions and physical action from a
common initial state. It does not establish that all learned organization is in
those weights, or that birth circuitry, supplied context and sensory encoding
are dispensable. It is not a learned dynamical attractor, semantic recognition,
full-body skill, inferred context or consciousness result.

## What this changes next

The earlier preparation's apparent recency-only failure is no longer a complete
description. Both event directions now coexist in three of four graph seeds,
and their physical contribution survives removal of acquired activity and body
history. But the remaining seed-dependent interference and order effects are
unresolved. Presentation order is balanced across seeds rather than crossed
with every seed, so graph dependence and order cannot be fully separated.

The next useful challenge should require combining senses, not polishing this
joint's endpoint. The current two audiovisual recordings each identify their
load on their own; either sense could be redundant. A crossed four-pairing world
can associate the same video with opposing loads depending on the sound, and
the same sound with opposing loads depending on the video. That creates a
measurable need for a joint representation. Test each single-sense deletion,
held-out temporal fragments and selected-weight transfer under the same physical
feedback delay. Do not supply a pairing label or decoded target to the brain.
Keep physical afferents and any schedule cues in the bypass audit.

This will test whether the current populations can acquire useful combinations,
and where hierarchical or supervisory circuitry is actually necessary. The
present timing intervention alone has not justified an adaptive timing
supervisor, a workspace clock or a larger body.

## Reproduction and inspection

The producer is `temporal_verification.py`; `--unmatched` selects the timing
control. Use seeds/order pairs 11/0, 23/1, 44/0 and 77/1. Each source folder is
`.live/research/20260909_temporal_verification_{aligned,unmatched}_orderO_seedS`.
`temporal_memory_transplant.py SOURCE OUTPUT` creates the paired birth-state
assay from each completed source. It checks source and checkpoint hashes and
reproduces the original acquisition prefix before probing.

`temporal_verification_analysis.py --output OUTPUT ROOT...` audits acquisition
and full-state probes. Add `--transplants` for the eight transfer folders.
The analysis stores every probe tick, both prediction channels, teaching,
learning rates, selected-weight changes and physical trajectory with explicit
column names. It rejects missing pairs, duplicate conditions, changed sources,
nonmatching initial histories and premature somatic evidence. The generated
analysis folders are `20260909_temporal_verification_analysis` and
`20260909_temporal_memory_analysis` under `.live/research/`.

`temporal_verification_plot.py ANALYSIS_DIRECTORY` renders both timing conditions
with all four seeds and all four context/recording combinations. Curves are
direction-folded using the imposed load only for offline analysis; the brain
never receives this plotted quantity. Dotted birth curves overlap before
feedback. No raw recordings or media are committed with this result.

The final focused suite passed 26 tests in 16.66 seconds: temporal verification,
opponent context, context organization, predictive receptors and runtime
checkpoints. It includes an exact impulse-delay comparison, continued plastic
checkpoint replay, deliberate weight corruption, selected-only transfer and
empty-evidence rejection. This test result covers the experimental machinery,
not mammal-level behavior or consciousness.

## Relation to the ALERM specification

The current `al-paper/alerm.tex` manuscript was reread in full. Its prospective
claim concerns composing local regulation into useful embodied organization,
not merely producing individually stable cells. The present test examines one
temporal interface in that composition. It does not reproduce the manuscript's
two-channel supervisory control law: this existing predictor amplifies plasticity
with error magnitude, whereas the manuscript distinguishes stress-related search
from consolidating regulation. Its metabolic constraint is also absent from this
one-joint preparation. These differences prevent treating a success or failure
here as validation or refutation of the whole framework.

The immediate architectural question is whether a current prediction can serve
action while a differently timed projection supports learning from delayed
evidence. A future regulator may control the strength or timing of those
relationships without representing every cue identity itself. That remains a
hypothesis; no adaptive timing supervisor is installed in this experiment.
