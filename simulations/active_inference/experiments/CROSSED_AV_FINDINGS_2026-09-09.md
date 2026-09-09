# A physical task that requires both senses

## Declared question

The previous 590-cell preparation learned cue-specific physical predictions,
but each of its two audiovisual recordings could identify the event alone.
It therefore did not establish a need for multimodal integration. This world
crosses their video and audio streams into four equally exposed pairings.
The imposed load is positive for matching source indices and negative for
different indices. A paired world reverses all four signs. Source indices are
experimental bookkeeping; no index or correct action enters the brain.

The same video now occurs with opposing loads depending on the sound. The same
sound occurs with opposing loads depending on the video. In a resting-state
probe, 64 ticks pass before new somatic evidence arrives. Removing either sense
makes the complete delivered input histories identical for a pair requiring
opposite responses during that interval. Unit tests also check identical neural
trajectories in this intervention. This is a conditional impossibility result
for those matched histories, not a universal limit on a single sensory system.

## Brain and experiment

The brain configuration remains the previous delayed-verification architecture,
including its existing graded/predictive PAULA extensions and positive local
adaptation. No subclass or core equation is added. The context lamp stays at 0
and its second mixed bank remains suppressed. Retaining that redundant bank
preserves the earlier configuration for this first world-change test; the full
590-cell count must not be confused with 590 independently useful units.

Eight courses cross graph seeds 11, 23, 44 and 77 with normal/reversed physical
assignment. Each has four balanced acquisition blocks, each containing all four
pairings in a seeded order independent of graph seed and assignment. The body,
brain and afferent queue continue between episodes. Every episode runs for
364 ticks, comprising 300 stimulated ticks and 64 withdrawal ticks.

Each final state supplies eight 96-tick embodied probes with selected prediction
weights intact or reset. There are 24 further 96-tick probes from the birth brain
and a resting body, crossing selected birth/learned weights with both senses,
vision only or audio only. Four learned-weight probes shift sensory onset by
64 recorded samples. Those are temporal transfer tests on previously encountered
samples, not novel-object recognition. An exact 364-tick birth replay checks the
first acquisition episode. Each run executes 9,644 ticks, of which 9,280 are
recorded as new data. All eight execute 77,152 ticks if they finish.

Keep every onset, opponent prediction, teaching arrival, selected update,
actuator command and physical trajectory. In the common-resting probes, inspect
the first 64 ticks separately from subsequent correction. Compare learned to
birth weights, both senses to each sensory deletion, and the two learned
assignments on identical sensory streams. A four-class host decoder or endpoint
score is not a substitute for neural use. Acquisition order is not factorially
counterbalanced; its effects remain an explicit limit.

## Biological relation

[Rigotti et al., 2013](https://www.nature.com/articles/nature12160) found that
nonlinear mixed selectivity in monkey prefrontal populations supports a richer
set of readout functions than specialized responses. The publisher abstract was
reviewed; an author-hosted full-text request timed out. That result motivates
testing whether combinations remain usable by neural consumers, not assuming
that random mixed projections learn this physical task. The present circuit is
not a reconstruction of the recorded prefrontal architecture.

A search also located a 2025 study of locomotion-dependent auditory gating to
parietal cortex. Its full text could not be retrieved in this turn, so it is a
follow-up lead, not evidence for a new gate installed here. Our current body
does not change the prerecorded audiovisual stream. That open sensory loop is
a real limitation before any claim of general active perception.

The source check also found that Okray et al.'s 2023 cross-modal engram paper
was [retracted on 25 March 2026](https://www.nature.com/articles/s41586-026-10355-4.pdf).
The notice reports failed replication and reanalysis of the voltage-imaging
results, with possible analysis-pipeline contamination. It separately states
that the authors replicated the behavioral and connectomic data. The original
binding mechanism must therefore not be treated as established physiological
evidence here. No dependency on this paper was found in the searched custom-agent
Markdown/Python/HTML or neuron-model Markdown/TeX/Python sources. The running
experiment does not implement its proposed DPM bridge.

## Status

All eight courses completed, executing 77,152 ticks. The full analyzer checked
all 74,240 recorded ticks against the learning recurrence, physical replay,
afferent delay and independent audiovisual/load reconstruction. It also checked
acquisition continuity and the matched sensory-deletion histories. Each producer
replayed its first 364-tick episode exactly. The final focused suite passed
35 tests in 18.61 seconds. These checks establish valid experiments, not
successful conjunctive behavior.

## The harder world breaks the earlier result

Every graph seed fails the conjunctive rule. Under normal assignment the
acquired network predicts a negative load for all four pairings. Reversing
acquisition reverses that overall prediction, without learning which pairing
requires which sign. The responses have different magnitudes, so this is not
literally constant activity. Their sign fails to express the required relation.

| Input pairing | Required load, normal world | Learned prediction in all four seeds |
| --- | --- | --- |
| Video 0, audio 0 | Positive | Negative |
| Video 0, audio 1 | Negative | Negative |
| Video 1, audio 0 | Negative | Negative |
| Video 1, audio 1 | Positive | Negative |

The reversed world negates both the required signs and the learned predictions.
After direction-folding by the actual load, its resting learned trajectories
differ from the normal world's by less than 0.000000083 across the measured
prediction/displacement columns. This symmetry supports an acquired assignment
effect, not successful integration. All four acquisition schedules end with
pairing 0/1; a recency contribution is not ruled out.

In both assignments, the same two pairings fail throughout each checked interval
16–31, 32–63 and 64–95. This occurs in acquired-state, resting weight-transfer
and shifted-onset probes. Across the resting comparisons, learned weights
increase displacement at 256 ms by 21.89–43.95 percent for the matching-source
pairings, and reduce it by 32.12–38.64 percent for the mismatching pairings.
The 50 percent directional result is not accepted as partial conjunctive
success: a single learned direction already produces it in this balanced task.
Sensory deletions preserve the predicted indistinguishability before bodily
feedback. They do not repair the missing relation.

## The available interface is not simply missing the distinction

The actual incoming mixed-population histories have a nonzero four-pair
interaction before acquisition and after weight transfer. The largest absolute
channel interaction ranges from 0.06995 to 0.09666 across graph seeds. This
establishes nonadditivity, not a learned representation or useful readout.

`crossed_av_capacity.py` asks a narrower diagnostic question. Hold those recorded
incoming histories fixed and allow any legal constant pair of selected weight
vectors. Could their differential prediction have the correct sign for all four
pairings? The tool first reconstructs both actual predictor trajectories from
the recorded arrivals, weights, one-tick dendritic delay and four-tick somatic
integration. All 8,192 checked predictor ticks pass that reconstruction.

For the real-valued interface, the filtered context obeys
`B[t] = 0.75 B[t-1] + 0.25 * 0.99 * arrivals[t-1]`.
Each selected weight is between zero and one, so their source-aligned difference
lies in `[-1, 1]`. An offline linear program finds lower and upper bounds on the
minimum correct-sign output across every tick and pairing in a declared window.
The upper certificate is a convex combination of event constraints; its
coefficients and the complete input matrix are retained. No optimized neural
weights are saved or installed in the brain.

For the learned-weight factual histories over ticks 16–63, feasible minimum
outputs range from 0.27987 to 0.34507 release units across seeds. The upper/lower
gaps are below 0.000000313. A conservative two-channel native-arithmetic guard
is below 0.000431. Thus the recorded interface can support a correctly signed,
nontrivial differential response under a different weight placement. This is
not evidence that the current local learning law will find those weights.

The first reconstruction attempt failed by 0.000000012 because a float64 matrix
product did not reproduce PAULA's float32 local potentials, summation order and
somatic arithmetic. The corrected factual replay reproduces those operations;
the strict 0.000000000002 check was not relaxed. The ideal linear capacity
calculation separately records its numerical discrepancy and rounding guard.

This remains a conditional capacity result. Counterfactual weights can change
the incoming histories through native retrograde paths. It does not prove
capacity of the entire coupled brain, a trainable optimum, or a requirement
for perfect compensation. Selected weights also remain below their cap in the
actual runs, with final maxima between 0.23795 and 0.33292. Both positive and
negative source-aligned differential weights survive; complete erasure of one
opponent bank is not an adequate explanation either.

## Consequence for the next intervention

Do not add neurons merely because the crossed task failed. The recorded
interface already contains a usable joint distinction. First measure acquisition
over longer balanced experience, with intermediate weight-transfer probes and
continued physical/neural state. Four exposures per pairing are not a biological
deadline. Check whether the joint response strengthens, remains dominated by
the last episode, or is repeatedly overwritten, and inspect the local teaching
and credit trajectories at those transitions.

If interference persists, population competition or selective plasticity
regulation becomes a motivated intervention. A global workspace clock or
an entirely new body is not yet justified by this failure. The full objective
still requires learned hierarchy, active perception, autonomous action and
much broader transfer. This one-joint experiment completes none of those.

## Evidence locations

Raw courses are under
`.live/research/20260909_crossed_av_{normal,reversed}_seed{11,23,44,77}`.
`crossed_av_analysis.py --output OUTPUT ROOT...` produces the checked trajectories
and sensory-intervention comparisons. The completed analysis is
`.live/research/20260909_crossed_av_analysis`.
`crossed_av_capacity.py --output OUTPUT ROOT...` produces the conditional input
matrices and certificates. Its completed output is
`.live/research/20260909_crossed_av_capacity_checked`. This supersedes the first
capacity output by including Python-scalar-to-float32 casts in the conservative
rounding guard; the fitted bounds and factual replay are unchanged.
Raw media, checkpoints and trajectory arrays remain local.
