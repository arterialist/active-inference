# Branch-local APL in the connected FlyWire preparation

The same 2,755 biological cells now run with an optional spatial APL. Its
360,309 reconstructed tree nodes carry intracellular voltage state within
one PAULA neuron. The original graph, input/output port identities, boundary
slots, weights and native plastic return paths are retained. This is a working
intracellular implementation, not yet a reproduction of APL physiology.

Two 224-tick recordings compare intact transmission with APL-output blockade.
Replacing the global APL with the local model changes actual KC and PN input
signals, followed by membrane and spike changes. Under APL-output blockade,
every recorded KC and PN field remains exactly identical to the global-model
control. The effects of the replacement therefore depend on APL's forward
release in this course. This does not isolate every subsequent feedback path.

## What is anatomical, and what was assumed

The preparation uses the left FlyWire m783 APL, root `720575940624547622`,
PAULA ID `66912`. All 128,435 linked contacts match the 6,315 original incident
pair counts. The adapter places incoming current at 68,332 reconstructed
contact attachments and reads release at 60,103 linked outgoing attachments.
Open connectors are retained as evidence, without inventing missing neural
partners. The whole-cell output diagnostic includes all 100,093 outgoing
contact entries, including unlinked entries; they do not acquire consumers.
Contact attachments and skeleton radii are reconstruction data, not measured
conductances. See [the spatial audit](SPATIAL_FINDINGS_2026-09-10.md).

The opt-in `LocalCableGradedNeuron` inherits PAULA input processing and learning.
It adds sealed-end passive branch exchange. Each edge is a cylinder at the mean
of its endpoint radii, with half its lateral area assigned to each endpoint.
The source contains 6,940 zero-radius nodes, including 44 internal nodes. These
are not deleted or silently repaired. No edge has two zero endpoints. The
modeled membrane area is 152,911.84 µm²; this depends on that cylinder rule.
There is no claim of spatial discretization convergence.

For normalized membrane areas `C`, axial Laplacian `L`, arriving current `I`
and `a = dt/lambda_param`, the model updates

```text
(C + a L) v_next = (1-a) C v + a I
```

The explicit membrane leak follows PAULA; axial exchange is implicit. Each
edge contributes `(Rm/Ra) π r² / (length × total_area)` to `L`. The sum of
`C*v` follows the aggregate membrane equation because axial current cancels
in the whole-cell balance. That conservation does not establish accurate local
voltages or timing.

Each input port's native delayed potential is divided among its contact entries.
The adapter reads the already queued, pre-learning potential; it does not
multiply a past arrival by the current learned weight. Each terminal reads the
mean capped, rectified local release across its contact entries, then multiplies
by its native terminal coefficient. The receiving pair keeps its count-scaled
weight, so contact count is not multiplied twice.

`S` is now area-weighted voltage, not a measured soma potential. `O` is mean
release over all outgoing contact entries, not a spike or calcium signal.
Actual terminals can release different amounts. During the output-block lesion,
intracellular activity and attempted release remain visible, but forward events
are removed. Native return events and positive adaptation remain active.

The course retains `lambda=20` ticks, two dendritic ticks plus one network tick,
count conversion `0.075`, graded gain `0.01`, release cap `1`, postsynaptic rate
`1e-8` and retrograde rate `1e-6`. The local model adds `Rm/Ra=25,000 µm`.
The ordinary builder still defaults to the global model and count conversion
`0.02`. No shared `neuron.py` equation was edited.

[Amin et al., 2020](https://doi.org/10.7554/eLife.56954) measured localized APL
activity and inhibition. Their radius-dependent spatial fit motivates the
resistance-ratio hypothesis, but did not model branch loading or dynamic
feedback. APL membrane and axial resistances were not measured. Their calcium
imaging does not identify our current scale, voltage scale or time per tick.
Consequently, neither importing that ratio nor observing local activity is a
quantitative replication of the paper.

## Traceable effects in the actual connected graph

All 174 PNs receive artificial current levels 1.25, 2.5, 5 and 10 in successive
40-tick windows starting at tick 32. This includes non-olfactory ALPNs and is
not an odor pattern. The levels belong to one adapting trajectory, not separate
dose-response trials. The preceding and following 32 ticks have no experimental
drive. Anatomy and initial state are deterministic; there is no independent
specimen or seed replication claim.

At tick 66, the first APL response occurs with all previous branch voltages
exactly zero. PNs supply it before KCs fire. At tick 67, actual receiving
information differs between the global and local APL models:

| Receiving cell and port | Global APL input | Local APL input |
| --- | ---: | ---: |
| KC `720575940602815200`, port 46 | 0.0484304093 | 0.0064543588 |
| DA1_lPN `720575940603231916`, port 95 | 0.0484304093 | 0.2956255674 |

KC and PN membrane state first differ at tick 69. Local inhibition is not
uniformly stronger: the first example receives less and the second more.
The changed PN activity then changes APL's own next input. DP1m_adPN
`720575940618308825` spikes at tick 73 in the global model but tick 75 in the
local model. Its signal to APL port 858 is therefore present at tick 74 only
in the global run. This is a preserved PN→APL→PN loop, not a feedforward
textbook replacement. Other routes and ongoing return events also contribute.

The interval counts below summarize the full traces. They are not acceptance
scores for sparseness, odor discrimination or learning.

| PN drive / ticks | Global intact KC spikes | Local intact KC spikes | APL blocked, either model |
| --- | ---: | ---: | ---: |
| 1.25 / 32 through 71 | 0 | 0 | 0 |
| 2.5 / 72 through 111 | 6 | 0 | 56 |
| 5 / 112 through 151 | 701 | 263 | 3,292 |
| 10 / 152 through 191 | 1,780 | 1,555 | 6,623 |

The first KC spike moves from tick 101 to tick 116. Under blockade, equality
holds for every recorded KC/PN soma field, neuromodulator, input channel,
local potential, postsynaptic information coefficient and terminal information
coefficient throughout all 224 ticks. Equality of the counts alone would not
have established that. The local blocked APL still produces 14,345 native
return events; 484,232 attempted forward terminal events are removed.

## Early saturation and its cause

At the first response, the current has 129 positive ports totaling 101.19281
and 11 negative ports totaling -4.33200. Its area-weighted voltage is only
4.84304, but individual voltages range from -220.41384 to 299.08276. At the
assumed gain and cap, 509 nodes and 161 outgoing contact entries already reach
the release ceiling. Inferring saturation from `S` alone would miss this.

An exact conditional linear decomposition attributes 279.12756 of the positive
peak's 299.08276 units to port 1028 from VC3l_adPN `720575940619928429`.
The peak is at tree node `509295077`, whose source radius is 49 nm. At the
negative peak, tree node `509214293`, ports 154 and 2476 contribute -137.63523
and -82.78083 units. Their sources are the two selected M_vPNml50 cells
`720575940608124338` and `720575940631630021`. Negative input comes through
the source model's inhibitory sign convention, not negative release.

This decomposition solves the first linear cable transition and sums every
port's contribution at the chosen node. It is conditional on the recorded
currents and exact resting state, not a full-network path lesion. Later activity
changes input currents and learned coefficients, so this decomposition cannot
be extrapolated across the whole course.

A numerical refinement then held this same current for one tick, using 1, 2,
4, 8, 16, 32 or 64 passive substeps. Geometry, current, gain and cap did not
change. The original one-step transition reproduced every node bit exactly.
Every node voltage and terminal readout is retained at every finer substep.

| Substeps per original tick | Peak voltage | Capped nodes | Capped output contacts |
| --- | ---: | ---: | ---: |
| 1 | 299.08276 | 509 | 161 |
| 8 | 342.74697 | 651 | 200 |
| 32 | 348.34635 | 684 | 207 |
| 64 | 349.30581 | 691 | 207 |

The coarse step understates this peak by 16.8% relative to its own value; it
does not create the saturation. The largest terminal-release change from
one to 64 substeps is 0.09571 on the unit release scale. From 32 to 64 it is
0.00216. These are numerical comparisons, not biological error tolerances.
Substepping also changes the explicit leak approximation; no substep output
was supplied to the surrounding network. This test does not prove the finer
scheme correct, establish spatial convergence, or rerun adaptation.

The immediate issue is the translation from pair-count current to local current
density and release. Conserving a whole-cell current while concentrating it on
a small membrane region does not conserve its local effect. The resistance
ratio constrains neither absolute current nor voltage-to-release gain. Lowering
gain until the cap disappears would not supply those missing measurements.

## Verification and limits

The intact local course replayed without instrumentation with exact agreement
for 81,069,525 recorded branch-voltage values, 80,709,216 node-current values,
all recorded terminal releases and port currents, and all ordinary recorded
neural states and weights. This includes initial state and every tick, not just
the endpoint. A later replay with the opt-in network reset hook also matched.

The full-record auditor recomputes each node equation by sparse multiplication,
and reconstructs current placement and terminal readouts independently from
contact lists. Maximum equation residual is `4.33e-14` intact and `9.51e-14`
under blockade. Corruption tests reject spatially wrong current even when its
total is conserved, and reject a false port assignment even when the recorded
voltage still satisfies the cable equation. Tests do not establish physiology.

Recorded coefficients change at both positive learning rates. However, the
inherited nonspiking APL never advances `t_last_fire`; its native timing rule
is not a validated graded-cell learning rule. There is no local calcium,
vesicle state, active channel, contact-specific weight or local neuromodulation
model. Local voltage reaches 526.04 intact and 1,061.38 under blockade later
in the course. Local voltage is not silently clipped to the scalar soma bound.
These values are uncalibrated model units, not membrane millivolts.

The records omit complete in-flight queues, individual retrograde error vectors
and terminal modulation coefficients. They are not executable checkpoints.
Current boundaries remain undriven. Sparse useful odor coding, measured calcium
responses, acquired discrimination, long-run plastic stability and successful
reunion with missing circuits remain unproved.

## Reproduction and retained data

From `active-inference`, with the optional `flywire` and `dev` dependencies:

```sh
uv run python -m simulations.drosophila.intervention_probe \
  .live/research/flywire783/kc-apl-left-v1 NEW_LOCAL_RECORD \
  --condition intact --weight-per-count 0.075 --apl-representation local_cable \
  --spatial .live/research/flywire783/apl-left-spatial-bound-20260910
uv run python -m simulations.drosophila.intervention_analysis \
  NEW_LOCAL_RECORD NEW_LOCAL_RECORD/analysis.json
uv run python -m simulations.drosophila.intervention_analysis \
  NEW_LOCAL_RECORD NEW_LOCAL_RECORD/replay.json \
  --source .live/research/flywire783/kc-apl-left-v1
uv run python -m simulations.drosophila.cable_response_probe \
  NEW_LOCAL_RECORD NEW_REFINEMENT_RECORD
```

Repeat the course at a new path with `--condition apl_release_block` for the
lesion. To compare a local run with a matched global run, use the analysis
command with `--reference GLOBAL_RECORD --compare-models`.

Raw records remain local in `.live/research/flywire783/`:
`pn-course-local-cable-intact`, `pn-course-local-cable-apl-block` and
`apl-first-response-refinement-20260910`. The two courses total about 1.06 GiB;
the full-substep refinement adds 424 MiB. Existing source data and records
were not overwritten. The intact/block courses took 71.6/80.5 seconds with
about 1.40 GiB peak RSS; the refinement took 18.8 seconds and about 1.20 GiB.
No live servers or embodied suites were started.

The immediate physiological work is to define a stimulation and observation
model that can be compared with published local APL responses, then obtain a
measured odor-to-PN mapping. Current, release, geometry and time assumptions
must remain identifiable. A lower KC count under uniform drive is not a
substitute for either task. The global and local models remain separate
references until those tests discriminate between them.
