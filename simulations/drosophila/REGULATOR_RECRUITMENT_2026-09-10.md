# Can measured sensory inputs recruit the regulator without an electrode?

## Question and decision made before connected recordings

Brief electrode assistance did not preserve recovery in the antennal cut.
The next experiment replaces that assistance with stronger initial receiving
weights on the existing ORN_DL5 inputs to the identified lLN2F_b regulator.
The graph contains 35 such pairs and 178 contacts. No pair is added. No direct
electrode current reaches the regulator. The positive LN/PN-to-LN feedback scale
remains 0.5, sensory command 100, seed 11, and every native learning rate stays
positive. All other initial weights and cellular parameters remain unchanged.

This is a sensitivity experiment on the assumed contact-to-weight conversion,
not an anatomical change, fitted physiology, or a proposed permanent agent
default. Isolated receptor gains 1, 2, 4 and 8 will be tested first. Among these,
the setting with regulator stimulus spike count closest to the assisted
reference's 80 spikes will be carried into the antennal cut unchanged, provided
the isolated target PN still responds during the stimulus and stops firing by
tick 1400. Ties favor the smaller scale. This matches an experimental functional
reference, not a biological firing-rate target. If no candidate meets those
conditions, no connected candidate is selected by this rule.

The connected test must inspect activity during ongoing stimulation and after
it stops, in every population, not only whether the target PN count resembles
the reference. Even a successful result at this one input and seed would need
pathway ablations and varied sensory conditions before supporting autonomous
gain control. The source-model transmitter signs remain uncertain.

## Execution

`gain_reunion.py --regulator-afferent-scale N --lateral 0` changes initial
receiving coefficients only on those internal measured sensory pairs. Default
scale 1 does not write a changed value. Initial reference/scaled weights and
final learned weights are retained in `regulator-afferent-*.npz`; the analysis
checks their identities and values against the graph. Passing a differently
scaled isolated reference is rejected.

## Isolated selection and unchanged reunion

| Initial sensory receptor scale | Regulator stimulus spikes | Target PN stimulus spikes | Target PN recovery spikes | Target PN last spike |
| --- | ---: | ---: | ---: | ---: |
| 1 | 33 | 331 | 12 | 1291 |
| 2 | 99 | 218 | 4 | 1240 |
| 4 | 198 | 126 | 2 | 1236 |
| 8 | 299 | 86 | 1 | 1202 |

Scale 2 was selected by the rule above before its connected recording. In the
424-cell antennal cut it produces 219 target PN stimulus spikes and five offset
spikes. Every recorded spiking cell has stopped by tick 1259. Fourteen LNs and
five PNs participate early; eleven LNs and four PNs participate during the later
stimulus. The regulator itself has 98 stimulus spikes and no recovery spikes.
Its spike timing differs from the isolated course beginning at tick 253, so this
is not a fixed prescribed regulator train hidden behind a sensory connection.

The unchanged scale also preserves response and recovery in the full 3,005-cell
preparation: 219 target PN stimulus spikes, four offset spikes, and no LN
recovery spikes. APL has graded activity with maximum local release about
0.108. Only one of 2,580 KCs spikes once. This does not demonstrate useful
mushroom-body coding, memory, or a functioning whole fly brain.

All regulator electrode commands are exactly zero. All 35 selected receiving
weights change through native learning, with maximum absolute change about
1.7e-6 at input 100. In the connected intact course, 736 selected positive
feedback receiving weights also change. This establishes ongoing adaptation,
not useful learned regulation. No receptor is clamped to a target value during
the recording. The initial strength choices remain experimental assumptions.

## The lower-input counterexample

The same parameter setting was then tested at sensory command 50, without
retuning. It fails in the connected cut despite recovering in isolation.

| Sensory command | Isolated regulator stimulus spikes | Isolated PN stimulus / recovery spikes | Antennal PN stimulus / recovery spikes | Antennal LN recovery spikes |
| --- | ---: | ---: | ---: | ---: |
| 50 | 33 | 226 / 6 | 256 / 167 | 28,446 |
| 100 | 99 | 218 / 4 | 219 / 5 | 0 |

Even in isolation the response curve differs from a simple monotonic
sensory-to-PN rate relation. At input 100 the regulator fires every ten ticks.
At input 50 its intervals are predominantly 27 and 33 ticks, and its first
spike moves from tick 213 to 228. Mean late sensory release fraction changes
from about 0.145 to 0.327. Doubling input therefore triples inhibitory-cell
firing and slightly reduces the target's integrated spike count. This is an
observed operating-regime change, not proof of a biological bifurcation or a
failure of every possible temporal readout.

The connected lower-input failure is more decisive than that small count
reversal. It recruits 133 LNs and 151 PNs in the first 200 stimulus ticks.
During recovery, 132 LNs and 151 PNs remain active despite zero ORN spikes.
Conditional target replay without ORNs retains all 167 recovery spikes;
excluding positive-model LN input removes them. The sensory gate still executes
and is strongly engaged. Recurrent target drive again bypasses it.

Thus the electrode-free result is a demonstrated operating point with neural
recruitment, not robust autonomous gain control. The isolated reference-matching
rule was deliberately a selection rule for this experiment. Matching one
regulator spike count cannot certify the response curve or stability after
composition. The candidate is not adopted as an agent default.

## Pathway controls

Blocking the regulator's outputs to ORNs at input 100 raises the target's
stimulus count from 219 to 331, with 28 offset spikes. Blocking only its output
to the target PN instead gives 220 stimulus spikes and four offset spikes.
Both connected controls recover. Thus the ORN-directed pathway contributes
substantially to sensory gain in this course, while its direct inhibitory
connection to the target has little effect on these counts. Neither selective
block alone makes this high-input network persist.

The all-regulator-release control does produce persistent activity. Its
regulator still fires, but its forward terminal events are withheld while
native return events remain enabled. Initial weights, bindings, sensory inputs
and the complete recorded past before the first withheld event at tick 213
match the intact condition.

| Connected control at sensory 100 | PN stimulus spikes | PN recovery spikes | All LN recovery spikes |
| --- | ---: | ---: | ---: |
| Intact | 219 | 5 | 0 |
| Regulator-to-ORN output blocked | 331 | 28 | 7 |
| Regulator-to-target-PN output blocked | 220 | 4 | 0 |
| All regulator forward outputs blocked | 331 | 249 | 29,344 |

The two controls with 331 stimulus spikes have different collective outcomes.
One recovers; the other does not. Sensory gain and recurrent stability are
therefore distinguishable functions in these interventions, even when the
target's integrated stimulus response is identical. The all-output block's
first network spike difference appears at tick 221. It does not depend on
turning off learning or waiting until the stimulus ends.

These controls implicate the regulator's collective outputs in the high-input
recovery result. They do not identify a minimal stabilizing pathway: neither
single selective block is a joint block of all candidate routes. The relative
roles of its remaining LN and PN targets need separate interventions. Conditional
target replay also cannot replace those closed-loop tests.

## What this changes

The preparation can recruit regulation through measured sensory connections,
without a prescribed inhibitory train. What remains unresolved is the range
over which sensory recruitment and recurrent inhibition balance one another.
The next useful physiological target is the regulator's input/output timing
and its inhibitory effects on other recurrent cells across input strengths.
Repeating a fit to one output count, or building associative consumers on this
input-dependent instability, would leave the demonstrated failure in place.

Records are under `.live/research/flywire783/`, with prefix
`dl5-regulator-afferent`. Each completed course retains tick arrays, receiving
weights and a matched isolated `comparison/analysis.json`. The analysis replays
the target's complete receiving history and checks local gate routing. Source
membrane checks use recorded currents and thresholds and find no clipping or
threshold ambiguity in the completed courses. They are not full replays
of every source receptor, return event, or unrecorded intracellular variable.

The four initial isolated courses use names
`dl5-regulator-afferent{1,2,4,8}-isolated-20260910`. The selected intact antennal
course is `dl5-regulator-afferent2-antennal-20260910`; its full-network counterpart
is `dl5-regulator-afferent2-intact-full-20260910`. The matched controls use
`dl5-regulator-afferent2-{low,ornblock,pnblock,releaseblock}-{isolated,antennal}-20260910`.
`dl5-regulator-afferent-pathway-controls-20260910.json` retains shared-past checks,
intervention onset, event counts and first per-tick differences for each lesion.
Forward-event counts include attempted events on disconnected boundary terminals;
they are not counts of affected internal synapses or evidence of effect size.
