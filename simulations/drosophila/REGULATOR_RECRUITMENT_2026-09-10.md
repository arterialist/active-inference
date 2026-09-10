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

## Follow-up declared before the branch recordings

At the existing high-input operating point, the next three recordings block
the regulator's internal outputs to ALLNs, to all ALPNs, or to both populations.
The original intact course supplies the fourth condition. ORN-directed output
remains available in all four conditions. APL and disconnected boundary targets
are excluded from these branch selections. Source identity, not a new inferred
transmitter label, selects the regulator. Forward events are withheld from
tick zero; native return events and ongoing learning remain enabled.

This is a two-factor loss-of-function experiment, not a repair or a claim that
the two populations are independent modules. If only one branch block causes
persistence, it identifies a conditional requirement at this operating point.
If both single blocks recover but the joint block does not, the outcome is
consistent with compensating routes and requires further interventions. No
conclusion is selected from target-PN totals alone. All population traces and
the complete recovery period are inspected. The isolated gain preparation is
an intact functional reference, not a test of all the newly included LN/PN
partners.

## Completed branch test: the local-neuron branch preserves recovery

The measured regulator projects to 93 internal ALLNs through 5,182 counted
contacts and 137 internal ALPNs through 1,436 contacts. These selections are
disjoint; the joint intervention is their exact union. Every selected pair has
a negative source-model sign. This does not establish receptor identity.

| Forward output blocked, sensory 100 | PN stimulus spikes | PN recovery spikes | All LN recovery spikes | Last LN / PN spike |
| --- | ---: | ---: | ---: | --- |
| None | 219 | 5 | 0 | 1193 / 1259 |
| Regulator to ALLNs | 287 | 244 | 29,300 | 2199 / 2199 |
| Regulator to all ALPNs | 220 | 5 | 2 | 1201 / 1277 |
| Regulator to both populations | 294 | 236 | 29,367 | 2199 / 2199 |

All conditions have zero ORN recovery spikes. In the failed LN-branch and
joint-branch conditions, mean sensory release fraction during recovery is about
0.0492. The sensory gate has not disappeared or failed its recurrence equation;
the connected network remains active in spite of it. The ALPN-branch control
recovers with almost the intact target response. This establishes a conditional
requirement for the identified regulator-to-ALLN branch in this preparation.
It does not show that all 93 contacts are necessary, that the ALPN branch is
generally dispensable, or that the lower-input failure has been repaired.

The complete recorded history is identical until the first withheld release at
tick 213. The first state differences occur at 216. The LN-branch and joint
controls first change spikes at 221, in lLN2T_b root `720575940616169578` and
il3LN6 root `720575940623636701`. This is an early change in the recurrent
population, not an effect inferred only from final output counts. Native return
events, learning, incident ports and all unblocked forward paths remain enabled.

The two earliest changed cells also expose an interpretation limit. lLN2T_b
has a curated acetylcholine annotation, despite a low-confidence serotonin
prediction; its source-model sign is positive. il3LN6 has a curated GABA
annotation but a positive source-model sign. The latter belongs to the
previously documented sign-uncertainty group. The earlier negative-current
sensitivity control did not resolve persistence in a different preparation,
but that does not establish robustness of this new branch result to polarity
assumptions. These cells must not both be presented as biologically established
excitatory neurons. The present causal claim concerns the retained source-sign
model, and needs that distinction before further biological interpretation.

The new records are `dl5-regulator-branch-{ln,pn,both}-20260910` under the same
research directory. Each analysis includes the matched connected reference via
`--reference`. `dl5-regulator-branch-factorial-20260910.json` retains the four
conditions, anatomical counts, exact selection union, recovery trajectories and
causal comparisons. The runner's three new `regulator_to_*` pathway choices
select measured internal pairs by source identity and target population. They
do not change the default intact preparation.

## Physiological direction after this result

Inhibitory-to-excitatory-LN coupling is supported by paired fly recordings;
the same work also shows that excitatory LNs recruit inhibitory LNs. That is
a relevant reciprocal motif, rather than a reason to model inhibition only at
the principal output neuron. The present 93-pair intervention is not a
quantitative reproduction of those recorded cell pairs.
[Yaksi and Wilson, 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2954501/).

Muscarinic regulation offers a specific candidate for replacing an assumed
constant recruitment gain with local, activity-dependent dynamics. Fly work
found mAChR-A effects on inhibitory-LN odor responses; later receptor
manipulations connected voltage-dependent potentiation to habituation.
[Rozenfeld et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6900217/),
[Rozenfeld et al., 2021](https://www.nature.com/articles/s41467-021-27593-x).
This motivates testing a receptor-level PAULA mechanism against physiological
interventions before introducing it into this loop. It does not justify an
observer-controlled learning rate, a universal muscarinic gain constant, or
assigning that receptor to this FlyWire cell without further evidence. A
network-level improvement would then have to survive the lower-input challenge
and preserve sensory function, with ongoing adaptation still enabled.

One assembly limitation also remains explicit: the existing presynaptic gate
reads only the identified regulator's receptors and affects ALPN-directed ORN
terminals. Other inhibitory sources and ORN-to-LN terminals are not included in
that mechanism. Their ordinary somatic interactions remain in the network.
This is a hypothesis boundary, not a claim of complete presynaptic physiology.
