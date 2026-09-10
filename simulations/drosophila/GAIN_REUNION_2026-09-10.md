# Sensory regulation survives locally but fails to control recurrent output

The isolated presynaptic mechanism still gates sensory release after its actual
partners are restored. The failure is elsewhere in the composed circuit:
recurrent input sustains PN firing after the experimental drive stops, while
also over-recruiting the LN that suppresses sensory transmission. These effects
coexist. Counting only PN spikes during stimulation conceals the distinction.

## What was held fixed

The [isolated assay](LN_GAIN_CONTROL_2026-09-10.md) supplies the same one-tick
current commands to 42 DL5 ORNs and lLN2F_b `720575940623000858`. The target PN
is `720575940617207185`. Commands establish the input boundary, not imposed
spike trains or independently generated public-odor input. Native spikes and
weak positive postsynaptic and retrograde learning remain active.

The gate has gain 1 and decay 100 ticks. No gain, current, cell or synaptic
parameter was retuned on reunion. The sensory command is 50 or 100 pulses per
nominal second; the LN command is 80. All courses use timing seed 11, baseline
ticks 0–199, stimulation 200–1199, and recovery 1200–2199. Only one channel and
one train per course are tested. This is not an odor-discrimination assay or
an acceptance distribution.

The full preparation contains 3,005 cells and 213,855 measured internal pairs.
The antennal-only preparation contains 42 ORNs, 208 LNs and 174 PNs. KCs and APL
are absent in that cut, not simulated as silent cells. Incident boundary ports
remain in both preparations. The full preparation retains the existing spatial
APL model. Its local release reaches its imposed cap, which is not a verified
physiological response.

## Stimulus output and recovery are different tests

| Preparation | Gate gain | Sensory command | PN stimulus spikes | PN recovery spikes |
| --- | ---: | ---: | ---: | ---: |
| Isolated ORN/PN/LN | 1 | 50 | 129 | 3 |
| Full reunion | 1 | 50 | 187 | 167 |
| Antennal-only reunion | 1 | 50 | 251 | 168 |
| Isolated ORN/PN/LN | 1 | 100 | 251 | 6 |
| Full reunion | 1 | 100 | 263 | 177 |
| Isolated ORN/PN/LN | 0 | 50 | 330 | 18 |
| Full reunion | 0 | 50 | 330 | 185 |

The zero-gain isolated and full runs have the same stimulus spike count, 330.
Nevertheless, their PN state differs from tick 215, and their recovery differs
substantially. Equal counts do not establish interchangeable circuit behavior.

In the full gain-1, sensory-50 course, 169 LNs emit 43,077 recovery spikes and
562 KCs emit 16,695. The antennal-only course has 169 LNs emitting 43,060 recovery
spikes, without any KCs or APL. Mushroom-body feedback is therefore unnecessary
for this finite-window persistence. It does affect responses: the full PN has
187 stimulus spikes versus 251 without KCs/APL. This comparison does not assign
that difference exclusively to APL because both populations were removed.

Both courses drive the regulating LN to 334 recovery spikes, close to the
model's refractory limit. Its gate remains strongly engaged, with mean release
fraction about 0.049 during recovery. Three ORNs also keep firing after the
electrodes stop. Absence of experimental current is not absence of neural input.

## Receiving histories locate the competing drives

Fresh PAULA target-PN cells replay the recorded inputs with native ongoing
learning. The all-input replay exactly reproduces every recorded PN soma,
current, intrinsic and receiving-weight value for 2,200 ticks per course.
Masks then remove selected receiving histories without changing the remaining
histories. These tests locate conditional input effects, not closed-loop lesion
outcomes.

| Full gain-1, sensory-50 receiving replay | Stimulus spikes | Recovery spikes |
| --- | ---: | ---: |
| All recorded inputs | 187 | 167 |
| ORN inputs only | 54 | 1 |
| Without ORN inputs | 154 | 167 |
| Without positive-model ALLN inputs | 21 | 0 |

Positive-model ALLNs supply a recovery signed-current sum of about +24,425,
versus −5,712 from negative-model ALLNs, −1,105 from APL and +120 from DL5 ORNs.
These are model-current sums over ticks, not measured charge in physical units.
The antennal-only replay gives the same qualitative result: removing positive-
model ALLN input eliminates PN recovery firing.

Two effects are now distinguishable. Stronger recruitment of the regulator
reduces the sensory component below its isolated level. Other LN pathways
simultaneously supply the PN with recurrent excitation. A working sensory gate
cannot by itself regulate every other route into the consumer.

The gate-state recurrence, local release multiplication and next-tick target
routing also match their recorded values. One selected ORN has no direct pair
to the target PN. Its missing target terminal is explicitly recorded as absent;
its real connections to other PNs remain present. No edge was fabricated to
make the audit uniform.

## Closed-loop interventions separate the source from its delivery

Both late interventions are complete. They start at tick 600, after recruitment,
while sensory stimulation continues until tick 1200. The LN-to-LN intervention
withholds 8,849 positive-model directed pairs from 120 source cells. The
target-only intervention withholds 58 pairs from 58 source cells. All anatomy,
native learning and return-event transmission remain present.

| Outcome | Intact antennal cut | Positive LN→LN block | Positive LN→target PN block |
| --- | ---: | ---: | ---: |
| Target PN spikes, stimulus ticks 600–1199 | 147 | 153 | 1 |
| Target PN spikes, recovery ticks 1200–2199 | 168 | 172 | 0 |
| All LN recovery spikes | 43,060 | 20,543 | 43,257 |
| LNs firing in recovery | 169 | 111 | 169 |
| Regulating LN recovery spikes | 334 | 65 | 334 |
| Mean sensory release fraction in recovery | 0.0491 | 0.1901 | 0.0491 |

Blocking positive LN-to-LN transmission reduces total LN recovery activity by
about half, but does not restore PN termination. It also weakens the regulator
and the negative-model LN current arriving at the PN. Remaining positive-model
LN input still dominates the target's recovery drive. Less population activity
does not imply a more useful consumer signal.

Blocking positive LN transmission only onto the target eliminates its recovery
spikes, but also removes almost its entire response to the continuing sensory
input. The full stimulus count is 105, of which 104 occurred before the block.
A total-count and recovery-only acceptance test could misclassify this as a
good result. The interval aligned to the intervention exposes the failure.
The rest of the network remains active, so a quiet target also cannot certify
that the recurrent regime has been resolved.

Each intervention matches 1,956,000 recorded pre-intervention values exactly,
including the entire population's soma state. The commands match throughout.
The recurrent block first changes soma state and spikes at tick 603. The
target-only block changes state at 603 and spikes at 605. Receiving-history
replay and gate/routing checks pass for all 2,200 ticks in each course. The
first block withholds 2,295,397 selected forward events; the second withholds
28,625. Neither removes native return events.

These are closed-loop consequences, unlike the conditional receiving masks.
They identify necessary delivery to this PN and a contribution of LN-to-LN
transmission to population maintenance. They do not identify a minimal
sustaining loop, prove indefinite persistence, or establish a functional repair.

## Two routes can sustain the unwanted activity

Two further completed courses block positive-model PN-to-LN transmission, or
the union of that pathway and positive-model LN-to-LN transmission. Together
with the intact and LN-only conditions they form a two-factor intervention.
All four share the same past until tick 600 and the same ongoing stimulation.

| Positive LN→LN blocked | Positive PN→LN blocked | LN recovery spikes | LNs firing in recovery | Target PN stimulus spikes after block | Target PN recovery spikes |
| --- | --- | ---: | ---: | ---: | ---: |
| No | No | 43,060 | 169 | 147 | 168 |
| Yes | No | 20,543 | 111 | 153 | 172 |
| No | Yes | 38,710 | 154 | 146 | 167 |
| Yes | Yes | 0 | 0 | 52 | 2 |

Neither individual block ends the target PN's persistence. Combining them does,
while preserving a response to continuing sensory input. The combined course's
last ORN spike is at 1199, its last LN spike at 1193, and its last PN spike at
1217. It retains 41 target-PN spikes during the settled stimulus interval
800–1199. This is different from the earlier target-only block, which silenced
the output despite continuing stimulation.

This establishes conditional dependence in the modeled circuit. PN-to-LN
transmission is required to maintain the broad activity when positive-model
LN-to-LN transmission is absent; either remaining network can sustain substantial
activity when only the other route is blocked. It does not establish that either
pathway is sufficient without the other retained connections or identify a
minimal cycle. The result also does not separate synaptic adaptation from fast
state as the origin of recruitment. Learning remains active in all conditions.

The PN-only block selects 8,108 measured pairs from 158 source cells. The combined
block selects the disjoint union of 16,957 pairs from 278 sources. Each new run
matches 1,956,000 pre-intervention values and all commands against intact. Both
first change neural state and spikes at tick 603. The PN-only course withholds
2,720,305 forward events. The combined course withholds only 23,203 because the
network stops generating the sustained releases that would otherwise follow.
Independent target receiving replay and gate/routing checks pass throughout
both 2,200-tick courses.

The combined lesion is a diagnostic reference, not the proposed final circuit.
It retains the controlled LN electrode and removes broad classes of functional
transmission. It has not demonstrated autonomous lateral normalization, odor
identity discrimination, multi-intensity robustness or learning. It shows that
the present persistent activity can be ended through recurrent input to LNs
without disabling the target's sensory response or resetting neural state.

## Biological interpretation and next functional bottleneck

The source graph's positive/negative column is a model assumption, not a
measurement of receptor effects. The [earlier identity audit](ANTENNAL_REUNION_2026-09-10.md)
found conflicts between imported signs and curated transmitter annotations.
Those remain unresolved. This experiment does not justify converting every LN
to inhibition or discarding measured recurrent connections.

[Nagel and Wilson, 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4829653/) measured
diverse LN onset, offset and rebound dynamics. Excitatory and inhibitory inputs
help determine response timing; intrinsic properties contribute to preferred
timescales. Their experiments also found depressing ORN input and slowly
developing inhibition between LNs. Our homogeneous recurrent coupling is not
a reconstruction of those diverse mechanisms. A repair must preserve useful
responses to changing inputs, rather than merely make this network quiet.

The PN feedback test above resolves the earlier question of why positive-model
LN-to-LN blockade alone did not end persistence. The repair target is now the
joint dynamics of both recurrent routes and their recruitment of inhibition.
Simply strengthening the sensory gate or cutting input to the consumer is not
supported as a solution. Broad permanent pathway removal would evade the
composition problem rather than reconstruct a functional biological circuit.

[Yaksi and Wilson, 2010](https://www.sciencedirect.com/science/article/pii/S0896627310006847)
found PN-to-excitatory-LN transmission with chemical and electrical components.
Excitatory LNs also drove inhibitory LNs, giving that network opposing effects
on PN output. This supports studying feedback and inhibition together. It does
not establish an electrical connection for each of our chemically reconstructed
pairs, nor map their recorded cells to these specific FlyWire identities.

The five largest positive-model LN contributions to target-PN recovery remain
unchanged after the LN-to-LN block. Their roots are `720575940611671506`,
`720575940644704160`, `720575940631586858`, `720575940633483807`, and
`720575940640830453`. All five have a serotonin prediction in the retained
annotation, while the two lLN2T_a cells have curated acetylcholine annotations.
This is a reason to investigate transmitter and receptor assumptions on the
causally implicated paths, not to relabel these cells automatically. Predicted
transmitter, source-model sign and measured postsynaptic action are different
evidence categories. Fast positive current is not justified by the sign column
alone.

This uncertainty is explicitly discussed in the prediction paper, not merely
an objection inferred from our simulation. [Eckstein et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11106717/)
identify antennal LNs as a difficult case. They suggest that local neurons
predicted as serotonergic or dopaminergic should often instead be GABAergic or
cholinergic, with morphology supporting the latter in many cases. Serotonin
was their least reliable transmitter class. The five cells above have retained
confidence scores between 0.297 and 0.359. These scores are not independent
receptor measurements.

The imported binary signs also need their own provenance. [Shiu et al., 2024](https://www.nature.com/articles/s41586-024-07763-9)
assigned inhibition when a majority of presynaptic sites voted for GABA or
glutamate, and placed monoaminergic predictions in the excitatory category.
That convention helps explain why a binary model sign and the separate
annotation's winning transmitter label need not match. It is not a justification
for assigning every connection identical fast current dynamics. Nor would it
justify turning every serotonin-predicted LN into a neuromodulatory neuron.

## Evidence and reproduction

Local recordings are under `.live/research/flywire783/`:

- `dl5-gain-reunion-full-50-20260910/`
- `dl5-gain-reunion-full-100-20260910/`
- `dl5-gain-reunion-full-50-zero-20260910/`
- `dl5-gain-reunion-antennal-50-20260910/`
- `dl5-gain-antennal-recurrent-block600-20260910/`
- `dl5-gain-antennal-target-block600-20260910/`
- `dl5-gain-antennal-pn-feedback-block600-20260910/`
- `dl5-gain-antennal-ln-pn-feedback-block600-20260910/`

Each contains chunked tick recordings, structure, a run manifest and a
`comparison/` directory with receiving replays and analysis. Source hashes are
historical provenance, not rewritten when the driver evolves. Analysis checks
retained record bytes and independently replays the target receiving dynamics.
It does not verify every unrecorded intracellular state in the preparation.

Use `python -m simulations.drosophila.gain_reunion --help` and
`python -m simulations.drosophila.gain_reunion_analysis --help` from the project
environment. Both require a new output directory. `--scope antennal` makes the
boundary cut explicit. `--pathway positive_LN_to_LN` or
`--pathway positive_LN_to_target_PN`, with `--block-start 600`, defines the late
intervention. Analysis accepts `--reference` pointing to the intact course and
checks pre-intervention parity, anatomical bindings and transmission counts.
The two-factor follow-up uses `--pathway positive_PN_to_LN` and
`--pathway positive_LN_or_PN_to_LN` with the same onset and intact reference.

No learning, memory, behavioral or consciousness capability is established by
these reunion experiments. They identify why an isolated functional component
does not yet supply a usable signal in the interconnected preparation.
