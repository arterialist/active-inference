# Memory that can guide movement and teach another cue

This experiment asks whether an acquired PAULA memory can affect both behavior
and further learning while adaptation continues. The first requirement is an
A-to-nutrient memory that survives a delay and changes the response to A. Only
then should B-before-A, without nutrients, test memory-mediated teaching.
Second-order conditioning already exists in biology and computational models.
Success here would establish a composition in PAULA, not a new discovery about
flies or evidence of consciousness.

The files live in the existing active-inference repository. They do not change
the sensory preparation, its regulator experiments, or neuron-model defaults.

The completed bounded comparison establishes a stored, causally expressed cue
effect with continued adaptation and preserved imported inhibitory signs. It
does not establish useful movement or recruitment of the teaching route.
The [rule-location comparison below](#restricting-the-rule-to-memory-inputs)
is the latest result. The second-order experiment remains untested.

## Biological choice and limits

[Yamada et al., 2023](https://elifesciences.org/articles/79042) identifies a route
from the alpha1 memory output through SMP353/354 and SMP108 to dopamine cells in
other compartments. Its behavioral and dopamine measurements motivate the
feedback hypothesis. [Aso et al., 2023](https://elifesciences.org/articles/85756)
links alpha1 output to movement through UpWind Neurons, including SMP353/354.
These studies use hemibrain anatomy and separate experimental animals. Our
reference below uses FlyWire 783. Type correspondences do not make their cells
the same specimen.

The reference contains both left MBON07 cells, all eight left PAM11 cells, left
SMP353, SMP108, APL, and 64 identified alpha/beta KCs. KCs are ranked by their
summed measured connections to the two MBON07 cells, before simulation. Alternate
ranks form two disjoint controlled input codes. This samples strong afferents;
it does not reproduce natural odor tuning or the full mushroom body. Every
directed pair among the 77 selected cells is retained, including reciprocal
MBON, DAN and KC connections. There are 1,045 internal pairs representing 6,140
synapses. Another 5,958 boundary cells retain their identities and incident
connections. Outside neurons do not run, and their input ports remain undriven.
No exact SMP354 type appears in the pinned annotation table; no cell is silently
substituted. Student-compartment targets of SMP108 remain a recorded boundary
until the alpha1 prerequisite is usable.

[Springer and Nawrot, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10806035/)
compares simplified rate-network motifs using selected plasticity and optimized
parameters. That is useful precedent, not evidence for this PAULA preparation.
[Rachad et al., 2025](https://doi.org/10.1016/j.celrep.2025.115593) studies aversive
higher-order conditioning. The abstract and publication metadata were accessible;
publisher full text was not. Its mechanism has not been adopted from the abstract.

[Ichinose et al., 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4643015/) provides
another constraint on the reciprocal alpha1 circuit. Its manipulations support
reward augmentation through MBON-alpha1 feedback and NMDA-receptor signaling in
PAM-alpha1. The imported source assigns negative fast currents to all selected
MBON07-to-PAM11 pairs. That uniform sign is therefore not a sufficient account
of the proposed biological excitatory feedback. This experiment preserves the
imported reference and does not claim to reproduce that consolidation mechanism.

## First bounded comparison

The native fly builder gives all transmitting modulation vectors and receiving
modulator sensitivities zero. It therefore has no dopamine teaching channel.
The isolated experiment assigns existing M[1] transmission to the measured
PAM11-to-MBON07 pairs. Release starts at one and receiving sensitivity at the
original count-to-current magnitude, count times 0.02. This is an explicit,
unfitted transmitter/receptor assignment. Signed fast currents remain intact.
Both receiving and returning adaptation stay positive in every phase.

The first course uses PAULA's native multiplicative learning rule. MBON receiving
learning uses the existing NeuronParameters default, 0.01, while other parameters
retain the fly builder's settings. This tests learning at a usable rate without
claiming that the sensory execution defaults already learn memory. Six-block
expansion is available, but the first decision uses two blocks.

The bounded alternative uses the already implemented reward_hebb equation with
kappa=-10 and rh_decay=1 at MBONs. Dopamine suppresses its positive coincidence
term while active-input decay continues. This tests whether the existing rule
can express the depression needed for alpha1 disinhibition. It adds no equation.
The parameters are a declared engineering hypothesis, not a dopamine receptor
fit. The native rate-boost extension alone only changes update magnitude and
does not by itself supply this depression sign. Spike eligibility extensions
likewise do not identify a dopamine-dependent depression rule. They are not
automatically composed into this test.

Pairing is compared with the same nutrient quantity displaced into a gap.
The completed comparison uses 1,000 cue-free ticks on either side of the displaced
nutrient pulse. A short first execution used a pulse immediately before C and
exposed a backward-pairing confound from lingering modulation. Those four pilot
directories are retained locally but are not the completed comparison below.
The longer gap reduces modulation substantially, without pretending that an
exponential tail becomes exactly zero.
Naive cue responses, acquisition, 1,000 ticks of retention, and later A/C responses
occur in the same continuing neural preparation and physical body. A and C each
receive equal exposure durations. No state or weight reset occurs between phases.
Probes continue adaptation and can themselves change the memory. Differential
weights alone will not count as learned action. If a response changes, a separate
checkpoint branch will restore only the selected KC-to-MBON receiving weights
and corresponding release coefficients to their pretraining values, preserving
the remaining acquired state, to test expression by those stored coefficients.

The main second-order comparison, once acquisition works, will interrupt the
SMP108-to-student-DAN teaching route during B-before-A exposure. It must preserve
the original A memory and basic motor function. A broad brain lesion would not
answer that question. This intervention is not yet implemented or tested.

## Physical and sensory boundaries

The existing LoadedHinge MuJoCo body and EnergyBudget organs are reused. A
nutrient pump supplies the gut and actually increases available energy; accepted
nutrient drives the PAM11 experimental inputs. The environment supplies no
target response or learning update. Each controlled KC receives current 40 when
its cue is present. This bypasses unresolved upstream sensory processing openly.

SMP353 receives constant current 1.4. Its actual PAULA spikes pass through a
fixed muscle filter into the hinge actuator. This is an engineered replacement
for downstream motor circuitry, not fly walking or a chosen action. The spring
provides passive return force. A PAULA tick is mapped to the body's existing
4 ms physics step for this experiment, without asserting fly physiological
timing. The organism does not yet seek a spatial food source. Useful movement
must be established before this preparation can support an embodied competence
claim.

## First decision, 10 September 2026

Five completed courses each contain 10,120 continuing ticks and two acquisition
blocks. They compare native paired and unpaired nutrients, the existing
alternative rule with paired and unpaired nutrients, and that alternative with
only the receiving dopamine sensitivity removed. The last control retains
PAM11 activity, its fast anatomical currents and every positive basal learning
rate. Every course accepts and digests 0.96 J. None accumulates energy debt.
The body moves in all courses, but this is not evidence of useful learned action.

![Recorded learning, expression intervention and physical response](evidence/first-acquisition.png)

The native rule produces exactly identical selected KC-to-MBON receiving weights
at every recorded tick in paired and unpaired courses. After retention, mean A
weight is 0.613448 in both, and each produces six MBON07 spikes during the later
A presentation. Dopamine arrives, but the native update does not store this
contingency in the selected connections under the tested settings. Other state
variables can differ; this is not a claim that both entire organisms are equal.

The existing alternative retains mean A weight 0.618937 with pairing, compared
with 0.882984 for displaced nutrients and 0.883003 with the dopamine receptor
disabled. C weights are 0.898878, 0.886885 and 0.898937, respectively. The small
C difference in the displaced-nutrient course is consistent with its remaining
dopamine tail and is not treated as perfect isolation. Selected receiving weights
do not change during the 1,000-tick retention period because these silent ports
receive no new input. Positive adaptation remains available throughout. This
establishes storage over that interval, not biological long-term consolidation.

Stored coefficients affect expression. During A, the paired preparation emits
83 MBON07 spikes. Replacing only its 128 selected receiving coefficients and
128 corresponding terminal release coefficients with those from the unpaired
history raises that count to 132. Transferring the paired coefficients into the
unpaired preparation lowers its response from 132 to 81. Body state, organs,
other neural coefficients and queued events remain those of the receiver.
The analogous C probe remains at 132 spikes. Both unchanged A branches reproduce
their recorded neural and physical trajectories exactly. All 64 currently active
selected receiving coefficients continue changing during each probe, so the
intervention establishes expression by acquired coefficients under continued
learning, not a frozen readout.

That partial result fails the composition requirement. In the paired-state
intervention, SMP353 emits seven spikes with either coefficient set; their timing
differs enough to change mean hinge angle from 0.052600 to 0.051738 radians.
This small causal physical effect is not food seeking or demonstrated useful
movement. SMP108 emits zero spikes in every complete course. Its maximum somatic
state is about 0.060 against threshold 1. The retained SMP353-to-SMP108 connection
has 48 measured contacts and initial effective weight 0.96. The upstream cell's
sparse activity does not recruit the next cell under this count scale and the
declared undriven boundary. Adding student compartments would not establish a
working teaching signal.

The alternative also exposes a separate defect in applying one learning rule
to every MBON input. The two initially negative MBON07 reciprocal coefficients,
-1.1 and -1.0, become approximately +0.706 and +0.703. Both APL-to-MBON current
coefficients also cross zero. These are consequences of the existing additive
rule, not new anatomical connections. The coefficient-swap result remains a
model result, but the altered reciprocal signs prevent accepting this as the
intended inhibitory organization. The native course preserves those signs.

The direct-learning prerequisite therefore remains unresolved as a useful,
anatomically constrained composition. No B-before-A trial or second-order
conclusion is justified yet. Architecture expansion stops here. The specific
next decision is whether the already implemented dopamine-sensitive rule can be
restricted to the intended excitatory memory inputs while the reciprocal inputs
retain their own ongoing native adaptation. Existing NeuronParameters select a
single rule for the whole cell; the existing port-modulation extension changes
native learning rate rather than selecting this alternative rule by input.
Any such composition must be a separate opt-in comparison, with the present
courses preserved. It must then be judged by useful output and recruitment of
the proposed feedback route, not only by weight separation. Adding glia or a
larger brain is not supported by these recordings.

This is a feasibility result and a localized composition failure, not a
publication claim. The tested ingredients and causal substitution are useful
for the next decision. There is no evidence yet for a contribution beyond those
already known ingredients, for higher-order learning in PAULA, or for consciousness.

## Restricting the rule to memory inputs

The specific comparison proposed above is now complete. `input_rule.py` composes
the two existing equations by receiving input, without editing neuron-model.
Selected KC-to-MBON inputs use the unchanged reward_hebb update. Other inputs
use the unchanged native multiplicative update, including their positive
learning rate. Somatic processing, dopamine integration, propagation and native
returning errors keep their existing order. No eligibility trace, behavioral
target or new weight equation is added. Empty port declarations use the ordinary
native tick. This is an opt-in hypothesis about where a rule applies, not a
new accepted PAULA default or a fitted receptor mechanism.

![Restricted rule, causal expression and physical limit](evidence/selected-rule.png)

Three otherwise matched 10,120-tick courses use paired nutrients, displaced
nutrients and the dopamine-receptor control. No internal pair reverses its
imported current sign at the retained checkpoint. Mean retained A weights are
0.618937, 0.882984 and 0.883003. Thus the pairing-dependent weight difference
survives preservation of the inhibitory reciprocal organization. In the later
A presentation, MBON07 emits six spikes with pairing, 22 with displaced nutrients,
and 23 with the receiving dopamine channel disabled.

The coefficient substitution again establishes expression. Replacing the paired
preparation's selected coefficients with unpaired coefficients changes its A
response from six to 23 spikes. The reverse transfer changes the unpaired
response from 22 to five spikes. C responses are 21 spikes before and after the
same substitution in probes from the same retained paired state. Both A sham
branches reproduce their recorded somatic and physical trajectories exactly.
Every probe keeps learning active and changes all 64 active selected receiving
coefficients. Transfer includes receiving weights and corresponding release
coefficients jointly; it does not distinguish their individual contributions.

Preserving inhibition therefore repairs the unwanted rule effect without
eliminating the acquired cue effect. It does not repair the output failure.
SMP353 emits eight spikes in both paired-state A probes. Their timing changes
mean hinge angle from 0.053211 to 0.052914 radians. The corresponding C means also
change slightly, from 0.053032 to 0.052939, despite equal MBON spike counts. These
timing effects must not be inflated into learned food-seeking competence.
SMP108 remains silent in all three full courses.

The preparation now supports a limited direct-memory claim. Nutrient pairing
leaves a connection-dependent cue response, persists through the specified
retention interval, and weakly affects the continuously actuated physical body.
It does not yet support the task's useful embodied-memory prerequisite. The
specific remaining failure is transmission from that stored response into a
useful movement change and an active memory-to-learning feedback route under
the declared sensory and tonic boundaries. No larger circuit, B training,
pretrained controller, hardcoded action menu or additional biological subsystem
was added to hide this failure. Further work must resolve that functional
interface before expanding the architecture.

The [restricted-rule evidence](evidence/selected-rule.json) records the three
courses, six expression branches, trajectory hashes, positive learning rates
and current-sign comparison. Their local prefixes are `memory-selected-` and
`memory-selected-expression-` under the same raw-record directory. The original
whole-cell comparisons and their code remain executable. The focused tests now
include the rule-location contract and pass, 29 tests including the anatomical
loader tests. This result remains insufficient for a publication contribution
claim or for any conclusion about learning mechanisms in real flies.

```sh
uv run --no-sync --with cloudpickle==3.1.2 python \
  -m simulations.drosophila.memory_feedback.input_rule \
  .live/research/flywire783/memory-alpha1-cut-20260910 \
  .live/research/flywire783/memory-selected-paired-20260910
uv run --no-sync --with cloudpickle==3.1.2 python \
  -m simulations.drosophila.memory_feedback.selected_analysis \
  .live/research/flywire783 simulations/drosophila/memory_feedback/evidence
```

## Reproduction and records

Run from active-inference. Source files are already pinned and hash-checked by
the existing FlyWire loader.

```sh
uv run python -m simulations.drosophila.memory_feedback.anatomy \
  .live/research/flywire783 \
  .live/research/flywire783/memory-alpha1-cut-20260910
uv run --no-sync --with cloudpickle==3.1.2 python \
  -m simulations.drosophila.memory_feedback.acquisition \
  .live/research/flywire783/memory-alpha1-cut-20260910 \
  .live/research/flywire783/memory-native-paired-spaced-20260910 --blocks 2
```

The four other completed directories replace `native-paired` with
`native-unpaired`, `hebb-paired`, `hebb-unpaired` or `hebb-receptor-block`.
Use `--unpaired` for displaced nutrients, `--rule dopamine_hebb` for the
alternative, and `--no-dopamine-receptor` for its receiving-channel control.
The completed protocol is `spaced-nutrient-v2` in each manifest.

```sh
uv run --no-sync --with cloudpickle==3.1.2 python \
  -m simulations.drosophila.memory_feedback.expression \
  .live/research/flywire783/memory-hebb-paired-spaced-20260910 \
  .live/research/flywire783/memory-hebb-unpaired-spaced-20260910 \
  .live/research/flywire783/memory-expression-paired-unpairedweights-A-20260910
uv run --no-sync --with cloudpickle==3.1.2 python \
  -m simulations.drosophila.memory_feedback.analyze \
  .live/research/flywire783 simulations/drosophila/memory_feedback/evidence
```

The [compact numerical evidence](evidence/first-acquisition.json) names every
course and expression branch, includes underlying trajectory and checkpoint
hashes, and records the sign changes with exact neuron identities and source
rows. The figure shows recorded trajectories and diagnostic branches; it is
not a synthetic animation. The focused causal-contract and anatomical-loader
tests pass, 28 tests total. A 77-cell run used about 176 MiB resident memory in
an observed early run; completed longer courses took about 53–57 seconds while
other independent courses ran concurrently. These are operating observations,
not a controlled performance benchmark.

Existing output directories are rejected. The run stores every tick's somatic
state and modulators, selected KC-to-MBON weights and releases, and physical
trajectory in disk-backed arrays. Exact neural checkpoints preserve queues,
buffer aliases and adaptation state at important phase boundaries. Physical
and organ state are separate. These are trusted local executable checkpoints;
raw runs remain under the existing ignored `.live/research/` directory.
