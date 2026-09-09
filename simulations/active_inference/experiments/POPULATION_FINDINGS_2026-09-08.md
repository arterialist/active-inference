# Population composition and real audiovisual input

## Continuation: metabolic regulation preserves a rhythm but not its useful body coordination

The next experiment combines the acquired predictive motor loop with a slow
physiological loop. Its question is whether the adaptive whole can reduce
expenditure during scarcity and resume useful activity after digestion without
a host rest/wake selector. The components are provisional building blocks;
demonstrating that an inhibitory connection inhibits is not the research payoff.

The existing V3 normalized organ interface is reused. V3's speed-based activity
cost is not reused: low displacement can conceal vigorous or isometric muscle
activity. The new explicit budget accounts for positive actuator work and
activation separately, with finite gut and usable-energy stores. Parameters
are illustrative, not measured mammalian physiology: 12 J usable capacity,
48 J gut capacity, 3 W digestion, .2 W basal demand, .25 mechanical efficiency
and .5 W activation coefficient. Mechanical work remains endpoint-sampled.
Unmet demand accumulates as failure debt rather than disappearing at a zero
reserve. There is no reserve-dependent host motor brake. Motion after failure
is retained for diagnosis and does not count as viable locomotion.

The 1,945-cell acquired runtime receives two graded energy afferents and one
signed half-reserve comparator, becoming 1,948 cells. Its output inhibits all
four muscles through PAULA ports. This fixed anatomical reference is not a
learned preference, hypothalamic reconstruction, sleep or torpor mechanism.
The old V3 arbiter depends on its separate hunger/search/WTA graph and fixed
IDs, so it was not transplanted as a self-contained regulator. All original
runtime objects, queues, learned weights and positive adaptation survive the
additive construction. Matched controls retain every new cell and edge but
zero the four new motor weights; native plastic return paths remain active.

Four courses each run 2,048 ticks, 8.192 seconds, beginning with 9 J usable
energy and an empty gut. Half receive an external 48 J nutrient bolus at local
index 1024. This is laboratory refeeding, not autonomous foraging. Intact and
disconnected variants test both refeeding and continued scarcity. All neural,
energetic and physical audits pass; the largest predictor reconstruction
residual is 1.14e-13. The energy observer is behaviorally passive in its
regression tests. Four new tests and 21 focused integration tests pass.

With feedback intact, energy never reaches zero. The refed branch falls to
2.566 J before recovering to 5.203 J. The unfed branch conserves its reserve
but continues to pay basal cost, ending at 1.734 J. Without feedback, both
courses first incur unmet demand during the step from .748 to .752 seconds.
Their identical movements thereafter must not be interpreted as successful
survival. Refeeding changes their organs and energy neurons but never their
actuator commands in this course.

The meaningful result is the relationship between the ongoing rhythm,
activity and body motion. Phase-0 events stay at indices 19, 183, 347, and
every 164 ticks thereafter in all four courses. The clock is not restarted.
In intact scarcity, actual work per complete cycle falls from 1.259 J in the
first cycle to .00349 J just before refeeding. The refed branch later sustains
roughly .45 J per cycle, whereas the unfed branch approaches quiescence.
Predictor weights keep changing through both regimes, without a learning-mode
switch or a state reset. The first refeeding effect on energy afferents occurs
at 1025, the neural regulator at 1028, actuator output at 1087, joint position
at 1088 and local prediction error at 1092. The delay includes the current
motor phase and neural gating, not just transmission time.

However, every complete post-meal cycle has negative forward displacement.
The last complete cycle has substantial bilateral excursions, .328 and .327
radians, yet moves backward .472 mm. The complete refed course ends .517 mm
behind its starting coordinate. Thus preserving cadence and reopening activity
does not preserve the functional sensorimotor regime. Energy regulation has
changed the transmitted muscle waveform and the body's operating trajectory.
This does not establish that the predictor forgot its memory or that the
regulator alone explains every physical effect; all remain coupled.

The next high-information architectural question is whether a shared
supervisory population can preserve and select between several learned,
usable sensorimotor organizations through such state transitions, rather than
merely stabilizing average activity or an internal clock. Treat the existing
association, prediction, rhythm and body mechanisms as provisional parts for
that whole-network experiment. Do not require their perfect characterization,
or turn this result into another gain search. Neither the current low-activity
regime nor the return of movement demonstrates consciousness or mammalian
competence.

Biological boundary: Cheng and colleagues (2025),
[Nature Communications 16, 5954](https://www.nature.com/articles/s41467-025-61179-1),
describe fasting-related brainstem control involving temperature, heart rate,
thermogenesis and activity. Its abstract and initial experimental results were
read. Those coordinated physiological changes are absent here, so a torpor or
sleep label would be unjustified.

Records: `.live/research/20260909_metabolic_population_{intact,cut}_{refed,unfed}_seed11/`.
The analysis folder `20260909_metabolic_population_analysis_seed11/` retains
complete per-tick contrasts, summaries for each actual CPG cycle, and
`regimes-shared-scale.png`. Four copied manifests initially listed the parent's
1,945 IDs. Their ID lists were corrected against both raw records and configured
allocation order; `manifest-as-recorded.json` and its hash preserve each original.
No neural record, physical record or checkpoint changed. The immutable probe
source retains that metadata-copy limitation; use corrected manifests or raw
record IDs when resuming these courses. New raw data total approximately 133 MB.

## Continuation: acquired temporal placement changes action, with a cost/progress tradeoff

Resetting prediction weights changes their strength. The next intervention
therefore rearranges weights only among the four delayed histories of the same
motor source and membrane timescale, separately for each predictor. Each
source/timescale weight distribution remains exactly unchanged. Source identity,
cell count, wiring, neural fast state and physical initial state remain fixed.
Birth weights were uniform, so this rearranges acquired deviations rather than
an arbitrary birth pattern. It does not preserve effective current: delayed
signals have different attenuation and timing, and that is part of what the
intervention tests. This is not a complete removal of learned temporal structure;
differences across sources and timescale bands survive.

Four permutation seeds, 23, 44, 77 and 101, each have sham and .5 Nm load
branches. They are four interventions on one acquired graph, not four graph
replications. All eight 640-tick runs complete, together with eight exact
32-tick unchanged replay checks. Restoring only the original selected weight
objects recovers the complete serialized runtime hash before the intervention
is reapplied. A separate analysis recomputes all four local/physical audits,
verifies the actual permutations and undeclared-state invariants, and confirms
exact sham/load prefixes through index 255. All residuals are zero. No neuron
equation, motor gain, sensory input schedule or basal plasticity rate changes.

Every permutation has the same sequence of first observable effects:
predictor membrane index 1, neural error-receptor arrival 4, muscle actuation
5, and pre-step joint position 6. Full positive and negative trajectories
remain in the audit, including differences that appear well before the load.
This establishes a causal influence of learned delayed-weight placement on
embodied action beyond merely changing each predictor's total weight.

The original placement uses less sampled positive mechanical work under load,
7.162 J versus 7.872 through 8.255 J for the four permutations. Its final-period
load/sham joint distance is also smaller, .04536 rad versus .04696 through
.05342 rad. Neither result establishes a universally better action policy.
All four rearrangements finish farther forward and make more late forward
progress. Original loaded endpoint is -.00607 m, versus -.00507 through
-.00368 m after rearrangement. All still finish behind their starting point.

Original placement also rejects the load less well during some intervals.
Indices are inclusive and relative to the branch checkpoint:

| Permutation | Intervals with larger original load/sham joint distance |
| --- | --- |
| 23 | 257..321, 355..382, 604..618 |
| 44 | 257..326, 404..433 |
| 77 | 257..322, 415..442, 507..538 |
| 101 | 257..326, 402..437 |

The architectural implication is narrower than a learned controller repair.
The circuit has acquired a motor-history-dependent reference and that
reference now participates in the body loop. It has not learned which tradeoff
between movement, effort and stability matters to the organism. Reducing
prediction error does not specify a desired destination or a biological need.
The next behavioral integration should give movement and expenditure physical
consequences and connect those consequences to neural regulation. It should
not select another gain or permutation because its endpoint looks better.
Existing V3 metabolic/body components are candidates to reuse, not evidence
that this population prototype already has that regulation. The old visual
and auditory populations are retained but receive no media playback in these
load experiments; this is not visually guided or multimodal action.

New reusable instrumentation: `sensory_motor_placement.py`, its data-only
analysis `sensory_motor_placement_audit.py`, and six regression tests. The
unchanged prefix initially rejected a schema mistake that sliced static
feedback connection IDs as if they were time samples. The mistake was fixed
before production and has a regression test. No failed attempt is counted as
a completed research run. Existing focused tests: 86 passed in 48.93 s; new
tests: six passed in .27 s. Current raw records total about 94 MB and all are
retained with initial/final executable states. Disk remains at 25 GiB free.

Evidence: `.live/research/20260909_sensory_motor_placement{23,44,77,101}_{sham,pulse}_seed11/`
and `20260909_sensory_motor_placement_audit_seed11/`. The full objective,
including learned hierarchy, reasoning, mammal-level breadth and consciousness,
remains unachieved.

## Continuation: a learned prediction now affects the physical body

The sensory-correction component adds four graded position filters and six
incoming ports to each of four existing antagonistic muscles. The acquired
1,941-cell brain becomes 1,945 cells. Original neurons, synaptic objects,
in-flight signals and extension traces remain; no neuron equation changes.
Muscle input buffers expand from two to eight ports and connection caches are
rebuilt with alias checks. Their input-count-dependent upper t_ref bound also
changes, while the existing t_ref value survives until its next native update.
Every basal learning rate remains positive. The new motor conductance is an
explicit increase in control authority, not conductance-normalized scaling.

All conditions contain the same new neurons and edges. New weights select no
corrective current, filtered signed-position feedback, or signed neural
observed-minus-predicted feedback. The last route uses the existing local
motor-history predictor. It is not a learned upper controller. A fourth
condition resets only the predictor's selected incoming weights to birth
values. This changes weight strength as well as acquired structure and cannot
by itself distinguish useful prediction from a gain or posture effect.

Eight full-state branches each run 640 ticks, 2.56 seconds. Each controller
has a sham course and a .5 Nm left-joint torque during indices 256 through
303. Matched courses agree exactly before force onset. Independent checks of
delivered inputs, muscle dynamics, ongoing learning, temporal-history dynamics
and every physical integration step pass. Raw records and initial/final
executable neural checkpoints remain available. No live demo was restarted.

The physical joint first differs at index 257. Corrective actuator commands
first differ at 262 in all three active feedback variants; commands never
differ in the no-current condition. Learned versus reset prediction weights
first change actuator commands at index 5 and pre-step joint position at 6.
These are exact trajectory divergences, not general latency constants.

| Controller | Mean load/sham joint distance in final 164 ticks, rad | Loaded forward change during final 164 ticks, m | Loaded sampled positive work, J |
| --- | ---: | ---: | ---: |
| No corrective current | .83851 | -.003659 | 3.874 |
| Direct position reflex | .02977 | +.003949 | 9.430 |
| Learned expectation | .04536 | +.001430 | 7.162 |
| Prediction weights reset | .05325 | +.002858 | 8.085 |

The distance is the Euclidean difference between the two joint trajectories,
not error against a prescribed desired joint angle. Full differences and every
unfavorable interval are retained in `20260909_sensory_motor_audit_seed11`.
All corrective variants reduce this load-induced distance throughout the
post-force course relative to no correction. They continue moving; the direct
reflex actually increases stroke amplitude and uses much more work. Work is
an endpoint-sampled mechanical estimate, not exact RK4 work or metabolism.

All loaded courses still end behind their initial forward coordinate. This
does not mean recovery never occurs: all three corrective variants make
positive late forward progress while the no-current body still moves backward.
The direct reflex gives the strongest recovery here. The learned route uses
less work but makes less late progress. Its endpoint forward load penalty is
also slightly worse than the reset route. There is no single winning controller
across these measures, and one acquired graph is not replicated acceptance.

Actual saved-state MuJoCo videos use the same fixed camera in four panes,
with no neural or physical integration during rendering and no interpolation
or motion magnification. They show one initial frame and every fourth tick at
quarter speed. See `.live/research/20260909_sensory_motor_pulse_video_seed11/`
and `20260909_sensory_motor_sham_video_seed11/`, each containing
`comparison.mp4`, three stills and a fidelity manifest. They depict the
controller comparison above, not the later weight-placement intervention.

Relevant primary research: Zobeiri and Cullen (2024),
[Nature Communications 15, 4003](https://www.nature.com/articles/s41467-024-48376-0).
Its active/passive and attempted-movement results distinguish motor-related
signals from movement-generated sensation, and its population model shows
why heterogeneous responses matter. Abstract and the relevant results were
read for this continuation. Their experiment has no explicit learning task;
their fitted population weights do not establish PAULA's local learning rule.
Our graded filters and signed feedback are an experimental construction, not
a reproduction of Purkinje-cell physiology.

## Continuation: descending feedback changes sensory dynamics, but not action in this course

The next experiment tested existing hierarchical communication rather than
adding cells or refining prediction. Four 384-tick branches start from the
same acquired 1,941-cell runtime and physical state. The executable initial
checkpoint payload hashes are identical across all four branches. Factors
are a .5 Nm external torque on the left joint during local indices 64 through
111, and intact versus interrupted upper-to-sensory information. One tick is
4 ms. The disturbance is a laboratory intervention independent of neural
activity, not a controller. Every basal learning rate remains positive.

The cut acts on 768 declared upper-to-visual/tactile ports. It removes arriving
information before native input processing, while leaving modulation and
already queued dendritic currents alone. It therefore also changes
information-triggered plasticity and retrograde signaling. It is not a
forward-current-only lesion. The recorder retains both sides of this
intervention, all population fields, actual forces, full physical integration
state, predictor internals and motor-history dynamics. An uncut observer
matches its unobserved control; deliberately corrupted arrival and force
records fail validation. All four production runs and their independent
factorial analysis pass with zero intervention, history and prediction
equation residuals.

### The physical disturbance reaches and changes the hierarchy

These are first differences between full trajectories, not average latencies
or general conduction constants. Indices start at the branch checkpoint.

| Contrast and measured response | First changed index |
| --- | ---: |
| Torque versus sham, joint sensation | 65 |
| Torque versus sham, tactile membrane / spikes | 68 / 72 |
| Torque versus sham, local predictor error / weights and rate | 69 / 70 |
| Torque versus sham, activity-regulator membrane / spikes | 80 / 84 |
| Torque versus sham, upper membrane / spikes | 86 / 187 |
| Cut versus intact under torque, tactile spikes | 192 |
| Cut versus intact under torque, upper spikes | 199 |
| Cut versus intact under torque, regulator spikes | 213 |
| Cut versus intact, actuator commands or joint trajectories, either force condition | No difference in 384 ticks |

The unperturbed cut removes 17 nonzero arriving values without changing any
population's spike output. Its recorded non-spike state nevertheless differs
from index 51; unchanged spikes do not establish unchanged neural dynamics.
Under torque it removes 1,113 and changes many
sensory and upper-cell spike times in both directions. Tactile cut-minus-intact
output contains 939 positive and 960 negative cell-ticks; upper output has
244 positive and 258 negative cell-ticks. This is state-dependent influence,
not evidence of uniform suppression, useful recovery or content preservation.
The old `mismatch_candidate` population changes its membrane but never fires.
The newer local opponent circuit reacts well before upper spikes change.

The torque ends after 192 ms, but the body remains differently displaced.
At branch index 383, the left joint is 1.2241 radians versus .3726 without
torque. Neither course exceeds the declared +/-1.8 radian joint range.
Persistent sensory and upper activity therefore cannot be attributed to an
autonomous neural regime merely because the applied force has stopped.
The world and body carry state too.

As a physical diagnostic, projection of endpoint torso displacement onto its
initial forward axis is +.00501 m without torque and -.01652 m with torque.
These values are reconstructed from the recorded full body states, not a
new simulation. They identify a candidate load-sensitive movement task, not
an accepted locomotion benchmark or proof that the shorter baseline is robust.

### Where this composition stops

The configured forward graph has no path from `upper_core` to either motor
population. The new prediction consumers also have no forward route into
`upper_core`. Native plastic return paths remain, so the entire system is not
feedforward or causally disconnected. Indeed, the feedback cut changes the
new predictor weights by up to 4.52e-7 under torque. Yet that influence does
not change actuator commands or body motion anywhere in the measured course.
This is an identified action-interface gap in the current population research
brain, not a claim about the separate V1-V4 demo agents.

The same records separate temporal persistence from state information. Joint
input differs from index 65, while the complete selected contextual-arrival
prefix remains identical through index 258. Motor-history input first differs
at 257, after a tiny muscle-terminal difference at 256. However, local error
already differs at 69 and predictor weights at 70. Identical motor-history
inputs do not imply identical complete adaptive states. The weights can carry
bodily history; claiming that this entire model is unobservable would be wrong.

The scientific decision is to stop this diagnostic without tuning feedback
gain. Descending coupling exists and affects lower dynamics, but useful
regulation of action has not been built. The next architectural hypothesis
should give sensory populations an explicit neural route to the antagonistic
motor plant under a physically achievable load challenge. A competent direct
proprioceptive reflex is the necessary comparison if a learned upper route is
proposed. If the reflex suffices, accept sensorimotor closure without claiming
that it establishes a necessary hierarchy. More prediction precision or more
audit tests alone would not resolve this missing action interface.

This interpretation is consistent with two published distinctions, not a
replication of either model. The abstract of
[Finkelstein et al., 2021](https://pubmed.ncbi.nlm.nih.gov/33875892/)
reports changing behavioral sensitivity despite continued transmission of
distractor activity. Useful dynamical gating need not mean suppressing the
incoming signal. The indexed results and discussion of
[Ji et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8139836/)
describe motor feedback supporting sustained navigation through sensory
fluctuations. These sources motivate testing functional sensorimotor coupling;
they do not validate the PAULA topology or a consciousness claim. Full-text
access attempts were blocked, so no complete methods review is claimed here.

The retained runner is `hierarchical_body_perturbation.py`, with
`hierarchical_body_audit.py` for the four-condition analysis. Raw runs are
`20260909_hierarchical_body_{intact,cut}_{sham,pulse}_seed11` under
`.live/research/`. The current combined analysis is
`20260909_hierarchical_body_integrated_audit_seed11`; the earlier analysis
without the explicit context-observability table is retained separately.
Neither is a new neural run. The four new raw recordings total 26,999,678
bytes. This is a one-graph, one-state mechanistic result, not replicated
hierarchical regulation, multimodal action selection or mammalian cognition.
All 82 focused regression tests pass, including six new intervention and
interpretation checks. Experiment workers and tests have completed. Available
disk space remains 25 GiB; no live agent or acceptance matrix was restarted.

## Continuation: a temporal population repairs missing support, not the whole predictor

The matched motor-history experiment is complete. Each variant has 1,941 PAULA
cells: the retained 1,793-cell embodied preparation, 128 added graded history
cells and 20 added predictor/comparison/readout cells. Each of eight motor
sources projects into sixteen history cells. Their dendritic delays are 1, 8,
24 and 48 ticks. The multiscale variant uses membrane constants 2, 8, 32 and
128; the short variant uses 2 throughout. Cell count, connections, delays,
birth weights, positive basal rates and maximum weight capacity match exactly.
Only the history cells' membrane constants differ. No new neuron equation,
special hold timer or host-computed error was introduced.

The bank uses ordinary propagation and integration through the existing
graded PAULA extension. Its isolated impulse test verifies each member's
latency. After index 127, maximum short-bank output is below 1e-20 while the
multiscale bank still exceeds .001. Passive-observer tests and independent
full-tick audits verify input delivery, synaptic currents, bounded native
weight changes, membrane dynamics and release. Counterfeit weight changes
on inactive ticks are rejected too, rather than checked only when an input
happens to expose them.

This structural hypothesis is motivated by
[Kennedy et al., 2014](https://www.columbia.edu/cu/neurotheory/Larry/KennedyNatNeuro14.pdf).
Their fish recordings identify delayed and diverse corollary-discharge
responses, including a contribution from unipolar brush cells. Their methods
combine measured responses with fitted synaptic filters and a different
anti-Hebbian learning model. This PAULA filter bank does not reproduce that
cellular mechanism, rebound firing, fitted parameters or their stability
analysis. Relevant results and modeling methods were read from the authors'
paper copy after the PMC page failed to load.

### Embodied results and their limits

Both variants ran for 1,024 ticks in the same radians-explicit, .08-force-gain
body. Each then produced three 512-tick complete-state branches with acquired,
birth-reset or within-predictor shuffled contextual weights. All plasticity
continued. The 5,120 recorded ticks pass independent history, predictor and
physical-integration checks with zero residual. Physical and sensory histories
are exactly equal across the two acquisition conditions and across the
within-condition interventions in this course.

The new bank increases each new predictor's input count from eight to 128,
while keeping unit total birth conductance. Its maximum permitted total
conductance is consequently larger. Improvements over the old predictor are
NOT a timescale-only result. The short-versus-multiscale comparison controls
for that expansion.

| Measurement | Short bank | Multiscale bank |
| --- | ---: | ---: |
| Minimum summed contextual arrival after tick 255 | .0245156 | 1.00747 |
| Target exceeds conditional output envelope by .01, channel counts after index 63 | 44, 0, 51, 0 | 0, 0, 0, 0 |
| Acquired-weight benefit over birth, last 164 branch ticks | .0157785 | .0176019 |
| Acquired-weight benefit over shuffled placement, same interval | .0068310 | .0074698 |

The envelope gives every contextual weight its allowed maximum and includes
an initial membrane bound and conservative numeric slack. Removing an envelope
violation establishes available forward support under the recorded inputs,
not that learning has found an accurate predictor or that arbitrary body
states are representable.

Both banks show a learned weight-placement contribution to actual opponent
error. It first appears at branch index 3. The multiscale-minus-short advantage
is much weaker than either bank's learned-versus-birth contrast. At the end
of acquisition, the mean short-minus-multiscale error over the last 164 ticks
is only .000256752. Channel-specific values are -.000678684, .000825051,
.00182120 and -.000940555. Two channels favor each bank. The multiscale bank
has higher error on many earlier intervals, with the mean contrast reaching
-.0365697 during the trajectory. It is not a uniformly better predictor.

The multiscale learned-minus-shuffled placement test also has unfavorable
aggregate intervals: branch indices 9-21, 26-30, 175-185, 192-195, 338-350,
358-360 and 501-511. Against birth weights its aggregate benefit never becomes
negative, but individual channels still have 248, 173, 0 and 0 unfavorable
ticks. All intervals and per-channel effects remain in
`20260909_temporal_body_audit_seed11/effects-per-tick.npz`. The final window
alone would overstate success.

### What this changes in the next architectural decision

Temporal support and useful state representation are different requirements.
This bank keeps motor information available between commands and lets acquired
weights use it. That does not make motor history a sufficient description of
the physical system. A motor command's consequence depends on current joint
state and external conditions as well as recent commands. Those quantities
are absent from this predictor's context, even though physical afferents reach
other populations in the larger network.

The next hypothesis is mixed motor and sensory-state context, with explicit
tests of motor information's contribution beyond delayed sensory copying.
Movement omission, altered initial physical state and perturbations are needed
to distinguish a predictive state from a learned response to one periodic
course. Further acquisition, independent whole-graph replication and changed
movement timing also remain unresolved. No single new population has been
accepted as a general forward model, neural supervisor, global workspace,
learned action selector or consciousness mechanism.

The runnable path is `temporal_body_probe.py`, `temporal_body_branch.py`, then
`temporal_body_audit.py`. Each records explicit IDs and full selected dynamics,
uses exact runtime checkpoints, checks source hashes and refuses to overwrite
an output directory. These experiments use the corrected physical auditor and
complete normally; they do not require the historical recovery workaround.

## Continuation: physical consequences of the population brain's motor output

The 1,761-cell audiovisual preparation now has a separate physical research
composition with 1,793 PAULA cells. It adds the retained four-cell CPG and four
muscle cells, four opponent joint-position afferents, four motor-context
predictors, eight opponent comparison cells and eight prediction consumers.
Physical afferents also project into the existing 192-cell tactile core.
Every old audiovisual neuron remains, and the selected acquired audiovisual
weights transfer. Other states start from the configuration. This experiment
does not play audiovisual media or demonstrate multimodal action selection.

The host only converts actual joint positions into sensory currents and muscle
membranes into actuator controls. It supplies a single birth kick at tick 3.
Predictions, opponent comparisons and local rate-dependent plasticity run in
PAULA. The reference motor configuration defaults to frozen learning, so the
new composition explicitly uses positive basal rates. Reserved zero-current
CPG ports repair the one-input learning-window bounds in this composition.
Muscles gain graded neural release while retaining their membrane-driven
physical output. Old reference code, demos and other organisms are unchanged.

### The body and transducer initially hid most of the movement

The reference XML's `range="-1.8 1.8"` means degrees, since it has no explicit
compiler angle convention. Compiled limits are +/-0.03141593 radians. This was
verified in the installed MuJoCo 3.5.0 model, not inferred from the displayed
animation. [MuJoCo's XML reference](https://mujoco.readthedocs.io/en/3.3.1/XMLreference.html)
documents the default. The new RadianResearchRower explicitly changes only
that convention; it is a different body, not a silent rewrite of old evidence.

| Body / muscle conversion | Ticks outside left/right limits | Maximum absolute angles, radians | Fractional afferent samples, L+, L-, R+, R- |
| --- | --- | --- | --- |
| Reference degrees, gain 8 | 594 / 594 | .80804 / .80562 | 0, 0, 0, 0 |
| Explicit radians, gain 8 | 446 / 477 | 3.27539 / 2.90785 | 55, 93, 96, 21 |
| Explicit radians, gain .08 | 0 / 0 | .52100 / .60791 | 512, 82, 594, 0 |

Each row contains 640 recorded ticks. The last change reduces force conversion
100-fold without changing neural weights or motor wiring. At the reference
gain, peak muscle output maps to about 267 Nm through gear 25. These soft-limit
excursions are a physical calibration and sensory-information problem, not an
integration crash. Changing units alone was insufficient. The quieter body
provides graded information, but its first course never excites R-; it is not a
complete exploration of the body's state space.

The reference-gain recording locates motor onset at tick 45, physical sensory
input at 46 and tactile-core release at 49. Disconnecting muscle-to-force
conversion leaves the CPG and muscles active but removes motion, joint input
and tactile-core release. The CPG has a 164-tick cycle with 41 ticks between
successive phase cells. This composition uses one neural tick per 4 ms physics
step, unlike the reference's default six neural ticks per body step. That
timing change is explicit; reference behavioral equivalence is not claimed.

### Learned physical prediction has a measurable contribution and a temporal limit

After the .08-gain course, three 512-tick branches start from the same complete
neural/body state. Selected motor-context weights remain acquired, return to
birth values, or are shuffled within each predictor. Learning remains active.
The independent audit verifies applied sensory transduction, physical replay,
the neural comparison/learning equations and nonselected initial fields. All
residuals are zero. Physical and sensory trajectories happen to remain exactly
identical across these three branches, so movement differences do not explain
their error contrast in this test.

Acquired weights first change opponent error at branch index 3. Aggregate
error is never worse than the birth-weight control; the late 64-tick mean
reduction is .00430006. That aggregate conceals unfavorable channel-specific
ticks: 186, 111, 5 and 0 across the four channels. Against shuffled weights,
the late reduction is .00263900, but aggregate error is worse at indices
22-24, 82-146, 246-310 and 411-474. These recurring intervals are retained in
`20260909_proprioceptive_learning_audit_seed11/effects-per-tick.npz`.

The trace explains why more acquisition alone cannot provide a precise
position prediction. At global tick 760, physical positive joint inputs are
approximately .2504 and .3442, but each predictor's total motor-context arrival
is only .00007. Muscle activation and CPG impulses have faded while physical
displacement persists. The local context trace influences weight updates; it
does not itself provide persistent forward current.

The capacity auditor gives every selected contextual weight its maximum
permitted value and begins with PAULA's maximum membrane potential. A
conservative float-rounding envelope then bounds forward output under the
recorded motor releases. After index 63, actual target receptor output exceeds
that envelope by more than .01 on 177, 66, 325 and 0 ticks across the channels.
All recorded predictor outputs lie below the envelope. This is conditional on
the present inputs and circuit, not an impossibility result for PAULA, and
zero instantaneous error is not the acceptance criterion. It identifies a
missing temporal representation rather than merely weak training.

The next structural hypothesis is a population that preserves recent motor
and sensory history on several timescales. Motor command alone also omits
initial body state and external forces. Future tests must include those
conditions, movement omissions and independent graphs, then test whether
prediction changes useful neural action. Simply increasing weights or calling
all error cells a supervisor would not resolve this evidence.

Zobeiri and Cullen's [2024 macaque study](https://www.nature.com/articles/s41467-024-48376-0)
supports testing motor-related signals separately from physical feedback.
They compared active, passive, combined and attempted-but-restrained movement.
Their population model used fitted inhibitory combinations of heterogeneous
Purkinje responses. It motivates those experimental distinctions, but does
not establish this PAULA rule, its local learning mechanism or this body's
biological fidelity. Relevant results and methods were read; the ELL paper
found in search was inaccessible and has not been treated as a reviewed result.

### Instrumentation correction and reproducibility

The three branch simulations recorded all requested ticks, then their original
post-run actuator audit failed. It multiplied a float32 muscle array before
conversion, whereas the physical bridge converts to float64 first. Birth
records had promoted to float64 and concealed this distinction. The independent
auditor uses the actual conversion order and retains exact equality. A
regression test rejects even a one-ULP actuator change. Raw evidence is not
rewritten. `proprioceptive_recovery.py` restores the original neural and full
MuJoCo integration states, requires every replayed array to match exactly, and
only then writes the missing final runtime and a recovery record.

Recovery is complete for all three branches: each replayed all 512 ticks and
all 31 recorded arrays exactly, including physical integration state, neuron
fields, selected weight/receptor dynamics and terminal information. Their
`recovery.json` files identify the verified final checkpoints. The earlier
independent audit's statement that those checkpoints were missing describes
its execution time, before recovery. Seventy-one focused regression tests pass.
All experiment workers have exited; no live server or agent suite was started.
The disk still has approximately 25 GiB free.

This is motor-driven embodied learning in a larger connected preparation, not
an accepted autonomous brain. There is no learned action selector, generalized
hierarchical recall, complete supervisor, mammalian capability or demonstrated
consciousness. The full research goal remains unchanged.

## Continuation: timing transfer and the neural use of learned expectations

The four sensory-timing courses have finished. They contain 10,112 new recorded
ticks across both acquisition mappings and orders. The two independent timing
audits reproduce applied sensory inputs, initial-state checks, reference replay
prefixes and the declared predictive-path equations exactly. The native
300-tick baseline is reused from retained transplant recordings, not inferred
from the 32-tick replay controls.

The learned-minus-shuffled assignment-by-cue contribution stays positive on
the declared physical sound-profile axis at every index from 10 through 299
under native, one-phase-shifted and continuous visual drive, in both orders.
The following values locate intervals in the full per-channel evidence; they
are not recall accuracies.

| Input timing | Order 0, indices 32 to 63 | Order 0, indices 288 to 299 | Order 1, indices 288 to 299 |
| --- | ---: | ---: | ---: |
| Native | .0419548 | .0356616 | .0356372 |
| Shifted by one phase | .0419966 | .0356936 | .0356693 |
| Continuous | .0421355 | .0357310 | .0357074 |

This association contribution does not depend on the original four-tick input
phase. It does not establish invariance to all timing changes. Continuous
per-channel input dose is between .98699 and 1.00948 times native dose across
the two changing movies; shifted dose is between .97807 and 1.02195 times native.
Equal dose was established only in the earlier constant-image experiment.
Neither that externally driven rhythm nor this successful transfer demonstrates
a learned recurrent regime.

The pooled interaction also conceals unequal individual contributions. Under
native input, paired-history cue 1 and swapped-history cue 0 have consistently
negative learned-minus-shuffled projections after index 31. Paired cue 0 is
small and sometimes negative in order 0; swapped cue 1 crosses zero in both
orders. These signs describe individual weight-placement effects on one fixed
physical axis. They are not four independent successes at recalling a sound.
All individual effects remain in `effects-per-tick.npz` under
`20260909_sensory_timing_audit_order{0,1}_seed11`.

### Test the error circuit before building a supervisor on it

The new `associative_mismatch_probe.py` crosses both visual clips with both
auditory clips. Every physical event is presented to networks carrying weights
from both acquisition mappings, with learned and within-cell shuffled placement.
Only selected contextual weights are transplanted into the original executable
state; ongoing adaptation remains active. Familiarity is an analysis label,
never an input to the brain. Both orders are tested, still on one original
graph seed. All four courses are complete, with 10,112 recorded ticks and
666,098,579 bytes of raw data. Sixteen native visual-only replay prefixes match
the prior records exactly. Both completed mismatch audits have zero residual
for the declared predictive-path equations and pass the initial-state and
physical-input checks. All workers have exited.

The question is whether the SAME physical event produces less actual opponent
error with familiar weights than with weights from the contrary history, and
whether local learning rates reflect that difference. The independent auditor
keeps each physical pair, each opponent channel and each tick. It checks every
nonselected initial weight as well as activity, terminals and receptor state.
Lower aggregate activity alone is not sufficient: silence, physical loudness,
recency or a change in sensory input must not substitute for learned expectation.

### Completed result: useful context, imperfect event timing

The learned histories change opponent comparison outputs at index 12 and the
local plasticity rates at index 14. Every physical pairing has a positive
late familiar-history benefit in both orders, and subtracting the shuffled
weight-placement control preserves that benefit. Its late means are below,
averaged across channels at indices 288 to 299. A positive value means that
learned feature placement contributes to lower error for the familiar history
when both histories receive precisely the same physical input.

| Physical event | Order 0 | Order 1 | Indices with negative placement benefit |
| --- | ---: | ---: | --- |
| Visual 0, audio 0 | .00541480 | .00565658 | 224 to 234, both orders |
| Visual 0, audio 1 | .00501969 | .00524911 | None |
| Visual 1, audio 0 | .00405103 | .00380023 | 224 to 232, both orders |
| Visual 1, audio 1 | .00386983 | .00363110 | None |

The negative intervals are real, not floating-point sign noise. For visual 0 /
audio 0 / order 0 the channel-mean placement benefit reaches -.00230349.
Individual channels also disagree outside these intervals. For that event,
75 channels have positive and 21 negative late placement contributions. These
are distributed signals, not a uniformly correct population of anomaly cells.
Local rate contrasts first change two ticks after the opponent outputs and
have additional brief sign reversals. The exact audited receptor rule explains
them; a lower mean comparison output does not force every neuron's filtered
error magnitude or plasticity rate to be lower.

The raw trace identifies the sensory event behind the major reversal. Sound
clip 0 is almost silent near index 216, then has a burst. Its mean physical
auditory feature rises to .374204 at index 220. In the visual-0 learned probes,
mean auditory arrival at the comparison cells is .103997 at index 222. It is
the same to the displayed precision for both acquired histories. Predictor
arrival changes much less: .027369 for paired weights and .032934 for swapped
weights. At index 225 the paired history's positive and negative comparator
outputs average .020754 and .013381; swapped weights give .017100 and .015309.
Thus a stronger contrary-history prediction briefly fits the burst better,
whereas the quieter familiar prediction fits the surrounding quiet intervals.
All channel-level inputs, delayed synaptic currents and outputs are retained;
the comparator equation audit reconstructs the integration exactly.

This is evidence that acquired associations are used by a neural error and
learning-rate pathway. It is not evidence that the circuit predicts the
unfolding acoustic sequence. Neither is a short unfavorable error interval by
itself proof of failed statistical learning: a sensible expectation can be
wrong on an individual observation. Requiring smaller familiar error on every
tick would impose a stronger criterion than justified by this task. Conversely,
using only the favorable endpoints would hide this circuit's temporal limits.

The earlier physical-profile interaction also has a nonzero channel-centered
component. Splitting the fixed sound-profile axis into uniform and centered
parts gives late native contributions .01398795 and .02167360 in order 0,
and .01397423 and .02166297 in order 1, with the original normalization.
The centered contribution is positive at indices 10 to 299 for all three
tested input schedules. This descriptive decomposition does not fit a decoder
or change the brain. It also does not exclude gain on a nonuniform baseline
profile, or establish category or waveform recall.

Evidence is under `20260909_associative_mismatch_{paired,swapped}_order{0,1}_seed11`
and `20260909_associative_mismatch_audit_order{0,1}_seed11`. The audit records
every fixed-input crossed-history comparison, including the unfavorable ones.
Six new fixture tests verify stimulus crossing, passive audiovisual recording,
positive ongoing learning, initial weight column identity and comparison sign.
The combined focused regression has 62 passing tests. Shared neuron sources,
live servers and body agents were not changed or started.

The next architectural experiment should test whether temporal context carries
information that improves prediction, rather than making a sharper fitted
readout of these two clips. First establish that the available sensory history
actually predicts an event. Action-generated sensory consequences offer a
stronger next embodied test: neural motor signals can precede physical feedback,
and delayed or blocked consequences can violate a learned relation. Reuse the
existing body and motor infrastructure, retain neural-only control and positive
plasticity, and compare matched physical replays with the closed loop. This
is a proposed next experiment, not an implemented sensorimotor result. Whole
graph replication and unfamiliar examples remain required; order replication
on seed 11 is not independent-network replication.

### Literature and the unresolved supervisory input

[Garner and Keller, 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8737331/)
report association-specific suppression in mouse visual cortex after auditory
cues in a behaviorally relevant conditioning paradigm. Their unpaired-cue and
optogenetic controls motivate testing neural use of cross-modal predictions.
Their result supports neither locating every association in an upper layer
nor treating our visual-to-auditory preparation as a replication. The directions,
stimuli, reinforcement, timing and neuron models differ.

[Meng and Wang, 2026](https://www.nature.com/articles/s41467-026-70354-x.pdf)
study local inhibitory learning guided by a global mismatch-dependent factor.
The methods define that factor using the discrepancy between stimulus and
prediction; their discussion explicitly leaves its biological generation open.
That model helps frame inhibitory alignment but does not supply a completed
neural supervisor. Its simplified rule is initially instantaneous, and its
biophysical formulation is not PAULA. I inspected the relevant results,
learning equations and discussion, not every supplementary derivation. No
equivalence theorem from that paper is being asserted for this preparation.

For PAULA/ALERM the distinction matters: comparison, prediction acquisition and
regulation of the conditions for useful dynamics are separate causal jobs.
Our channel-aligned comparators are anatomically specified; the contextual
predictions are learned. A self-regulating hierarchy must acquire and use
reliable relationships without a host deciding what counts as surprising.
That remains part of the full research objective, alongside action, energetics,
retention, generalization and independent-graph replication.

## Continuation: weight-carried association and an input-clock confound

Both orders of the 1,761-cell contrast/predictor course are complete. All four
histories, 15,936 recorded ticks, pass the predictive pathway's independent
learning, delivery and comparator audit with zero residual. This reconstructs
those declared equations, not every intracellular variable of all 1,761 cells.
The full-channel assignment-by-cue interaction has a positive physical-profile
projection at indices 10 through 299 in both orders, and zero before that.
`20260909_contrast_prediction_order_audit_seed11` retains both full trajectories
and their order-balanced values. The earlier entry's running status is historical;
all acquisition and probe workers have now finished.

### Does the acquired information survive without acquired activity?

`predictive_weight_transplant.py` transfers only the learned contextual input
weights into each source run's original, unexperienced executable checkpoint.
All other weights, terminals, activity, modulation, receptor traces and queued
events remain at birth. Learning continues during the new probes. The second
condition shuffles the same weights within each predictor with permutation seed
23, preserving each cell's exact weight multiset and fan-in while changing its
feature assignments. This is a causal preparation, not a biological reset or
shuffle mechanism to be installed in an agent.

There are four completed courses, two pairings by two orders. Each includes two
32-tick birth replay controls and four 300-tick visual-only probes. These 5,056
new ticks are fully recorded for the existing cell observables and the selected
learning pathways. All eight control prefixes match the previous recordings
exactly. Both per-order transplant audits have zero local equation residual.
The unexperienced initial network responds differently after receiving weights
from the two different audiovisual histories. Thus the measured learned
contribution does not require carrying over the acquired activity state.

The weight-placement effect, learned transplant minus shuffled transplant in
the crossed assignment-by-cue comparison, projects positively on every index
10 through 299 in both orders. Its downstream consumer interaction first differs
at index 12. Window means on the declared physical spectral-profile axis are:

| Training order | Indices 32–63 | Indices 288–299 |
| --- | ---: | ---: |
| 0 then 1 | .0419548 | .0356616 |
| 1 then 0 | .0419621 | .0356372 |

These are normalized activity contrasts, not recall accuracies. In order 0,
the shuffled pathway's remaining assignment interaction lies between -.000491
and .000845 after index 31, whereas the learned-minus-shuffled contribution
lies between .03270 and .04547. This supports a role for learned feature-to-port
placement beyond the per-predictor weight distribution. Do not turn the ratio
into a percentage of memory stored there: the responses evolve with active
plasticity and return pathways. One network seed, one permutation seed and two
familiar recordings are still insufficient for a general association claim.
No new category, noisy-cue generalization, useful decision or body action was
tested here. The result is a weight-carried, cue-dependent contribution to a
neural auditory-channel prediction, not a replayed sound waveform.

The birth replay controls caught a measurement failure before accepting these
results. The original contrast recorder flattened weights in population
allocation order; JSON later alphabetized population keys. Reloading those keys
reordered the columns without changing the neural weights. Recovering the
original ascending neuron-ID order restored exact agreement. The adapter now
records an explicit neuron sequence, and a JSON round-trip regression preserves
the layout. Failed attempts are suffixed `_layout_failed` and contain setup
metadata, not accepted probe results. No old source or raw trace was overwritten.

### Input scheduling creates a rhythm even for a uniform image

The actual media encoder drives each sensory channel only when the current
tick modulo four matches its neuron ID modulo four. Earlier constant-current
contrast fixtures drove all channels together and did not test that interface.
A small diagnostic exposed nonzero, four-phase contrast responses to a spatially
uniform image. The new `population_input_phase_probe.py` then tested the same
question in the full 1,441-cell contrast preparation, with native learning and
return pathways active.

Both conditions present a uniform value of .6 for 300 ticks. The existing
encoder supplies 1.2 once every four ticks per receptor; the comparison supplies
.3 on every tick. Their integrated external information is equal for each
receptor. The time course and local learning exposure are not equal, which is
part of the interface intervention. Every external input, cell observable,
information terminal and contrast weight is retained. The independent delayed
contrast-equation audit has zero residual in both conditions.

After index 63, the four-phase condition's maximum contrast output is .01411;
the continuous condition's is .000305. For the first above-pool cell, mean output
at the four input phases is approximately [0, 0, .00637, .00835] under scheduled
input, versus [.000148, .000149, .000150, .000150] under continuous input. Full
trajectories support the phase relationship; these means only index the records.
The smaller continuous response is not identically zero. The full graph has
adaptation and heterogeneous return pathways, unlike an ideal symmetric static
subtraction. This is not evidence that all rhythmic activity is artificial, nor
that timed sensory encoding is inherently invalid. It is a demonstrated source
of externally paced activity that must be distinguished from learned or
autonomous neural dynamics. The earlier real-media capacity result remains a
result about the observed neural representation, not pure spatial contrast.

### Consequence for the ALERM research target

I re-read the complete current `al-paper/alerm.tex`. Its formalization describes
recall as a gradient flow while also describing memory as crystallized
limit-cycle pathways. Those descriptions need explicit scope. With a fixed
smooth potential and strictly positive scalar mobility, pure gradient flow
obeys dF/dt = -mu times the squared norm of grad(F), so it cannot sustain a
nonconstant autonomous periodic orbit. Changing inputs, adaptation, delays,
non-gradient circulation or a restricted-state description can change that
conclusion. The equation alone does not demonstrate any of them in PAULA.

A useful mathematical distinction is stability transverse to a regime versus
motion along it. For example, with J a 90-degree rotation, the two-dimensional
system dx/dt = (a - norm(x)^2)x + omega Jx, a > 0, has an attracting cycle of
radius sqrt(a). The radial potential V = (norm(x)^2 - a)^2/4 decreases with
dV/dt = -norm(grad(V))^2, yet the state keeps circulating on that cycle because
the rotational term remains nonzero there. This is an illustrative dynamical
system, not a PAULA extension, a new neuroscience result, a fitted brain model,
or an identification of V with metabolic or variational free energy.

For this project, the design question is whether neural regulation can restore
an information-carrying regime without erasing its content or stopping useful
evolution along it. Continued weak learning may slowly change that regime.
The present weight-transplant result establishes neither a recurrent basin nor
that regulatory capability. The input-phase control also shows why a wave-like
display would not establish either. Next test the learned relation's robustness
to input timing and nearby physical cues, replicate graph construction, then
test neural use and regulation of a recurrent population state. A learned
association and an internally generated regime are different requirements;
neither may be substituted for the full embodied goal.

New artifacts are `20260909_contrast_transplant_{paired,swapped}_order{0,1}_seed11`,
`20260909_contrast_transplant_audit_order{0,1}_seed11`, and
`20260909_population_input_phase_seed11` under `.live/research/`. The transplant
and timing courses produced 314,567,091 bytes of new raw data. There are 50
passing focused tests; all workers and auditors finished. The last disk check
showed 24 GiB free. No shared neuron-model source, live server, body simulation
or agent-version suite changed, and no additional old data was deleted.

## Continuation: presentation order and the input representation, 9 September

The raw-input predictive bridge's reversed-order recordings are complete. The
new audit reconstructs all 32 records with zero local equation residual. Together
with order 0, this is four acquisition histories and 15,936 recorded neural ticks,
not four independent network seeds. The full channel trajectories show the
assignment-by-cue interaction projecting positively onto the fixed physical sound
profile difference at every probe index 6 through 299 in both orders. The first
six indices are zero. The order-balanced projection's mean is .11525 at indices
32–63 and .02412 at 288–299. This weakens a simple final-sound explanation but
does not remove nonlinear history-dependent gain, demonstrate semantic recall,
or establish useful action. Correlated ticks are not statistical replications.
`predictive_order_audit.py` verifies the graph, acquisition schedule, media,
per-order audit and raw records before combining their full trajectories.

A separate diagnostic identifies an input-representation limitation. A fixed
nonnegative linear readout of two cue means can only produce a pair of outputs
inside the positive cone of its input-channel mean pairs. For the original
32-channel random visual projections, 63 paired and 66 reversed auditory channel
mean pairs fall outside that cone at a squared-residual tolerance of 1e-12.
Allowing all 96 physical visual channels leaves 62 and 63 outside. This is not
an impossibility theorem for the adapting nonlinear brain. It concerns a stated
static family and does not make exact reconstruction a biological requirement.
Finite-time responses, changing weights, recurrence and different neural codes
are outside that bound. The diagnostic weights are never supplied to neurons.

`components/sensory/population_contrast.py` adds a neural input representation:
one graded pool, 96 matched channel relays, 96 above-pool cells and 96 below-pool
cells. Positive and inhibitory projections form the difference; graded release
rectifies it. There is no host-computed surround, stimulus identity, tonic label
or fitted embedding. The existing 1,152-cell graph remains active, and all added
postsynaptic and retrograde rates are positive. This is a global surround over
the supplied channel list, not a reconstructed retina or spatially local
receptive field. The unchanged base `neuron.py` and `network.py` were read in
full before interpreting the transmission and plasticity.

The biological motivation is limited. Pitkow and Meister's 2012
[retinal decorrelation study](https://pubmed.ncbi.nlm.nih.gov/22406548/) reports
that nonlinear processing accounts for much of the decorrelation, beyond the
center-surround filter alone. That supports investigating transformed sensory
codes, not claiming these particular PAULA cells reproduce the retina. The
primary abstract and indexed passage were accessible; full HTML access returned
a browser challenge. This continuation does not claim a full-paper replication.

The 1,441-cell contrast preparation completed 600 real-input ticks, 300 for each
recording. Its independent arithmetic auditor reconstructs every added cell's
delayed graded state and output with zero residual. All cellular state fields,
information terminals and added incoming information weights are recorded at
each tick. The observed contrast-output mean pairs remove the unbounded-cone
residual for both sound assignments. That is representation capacity, not learned
association. Uniform-input and unequal-input fixtures check common-input
rejection, opposite-polarity response and the first response at tick index 5.

The next preparation composes that contrast population with the 320-cell
predictor/comparator/consumer bridge, giving 1,761 neurons. It retains 32 context
inputs per predictor, unit total initial context weight and the original local
learning constants. It changes source amplitude, representation and delay, so it
is not a scale-only control. Two workers run the two pair assignments through
both orders. Each run records the same acquisition, initial/trained visual and
blank probes, and selected-weight-reset probes. The added recorder captures all
contrast weights as well as the existing full-cell and predictive-state arrays;
tests verify that it does not alter neural state, including across a recording
boundary with signals in flight. Initial launches failed before neural ticking
because the optional checkpoint serializer was absent. Relaunching with
`uv run --offline --no-sync --with cloudpickle==3.1.2` supplied it. Setup-only
directories are explicitly suffixed `_setup_failed` and are not results.

An additional bounded-capacity check uses actual predictor-port arrival means
and actual auditory terminal-release means from the first two completed
audiovisual presentations. It retains the declared per-synapse cap. The summed
static mean-map squared-error bounds are:

| Experienced assignment | Raw neural input | Contrast neural input |
| --- | ---: | ---: |
| Original pairing | .01882976572 | 8.6118747e-10 |
| Reversed pairing | .04330148467 | .00022526968 |

These are conditional mean-map approximation errors, not behavior errors,
accuracies, or bounds on the changing dynamical system. The probe histories can
also change their input sources through native return pathways. Each numerical
solution includes an independently recomputed convex tangent-plane lower bound
and feasible upper-bound witness; the largest gap here is below 2.9e-14. One
optimizer boundary value was -1.39e-17. The diagnostic now projects such roundoff
back into the feasible box and recomputes both bounds, with a regression test.
No optimization result is installed in a brain. This checks that the promising
unbounded-cone result was not solely an artifact of permitting unlimited weights.
It still does not show that the local learning dynamics reach those weights.

The composed order-0 course has now completed both assignments, 7,968 ticks in
32 records, with zero equation residual. Both selected-weight interventions
pass the same retained-state and all-input-weight audit as the raw bridge.
The predictor's assignment-by-cue interaction becomes positive at index 10;
the consumer's full-channel interaction first differs at index 12. Predictor
projection remains positive through index 299. Relative to the raw bridge's
same-order interaction, it is smaller on every index 32 through 191 and larger
on every index 256 through 299. Its window mean is .04246 at indices 32–63,
.04507 at 96–127, and .03580 at 288–299. The raw bridge's corresponding early
and final means were .12113 and .02021. This is slower onset and a smaller early
effect with a larger late contribution, not an unqualified improvement. Reduced
source amplitude also changes local learning exposure and can slow extinction.
Some individual signed effects remain negative relative to birth; the positive
crossed interaction does not by itself establish correct total predictions.
Full effects are in `20260909_contrast_prediction_audit_order0_seed11`.

Both composed order-1 workers are still running at this entry; each has completed
the first 792 acquisition ticks. Do not count capacity witnesses, partial
recordings or passing recorder tests as accepted recall. The next evidence is
the completed learned-weight intervention trajectory across both orders,
followed by replication on independent graph seeds and functional use.
Silence remains actual negative auditory evidence in this
architecture. Retaining weak plasticity while distinguishing absent input from
internally generated content remains an unresolved neural design problem, not
a reason to disable learning during probes. Generalized multimodal learning,
recurrent hierarchical state formation, action/perception and embodiment remain
open. This work does not demonstrate consciousness or mammal-level abilities.

There are 45 focused passing tests. No base neuron source or live/body version
changed, and no live/body server was started. The last disk check showed 24 GiB
free. Each new course has a 320 MiB raw budget and a free-space reserve; the two
active workers were approximately 0.52 GiB RSS each and one CPU core each at the
sampled instant. No further old records were deleted in this continuation.

Artifacts under `.live/research/` are
`20260909_predictive_bridge_audit_order1_seed11`,
`20260909_predictive_bridge_order_audit_seed11`,
`20260909_predictive_context_capacity`,
`20260909_population_contrast_seed11`,
`20260909_predictive_bounded_capacity_order0_seed11`,
`20260909_predictive_bounded_capacity_swapped_order0_seed11`,
`20260909_contrast_prediction_audit_order0_seed11`, and
`20260909_contrast_prediction_{paired,swapped}_order{0,1}_seed11` courses,
complete for order 0 and running for order 1.
The new course entrypoint is `contrast_prediction_probe.py`; completion is
indicated by `completion.json`, not mere directory existence.

## Latest continuation: an explicit predictive bridge and raw-data retention

The new candidate adds 320 cells to the existing 1,152-cell birth graph.
Ninety-six graded predictors receive sparse visual-receptor projections. Each
predicts one specified auditory-receptor channel. Two opponent comparison cells
per channel receive observed and predicted information with equal initial
weights and delays, with opposite signs. Thirty-two graded neural consumers
receive predictor output. The existing graph is preserved, including its older,
unaligned mismatch candidate. That older population is not relabelled as a
validated error detector. This screen starts with receptor representations;
it does not claim the higher populations have learned a compositional hierarchy.

`components/learning/predictive_bridge.py` declares the wiring.
`neuron/extensions/experimental/predictive_receptor.py` in the sibling model
adds an opt-in local learning rule. The default inherited path is unchanged.
Selected contextual weights follow a bounded additive update using local
presynaptic amplitude traces and the previous completed signed error-receptor
trace. A positive basal rate is amplified by locally present error magnitude.
No stimulus identity, decoded loss, training flag, or recall flag reaches a
neuron. Native incoming learning elsewhere, delayed propagation, modulation,
terminal adaptation, and retrograde messages remain active. Zero task error
can produce zero update without freezing the learning rate.

This is a phenomenological extension, not a cellular reconstruction or a
reproduction of a published two-compartment model. Brea and colleagues' 2016
[Prospective Coding by Spiking Neurons](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005003)
distinguishes contextual dendritic input from somatic teaching input and leaves
learning active during recall. Its conductance equations, prospective temporal
rule, and modeled short-term-memory input differ from this experiment. The
accessible methods also make clear that its paired-associate memory population
is modeled with stimulus-dependent stochastic rate trajectories. We cannot
import that result as proof that our PAULA population generates such dynamics.
Direct access to the earlier Urbanczik/Senn full paper failed; no full-paper
replication claim is made.

There are two important implementation checks. Native graded release scales
information but does not scale the terminal modulation vector by release
amplitude. The new signed receptor therefore receives information from explicit
positive/negative neural error pathways, with zero membrane throughput. Also,
PAULA clears delivery buffers before returning from a tick and retains float32
operations in this part of its membrane dynamics. The passive recorder captures
arrivals before that clear. Its independent auditor reproduces dendritic
attenuation and integration in their actual arithmetic order, rather than
loosening an error tolerance.

The controlled numerical learning fixture passes with four heterogeneous birth
weight settings, seeds 11, 23, 44, 77. Swapping experienced cue/signal assignment
changes later cue responses. Transferring only the selected learned weights to
otherwise fresh neural state preserves assignment-specific effects throughout
ticks 32 through 95 while plasticity remains active. These are small mechanism
fixtures, not four independent replications of a working full brain.

The real-media screen completed 7,968 neural ticks across paired and swapped
histories, one presentation order, one existing graph seed. Each history has
two presentations of each physical pair, withdrawal intervals, initial and
trained visual/blank probes, and two selected-weight-reset probes. Both runs
first stopped at their explicit recording-size cap. A separate continuation
loaded their trusted-local checkpoints and completed only the missing probes.
The original progress files were not overwritten. All 32 resulting records
pass the independent equation/delivery audit with maximum residual zero.
The reset audit checks retained cellular, terminal, receptor and comparator
state and all incoming information weights for undeclared changes.

The full per-channel, per-tick learned-weight effects reach the added consumer.
A same-cue, crossed-assignment contrast further subtracting the other cue is
positive along the declared physical spectral-profile axis on 294 of 300 ticks,
zero on the first six, and never negative in this screen. Its window mean falls
from .1211 at ticks 32–63 to .0202 at ticks 288–299. This is a learned,
cue-dependent contribution under the chosen observer, not sound-waveform
reconstruction or semantic recall. Individual signed weight effects differ:
the swapped cue-0 effect points away from its assigned mean sound profile on
all 294 nonzero ticks. That sign alone is not a rejection criterion because
common depression relative to birth can obscure selective learning. Conversely,
the positive interaction alone does not remove recency, presentation-order or
shared-gain explanations. Do not select one of those views to declare success.

Records are `20260909_predictive_bridge_{paired,swapped}_order0_seed11` under
`.live/research`, indexed by `completion.json`. The independent audit and full
effect trajectories are in `20260909_predictive_bridge_audit_order0_seed11`.
Sources are `predictive_bridge_probe.py`, `predictive_bridge_resume.py`,
`predictive_bridge_audit.py`; mechanism tests are `test_predictive_receptor.py`.
Next evaluate acquisition versus expression under counterbalanced order and
reference-independent channel comparisons before extending this mechanism into
recurrent generative populations. Silence currently supplies negative evidence
and may alter memory during recall. Distinguishing absent sensation from a
deliberately internally generated state needs a neural mechanism, not a host
switch that disables learning. No body/live/V1–V4 simulations were started.

At the user's request, 4,212 older raw NPZ traces from 18 completed evidence
population replications were permanently removed, freeing 7,353,044,714 bytes,
or 6.848 GiB. The full seed-11 runs remain, as do each pruned run's config,
manifest, summaries, state snapshots, first/last training traces and synchronous
clean probes. Original media, source code, audit outputs, recent real-media
courses and checkpoints were not deleted. Exact inventory, retained-file hashes
and the deletion journal are in `docs/research-retention/`. Each affected run
has `RAW_RETENTION.md`; original summaries keep their historical references.
Full raw re-audits of the pruned replications now require regeneration and
verification, not an assumption that the files are still present. The disk had
about 7.8 GiB free after cleanup and the bounded probe continuation.

## Latest continuation: same-input regulatory tests, 9 September

The complete simultaneous audiovisual factorial now exists at the learned
sixteen-presentation checkpoint for both pairings and both presentation orders.
Each history includes all nine combinations of two visual clips or silence and
two sound clips or silence. Twenty-four new 300-tick branches complement twelve
reused visual/blank branches. Four repeated visual controls match every earlier
recorded field. Acquisition was not replayed, and adaptation remained active.
Unlike earlier delayed-context tests, the probe timing matches simultaneous
training. These are histories of graph seed 11, not independent graph seeds.

The audit compares the same physical pair under the history that contained it
and the history that did not. It also subtracts each history's unimodal and
blank responses. This removes additive contributions, not nonlinear gain.
After balancing all four physical pairs, the raw and adjusted contrasts are
algebraically identical. They must not count as two confirmations.

There is a small positive balanced unfamiliar-minus-familiar mismatch response:
143 positive ticks, 96 negative and 61 zero. Its first 32-tick window is negative
and the nine later windows positive. Both presentation orders have positive
post-onset means. This is a history-dependent interaction worth investigating,
not a validated novelty detector. Inspecting individual pairs exposes a failure.

| Physical pair | Mismatch spike contrast after index 31, order 0 / order 1 | Auditory plasticity multiplier contrast, order 0 / order 1 |
| --- | ---: | ---: |
| Visual 0, sound 0 | +0.000525 / +0.000758 | +0.408 / +0.867 |
| Visual 0, sound 1 | -0.004664 / -0.006472 | -7.171 / -9.892 |
| Visual 1, sound 0 | +0.002740 / +0.003323 | +7.923 / +7.309 |
| Visual 1, sound 1 | +0.005830 / +0.005830 | +9.572 / +8.629 |

Mismatch values are differences in spikes per cell per tick, not accuracies.
Rate values are differences in the local learning multiplier, not eta itself.
For visual 0/sound 1, the order-balanced mismatch contrast is negative in every
declared time window. Its auditory rate contrast is negative on 288 of 300
ticks. Visual 1 with the same sound has the opposite contrast: its mismatch
response is positive in every window and its rate contrast positive on 282
ticks. Both physical inputs therefore tend to elicit the greater response in
the reversed-pairing history, although only one is unfamiliar there. A positive
pooled unfamiliar-minus-familiar score hides this history-specific bias.

### Causal localization on the acquired cross-sensory pathway

Eight further probes reset only the 768 selected visual-to-auditory incoming
information weights to birth values. Both sound-1 pairs are tested in all four
histories. Four repeated intact audiovisual controls match every field.
All other state and ongoing adaptation remain. This is a conditional pathway
intervention, not complete erasure of experience or a proposed brain mechanism.

The regulator's same-input asymmetry remains. For visual 0/sound 1, its balanced
post-onset spike contrast changes from -0.005568 to -0.004314. For visual 1/sound
1 it changes from +0.005830 to +0.005306. Every time window retains the respective
negative or positive sign after resetting the pathway. The reset affects the
responses but does not remove their opposing familiarity contrasts. The pathway's
contribution on visual 1/sound 1 even changes sign between presentation orders.
Do not interpret ratios of these contrasts as fractions of memory stored there.

For original pairing/order 0/visual 0/sound 1, the first intervention effects are
selected current at index 7, terminal information/modulation at 8, auditory
spikes at 11, upper spikes at 23, mismatch spikes at 24 and auditory plasticity
multiplier at 26. This establishes a coupled route of influence without proving
that the route carries a useful prediction or that each first difference is a
direct connection. Full per-cell differences are retained.

The independent audits reconstruct selected currents exactly, selected learning
within 1.11e-16 for the factorial and 5.55e-17 for the reset probes, and graded
sensory trajectories. They check physical assignments, matched birth/configuration,
initial intervention weights and selected/sensory endpoints.

### Architectural consequence

The existing mismatch bank has randomly sampled positive sensory inputs and
negative sensory-core inputs, labelled `observed_sensation` and
`predicted_suppression`. Those labels do not establish that the two sides encode
the same feature. The regional feedback component aligns the separate activity
regulator's outputs, not these prediction comparisons. This does not prove that
random recurrent wiring could never learn a useful comparison. It identifies a
missing demonstrated transformation in this implementation.

The next architecture must make explicit what quantity is predicted, which
neural population represents that prediction, and how it is compared with the
observed quantity. For example, a visual-driven population could learn to
predict auditory-core activity; an error circuit would then compare homologous
observed and predicted auditory channels, rather than unrelated sensory and
core samples. Correspondence may be learned or anatomically structured, but
must not come from stimulus labels or an external decoder. This requires a
learning and neural-use experiment, not merely an aligned subtraction demo.
Keep the full population, positive adaptation and return pathways. Treat
input-specific plasticity, competition, regulatory routing and neural use as
distinct interventions rather than change all four until an observer improves.

[Parras et al., 2017](https://www.nature.com/articles/s41467-017-02038-6)
separate neuronal mismatch from repetition suppression using same-tone controls
with matched presentation rates and contextual structure. Their approach supports
our requirement to distinguish history and adaptation effects from prediction
error. The current two-clip pairing factorial is not a reproduction of their
auditory oddball, many-standard or cascade experiments.

Sources are `simultaneous_regulation_probe.py`, `simultaneous_regulation_audit.py`,
`regulatory_weight_probe.py` and `regulatory_weight_audit.py`. The corresponding
`.live/research/20260909_simultaneous_regulation_*` and
`20260909_regulatory_weight_*` directories contain manifests, full-tick records,
independent audits and exact source hashes. In total this continuation executed
9,600 new probe ticks and 2,400 exact-control ticks. All eight simulation workers
and both auditors finished successfully. No model equation, final-agent wiring
or live/body suite changed. The focused regression has 56 passing tests.
Disk space is about 1.2 GiB; further raw recording needs a strict budget.

## Latest continuation: recruitment survives loss of input-specific growth, 9 September

Forty-eight new 300-tick probes completed from the existing sixteen-presentation
checkpoints. There was no acquisition replay. Four repeated intact controls
match every earlier recorded field exactly. The unique protocol is 15,600 ticks,
including those controls, not the total work across an interrupted logging run
and its repeated validation. All adaptation and native return pathways remained
active. No neuron equation, sensory transducer or wiring changed.

The new interventions cycle birth weights within each target's four selected
inputs, or replace acquired changes on those inputs by their target-specific
mean. The latter preserves birth differences and the target's total selected
weight at intervention onset. It does not preserve weighted currents, later
weights or feedback. Its name in the records is `mean_growth`.

| History | Intact auditory events, cues 0 / 1 | Mean-growth events, cues 0 / 1 | Mean-growth events after index 31, cues 0 / 1 |
| --- | ---: | ---: | ---: |
| Original pairing, order 0 | 45 / 154 | 44 / 133 | 16 / 81 |
| Original pairing, order 1 | 32 / 145 | 29 / 130 | 13 / 79 |
| Reversed pairing, order 0 | 21 / 74 | 27 / 69 | 20 / 36 |
| Reversed pairing, order 1 | 19 / 63 | 15 / 63 | 9 / 39 |

All eight mean-growth cue branches have late auditory activity. The first
auditory event is at index 15 or 16 and the last is between 274 and 299.
All corresponding blank branches have zero auditory spikes. Removing
input-specific acquired changes therefore does not abolish recruitment.
It does change the trajectories. For original pairing/order 0/cue 0,
selected current first differs at index 7, auditory soma and visual terminal
release at 8, auditory spikes at 84, upper soma at 90 and upper spikes at 93.
The upper population is affected by the change; this does not establish that
it reads the remembered sound or uses it correctly.

The independent audit validates all intervention weights, reconstructs selected
local currents exactly, selected learning within 1.11e-16 and graded receptor
activity. Full per-cell trajectories and first-difference arrays are retained.
Birth-placement subtraction does not repair the unstable learned-placement
contrast. Its original-pairing sign still changes with presentation order.
The input-specific-growth assignment contrast on the original sound spatial
reference has 87 positive ticks, 111 negative and 102 zero. Four declared
time windows are positive and six negative. No associative success follows
from these observer-dependent differences.

Both reversed-pairing mean-growth probes again have negative late contrast on
the original sound reference and positive late contrast on the trained sound
reference. This observer disagreement survives the removal of input-specific
growth. Choosing the favorable reference would still give a misleading result.

### What the learning process actually accumulated

`learning_growth_audit.py` adds a data-only, full-tick reconstruction of the
selected learning process. For the observed trajectory, it partitions acquired
weight deviation from birth by the physical trial during which each update
happened. Each term undergoes the subsequent equation's attenuation. The sum
reconstructs every selected weight over all 12,672 acquisition ticks per history,
with maximum error below 2.9e-15. These are conditional algebraic contributions,
not causal credit, independent counterfactual experiences or memory fractions.

Separately, it resolves the acquired weight vector into a constant across all
selected inputs, variation between receiving neurons, and variation among a
receiving neuron's inputs. The components are orthogonal in weight coordinates,
not independent functions of the coupled neural system.

| History | Constant component of squared weight change | Between-target component | Within-target component |
| --- | ---: | ---: | ---: |
| Original, order 0 | 33.14% | 53.26% | 13.60% |
| Original, order 1 | 32.98% | 52.49% | 14.53% |
| Reversed, order 0 | 26.39% | 50.18% | 23.43% |
| Reversed, order 1 | 26.04% | 51.31% | 22.65% |

The largest component varies between receiving neurons. A target-specific gain
can carry learned information. These results do not justify calling all gain
changes meaningless, nor do they prove useful memory. They do rule out treating
the mean-growth intervention as one uniform gain over the entire population.

The dominant conditional contribution follows auditory clip 1 when audiovisual
pairings are reversed. Its final term norm ranges from 1.38 to 1.74, compared
with 0.059 to 0.107 for auditory clip 0. Terms are signed and need not add in
norm; those norms must not be presented as memory percentages.

The full acquisition records expose two sources of unequal learning opportunity.
Auditory clip 0 yields 7,268 to 8,084 auditory spikes across its sixteen
presentations, compared with 72,597 to 80,279 for clip 1. The event-weighted
effective learning rate is also larger for clip 1, 0.000624 to 0.000632,
compared with 0.000178 to 0.000233 for clip 0. Here the weighting is each
selected port's actual `Lplus + Lminus`, not wall time or the average rate
across silent neurons. Thus the total local learning exposure differs by
roughly 18 to 50 times, despite equal physical presentation duration.

This is not merely the onset transient. In original pairing/order 0, exposure
after index 31 within each presentation sums to 209.58 for sound 1 and 4.46
for sound 0. Both pairings and both orders retain the disparity after onset.
The audit retains the per-tick exposure trajectory and each presentation's
contribution. Spikes, incoming eligibility traces and eta are in the checked
source records, so the rate-weighted and unweighted calculations can be
reconstructed without another simulation.

### Consequence for the next architectural experiment

There are at least two unresolved processes: whether local learning develops
useful input competition, and how the coupled network allocates opportunities
to learn. Increasing competition before addressing the second could favor the
already dominant sound. The current evidence identifies unequal activity and
modulatory gain, not a causal proof that the supervisor is the source of failure.
Nor should biological learning be assumed to require equal rates for every sound.

The next comparison should distinguish sensory strength from learned novelty
in the regulator's response, using identical physical inputs under different
experience histories. A declared intervention on the neural rate-regulation
path should retain weak positive adaptation. Match or report total local
exposure before attributing an effect to its timing. Test learned content by a
neural consumer rather than optimizing an external sound-template projection.
Keep the existing full population and feedback, and replicate a promising
mechanism across graph seeds before acceptance.

[Albesa-González and Clopath, 2024](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012110)
provide a relevant candidate: strong competition during synapse selection and
weaker competition during stabilization. Their examples use a conductance-based
neuron with 1,000 excitatory and 200 inhibitory Poisson inputs, including
equal-rate correlation controls. They explicitly leave recurrent assembly
formation untested. The paper supports a mechanism to investigate, not a
justification for importing its results into this recurrent PAULA preparation.
No filopodium/spine or nonlinear-competition extension was implemented here.

Records are `20260909_cached_weight_factors_{paired,swapped}_order{0,1}_seed11`,
the combined `20260909_cached_weight_factors_audit_seed11`, and
`20260909_learning_growth_{paired,swapped}_order{0,1}_seed11` under `.live/research`.
The factor runner is `cached_weight_factors.py`; recovery is
`cached_weight_factors_resume.py`; independent validation is
`cached_weight_factors_audit.py`. Fresh workers must configure the local neuron
logger before loading checkpoints. The first run omitted this and printed
spikes; its four workers were stopped and the logs compressed losslessly.
Recovery reused eight completed probe files per history and computed the four
missing ones. It also repeated the intact control. No completed probe was deleted.

All simulation workers and both analyses have finished. There are 52 passing
focused tests. Disk space is about 1.6 GiB, so further experiments must budget
recordings and reuse checked acquisition data. No live/body/V1-V4 suite was
started. These results concern one graph seed under four histories, not four
independent graph seeds. Functional recall, useful hierarchy, embodiment and
the full research objective remain open.

## Latest continuation: learned-weight interventions, 9 September

Four histories of the same 1,152-cell graph completed another 68,688 neural
ticks. Acquisition replay accounts for 50,688; sixty matched diagnostic probes
account for 18,000. The two pairings and both presentation orders are retained.
This is one graph seed, not independent-seed replication. No neural equation,
sensory gain or wiring changed. All adaptation and native return pathways remain
active during the probes.

Each acquisition replay matched every field of the earlier recording. It saved
executable checkpoints after four and sixteen presentations. Reloaded intact
probes also match the source trajectories. Future interventions can start from
these actual object graphs without repeating acquisition. No acquisition trace
was duplicated on disk.

The experiment changes only the 768 incoming visual-core to auditory-core
information weights. Conditions are intact, reset to birth values, and three
cyclic permutations among each target's four visual inputs. Every nonidentity
cyclic offset is tested. Cycles preserve each target's final weight multiset;
they do not preserve its weighted current or temporal input correlations.
Each condition receives silence, visual clip 0 and visual clip 1 in separate
300-tick branches from the same sixteen-presentation parent state.

| History | Intact auditory events, cues 0 / 1 | Reset events, cues 0 / 1 | Range across three cycles, cues 0 / 1 |
| --- | ---: | ---: | ---: |
| Original pairing, order 0 | 45 / 154 | 0 / 3 | 39 / 115 to 135 |
| Original pairing, order 1 | 32 / 145 | 0 / 3 | 30 to 32 / 122 to 126 |
| Reversed pairing, order 0 | 21 / 74 | 0 / 4 | 19 to 27 / 55 to 66 |
| Reversed pairing, order 1 | 19 / 63 | 0 / 0 | 14 to 16 / 51 to 71 |

Every silence branch has zero auditory spikes. Seven reset cue branches have
no auditory spikes after index 31. The eighth has one late event at index 134.
All twenty-four cycled cue branches still have late auditory activity, in some
cases through index 299. Thus changes on the selected pathway causally enable
most recruitment in the trained network, while their precise final arrangement
is not necessary merely to recruit auditory activity. This does not establish
that their arrangement is irrelevant to information processing.

The independent audit reconstructs every selected input potential exactly using
the pre-update weight and native float32 input arithmetic. Selected learning
updates agree within 1.11e-16. It also reconstructs graded receptor trajectories
and checks that no nonselected incoming weights change at intervention onset.
This is not a reconstruction of every internal current or hidden state.

For original pairing/order 0/cue 0, both reset and cycle-1 interventions first
change selected local potentials at index 7. Auditory soma and visual terminal
information differ at index 8, selected arrivals at 10, and visual soma at 11.
Auditory spikes first differ at 16 for reset and 20 for cycle-1. Visual spikes
first differ at 21 and 45 respectively. These are measured first differences,
not universal latency constants or a complete path-lesion proof. They show why
the unchanged visual spike raster does not imply unchanged visual influence:
plastic return paths can change release amplitudes earlier.

### Do not make total-preference reversal a universal memory criterion

The auditor separates intact-minus-reset cue effects from intact-minus-mean-cycle
effects, retaining full cell-by-tick arrays. A small learned contribution could
exist under a stronger pre-existing visual bias without reversing the total
preference. A test fixture now demonstrates that distinction explicitly.

The selected-adaptation assignment contrast on the original auditory spatial
reference is positive on 107 ticks, negative on 58 and zero on 135. Its onset
window is negative, and the nine subsequent declared windows are positive.
This is a pairing-dependent effect of the intervention in these histories, not
yet proof of an association that a neural consumer can use. History-dependent
gain can contribute. The placement contrast is less consistent: its whole-probe
sign changes across the two original-pairing presentation orders. It also
contains birth-weight placement, because birth-weight cycles were not tested.

An apparent repair depends on which observer is chosen. All six reversed-pairing
cycled probes have negative late spatial contrast against the original sound
reference, while the original-pairing cycles remain positive. But all six have
positive late contrast against sound responses measured at the trained
checkpoint. Selecting the original reference would therefore produce a
misleading success claim. The probe does not establish that the network has
recalled the wrong physical sound either. It establishes observer dependence
while the sound-response representation changes with experience.

### Architectural consequence and next experiment

There is now a causal foundation for learned recruitment. There is not yet a
validated content-carrying interface for a higher neural consumer. Increasing
population size or imposing a persistent oscillator would not resolve that
distinction. The next structural question is whether local learning develops
useful contrast between competing inputs, with a neural-use test that tolerates
representational change instead of demanding an exact old sound raster.

[Gütig et al., 2003](https://brainworks.biologie.uni-freiburg.de/2003/journal%20papers/guetig-jns-2003.pdf)
distinguish stability from correlation-sensitive competition. Their linear
weight-dependence case can stabilize homogeneous weights without useful
symmetry breaking in the studied input models. Our selected rule uses that
linear dependence. This motivates a nonlinear-competition comparison; their
result is not a theorem about this recurrent PAULA graph. Native terminal
adaptation and the sensory population statistics also remain possible causes.
No new plasticity extension was implemented on this evidence alone.

Use the saved states to first separate acquired weight placement from birth
placement and target-level gain, then test any candidate learning change on
the existing assignment/order controls and independent graph seeds. Add a
downstream neural task rather than treating a fitted or fixed observer as the
brain's own interpretation. Shared source files must remain unchanged while
old checkpoints are needed; new extensions can be separate opt-in classes.

Sources are `experiments/media_weight_identity.py` and its independent
`media_weight_identity_audit.py`. Records are under `.live/research/` as
`20260909_media_weight_identity_{paired,swapped}_order{0,1}_seed11` and
`20260909_media_weight_identity_audit_seed11`. Checkpoints are named
`checkpoint-{4,16}.neural-checkpoint` within each history. All four workers
exited successfully. The current focused regression contains 45 passing tests.
Available disk space is about 2.0 GiB; further full traces need a space budget.
The full embodied, hierarchical and consciousness objective remains open.

## Latest result, 9 September: longer acquisition recruits sound populations but does not establish recall

Four full-neural courses used the unchanged 1,152-cell graded real-media graph.
Each tested the original and reversed audiovisual pairing in both presentation
orders, with checkpoints after four and sixteen presentations per pairing.
These are four histories of graph seed 11, not four independent graph seeds.
Each course executed 20,424 ticks, 81,696 in total. Every original acquisition
prefix, four-presentation trained-state record and ordinary graded visual probe
replayed exactly. Probe branches left their acquisition parent unchanged.
All learning rates remained positive. No gain, wiring or neuron-rule change
was made for the duration comparison.

The auditory population was previously silent in seven of eight ordinary
graded-cue probes; the eighth had two onset spikes. After sixteen presentations,
all eight cues produced auditory spikes beyond the first 32 ticks. The table
reports event counts only to locate the full trajectories, not memory accuracy.

| Pairing and order | Cue 0 events, total / after tick 31 | Cue 1 events, total / after tick 31 |
| --- | ---: | ---: |
| Original, order 0 | 45 / 17 | 154 / 100 |
| Original, order 1 | 32 / 14 | 145 / 90 |
| Reversed, order 0 | 21 / 13 | 74 / 41 |
| Reversed, order 1 | 19 / 12 | 63 / 36 |

The first auditory events occur at probe indices 15 or 16. Last events under
continued visual input occur at indices 251 through 299 of the 300-tick probes.
Matched no-input branches have zero auditory spikes throughout 364 ticks.
Thus the new activity is cue-driven rather than ongoing spontaneous firing.
This is an improvement in recruitment after longer experience. A selective
intervention is still needed to attribute it to particular learned parameters
rather than other changes accumulated during the longer course.

It is not a sustained autonomous auditory regime. In separate branches, the
same visual input ends after index 63. Five of eight branches have no subsequent
auditory spike; the other three have one, two and one events, all by index 70.
The remaining 293 ticks contain none. Their initial 64-tick cell traces match
the corresponding continued-cue branches exactly, including modulation fields.
This concerns spike output. It does not establish disappearance of synaptic,
subthreshold or other hidden memory.

The content test still fails. Projecting the two cue responses onto the spatial
difference between actual sound responses gives the same positive preference
under both assignments. This holds with original sound references and references
measured at the current checkpoint. For the original spatial reference, the
whole-probe contrasts are .01540 and .01329 for original pairing, versus .00384
and .00231 for reversed pairing. Late contrasts also remain positive. The second
visual clip produces more auditory activity under every history. Increased
auditory access therefore cannot be called retrieval of the associated sound.
This test does not exclude every possible temporal or distributed neural code.

### Contextual processing is measurable, but its role remains unidentified

At both checkpoints, every history includes the complete physical 3 by 3
factorial. The first 64 ticks contain visual clip 0, clip 1 or silence; the next
300 contain sound 0, sound 1 or silence, with visual input withdrawn. All nine
branches start from the same parent state. This delayed probe is a new temporal
condition; acquisition used simultaneous audiovisual input.

For each cell and tick, the audit retains
`I(v,a) = R(v,a) - R(v,None) - R(None,a) + R(None,None)`.
Additive sensory responses cancel. The assignment and order contrasts of these
interactions remain nonzero in auditory, upper and putative mismatch populations.
At sixteen presentations, the auditory assignment contrast summed across cells
is positive on 143 post-sound ticks and negative on 137. The upper contrast is
positive on 141 and negative on 134. There is no stable sign that justifies a
simple learned-suppression claim. Nor would a positive mean alone establish one.
Test fixtures demonstrate that history-dependent scalar gain can produce a
nonzero assignment interaction without a predictive circuit.

Garner and Keller's [2022 mouse study](https://www.nature.com/articles/s41593-021-00974-7)
tested experience-dependent sensory prediction in a behaviorally relevant task.
It motivates context-dependent processing but does not validate passive exposure
in this preparation as a reproduction of its mechanism. A neural consumer and
the organism's use of the association remain separate experimental requirements.

### Independent reconstruction and practical findings

`media_course_audit.py` reconstructs the selected learning and graded sensory
trajectories independently. Maximum selected-update discrepancy is 1.11e-16;
the conditional affine learning decomposition differs by at most 3.72e-15.
Median initial-weight coefficient A after sixteen presentations is .934 for
original pairing and .972 for reversed pairing, versus .972 and .987 near four.
Some ports have A near .012 while others remain near one. These are conditional
learning coefficients, not percentages of retained semantic memory or proof of
convergence. Effective learning remains highly heterogeneous.

Every training weight-health row is finite, with zero recorded input sign
changes, zero weight underflows and zero negative output terminals. This rules
out those observed numerical failures during these courses, not all instability.
The independent recruitment supplement retains full cue-minus-blank and
withdrawal-minus-blank output trajectories, not just the table above.

Records are `.live/research/20260909_media_course_{paired,swapped}_order{0,1}_seed11`.
The full audit is `20260909_media_course_audit_seed11`; its recruitment supplement
is `20260909_media_course_recruitment_seed11`. Sources and raw recordings remain
unchanged. Thirty-nine focused regression tests pass in 8.26 seconds.

`core/runtime_checkpoint.py` also now saves the executable neural state. A
full-graph active-media test preserved 1,344 in-flight signals and reproduced
every recorded field over 128 subsequent ticks exactly. Loading its 0.805 MB
checkpoint took 0.128 seconds in one measurement. This removes acquisition replay
from future branches once their checkpoints have been saved. The completed
four/sixteen courses have JSON inspection states, not these executable caches;
their first cache construction still requires one exact acquisition replay.
See `core/RUNTIME_CHECKPOINTS.md` for trust, version and body-state limitations.

The next causal test should localize the new recruitment before another scale-up.
Save exact four/sixteen-presentation checkpoints during a verified replay. Compare
intact probes with declared selected-weight reset and strength-preserving
within-target weight permutations, retaining the same fast state and positive
adaptation. Measure early current changes and subsequent return-path divergence.
If recruitment survives scrambling which visual source owns each learned weight,
it is evidence for nonspecific gain on that route, not its associative content.
If it does not, test assignment reversal and neural use before accepting memory.
No new upper population is justified merely by a larger response downstream.

The broader goal remains unachieved. No V1-V4 body simulation, live server,
visualization or shared base-neuron source was changed in this work.

## Architectural reference

The author's eight-page *Artificial Life notes and sketches (1).pdf* was read
visually in full. Page 1 describes small neural supervisors observing and
regulating modules; page 2 describes a low-level general connector exposing an
internal environment for deeper systems. Page 8 sketches recurrent composition
of lower population dynamics into further population dynamics. These are
architectural hypotheses, not evidence that any wiring with these labels works.

The implemented first preparation has 1,152 PAULA cells. It includes two
96-cell sensory sheets, two 192-cell local cores and their 48-cell inhibitory
populations, a 64-cell connector, a 256-cell upper core with 64 inhibitory cells,
64 putative mismatch cells and 32 activity-regulation cells. Connections are
sparse, recurrent and independent of the stimulus patterns. Scaling preserves
fan-in and the delay distribution. This is not 144 copies of an eight-cell loop.

All basal post- and presynaptic learning rates are positive. The sole subclass
is the already-existing experimental local plasticity-rate receptor. The upper
core and regulatory populations receive neural packets, never a host-decoded
state, stimulus label, prediction error or training-phase instruction. Their
names state proposed roles; the results do not validate all those roles.

## Binary-pattern results

One 288-cell pilot and three matched 1,152-cell conditions were completed.
The larger runs contain 2,496 ticks each. Cutting the upward connector silences
the upper population and changes lower activity through lost descending
feedback. Cutting recurrent/cross-channel pathways also changes the response.
This establishes causal coupling in the preparation, not useful hierarchy.

The missing tactile population remains silent under both post-experience
partial visual cues. Incoming-weight-only transplants into fresh fast states,
with adaptation still enabled, do not recover tactile completion either.
Those interventions test an information-weight contribution, not every possible
form of stored state. The silent intervals show no maintained population
activity. No learned attractor or autonomous supervisor is demonstrated.

## Real audiovisual preparation

The user requested recognizable sensory material instead of arbitrary feature
masks. The next runs therefore use *Labrador barking on command*, Sadie Campbell
/ Nonlinearmind, 2011, under CC BY-SA 3.0. The original video and soundtrack are
retained. Browser presentation is resized/transcoded with attribution. The clip
contains human commands as well as barking, so it is not a clean bark-only task.

The input path uses 12x8 darkness measurements and 32 logarithmic audio-frequency
bands at three fixed dB sensitivities. Audio analysis uses a trailing 32ms
window. Visual input has three ticks of declared latency. Inputs are staggered
across receptors every four neural ticks. This is crude physical transduction,
not a biologically validated retina/cochlea and not a pretrained recognition
system. No word, class label or semantic embedding reaches PAULA.

The common source interval is 476 neural ticks at 60 ticks per source second,
about 7.93 seconds; the source video is longer because its soundtrack ends
earlier. Each run has 4,000 ticks including silent video before experience,
audio-only reference, three audiovisual exposures, 192 ticks without sensory
drive, silent video, a still frame and a final audio-only reference. The shifted
condition changes the soundtrack's temporal alignment during exposure while
preserving each auditory receptor's input-amplitude multiset. It introduces a
circular wrap seam and is not a substitute for unrelated sounds or new clips.

| Observation | Aligned exposure | Shifted-sound exposure |
| --- | ---: | ---: |
| Auditory-core spikes during silent video, before exposure | 3,617 | 3,617 |
| Auditory-core spikes during silent video, after exposure | 4,137 | 4,249 |
| Auditory-core spikes during silent still image | 4,044 | 4,324 |
| Changed incoming information weights | 18,674 | 18,674 |

The increased firing does not demonstrate bark recall. It also increases under
the mistimed-sound control, and a substantial response exists before pairing.
Centered, neuron-specific temporal comparisons with the audio-only reference
do not establish faithful auditory reinstatement either. These are preliminary
measurements, not a universal requirement for exact phase-locked replay. A
learned regime could vary in latency or phase; further tests must accommodate
that without accepting generic excitation as memory.

The initial neural states and the first two complete trial recordings are
bit-identical between the aligned and shifted runs. The independent audit also
checks raw counts against summaries and the matched external auditory dose.
After stimulus withdrawal, 2,162 spikes occur while the aligned network drains
its recent activity; zero occur in the last 96 ticks of that silent interval.
Counting the whole withdrawal window alone would falsely suggest persistence.

## What the recordings expose

During the first paired movie exposure, visual receptors produce 9,224 spikes
and auditory receptors produce 386, a ratio of 23.90. Equal population sizes did
not establish comparable influence. This is a diagnosis of this encoder and
wiring combination, not a biological requirement for equal sensory rates.
The strong visual pathway can excite the nominal auditory core before any
pairing, through the pre-existing recurrent graph. Its presence is therefore
not evidence that the dog evokes a learned bark representation.

The next mechanistic step is to measure sensory-to-core and cross-core transfer
curves, activity saturation, inhibition and the local plasticity actually
engaged by transient sound. Test contrast adaptation and receptor sensitivity
as explicit sensory mechanisms before increasing associative gain. Then test
whether acquired cross-channel connections add selective completion beyond
the initial graph. Retain the raw-input baseline and controls when changing the
transducers. A quieter network or a larger learning rate is not by itself a fix.

Temporal attention, learned anomaly detection, recursively learned upper
organization, action–perception contingencies and robust self-regulation remain
unfinished. These runs neither verify nor refute the broader PAULA/ALERM or
consciousness hypotheses. One seed and one audiovisual recording are insufficient
for general claims. Three exposures are a bounded pilot, not a claim that this
is enough experience for every candidate mechanism.

## Evidence locations and cost

Raw directories are under `active-inference/.live/research/`:

- `20260908_population_pilot_288`
- `20260908_population_1152_intact`
- `20260908_population_1152_ascending_cut`
- `20260908_population_1152_recurrence_cut`
- `20260908_av_population_aligned`
- `20260908_av_population_shifted`

The aligned audiovisual run took 234.41 seconds wall time, 230.28 user CPU
seconds and about 1.12GB maximum resident memory on this machine. At most two
bounded simulation workers ran together. V1–V4 embodied suites were not started.

The data-only audit is `population_recording_audit.py`; it imports no PAULA
code. Cellular observables are recorded at every tick, with every incoming
information weight in the main runs. These are not complete per-tick restart
snapshots. Queues and the full plastic/adapt/retrograde state require a rerun or
the retained final snapshot. The viewer uses float32 presentation values while
raw records retain float64 values. Information-weight XOR encoding is bit-exact.

## Related literature, not validation of this implementation

[Zenke, Agnes & Gerstner, 2015](https://www.nature.com/articles/ncomms7922)
studied learned assemblies using interacting plasticity and stabilization
mechanisms in a different spiking model. It motivates testing their interaction,
not assuming a recurrent graph plus plastic weights will form usable memories.
[Keller, Bonhoeffer & Hübener, 2012](https://pubmed.ncbi.nlm.nih.gov/22681686/)
motivates manipulating expected sensory consequences in an actual sensorimotor
loop. The present passive movie exposure does not implement that loop.

## Follow-up: causal composition and architecture, 8 September

No frontend changes were made for this follow-up. The following experiments
test the 1,152-cell preparation directly. All basal plasticity remains positive;
the shared PAULA model and the embodied agents were not changed.

### Correction to the first control

Shifting a bark within a movie that continues to show the dog does not remove
the dog-sound association. It tests temporal alignment, not content association.
The similar response under shifted training therefore cannot rule out the
broader association the user requested. Exact waveform or zero-lag neural
replay is also not a necessary criterion for that association. The previous
measurements remain recorded, but must not be interpreted as those stronger
tests.

### A traced interaction, not a inferred label

`population_causal_probe.py` reruns the original silent movie. Its read-only
observer reconstructs arriving currents, membrane integration, firing and the
native plasticity direction for every auditory-core cell. Both the first
192-tick and full 476-tick instrumented baselines are bit-identical to the
original cellular recording. Double-precision current accounting has residual
rounding error below 1.4e-7, because native buffered arithmetic can be float32.
Firing reconstruction preserves native operation order and dtype.

The first auditory spike is neuron 495 at tick 23. Its threshold is 0.65.
Accumulated visual cross-channel input contributes 0.49083667 and descending
input contributes 0.17316033 to the membrane before reset. There is no auditory
sensory contribution. Over all 476 ticks, the intact auditory core emits 3,617
spikes. Removing either of those two incoming pathways eliminates every one of
them. Ports, weights and neuronal parameters are unchanged in the cuts.

This interaction also changes learning. Of 112,067 active cross-channel input
events in the intact run, 105,549 take the positive timing-credit branch. With
the descending path cut, all 111,450 remaining cross-channel events take the
negative branch. This is a network-induced change in local plasticity, not an
externally supplied teaching signal. Positive timing credit does not mean a
positive numerical weight increment for inhibitory synapses.

The source explains one limiting regime: `t_ref >= 2*c`. If a cell continues
spiking with intervals no longer than `2*c`, its active inputs always take the
positive timing branch after the first spike. Shrinking `t_ref` cannot restore
sign discrimination within that regime. This is a conditional statement about
the native rule, not a claim that PAULA cannot learn or that firing should cease.

### Different content and learned weights

`multimodal_pairing_probe.py` uses the dog recording and Ali T's CC BY 2.0
recording of a cat chirping. It tests paired versus swapped audio assignments,
with four presentations of each visual clip. Both channels are truncated to
300 ticks. Each condition executes 5,568 ticks, including eight independent
probes. The graph does not receive clip identities. Training inputs have exactly
matched per-receptor amplitude multisets across assignments.

Probes start in fresh activity states. They receive either original or learned
incoming information weights; queues, membrane, modulators, firing history and
learned outgoing terminal amplitudes are not transplanted. Adaptation continues
during each probe. This tests information-weight storage, not every possible
form of memory. Both conditions' initial probes are bit-identical.

The independent `multimodal_pairing_audit.py` reads only saved arrays. Rate and
activity-covariance readouts do not require exact temporal replay. Additional
population-centered versions were added after the first audit to exclude
uniform activity level. None of the four auditory readouts shows changes
following both assignments. For example, centered rate contrast changes by
-0.00660 under paired experience and -0.00889 under swapped experience. The
desired opposite directional changes are not established. These descriptive
one-seed results do not exclude other temporal codes or longer learning.

### Regulation exposes a second architectural problem

`population_regulation_probe.py` tests a two-factor intervention on existing
PAULA modulation, not a new neuron rule. The detector's threshold is either
the original `r=1.1, b=1.35` or `r=.2, b=.45`. Its receiving cells' M1 threshold
gain is either .6 or 12. The four conditions share all wiring and learning rates.

| Condition | Upper rate under silent movie | Upper cells near ceiling | Auditory spikes, silent movie / sound segment |
| --- | ---: | ---: | ---: |
| Original | .3345 | 100% | 1262 / 1813 |
| Sensitive detector only | .3311 | 98.8% | 0 / 499 |
| Strong receptor only | .3331 | 100% | 863 / 1141 |
| Both | .1052 | 0% | 0 / 0 |

Rates use ticks 32-191; finite windows can exceed the asymptotic 1/3 ceiling
slightly. Near-ceiling means rate at least .30. Sound receptors emit 26 spikes
during the tested 192-tick sound segment in every condition. The combined
intervention desaturates the upper population but suppresses that weak auditory
response too. It is not accepted as a fix. This is evidence for testing regional
control, not a proof that every globally regulated parameter setting must fail.

### Architectural decision after the user's request to rethink

Preserve the current graph as an experimental baseline. Do not promote it as
a working hierarchical brain or continue broad gain searches. Its sensory
cores, direct cross-modal drive, upper feedback and shared regulator do not yet
distinguish sensed input from internally generated activation. The upper core
also has no regulator observing its own activity specifically.

The next architecture hypothesis is a recurring local organization: structured
sensory afferents, an excitatory/inhibitory recurrent population, local
regulation, and separately addressable ascending and descending projections.
Separate pathways mean distinct synapses or interneuron routes, not a Python
controller deciding which state is true. A compartmental neuron extension is
an experimental alternative if ordinary PAULA circuitry cannot provide the
needed local distinction. No new neuron subclass is justified merely by naming
its output a prediction error.

Cross-modal learning should associate distributed population states through
reciprocal learned pathways. Descending activity can reinstate a missing
modality, but must not count its own reconstruction as independent confirming
evidence. Both unexpected presence and expected-but-absent input need neural
comparison routes. Their acquisition is an open mechanism, not guaranteed by
this wiring description. Higher levels should receive structured population
outputs, retaining distinctions and temporal relationships, rather than one
common activation pool. Each level has its own regulator, with slower context
modulating faster local loops and overlapping timescales rather than a shared
sample/hold clock.

The first controlled architecture experiment should compare the existing
pooled regulator with regional regulators at matched cell count, fan-in,
receptor gains and adaptation. Keep strong visual drive present while changing
weak sound. Test whether both remain represented, whether a perturbation in
one region is contained without silencing another, and then repeat the swapped
pairing and fresh-state memory probes. Regional control is not sufficient proof
of associative learning; passing these are separate questions. Action-dependent
sampling, metabolic regulation and another hierarchy level follow as distinct
causal extensions. No redesigned architecture has been implemented yet.

This tests ALERM's composition claim rather than assuming local stabilization
automatically survives interconnection. Related work provides hypotheses, not
validation: [Mikulasch, Rudelt & Priesemann, 2021](https://doi.org/10.1073/pnas.2021925118)
studies local dendritic balance and representation collapse in a different
model. [Murray et al., 2014](https://pubmed.ncbi.nlm.nih.gov/25383900/)
measures different intrinsic timescales across cortical areas. Neither supplies
a demonstrated learning rule for this proposed PAULA architecture.

### Reproducible evidence

All paths below are under `.live/research/`:

- `20260908_av_causal_currents_verified` and `20260908_av_causal_full_clip`:
  per-cell, per-family current and learning-credit arrays, configurations,
  exact-replay checks and source hashes.
- `20260908_content_pairing_seed11` and `20260908_content_swapped_seed11`:
  original sensory measurements, every tick's cellular observables, episode
  weight snapshots, training final state and fresh-state probes.
- `20260908_content_pairing_audit.json` and
  `20260908_content_pairing_centered_audit.json`: independent readouts and
  structural checks, with the original audit retained.
- `20260908_feedback_authority`: four declared parameter interventions and
  their per-tick records.

These are isolated research preparations, not embodied V1-V4 acceptance runs.
All completed; no new live simulation or matrix suite was started.

## Regional composition and unstable plasticity, next experiment

The regional component proposed above is now implemented in
`components/learning/regional_regulation.py`. The same 32 PAULA regulator cells
observe visual, auditory and upper populations separately. Their modulation
raises local excitability thresholds. The control scrambles outgoing regional
alignment with degree-preserving edge swaps. Individual source/target connection
counts, detector inputs, gains, delays and cell parameters match exactly. All
basal learning rates remain positive. There is no runtime host regulator.

Each seed receives independent fresh-network vision-only, audio-only and
audiovisual probes of the full 476-tick dog clip, then 96 withdrawal ticks.
The independent recording auditor verifies structure and recomputes observables.

| Seed | Conditional auditory gain, regional | Shuffled |
| --- | ---: | ---: |
| 11 | 1.332 | .425 |
| 23 | 1.332 | .511 |
| 44 | 1.338 | .465 |
| 77 | 1.333 | .503 |

Gain is auditory spike rate under both inputs minus rate under vision alone,
divided by audio-only rate. Silent vision produces zero auditory-core spikes
in these probes. Gain above one means amplification, not perfect fidelity.
Projection of the conditional auditory `F_avg` trace onto the audio-only trace
is 1.022-1.042 regionally and .410-.456 when shuffled. This supports local
regulatory alignment under unequal sensory drive, not recognition, arbitrary
scaling, perturbation recovery or lifelong learning stability.

### Association does not replicate

Regional paired/swapped learning executes 5,568 ticks per condition, including
four presentations per clip and independent fresh-state probes. Both seeds'
structural checks pass. Seed 11's auditory changes follow both assignments on
all four readouts. Seed 23's changes do not on any of them.

| Seed | Centered auditory rate contrast change, paired | Swapped |
| --- | ---: | ---: |
| 11 | +.004491 | -.001238 |
| 23 | -.003388 | +.000377 |

These small effects do not establish robust associative recall. The network
receives raw pixels and frequency-band activity, not dog/cat labels. Category
learning, held-out generalization, semantic recognition and retention remain
untested.

`multimodal_weight_path_probe.py` partitions learned incoming information weights
into visual/upper projections to auditory cells versus all remaining inputs.
Selective transplants leave every edge connected and every adaptation rate
positive. Fully learned replay is bit-identical in cellular fields and final
weights. At seed 11, restoring the selected inputs to initial weights removes
every auditory-core spike for both silent clips under both assignments. Keeping
only their learned changes produces 131/241 spikes for paired and 129/223 for
swapped experience, without assignment-specific signs on rate contrast. Fully
learned networks produce 23/68 and 21/21 spikes respectively. Learned inputs
therefore contribute causally, but their expression depends strongly on changes
elsewhere. A required connection family is not a sufficient associative memory.
The separate data-only auditor reports the full nonlinear factorial interaction.

### Adaptation changes the intended inhibitory organization

| Seed / assignment | Initially negative weights now positive | Information weights at a numerical bound |
| --- | ---: | ---: |
| 11 / paired | 93 | 29 |
| 11 / swapped | 86 | 30 |
| 23 / paired | 17 | 13 |
| 23 / swapped | 21 | 13 |

At seed 11 all sign changes occur in the inhibition family. Its absolute weight
change is 16,189 in the paired run, versus 82.7 for cross-channel weights and
39.0 for descending weights. The runtime network no longer has the inhibitory
organization described by its initial graph. There is no chloride-gradient or
receptor-switch mechanism in this preparation to interpret those reversals
biologically.

For inhibitory weight `w = -q`, native positive timing credit gives magnitude
growth `q_next = q * [1 + eta * (E - .02)]`. Since `E >= input + q`, growth can
accelerate. The soft bound is applied only to positive numerical increments.
On negative timing credit at a negative weight, its factor is `1 + q/10`,
amplifying the rebound. A sign reversal occurs when
`eta * [E * (1 + q/10) + .02] > 1`, before clipping. Weak basal eta does not
establish safety when neural modulation amplifies eta.

`synaptic_credit_trace.py` reconstructs the arithmetic inside the rate-gated
native tick and checks exact replay. It records effective eta, not the basal
value restored after the tick. This observer changes no neural state.

The completed trace of neuron 246, input port 21, matches every recorded cellular
field and every episode-end information weight through all 1,980 ticks. At
tick 1872 the weight first clamps to -100. At tick 1887, negative timing credit,
effective eta .0023763022 and error magnitude 101.00157 produce raw increment
+24.00103. The soft factor is 11, yielding +164.01604 before clipping to +100.
This is an arithmetic sign reversal, not an inferred biological transition.
Maximum reconstruction error is 1.42e-14. The full observer and intervention
test set passes 30 tests. All isolated workers completed; the static assessment
server is the only related server left running.

Regional stability under initial coupling is insufficient: adaptation must also
keep the coupling within a regime in which the regions can do their jobs. The
next candidate is locally bounded, sign-preserving magnitude plasticity, tested
against the retained native rule. Bounds alone will not establish learning.
No decoded target, label or host policy should repair the network's dynamics.

Compartment-specific learning signals remain another candidate.
[Mikulasch, Rudelt & Priesemann](https://doi.org/10.1073/pnas.2021925118)
derive distinct requirements for balancing somatic activity and dendritic
representation learning. Their equations do not automatically follow from PAULA.
Compare an explicit compartmental extension with ordinary neural circuitry.

Evidence is under `.live/research/`: `20260908_regional_aligned`,
`20260908_regional_shuffled`, `20260908_regional_audit_seed*.json`,
`20260908_regional_learning_{paired,swapped}_seed{11,23}` with their audits,
`20260908_regional_weight_paths_{paired,swapped}_seed11` with its audit, and
`20260908_inhibitory_credit_trace_seed11`. No embodied/demo suite was restarted.

## Bounded local learning, followed by a storage/retrieval distinction

The next goal turn is progress. It implements an opt-in neuron extension and
tests it on the actual audiovisual protocol, rather than accepting a stable
single-cell example as a brain result. The full mammal-level embodied goal,
including the consciousness question, remains incomplete.

### Mechanism and verification

`neuron/extensions/experimental/bounded_plasticity.py` in neuron-model adds
`BoundedPlasticityNeuron`, inheriting the existing local rate receptor. Its
disabled path is exact. For signed weight magnitude `q`, positive timing credit
uses `dq/ds = q * [e * (1-q/C) - .02]`, and negative credit uses
`dq/ds = -q * (e+.02)`. An analytic held-error update over effective eta avoids
Euler sign overshoot. The default magnitude scale is `C=10`, matching the
native positive soft-bound scale. This is a proposed phenomenological law, not
a reconstruction of inhibitory receptor biology or a claim that all plasticity
should follow one equation.

The neuron retains positive basal adaptation, local neural rate amplification,
native timing credit, delayed currents and retrograde signaling. Tests verify
pre-update causality. The inherited tick computes a provisional native weight;
the subclass replaces only the active incoming information weight before
returning to the network. Native DEBUG weight messages are provisional and
separate messages expose the final bounded values. No shared base-neuron code,
body, V1-V4 agent or external controller changed.

The scalar flow and enabled neuron pass seven mechanism tests. The previous
rate receptor's six tests also pass. The current native experimental runner,
including its new read-only health observer, matches all 1,152 cells for the
first 96 ticks of a previous recording exactly. The active-inference focused
test set passes 33 tests. The old pairing runner is retained at
`.live/research/source_snapshots/multimodal_pairing_probe.regional_v1.py`.

### Real-input results

Each bounded condition executes 5,568 ticks, including four presentations of
each audiovisual pairing and eight independent probes. Four runs cover two
graph seeds and both sound assignments. The independent audit verifies matched
sensory marginals and bit-identical initial probes. It checks every episode's
weight endpoints against the per-tick health observer. Interior weight arrays
are not all recorded; the observer reports their sign/range and actual changes.

| Seed / assignment | Final native inhibitory sign flips, previous run | Bounded sign flips, any recorded tick | Nonzero bounded weight updates during training | Final weights within 5% of magnitude cap |
| --- | ---: | ---: | ---: | ---: |
| 11 / paired | 93 | 0 | 7,092,731 | 24 / 22,144 |
| 11 / swapped | 86 | 0 | 7,175,253 | 14 / 22,144 |
| 23 / paired | 17 | 0 | 7,171,872 | 31 / 22,144 |
| 23 / swapped | 21 | 0 | 7,276,757 | 6 / 22,144 |

No initially nonzero incoming weight underflowed to zero in training. Maximum
absolute weights were 9.9224, 9.8585, 9.9690 and 9.8130 respectively. Spiking
continues and the tested upper-population exposure windows are not at their
firing ceiling. This addresses the observed local numerical failure without
freezing adaptation. It does not prove long-horizon stability, robustness across
four independent seeds or stability of every state variable. Terminal and
modulator plasticity retain their previous rules.

None of the four auditory recall readouts satisfies the stronger requirement
that both sound assignments move in their respective directions, in either
seed. For centered rate contrast, paired/swapped changes are +.02484/+.00839
at seed 11 and +.00745/+.00961 at seed 23. This is a failure of reliable selective
retrieval, not proof that the weights contain no learned information. The
opposite-direction requirement is stronger than any detectable association.

### An exploratory test in weight space

The new independent analysis projects changes in the 768 direct visual-to-
auditory weights onto products of separately evoked, population-centered visual
and auditory rate contrasts. It was introduced after seeing the recall results
and is explicitly exploratory. It is not a fitted classifier in the brain.

| Condition | Paired weight coefficient | Swapped weight coefficient |
| --- | ---: | ---: |
| Native seed 11 | +.3986 | -.5115 |
| Bounded seed 11 | +.5091 | -.5794 |
| Native seed 23 | -.4238 | -1.0803 |
| Bounded seed 23 | -.3967 | -1.0917 |

The relative effect of pairing is positive in both seeds, but common changes
dominate the absolute coefficient at seed 23. Input order can also contribute
to assignment differences. These measurements justify a causal test of stored
connection differences; they do not establish semantic learning or working
recall. In particular, calling every recall failure a failure to store any
relationship would discard this distinction.

`multimodal_weight_path_probe --donor` exchanges learned weights between matched
opposite-assignment networks. It swaps either the 768 direct visual-to-auditory
weights or their complement, preserving all neurons, connections, local rules
and fresh initial activity. Full learned replay must match the source records.
Complementary hybrids must also be bit-identical, since they instantiate the
same stored weights by opposite descriptions. The independent auditor verifies
those identities and that only the declared weights differ.

At seed 11, exchanging only these direct weights changes the paired host's
centered auditory rate contrast from .02484 to .00868, and the swapped host's
from .00839 to .01235. Thus these learned connection differences affect the
response in the predicted relative directions for that readout. Covariance
readouts do not give an equally simple result. This remains partial evidence,
not a replicated, generally functioning associative memory.

The completed seed-23 exchange does not replicate that centered-rate result.
The paired host changes from .007452 to .010896 and the swapped host from
.009610 to .010185 when each receives the other's direct weights: both increase.
Uncentered mean-rate contrast instead changes from .018626 to .013437 in the
paired host and from .011435 to .015697 in the swapped host. Which readout is
chosen therefore changes the apparent success. Do not select the favorable
readout separately for each seed. All 13 structural checks pass in both exchange
audits, including exact source replay, declared-weight-only intervention and
bit-identical complementary hybrids. The causal dependence on these weights is
real in these preparations; a robust content-specific retrieval direction is
not established.

This narrows the architectural question. Activity regulation, storage and
retrieval cannot be assessed independently and then assumed to compose. The
same learned projections alter the receiving population's operating state,
while the reference sound representations themselves can change with learning.
The current audit uses fixed initial sound references; compare those with
learned sound responses and actual downstream neural use before attributing
every mismatch to lost memory. Conversely, an observer that can extract a
pairing difference does not demonstrate that a neural consumer can use it.
These are competing explanations to distinguish, not established causes.

### Evidence and continuation

Under `.live/research/`, the bounded learning runs are
`20260908_bounded_learning_{paired,swapped}_seed{11,23}`, with corresponding
`20260908_bounded_learning_audit_seed*.json`. Weight-space analyses are
`20260908_{regional,bounded}_weight_geometry_seed*.json` where generated.
Counterfactual records are `20260908_bounded_exchange_{paired,swapped}_seed*`
with corresponding audits. Keep the native records as controls.

The next architecture work must distinguish acquired connection structure from
the recurrent dynamics that express it. Verify the weight-exchange effect,
separate common activation changes from content, and add neural consumers of
the relevant state before claiming useful hierarchy. Regional feedback and a
bounded learning rule are reusable parts, not a substitute for the sensorimotor,
metabolic, continuing-memory and reasoning loops in the full research goal.

All four bounded learning runs and all four donor-exchange runs are complete.
No experimental neural workers remain running. The existing static assessment
server remains available; no live embodied simulation or acceptance matrix was
started. Available disk space is approximately 2.8 GiB, so further recordings
need a storage budget before increasing duration or replication.

## Whole trained state and the granularity of adaptive coupling

The next goal turn is progress. It adds verified state-branching experiments,
tests whether two evaluation assumptions explain the recall failure, and builds
a native-PAULA wiring candidate. It does not establish selective recall,
embodiment of the population prototype, or consciousness.

### Test the observer and state partition before changing plasticity

The initial sound-reference axis can drift during learning. Recomputing it
from each learned network's audio-only responses changes the numerical scores
but does not rescue the intended paired/swapped rate reversal. For incoming-
information-only probes, centered-rate axis cosines between original and learned
references are .936-.940 across the four recordings. Centered covariance axes
move more, with cosines .499-.558. Both references remain available in the new
audit. No favorable reference or readout replaces the previous criterion.

Reconstruction from the training rasters also rejects a blanket explanation
that native timing credit is always positive. Across audiovisual exposure
episodes, the estimated positive-credit fraction on direct visual-to-auditory
inputs varies roughly from .50 to .94. This uses the recorded spikes, t_ref and
the known one-tick cleft delay, not a new online synaptic observer. It is a
diagnostic reason to avoid treating the whole run as a saturated timing window.

The former `learned_all` condition in the weight-path probe meant all incoming
information weights, not all learned neural variables. That remains useful as
a specified intervention, but it is not ordinary recall by the trained brain.
PAULA also adapts output terminals. `population_state_branch.py` now compares:

- `incoming_info`, the existing probe with only incoming information weights;
- `all_synaptic`, all incoming u_i and outgoing u_o/u_i_retro variables, with
  fresh activity, modulation and queues;
- `continuation`, a branch of the actual trained network with its ongoing state.

The tool reconstructs the trained network by replaying all 3,168 training ticks.
Every cell field, every episode-end incoming weight and every recorded weight-
health row must match. The final full snapshot must match too. It then branches
the in-memory network, preserving numeric types and buffer aliases within each
branch. It does not assume that JSON records encode every implementation detail
needed for exact restoration. Each parent must remain unchanged. Cleft delays
must be deterministic; stochastic delays need an explicit per-branch RNG design.

Each of four runs, two seeds and both assignments, executed 5,568 ticks including
eight fresh probe branches. All training replays and the independent state-
partition audits passed. The output records include each branch's starting
snapshot and every cellular tick. The data-only auditor exports instantaneous
spike-pattern projections and all five declared time windows, as well as the
four existing readouts. These projections are observers, not neural consumers.

### What the full-state comparison changed

Centered auditory rate contrast measured against each condition's current
sound responses is:

| Graph seed | State carried into recall | Paired | Swapped |
| --- | --- | ---: | ---: |
| 11 | Incoming info only | .028231 | .010845 |
| 11 | All synaptic variables | .018741 | .022707 |
| 11 | Intact continuation | .018741 | .022707 |
| 23 | Incoming info only | .015705 | .012741 |
| 23 | All synaptic variables | .019161 | .014808 |
| 23 | Intact continuation | .019161 | .014808 |

Both assignments still give positive contrasts. Against the original common
sound reference, the paired-minus-swapped difference changes sign when output
terminal learning is restored, in both seeds. At seed 11 it moves from +.016448
to -.007933; at seed 23 it moves from -.002159 to +.003226. Small partial-memory
effects therefore cannot be promoted to ordinary recall performance.

The parameter audit identifies the extra learned state precisely. All four
recordings changed 964 outgoing information values; none changed incoming
plasticity/adaptation vectors, outgoing modulation vectors or u_i_retro.
Incoming information weights were identical across the compared starting
partitions. Thus the all-synaptic versus incoming-info intervention isolates
outgoing information memory in these recordings.

Intact continuation and all-synaptic fresh activity give identical O rasters
for every neuron on every tick of all 16 probes. They are not identical full
states: small differences in continuous state and final weights remain. This
does not make ongoing dynamics generally irrelevant. The source protocol ends
with 96 silent ticks, leaving no queued events, maximum F_avg below .000071 and
maximum modulation below .000031. It is a test after that withdrawal, not a test
of working memory during ongoing stimulation.

Time matters even here. In seed 11's intact paired condition, centered-rate
contrast against the current sound reference is -.02119 over ticks 32-95 and
+.03701 over ticks 224-299. The swapped condition also changes from negative to
positive. Reporting only one favorable interval would misstate the result.

### A structural candidate before another neuron-rule extension

The population graph has one adaptive output terminal per neuron. Native
retrograde signals from multiple projection families therefore update shared
release state. A target in one family can alter subsequent transmission to a
different family. The experiment above establishes that terminal state matters;
it does not establish that sharing it causes the recall failure.

`components/learning/projection_terminals.py` provides the next intervention.
Its default preserves the graph exactly. The family-separated variant expands
1,152 terminals to 3,268 while keeping 1,152 neurons and 21,760 connections.
Every initial per-edge release vector, target weight, input count and delay is
unchanged. A shuffled control has the same terminal count and each terminal's
same fan-out, but mixes projection families within those adaptive channels.
This distinguishes a role-specific effect from merely spreading retrograde
updates over more variables. Both plasticity paths remain positive.

Four tests verify the builder's default, initial forward wiring, ID allocation,
matched control and local retrograde isolation. The combined focused test set
passes 44 tests. Audiovisual learning with this variant is not yet tested.
Do not call it a repaired memory or use it in the demo agents yet.

[Reyes et al., 1998](https://doi.org/10.1038/1092) provide biological motivation
for target-specific presynaptic behavior along one axon. Their experiments
concern short-term facilitation and depression, not this particular PAULA
partition or its learning efficacy. Separately, the June 2026 preprint by
[Miller, Miehl and Doiron](https://pubmed.ncbi.nlm.nih.gov/42395491/) reports that
matching excitatory plasticity to inhibitory homeostasis can preserve variable
coupling strengths with different response gains and timescales. Only its
abstract and indexed extracts were accessible in this turn. Treat it as a lead
for joint regulation/learning design, not an implemented rule or validation of
PAULA. Attempts to retrieve the complete article hit access checks.

The compositional issue is now more specific: a connection diagram does not
fully specify which pathways share adaptive state. Component contracts must
declare that sharing. The next experiment should compare shared, family-split
and shuffled terminals under the same audiovisual protocol, retaining both
incoming-only and full-state probes. Keep neural consumer tests and embodiment
in the full goal; none follows automatically from this wiring change.

Evidence is `.live/research/20260908_state_branch_{paired,swapped}_seed{11,23}`
and `.live/research/20260908_state_branch_audit_v2_seed{11,23}`. The earlier
seed-11 audit is retained as the pre-partition-count analysis. No source recording
was overwritten and no demo simulation or acceptance matrix was restarted.

## Terminal partition completed; return to the causal definition of association

The shared/family/shuffled hypothesis now has family and shuffled recordings
for both assignments at seeds 11 and 23. All four independent paired/swapped
audits pass their structural checks, including exact per-terminal fan-out in
the shuffled control. Each new run records 6,168 ticks and keeps positive
adaptation in full-state and restricted incoming-only probes.

There is no replicated recall repair. At seed 11 neither partition meets the
two-direction change condition on any of the four auditory or upper readouts.
At seed 23 auditory covariance meets it in both partitions, so that observation
does not distinguish role-based grouping. Upper centered-rate and centered-
covariance changes meet it in the family partition at seed 23 but not seed 11.
These are mixed descriptive effects, not grounds for accepting one favorable
readout per seed. Full traces remain available; a failed projection criterion
does not prove the absence of every possible temporal association.

Evidence: `.live/research/20260908_terminal_{family,shuffled}_{paired,swapped}_seed{11,23}`
and `.live/research/20260908_terminal_{family,shuffled}_audit_seed{11,23}`.

The user's questions expose a missing distinction in the research sequence.
Persistent activity is not necessary for every associative response, and
calling a population "upper" does not establish a learned joint representation.
The existing task measures cross-modal response under a continuing, familiar
visual cue. It has not demonstrated re-entry into a sustained learned regime,
or a neural consumer's use of recalled content. A feedforward association and
a recurrent auto-associative memory are different hypotheses; see the explicit
distinction in [this primary modelling study](https://pubmed.ncbi.nlm.nih.gov/34874938/).
Its abstract supports the distinction, not an implementation claim for PAULA.

`association_route_probe.py` therefore asks a narrower causal question before
another neural redesign. It reconstructs each original bounded network by
exact training replay, then compares four post-experience interventions:
intact, direct visual-to-auditory cut, upper-to-auditory cut, and both cuts.
The same factorial probes also run from the initial network. Every branch
starts with the same local state within its initial/trained condition. Cuts
retain neurons, input ports, delays and parameters but remove selected forward
delivery and corresponding retrograde registration. All remaining adaptation
stays active. The instrument refuses cuts with in-flight events rather than
silently dropping history. No new brain mechanism or decoder is introduced.

`association_route_audit.py` independently checks recorded edge selection,
starting weights, identical initial controls and original initial replay. It
exports per-tick changed-cell counts, signed spike differences, first divergence,
post-tick S, thresholds and fixed-reference projections. Equal spike totals
can conceal different active cells, so both are retained. A pathway effect is
not automatically an association: the paired/swapped history contrast and
the untrained controls must remain visible.

### Pathway dependence: measured results

All four runs complete, covering paired/swapped training at seeds 11 and 23.
They execute 31,872 ticks, including exact replay of the original training and
19,200 new probe ticks. Both independent audits pass. The eight intact trained
probe rasters and final incoming weights also match the earlier full-state
branch recordings exactly. No original recording is overwritten.

Auditory spike counts below cover all 192 auditory-core neurons over each
300-tick silent-video probe. Each entry lists clip 0 / clip 1; these are
individual cases, not seed means.

| Training | Intact | Direct visual input cut | Upper feedback cut | Both cut |
| --- | ---: | ---: | ---: | ---: |
| Seed 11 paired | 211 / 365 | 0 / 0 | 21 / 24 | 0 / 0 |
| Seed 11 swapped | 176 / 246 | 0 / 0 | 4 / 5 | 0 / 0 |
| Seed 23 paired | 226 / 390 | 0 / 0 | 37 / 42 | 0 / 0 |
| Seed 23 swapped | 184 / 311 | 0 / 0 | 23 / 22 | 0 / 0 |

The tick traces establish when the effects begin. Intact auditory firing starts
at tick 14 in seed-11 paired, tick 15 in seed-11 swapped, tick 12 in seed-23
paired and tick 15 in seed-23 swapped. Direct-route removal eliminates every
auditory spike, including those first events. Upper-route removal changes the
auditory raster beginning at tick 15 in all eight trained probes. It reduces
total auditory spikes by 83.6-98.0%, but preserves some direct-route activity.
Thus the descending route is effective in the composed network even though
it cannot initiate auditory firing without the direct route in these tests.

The initial network is nearly silent in the auditory core during video.
There is one exception: clip 1 at seed 11 produces one auditory spike at tick
21, eliminated by either cut. Seed 23 produces none. A readout discarding the
first 32 ticks misses that exception. Do not describe every initial trace as
perfectly silent or use a warm-up exclusion as the only evidence.

An arithmetic check explains why lack of standalone upper-driven firing is
not surprising. With local weights and releases fixed at their trained values,
the upper pathway's maximal-rate, phase-aligned linear membrane envelope ranges
from .1627 to .3661 across targets and cases, below base r=.65 for every target.
The envelope sums each positive pulse contribution W/lambda divided by
1-(1-1/lambda)^c, with the source cooldown c and dendritic attenuation. An
independent periodic-drive test verifies the formula. This is a fixed-
coefficient operating-range bound, not a freeze applied to any neural run and
not a bound on the full changing network. Actual direct-cut traces peak at
post-tick S=.2608-.2954, with no auditory firing. The upper route can still
help direct input cross threshold; eliminating it is therefore consequential.

Content remains unresolved. For example, seed-11 paired intact centered-rate
contrast is -.02552 in ticks 32-95 and +.03115 in ticks 224-299. Swapped changes
from -.00384 to +.03952 over those same intervals. Late positive projection is
therefore not assignment-specific recall. Removing upper feedback also changes
other populations through the remaining recurrent network; the spike reduction
does not localize memory storage to the cut connections.

The result narrows the architecture question. There is an experience-dependent
cross-modal response and a consequential descending contribution. A dead upper
layer is not the explanation. The next discriminating intervention should
separate upper feedback's content/timing from its permissive drive, using
recorded neural-output substitutions with appropriate activity-matched controls.
It must first reproduce intact delivery exactly and disclose that clamped replay
breaks the natural feedback loop. Do not simply amplify the upper connections
until the auditory population fires, or require autonomous persistence as a
condition for every form of associative response.

Evidence: `.live/research/20260908_association_routes_{paired,swapped}_seed{11,23}`
and `.live/research/20260908_association_routes_audit_seed{11,23}`. The auditor
retains every-tick changed-cell counts, signed spike differences, spike totals,
post-tick S, threshold extrema and fixed-reference projections. Original
per-cell traces remain in each probe. These two graph seeds are mechanistic
diagnostics, not the four-seed or embodied acceptance of a memory component.

The focused suite passes 52 tests before the envelope test was added; the new
envelope and related unit checks pass, and the latest end-to-end independent
audit/corruption test passes separately. No PAULA model or demo-agent code was
changed in this experiment.

The final combined rerun reports 50 passed and three recording tests refused
by their 600/650 MiB free-space guards. Available disk space fell to about
204 MiB after the research runs completed. This is not an all-green final
suite, and the guards were not bypassed. The earlier 52-test suite and the
separate latest end-to-end audit test had passed before the free-space drop.
The five non-recording route tests pass after the final auditor change.

The auditor now processes full cellular tensors one at a time, retaining
boolean rasters and its exact per-tick extrema rather than all sixteen tensors.
Sequential reruns on both real seed pairs reproduce all 512 exported tick
arrays per seed bit-for-bit, all result rows and all operating-range envelopes.
Measured wall times are 2.11 and 2.32 seconds; maximum resident sizes are
250.2 and 268.7 MiB, with zero swaps reported for those processes. The original
and streaming audits are both retained. Four route recordings occupy about
508 MiB in total. No further simulations were started after the space guard
failures; all research workers have exited. Storage must be resolved before
expanding experiments.

## 9 September: intervene on upper signals, not only pathway presence

Storage is available again, with 27 GiB free at preflight. No worker or server
from the preceding turn is running. The next experiment uses the existing
paired/swapped bounded networks at seeds 11 and 23, not a new tuned network.
It asks whether their post-experience auditory response depends on the upper
input's timing, assignment to dendritic ports, or source video. The initial
network receives the same interventions as a control.

`upper_signal_replay.py` records the 384 upper-to-auditory input rows after
network delivery and before local bounded-neuron computation. The process-local
hook observes or replaces those rows only. All neurons, topology, other inputs,
retrograde feedback and positive learning rates remain live. Counterfactual
replacement breaks natural feedback at the named delivery points; it is an
experimental input clamp, not a component of the proposed cognitive network.
Unchanged replacement must reproduce all cellular traces, final incoming
weights and the complete final snapshot. Native capture must reproduce the
earlier intact route-probe recording. Both equalities are tested before using
the altered replacements.

Three controls have deliberately different conservation properties. A shared
time permutation within each 32-tick window preserves every port's sample
multiset and the instantaneous joint upper-input patterns, but changes their
order and alignment with the rest of the network. Exchanging each target's two
upper afferents preserves its raw summed input at every tick, but not its
weighted and delayed current. Substitution of the other video's upper stream
changes dose as well as pattern and must not be called activity-matched.
`upper_signal_audit.py` checks those invariants independently and exports
per-tick spike differences and fixed-auditory-axis changes without driving the
network. A response to manipulation alone cannot establish remembered content.

The methodological motivation is specific. [Carrillo-Reid et al., 2019](https://pubmed.ncbi.nlm.nih.gov/31257030/)
report behavioral effects of experimentally recalling particular cortical
ensembles. Indexed primary abstract text was accessible; the PubMed open
request encountered a browser check. [Shahidi et al., 2019](https://www.nature.com/articles/s41593-019-0406-3)
report associations between precise multi-cell coordination and perceptual
accuracy beyond firing-rate modulation. The publisher abstract and figure
descriptions were accessible, not the subscription methods. These findings
motivate distinguishing activity amount, timing and ensemble identity. The
PAULA interventions do not reproduce those experiments or establish perception.

A separate read-only check reconstructs the native timing-direction condition
from the saved training O, t_ref and last-spike history. It rules out describing
all training as "the potentiation gate is always open." For example, seed-11
paired audio-0 exposure first has positive eligibility at about .1% and .5% of
cell-ticks in the first two 32-tick windows, rising to about 98% late in the
clip. Audio-1 exposure is broadly eligible earlier. This is the conditional
direction if a synapse is active, not a count of actual positive updates.
Actual active-port arrivals and update magnitudes still need to be traced
before attributing failed selectivity to this timing rule.

The focused suite, including the formerly space-blocked tests, now passes all
55 tests. Recording equivalence and deliberately corrupted replacement tensors
are covered. No existing neuron-model source or V1-V4 agent was changed.

### Upper-signal interventions: results

All four runs and both independent paired/swapped audits complete. They execute
36,672 ticks, including exact training replays, native delivery capture and 64
counterfactual probes. Every unchanged replacement matches its original
cellular trace, final incoming weights and complete recorded final snapshot.
Initial controls are bit-identical between the two training assignments.

Auditory-core spikes over each 300-tick trained silent-video probe:

| Training | Unchanged | Time shuffled | Target ports exchanged | Other video's upper stream |
| --- | ---: | ---: | ---: | ---: |
| Seed 11 paired | 211 / 365 | 231 / 375 | 245 / 369 | 207 / 361 |
| Seed 11 swapped | 176 / 246 | 176 / 224 | 185 / 263 | 193 / 220 |
| Seed 23 paired | 226 / 390 | 172 / 381 | 217 / 403 | 225 / 379 |
| Seed 23 swapped | 184 / 311 | 152 / 270 | 194 / 298 | 193 / 279 |

Entries list clips 0 / 1, not seed averages. Timing shuffle retains 76.1-109.5%
of the unchanged spike count; port exchange retains 95.8-116.1%. These effects
are much smaller than eliminating upper feedback, which left 2.0-16.4% of the
unchanged response in the preceding experiment. Initial networks remain nearly
silent under all replacements: seed 11 clip 1 produces one spike except with
time shuffle, and every other initial probe produces none.

Preserved spike count is not preserved dynamics. Time-shuffle effects first
appear in auditory rasters at ticks 12-14, port-exchange effects at ticks 15-16,
and other-video effects at ticks 15-17. Across these altered trained probes,
204-701 cell-by-tick spike entries differ from the unchanged run. Seed-11
swapped clip 0 has exactly 176 spikes before and after time shuffle, but 328
entries differ. The raw trace therefore rules out describing the control as
having no effect merely because its total count is unchanged.

The other-video substitution does not yield a reliable learned-content transfer.
Using the fixed initial auditory contrast axis, its change over ticks 32-299
points toward the donor video's associated sound in only two of eight trained
cases. Both seed-11 paired cases point in the opposite direction, with signed
changes -.00300 and -.00111 in reference-separation units. Seed-23 paired gives
-.00433 and +.01132; seed-23 swapped gives +.00148 and -.00342. This is a
descriptive diagnostic, not an eight-sample significance test: clips and
conditions are dependent, and dose differs between donor streams. All onset
and later windows are exported; no favorable interval replaces the full record.

The current evidence supports a distinction between effective descending drive
and demonstrated learned content. These particular pattern manipulations leave
most auditory activation intact while changing its detailed timing. That is
compatible with a permissive contribution plus temporal modulation. It does
not prove absence of information in the upper population, nor exclude an
invariant code. The familiar cue still drives the network throughout and the
readout is not a neural consumer of a learned representation.

The next architecture decision should follow a learning-credit audit rather
than another gain increase. Trace which visual-to-auditory and convergent upper
synapses actually receive positive/negative local updates during paired and
swapped experience, and whether that credit preserves the distinctions that a
neural consumer could use. Conditional timing-window openness alone is not an
update record. The signal-clamp tool remains available for causal tests after
a representation or credit mechanism changes; it is not added to the agent.

Evidence is `.live/research/20260909_upper_signal_{paired,swapped}_seed{11,23}`
and `.live/research/20260909_upper_signal_audit_seed{11,23}`. The original
recordings are retained, and all four research workers have exited. This is
two-seed mechanistic evidence, not memory acceptance, embodiment, or completion
of the mammal-level/consciousness research goal.

### Actual learning credit: a timing-gate limitation

The next experiment records active synaptic updates rather than inferring them
from postsynaptic firing. `association_credit_probe.py` observes the final
bounded update, including the arriving four-channel signal, previous weight,
native local error, effective learning rate, post-spike age, timing window,
direction and resulting weight. It observes 4,224 incoming ports across direct
visual-to-auditory, upper-to-auditory and connector-to-upper projections. The
intermediate update replaced by the bounded extension is not mistaken for the
effective final update. No neuronal state is written by the observer.

All four 3,168-tick training replays reproduce the source cellular records,
weight endpoints, health records and final snapshots exactly. They record
8,959,011 active updates in total. The independent data-only auditor checks
the complete selected port set, reconstructs timing from cellular rasters,
checks local errors and effective rates, and chains updates to every saved
weight endpoint. Its independent analytic solution differs from recorded
weights by at most 6.67e-16. This verifies the observed updates, not that a
particular input caused a postsynaptic spike. Endpoint reconstruction cannot
independently exclude omitted arrivals with exactly zero weight effect.

| Training | Active updates, all three pathways | Upper ascending positive direction | Minimum positive fraction within exposure windows |
| --- | ---: | ---: | ---: |
| Seed 11 paired | 2,207,878 | 98.65% | 98.29% |
| Seed 11 swapped | 2,256,421 | 98.80% | 99.81% |
| Seed 23 paired | 2,214,647 | 98.62% | 98.72% |
| Seed 23 swapped | 2,280,065 | 98.77% | 99.42% |

The window statistic includes all 72 nonempty windows per recording after
the first 32 ticks of each exposure, using 32-tick windows and the shorter
final window. The onset is retained in the event and tick records. It differs
from the sustained response: the first 32 ticks have 2,623-2,863 negative
ascending updates. Over the full recording, 2,137-2,343 of 2,406-2,411 ticks
with ascending arrivals have exclusively positive timing direction. Thus the
near-uniform sustained direction is visible in the trajectory and does not
depend on averaging onset together with the rest of training.

The auditory pathways differ. Direct crossmodal inputs receive positive
direction on 73.54-76.87% of updates, and descending inputs on 75.37-78.44%.
It would be false to describe all network plasticity as reinforcement-only.
Even among positive-direction upper ascending updates, 272,404-283,451 per
recording do not increase the weight. The bounded magnitude equation includes
decay, so timing direction and actual weight change are separate observations.
Mean effective rates per active update range from approximately .000845 to
.000987 across the three families, versus the basal .00001. The receptor's
rate amplification is active; lack of weight change is not the explanation.

A constructive unit test exposes a constraint in the native timing rule.
The lower bound is `t_ref >= 2*c`. With sustained inter-spike interval `c`,
post-spike age only traverses `0 .. c-1`, including the current spike reset.
Every active input therefore receives positive timing direction even with
the window forced to its minimum. The test uses the actual neuron, positive
plasticity and a local modulator at the minimum-window condition. It observes
three-tick inter-spike intervals and all three arrival phases. This is a
limitation of timing-based discrimination in that operating regime, not a
claim that PAULA cannot learn, that its other regulation is absent, or that
all synapses must strengthen. It also corrects the unconditional suggestion in
the old lab notes that shortening this window necessarily depresses more
inputs in an overactive cell. It can do so only if arrivals fall outside it.

This suggests a specific composition problem. Connecting populations changes
their activity, which changes which distinctions their plasticity can express.
A learning rule that distinguishes arrivals under sparse drive can lose that
timing distinction under sustained convergent drive. Adding a higher layer or
stronger descending gain does not by itself restore it. We have not yet shown
that this loss causes the failed auditory association, nor excluded a useful
code carried by magnitudes, inhibition or other temporal structure.

The next causal comparison should separate regulating the population's
operating regime from changing synapse-local eligibility. First measure
whether a neural inhibitory/regulatory intervention restores selective credit
while preserving distinct driven responses. Compare it with unchanged wiring
and a declared local eligibility extension, retaining weak positive baseline
adaptation. Require pairing reversal to reverse recalled auditory content,
an unpaired control, and a neural consumer that distinguishes the result.
Activity reduction or a reopened negative gate alone is not success. Replicate
a promising effect on at least four seeds before acceptance, then test its
composition with action and embodiment.

[Vogels et al., 2011](https://pubmed.ncbi.nlm.nih.gov/22075724/) provide a
specific biological-model lead: inhibitory plasticity can regulate sparse
responses and support stimulus-reactivated memory in their recurrent model.
The primary abstract was read here, not the full methods; no claim of
implementing or reproducing that model is made. The voltage-dependent
plasticity model of [Clopath et al., 2010](https://www.nature.com/articles/nn.2479)
is another lead for comparing local credit signals. Publisher indexing was
accessible but the direct page request failed; its equations need a full
reading before any implementation based on that paper.

Evidence is `.live/research/20260909_association_credit_{paired,swapped}_seed{11,23}`
and the strict-port-coverage audit outputs
`.live/research/20260909_association_credit_audit_v2_{paired,swapped}_seed{11,23}`.
Each audit exports full per-tick credit arrays, not only this table. All 58
focused tests pass. Four recording workers and four final auditors have exited.
No neuron-model source, demo agent, live simulator or embodied suite changed.
Recall remains unverified; this is a tested mechanistic constraint and a
reusable measurement tool, not a completed memory architecture.

### Inhibitory capacity changes activity but does not repair recall

The follow-up is a controlled change to the existing neural inhibitory route,
not a new plasticity model. For each upper target, `inhibitory_capacity.py`
computes the positive and negative initial mean-current capacities using actual
terminal release, source cooldown `c`, incoming weight and dendritic attenuation.
It rescales only the four negative incoming information weights so that these
capacities match. At seed 11 the factors range from 3.650 to 4.630. No neurons,
connections, delays, thresholds, modulatory gains or learning rates change.
This arithmetic assumes sustained maximal source rates for the comparison;
it does not guarantee balanced actual currents, synchrony or firing rates.
The matched inhibitory weights remain within the existing bounded interval.

Five complete runs execute 27,840 neural ticks: paired/swapped experience at
seeds 11 and 23 with the intervention, plus a fresh unchanged seed-11 paired
control. Every run includes training and eight full-tick probes, using initial
states or complete trained-state branches. The control reproduces the original
training cells, weight endpoints, health records and final snapshot exactly.
The paired/swapped modified runs have identical configs, runtime hashes,
initial-probe arrays and sensory marginals within each seed. Adaptation remains
positive during both training and probes. Five independent audits validate the
permitted config changes, initial capacity arithmetic, full training weight
continuity, selected active-update chains and probe records. Maximum analytic
update residual is 4.45e-16. All 61 focused tests pass, with the three capacity
tests re-run after adding deliberate-corruption checks.

The predicted timing-selectivity repair fails. Every ascending event after the
first 32 ticks of each exposure has positive timing direction in all four
modified runs. This includes every one of the 72 nonempty post-onset windows
per run. Whole-record fractions are lower because onset and withdrawal still
contain negative updates; they must not obscure the sustained result.

| Training | All observed updates on the three routes | Whole-record ascending positive direction | Positive ascending events older than the six-tick minimum, as fraction of all ascending events |
| --- | ---: | ---: | ---: |
| Seed 11 paired | 2,178,145 | 98.78% | 42.83% |
| Seed 11 swapped | 2,220,061 | 98.79% | 42.15% |
| Seed 23 paired | 2,180,793 | 98.70% | 41.79% |
| Seed 23 swapped | 2,252,853 | 98.78% | 41.53% |

The mechanism is visible within the first exposure, not only between runs.
At seed 11 paired, ticks 32-63 change from .154 to .118 spikes per cell per
tick, while mean t_ref grows from 42.32 to 51.43 ticks. At ticks 192-223 the
corresponding values are .143 to .104 and 45.78 to 54.01. The native formula
lengthens the timing window as F_avg decreases, subject to local modulation.
The exact event records show that the longer windows still include every
post-onset ascending arrival. For comparison, only 22.32% of all ascending
events in the unchanged seed-11 paired control are positive with age above
six, versus 42.83% after capacity matching. This is a diagnostic of the
recorded gate, not a simulated outcome with its window clamped to six ticks.

Thus the earlier maximal-rate counterexample is not the whole explanation.
The timing distinction also disappears at substantially slower activity.
Activity regulation and metaplasticity interact: lowering activity can lengthen
the credit window enough to preserve nonselective timing direction. Changing
one operating variable is not equivalent to restoring the learning conditions
of an isolated population. Actual synaptic changes remain nonuniform because
input timing, magnitude, existing weights and decay also enter the update.

Auditory spikes during complete-trained-state silent-video probes, clips 0 / 1:

| Training | Original upper inhibition | Capacity-matched upper inhibition |
| --- | ---: | ---: |
| Seed 11 paired | 211 / 365 | 147 / 288 |
| Seed 11 swapped | 176 / 246 | 164 / 202 |
| Seed 23 paired | 226 / 390 | 180 / 333 |
| Seed 23 swapped | 184 / 311 | 145 / 225 |

Activity persists but assignment-specific recall is not established. For the
auditory population-centered rate readout, trained-minus-initial contrasts are
+.01621 and +.01119 for seed-11 paired/swapped, and +.00137 and +.00887 for
seed 23. Reversing assignment does not reverse the change. Current trained
auditory reference axes do not rescue that direction: all four contrasts stay
positive. At seed 11 the paired and swapped trajectories remain positive in
each declared post-onset comparison window. Other readouts are retained rather
than selected for a favorable sign. Uncentered covariance follows assignment
at seed 23 but not seed 11; upper centered-rate changes likewise show a
seed-23-only effect. Neither is replicated recall. These offline projections
are not neural consumers or proof of absent memory in every possible code.

The useful next distinction is selective recruitment versus timing-credit
selectivity. Matching total inhibitory capacity is not a circuit that selects
different ensembles for different experiences. In the 64 full 32-tick
post-onset exposure windows per modified run, 245-256, 253-256, 249-256 and
255-256 of the 256 upper cells fire at least once, respectively for seed-11
paired/swapped and seed-23 paired/swapped. All 256 fire in 54, 58, 53 and 63
windows. This excludes the shorter final windows from the recruitment
comparison. Reduced rate has largely preserved broad recruitment; these
counts do not exclude information in precise patterns or timing.
Test a circuit-level competition
mechanism and a separately declared synapse-local eligibility mechanism against
the same current control, with matched input histories. Do not keep increasing
global inhibition until one readout happens to change sign. A positive result
must track reversed pairing, survive unpaired controls and support a neural
consumer; a reduction in activity or wider range of update signs is insufficient.

Recordings are `.live/research/20260909_inhibitory_capacity_{paired,swapped}_seed{11,23}`
and `.live/research/20260909_inhibitory_capacity_control_paired_seed11`.
Use the matching `20260909_inhibitory_capacity_audit_v2_*` directories for the
final auditors, which include current-reference and full weight-health checks.
All workers have exited. This intervention is retained as an optional research
component, not installed in V1-V4 or represented as a working memory repair.

### A local eligibility extension supports controlled cross-sensory recall

`neuron/extensions/experimental/eligibility_trace.py` now supplies an opt-in
local pre/post trace mechanism through subclassing. The base neuron and network
files are unchanged. Empty `eligibility_ports` preserves the previous bounded
neuron path. On selected excitatory ports, an arriving signal leaves an
exponentially decaying trace; a later actual somatic spike can strengthen that
input even without another arrival. Conversely, an arrival encounters the
decaying history of preceding somatic spikes. Zero-lag pairs are excluded.
There is no stimulus identifier, host-computed error, target response or hold
counter. The same positive basal learning rates and neural modulatory receptor
remain active. Nonselected inputs retain the previous bounded rule.

The selected weight obeys an exactly integrated held-coefficient affine flow,
`dq/ds = Lplus*(cap-q) - Lminus*q`, with `Lplus = spike*pre_trace` and
`Lminus = alpha*arrival*post_trace`. This changes both timing credit and weight
dependence; the comparison does not isolate timing as the sole cause of a
behavioral improvement. Native retrograde errors and output-terminal adaptation
remain a separate learning process. The extension's additional traces are saved
explicitly as dynamical state. Inherited bounded counters include provisional
selected updates and must not be interpreted as the effective extension ledger.

The methodological source is [Guetig et al., 2003](https://brainworks.biologie.uni-freiburg.de/2003/journal%20papers/guetig-jns-2003.pdf),
especially methods equations 1-2 and the all-pair timing construction. The
relevant methods were read from the full paper. This extension uses linear
weight dependence, not the paper's nonlinear competition mechanism, and retains
PAULA's neuron dynamics rather than reproducing their conductance model. This
is an established plasticity idea tested here, not a newly discovered general
learning principle or a faithful reconstruction of a particular cell type.

The test preparation contains 144 neurons: two 32-receptor sheets, a 32-cell
auditory population, 32 downstream consumers and 16 activity-driven modulatory
neurons. Each auditory cell receives all 32 visual afferents with initially weak,
identity-independent heterogeneous weights. Auditory and consumer projections
preserve receptor coordinates. Two disjoint synthetic receptor patterns per
sense provide deliberately distinguishable physical inputs. These are not
decoded real images, audio clips, learned features or semantic categories.
The cross-sensory assignment appears only in the sequence of sensory stimuli.

Sixteen recordings cover seeds 11, 23, 44 and 77: paired, reversed-pairing and
temporally separated exposure with the extension, plus paired exposure under
the preceding bounded rule. Every run uses 16 presentations per pairing and
positive plasticity during acquisition and recall. Both sensory conditions have
equal event counts; the separated condition delays the auditory pulse train by
32 ticks. It is a temporal-contingency control, not arbitrary independent
naturalistic experience. Within each seed, paired/reversed/separated configs,
input masks and initial probe arrays are identical.

Every seed gives the same neural-consumer outcome:

| Condition | Visual cue 0 | Visual cue 1 |
| --- | --- | --- |
| Before experience | No consumer spikes | No consumer spikes |
| Paired experience | Sound-0 coordinates only | Sound-1 coordinates only |
| Reversed pairing | Sound-1 coordinates only | Sound-0 coordinates only |
| Temporally separated experience | No consumer spikes | No consumer spikes |
| Previous bounded rule, same paired schedule | No consumer spikes | No consumer spikes |

Each successful probe has 64 consumer spikes, 16 cells responding at relative
ticks 5, 13, 21 and 29, and zero spikes in the alternative coordinates. Auditory
receptors are silent during these probes. Consumer spikes follow auditory-core
spikes by the two ticks implied by the declared connection and dendritic delay.
No learned Python readout drives the response. The fixed consumer projection
makes the auditory coordinate pattern available downstream; it does not show
a learned semantic interpretation or a higher-level concept.

Eight further replay experiments test paired and reversed training in all four
seeds. Each reproduces every saved training field and final dynamical snapshot.
Resetting only the 1,024 learned visual-to-auditory weights in the complete
trained state removes every consumer response. Transplanting only those weights
into a fresh network restores the complete consumer spike rasters in all 16
cue tests. No residual eligibility trace, modulatory state, changed output
terminal, or persistent firing is necessary for this particular reinstatement.
This necessity/sufficiency result is conditional on the constructed topology
and task. All branches continue adapting; no learning rate is frozen.

The data-only auditor reconstructs every selected eligibility weight trajectory
from actual local arrival masks, somatic spikes and effective rates. Maximum
residual over the 12 enabled acquisition records is 2.23e-16. All 16 record
audits pass; native-control weight trajectories are not independently derived
by that eligibility auditor. An additional read-only check validates the 48
state-branch records and their local updates with the same maximum residual.
Five extension tests, 13 existing bounded/rate tests and 63 focused simulation
tests pass. The observer itself has a full-dynamical-state equivalence test,
and deliberately corrupted arrival and weight records are rejected.

This is a working, causally checked cross-sensory association in a controlled
PAULA-extension preparation. It is not yet the requested hierarchical embodied
brain. The prior 1,152-cell real-media network remains unresolved. Transfer the
mechanism there with explicit state recording and preserve the current control;
then test overlapping and partial cues, interference, retention over ongoing
experience, selective recruitment and downstream action. The current native
control's failure does not establish that native PAULA cannot learn this task
under other rates, wiring or exposure durations.

Evidence roots are `20260909_eligibility_association_{paired,swapped,separated}_seed{11,23,44,77}`,
`20260909_native_association_paired_seed{11,23,44,77}` and
`20260909_eligibility_state_{paired,swapped}_seed{11,23,44,77}` under
`.live/research/`. Acquisition audit outputs use the matching
`eligibility_association_audit_v2` or `native_association_audit_v2` prefix.
The experiments total 80,896 neural ticks. All workers have exited, with no
live simulation or V1-V4 agent restarted or changed.

### Real-media transfer separates learned activation from selective recall

The controlled eligibility rule has now been transferred into the existing
1,152-neuron regional audiovisual graph. Only the 768 visual-core to
auditory-core incoming information updates change. The four selected inputs
per auditory cell use the same pre/post time constants of four ticks, alpha
one and cap one as the synthetic experiment. Wiring, initial weights, delays,
sensory transduction, media, exposure order, excitability regulation and other
learning rules remain unchanged. Both basal learning paths remain positive.
No new consumer or externally computed cognitive signal is introduced.

Five recordings cover paired and reversed audiovisual experience in seeds 11
and 23, plus one unchanged-rule control. The control uses the same subclass
with empty eligibility ports. Every training cell field, incoming-weight
endpoint, weight-health row and the complete original final state reproduce
exactly. Within each seed, the two assignments have identical configs, sensory
marginals and every initial probe array. This is a two-seed transfer screen,
not the four-seed acceptance test or an embodied result.

`eligibility_media_probe.py` records all eight cellular fields for all neurons
on every tick. For each selected synapse it additionally records actual arriving
information, effective learning rate, final weight and pre/post trace. Complete
initial and trained snapshots include the extension's additional state.
`eligibility_media_audit.py` independently reconstructs these local updates,
checks incoming spike masks against the graph's one-tick cleft delay, and
exports all four previous response measures against both original and current
sound references. No fitted decoder is placed in the brain. Other incoming
weights are checked at episode boundaries, with tested per-tick health
instrumentation; their full update equations are not reconstructed here.

The selected weights change, remain within bounds and support auditory
population activity during silent-video probes. Activity begins at relative
ticks 15-17. It is often dominated by the first 32 ticks, with much sparser
later firing. For example, seed 11 paired cue 0 produces 57 auditory spikes,
34 during ticks 0-31 and 17 during ticks 32-63. It produces no spikes in five
of the subsequent eight windows. Cue 1 produces 110 spikes, 57 in the first
window, with scattered later responses. This is not sustained replay of a
sound sequence. It also does not prove that every transient response is useless.

The centered auditory rate contrast after tick 32 is positive under both
assignments. Its values are 0.010541 versus 0.006601 in seed 11, and 0.013310
versus 0.008808 in seed 23, paired versus reversed, using original auditory
references. Thus the assignment changes a weak population contrast, but does
not reverse the overall response ordering. A full sign reversal would be
strong evidence; its absence alone is not evidence that no association exists.
The recorded per-tick projections change sign within the video. Covariance
measures and reference drift must also be retained, not replaced with the most
favorable score.

Four additional experiments reproduce all acquisition arrays and the complete
trained state, then preserve that state while resetting or exchanging only
the selected weights. `eligibility_media_state.py` retains eligibility traces,
modulators, all other synaptic quantities, and signals in transit. Its unchanged
branches reproduce every original probe array. The independent auditor checks
the complete branch-start state, the declared weight change and every selected
learning update during each probe.

| Seed and training | Intact auditory spikes, cues 0 / 1 | Selected weights reset | Weights from opposite assignment |
| --- | --- | --- | --- |
| 11 paired | 57 / 110 | 0 / 0 | 52 / 84 |
| 11 reversed | 58 / 89 | 0 / 2 | 70 / 112 |
| 23 paired | 65 / 134 | 2 / 6 | 59 / 104 |
| 23 reversed | 74 / 107 | 1 / 7 | 71 / 114 |

The reset removes nearly all visually evoked auditory firing. Selected learned
weights therefore have a causal role in activating the other sensory population.
The reset first changes auditory spikes at ticks 15-17; upper-population
differences follow at ticks 20-26. Weight exchange first changes auditory spikes
at ticks 15-18 and upper spikes at ticks 20-28. Upper rasters differ at more than
10,000 cell-ticks per probe. Coupling carries the intervention upward, but a
large raster difference is not evidence of a useful hierarchical representation.

The content result is weaker and reference-dependent. Weight exchange moves
the centered auditory rate contrast in the assignment-predicted direction in
three of four host networks against their original sound references, and four
of four against their current sound references. Seed 11 reversed is the
disagreement: its original-reference contrast changes from 0.006601 to 0.005546
when given paired weights, but its current-reference contrast rises from
0.007515 to 0.007861. Other response measures do not agree uniformly. This is
evidence worth pursuing, not a demonstrated robust association or grounds to
declare the model incapable of one. Exchanging weights also changes their
strength distribution. A gain-matched, source-assignment-disrupting control is
still needed before interpreting the small directional effect as selective
stored content.

The full-tick learning traces identify a difference from the synthetic
preparation. In seed 11's synthetic paired experience, integrated potentiating
and depressing coefficients on the intended connections sum to 91.31 and
15.70; the unrelated connections receive totals below 4e-7. Intended weights
finish near 0.177, unrelated weights near 0.020. In the real-media paired run,
the per-port median integrated coefficients are 0.09084 and 0.09446. The median
pooled ratio P/(P+D) is 0.488, and 80% of ratios lie between 0.410 and 0.560.
These are summaries of independently reconstructed tick-level coefficients,
not a replacement memory test or a stationary equilibrium claim. Coactivity is
much less segregated. The first real-media exposure recruits virtually every
visual and upper cell in each 32-tick window, though their precise spike times
and rates differ. Recruitment alone does not establish loss of information.

This fits the distinction between stability and competition discussed in
[Guetig et al., 2003](https://brainworks.biologie.uni-freiburg.de/2003/journal%20papers/guetig-jns-2003.pdf).
Their linear weight-dependent rule can stabilize weights while responding weakly
to correlation structure. It does not establish the cause in PAULA. The two
preparations also differ in fan-in, integration, recurrent feedback, exposure
count and sensory overlap. Do not attribute this transfer gap to recurrence,
population size or any one of those differences without a controlled test.

The next test should preserve per-target learned weight distributions while
disrupting which source owns each weight, then replicate promising effects on
new seeds 44 and 77 before changing architecture. A controlled overlap and
coupling series can subsequently locate which composition change destroys
selectivity. Do not demand perfect waveform reconstruction, but do require
experience-specific effects that survive alternative gain explanations and
reach a neural consumer. Adding another upper layer is not a substitute.

All five acquisition/control audits and four state-intervention audits pass.
The largest selected-update reconstruction residual is 1.12e-16. There are
47,712 executed neural ticks, 67 passing focused simulation tests, five passing
eligibility-extension tests and 13 passing bounded/rate tests. Evidence is under
`.live/research/20260909_eligibility_media_*`, with corresponding `audit`,
`state` and `state_audit` names, seeds 11 and 23 and paired/swapped assignments.
The explicit unchanged control is `eligibility_media_control_paired_seed11`.
The records occupy about 1.2 GiB. All workers have exited. No live simulation,
agent version, neuron base class or existing visualization was changed.

### Input-assignment controls weaken the selective-recall interpretation

Four further complete-state experiments preserve every receiving neuron's
selected raw-weight multiset. Three cyclic rotations move each of its four
weights to every other input once across the controls. A fourth intervention
orders the recipient's own weight values according to their ordering after
the opposite audiovisual experience. This rank transfer changes 242 of 768
weights in seed 11 and 265 in seed 23. No other neural state changes. Positive
adaptation continues throughout all probes.

These interventions reuse `eligibility_media_state.py --mode assignment` and
the existing data-only auditor, rather than introduce a different simulator.
The unchanged mode remains available. Full acquisition, including eligibility
ledgers, and the unchanged probes reproduce their source records exactly.
Independent branch validation rejects changes to hidden traces or to a target's
weight distribution. Raw distributions are preserved, but not the actual
time-varying input current: sources have different spike trains and the ports
have different attenuation and delays. Do not describe this as perfect current
matching. The three rotations are a balanced small control set, not a complete
permutation null distribution or a significance test.

The original-reference, population-centered auditory rate contrast after tick
32 is shown below. These numbers are geometric projections, not accuracy or
the percentage of a sound recalled.

| Host network | Intact | Range across three rotations | Opposite rank |
| --- | --- | --- | --- |
| 11 paired | 0.010541 | 0.007191 to 0.009276 | 0.008637 |
| 11 reversed | 0.006601 | 0.003694 to 0.009048 | 0.004639 |
| 23 paired | 0.013310 | 0.010421 to 0.015481 | 0.013470 |
| 23 reversed | 0.008808 | 0.001686 to 0.014239 | 0.011966 |

Opposite-rank transfer moves this contrast in the assignment-predicted
direction in two of four hosts with original references, and three of four
with current references. Full-probe projections give the same respective
direction counts. The other readout measures do not uniformly agree. The
arbitrary rotations retain substantial auditory activity and can increase the
candidate content contrast. For example, seed 23 reversed rises from 0.008808
to 0.014239 under rotation 1, without importing any opposite-assignment rank.
Thus the preceding weight-exchange result is insufficient to establish
selective stored content. It remains valid that learned selected weights
causally enable cross-sensory activation.

Continuing plasticity did not rapidly undo the intervention. Across all 32
non-control probes and every recorded tick, the selected-weight distance from
the corresponding unchanged branch stays between 0.998595 and 1.000111 times
its starting distance. This check uses the complete selected-weight arrays,
not only their final values. The rotations also change neural spike patterns,
so they are active interventions, not ineffective knobs.

The auditor now reports full-probe and onset contrasts in addition to the
previous post-32 measure and all 32-tick windows. Transport exclusion must not
become an assumption that early activity cannot express memory. For example,
seed 11 paired's onset contrast changes from 0.005314 to -0.009012 under rank
transfer, while seed 23 paired's onset contrast is unchanged. A transient can
be useful; neither this disagreement nor weak sustained activity proves that
the brain has learned nothing.

The next experiment should test the learned relationship through matched and
conflicting audiovisual inputs, with exactly the same physical combinations
before training and after both opposite training assignments. It should inspect
all populations and modulatory outputs, plus a selected-weight control, rather
than assume the population named auditory_core must literally reproduce its
sound-only activity. That is a specific reinstatement hypothesis, not a universal
definition of associative memory. An expectation could instead alter sensory
processing, suppress a predicted input, or affect a neural consumer. Require an
experience-specific interaction and subsequently a neural functional use, not
only a mean mismatch response or a fitted classifier. New seeds 44 and 77 are
still required before accepting a positive effect.

Two literature checks constrain this direction. [Wang et al., 2026](https://www.nature.com/articles/s41467-026-70347-w)
models mixed stimulus and prediction-error representations and compares them
with mouse recordings. The main methods derive recurrent dynamics from a
prediction-error and encoding-cost objective, assume linear readouts and
approximately stationary inputs on the neural timescale, and relate learning
to gradient descent plus homeostasis. This is not a demonstration that PAULA's
current local rule implements that objective. Its compartment-local extension
is in the supplement and has not been examined here. The paper motivates
testing functional roles instead of assuming dedicated error-cell identities;
it does not justify importing its optimized weights.

[Cayco-Gajic et al., 2017](https://www.nature.com/articles/s41467-017-01109-y)
distinguishes sparse connectivity, expansion, activity sparsity and decorrelation
in cerebellar-like networks. Its discussion explicitly finds that very sparse
activity is not required for pattern separation, and excessive sparsening can
lose information. The downstream learning test uses a classifier, not a PAULA
brain. This corrects a possible overinterpretation of our broad recruitment
measure. Dense recruitment alone is not an architectural failure criterion.

All four independent audits pass, with selected-update residual below 1.12e-16.
These experiments execute 24,672 neural ticks and retain about 449 MiB under
`.live/research/20260909_eligibility_assignment_{paired,swapped}_seed{11,23}`.
The matching `eligibility_assignment_audit` folders contain all readouts and
tick trajectories. All 68 focused simulation tests pass. No base neuron,
agent version or live visualization changed, and all workers have exited.

### Audiovisual interactions and the limits of the input shuffle

Four complete-state experiments now test every combination of the two physical
videos and two sounds. Each combination starts independently from the initial
network, the trained network, or the trained network with only its 768 selected
visual-to-auditory weights reset. Both opposite training assignments are tested
on graph seeds 11 and 23. Training is replayed exactly, including the selected
eligibility ledger and the complete recorded final state. All probes continue
positive plasticity. The experiments execute 27,072 neural ticks, including
48 probes of 300 ticks each. They are diagnostic, below the four-seed acceptance
requirement and far below a varied-media learning test.

For each neuron, field and tick, the data-only auditor computes
`J = (R01 + R10 - R00 - R11)/2`, followed by `Jpaired - Jswapped`.
All four physical combinations occur with equal weight. Separable video-only
and sound-only response terms cancel; nonlinear stimulus interactions do not.
The initial probes are identical across training assignments, in all eight
recorded fields. The visual and auditory receptor spike rasters also remain
identical across assignments in every corresponding trained and reset probe.
Thus differences in receptor spike timing do not explain the observed response
differences. This does not establish that every receptor field or outgoing
terminal weight is identical.

The auditory interaction differs between experiences from probe tick 10 in
seed 11 and tick 11 in seed 23. These are first differences in a factorial
trace, not isolated pathway transmission latencies. The signed population
trace alternates throughout the clips: seed 11 has 140 positive, 137 negative
and 23 zero ticks; seed 23 has 135 positive, 138 negative and 27 zero ticks.
The whole-probe sums are +22 and +43 weighted spike events. Resetting the
selected weights changes these sums to -16.5 and +2.5, respectively. These
small aggregate effects must be read alongside the retained per-cell traces.
They suggest a contribution of the learned projection to audiovisual response
interactions, not a reliable mismatch signal or sound reconstruction.

The upper population does not yield a consistent interpretation. Its intact
assignment-difference sums are -31.5 and +498.5, and its 32-tick windows change
sign in both seeds. The population named `mismatch_candidate` also disagrees
across seeds, at +2 and -22. Positive values in this convention mean relatively
greater response to combinations inconsistent with each training assignment.
Negative values could instead reflect match enhancement; neither sign alone
defines prediction error. Calling a group a supervisor does not establish that
it detects a learned violation or uses that detection to regulate another group.

An additional causal ambiguity remains. The protocol randomizes visual order
within each two-example repetition, but swapping the associated audio also
changes the audio sequence. Equal marginal exposure counts do not control
order, recency or their learned consequences. Counterbalancing training order
within each fixed graph, and testing an independently varied protocol order,
is required before claiming relationship-specific learning from these effects.
The present experiment also has no delayed outcome, omission condition or
demonstrated neural consumer. It does not locate recall in the upper population.

There is a correction to the preceding shuffle interpretation. Each receiving
neuron's input-weight multiset was preserved by those controls. Consequently,
they preserved the pattern of mean selected input strengths across receiving
neurons. That pattern could itself affect which ensemble responds. Decomposing
the paired-minus-swapped weight matrix into a uniform shift, target-specific
mean shifts and within-target differences assigns the following fractions of
its squared Euclidean norm:

| Graph seed | Uniform shift | Target-specific means | Within-target differences |
| --- | --- | --- | --- |
| 11 | 2.45% | 50.15% | 47.39% |
| 23 | 0.20% | 65.83% | 33.97% |

These percentages describe weight geometry, not memory content. Target-specific
mean input strength is not intrinsic excitability, nor necessarily effective
gain under unequal source activity and dendritic attenuation. The prior shuffles
weaken a within-target input-ranking interpretation; they do not eliminate an
ensemble-recruitment interpretation. The next controlled intervention exchanges
the per-target means, within-target differences, and both together, in otherwise
unchanged trained states. It preserves the distinction between stored parameters,
their dynamical expression and a consumer's functional use.

Both independent audiovisual audits pass, with maximum selected-update residual
1.11e-16. The complete records are under
`.live/research/20260909_audiovisual_expectation_{paired,swapped}_seed{11,23}`;
the two `audiovisual_expectation_audit_seed*` directories contain per-cell,
per-tick interaction arrays and all temporal windows. No base neuron or agent
version changed. The paired/swapped runs are completed; the separately bounded
weight-factor follow-up is recorded below when complete.

### A stored weight difference can be active or spike-silent depending on other weights

The weight-factor follow-up is complete: four runs, 22,272 neural ticks and
32 independent visual-only probes. For each target's four selected inputs,
write the weight row as its mean plus its zero-mean differences. The four
branches retain both original factors, exchange only the mean, exchange only
the differences, or exchange both with the opposite training assignment.
Only those selected information weights change at the branch point. Hidden
traces, modulation, queues and all other recorded state remain unchanged;
plasticity continues. No clipping is used. All transplanted weights lie
between 0.24199 and 0.51735, within the selected rule's [0,1] bounds.

The unchanged probes exactly reproduce acquisition controls. The both-factor
exchange also reproduces every recorded array of the earlier complete-weight
exchange in all eight cue cases. Four independent audits validate the new
branches and selected local updates, with residual below 1.12e-16. The selected
weight distance from the corresponding unchanged branch stays between 0.997643
and 1.005646 times its initial value across all changed conditions and ticks.
Thus continuing adaptation does not erase the interventions during the probes.

The mean-only exchange first changes auditory spikes at ticks 15-18. Its first
change matches the both-factor exchange in all eight cases. Within-target
differences alone first change auditory firing at ticks 16-69 and upper firing
at ticks 23-191. Mean-only changes upper firing at ticks 20-28. Neither temporal
priority nor the earlier weight-space norm fractions establish which factor
stores content. Mean exchange includes both a uniform shift and target-specific
shifts; these must be separated before attributing an effect to ensemble identity.

The auditory original-reference centered contrasts after tick 32 are:

| Host | Intact | Mean only | Within-target only | Both |
| --- | --- | --- | --- | --- |
| 11 paired | 0.010541 | 0.006178 | 0.005726 | 0.006425 |
| 11 reversed | 0.006601 | 0.005217 | 0.006347 | 0.005546 |
| 23 paired | 0.013310 | 0.005772 | 0.013878 | 0.008196 |
| 23 reversed | 0.008808 | 0.014338 | 0.007874 | 0.009205 |

Mean exchange moves this candidate readout in the assignment-predicted direction
in three of four hosts with both original and current sound references. It fails
the reversed seed-11 case. The reference dependence and weak content evidence
remain; this is not an accepted recall repair. The auditor preserves every
probe's spike raster and temporal projections, not only these aggregates.

One complete-state counterexample is more informative than the aggregate scores.
For paired seed 23, visual cue 0, exchanging only within-target differences
changes 34 auditory and 10,388 upper neuron-tick spike flags. The first spike
difference is auditory neuron 446 at probe tick 25. Exchanging the same weight
factor on the opposite-mean background changes **zero spike flags across all
1,152 neurons and all 300 ticks**. Nevertheless, membrane values differ at
127,802 neuron-ticks, starting at tick 8, and the selected weights remain
different. Their L2 distance starts at 0.389068 and ends at 0.388690. All other
seven recorded cellular fields agree exactly between those two branches.

At neuron 446, tick 25, the intact branch has stored potential 0.647417 and
does not fire. The mean-only branch has 0.636842 and does not fire. The
within-target-only branch fires and its stored potential resets to zero.
The both-factor branch has 0.639427 and does not fire. Recorded `r` is
approximately 0.650000 in all four. The post-tick zero does not reveal how far
above the active threshold the firing branch reached. An exact threshold-margin
claim requires a pre-spike observation, including the active cooldown threshold,
not reconstruction from the reset value. Nor does the first divergent spike by
itself prove it causes every later difference; that requires a selective
transmission intervention.

This establishes conditional expression of a parameter difference in this
preparation. It is a concrete example of why testing a learned projection apart
from its operating context can mislead. It is not yet a silent semantic memory,
a demonstrated hierarchical recall function or a novel general law. The next
mechanistic step is to measure the first threshold crossing and test its
downstream causal contribution. A functional recall claim still needs
order-balanced acquisition, additional seeds, varied/partial cues and an actual
neural consumer whose behavior follows the learned relation.

The reusable auditor now records both conditional effects of each weight factor,
with per-field, per-tick changed-cell counts, and full per-cell nonadditivity
arrays. Nonadditivity is `Rboth - Rmean - Rwithin + Rintact`; it is not an
information measure. Records are under
`.live/research/20260909_eligibility_factors_{paired,swapped}_seed{11,23}`.
The `eligibility_factors_audit_v2_*` folders include the conditional comparisons;
the first audit version is retained as an earlier derived analysis. No new
neuron rule, live agent, embodied suite or visualization was started for this
follow-up. The full embodied-brain objective remains unachieved.

Verification after the final auditor extension: 73 focused simulation tests,
five eligibility-extension tests and 13 bounded/rate tests pass. All eight
experiment workers and the derived-analysis workers have exited.

### A single near-threshold release accounts for the observed downstream divergence

The next experiment replays the paired seed-23 acquisition and the four
visual-cue-0 weight-factor probes exactly. A read-only native-frame observer
records neuron 446's integrated membrane state immediately before the firing
decision, including the active threshold, cooldown, current and scalar precision.
It does not replace the neuron equation or modify frame locals. All four observed
controls reproduce every previously recorded array. A fifth branch starts from
the same within-target weight exchange and blocks only neuron 446's outgoing
release at probe tick 25. The neuron still fires, resets and learns normally;
all retrograde events remain. Filtering after native event scheduling also
preserves the random-number draws.

The pre-spike states at the identified tick are:

| Weight condition | Integrated S before reset | Active threshold | Fires |
| --- | --- | --- | --- |
| Original | 0.6474170684814453 | 0.6500000039052308 | No |
| Opposite target means | 0.6368424892425537 | 0.6500000039052308 | No |
| Opposite within-target differences | 0.6500015258789062 | 0.6500000039052308 | Yes |
| Both exchanged | 0.6394268870353699 | 0.6500000039052308 | No |

The firing branch crosses by 1.52197e-6. It is 192 ticks after this cell's
preceding spike, so the active threshold is r, not the cooldown threshold b.
This narrow margin is observed in a case selected for its first divergence.
It does not establish that every neuron or the entire network is fragile, nor
that microscopic spike reproducibility is required for biological computation.

The block removes one tuple release from terminal 900, carrying information
amplitude 1.0104879140853882 and scheduled for absolute tick 3194. That terminal
fans out to 19 incoming ports. The before/after full recorded states differ
only in that queued release. The blocked and unblocked branches agree exactly
through the intervention tick, including the local spike and learning. Thereafter
the block removes all 29,238 later neuron-tick spike differences relative to the
original-weight branch. Only neuron 446's local spike at tick 25 remains different.
In particular, the upper population's 10,388 changed spike flags fall to zero.
This establishes dependence on that transmission over the recorded 300 ticks;
it is not an infinite-horizon claim or a demonstration of meaningful recall.

First differences without the block occur in the activity regulator at tick 27,
the connector at 29 and the upper population at 33. The actual graph contains
446-to-1135 with one dendritic tick, 446-to-706 with three dendritic ticks and
706-to-843/981 with three dendritic ticks. Adding the one-tick synaptic cleft
transit gives timing-compatible paths to those early differences. The whole
release was blocked, so individual edges along these paths have not separately
been proven necessary. Other regions diverge later through the coupled network.

Magnitude of raster change is a poor substitute for functional evidence here.
The original upper population emits 7,823 spikes and the weight-exchanged one
7,803, despite 10,388 differing neuron-tick spike flags. Blocking the release
restores its exact original raster, not merely its spike count. Timing could
still carry useful information; neither the large raster distance nor the small
count difference answers that question. The next functional experiment should
require a neural consumer to use a learned relation under related, perturbed
cues and counterbalanced experience. Do not "repair" the network by suppressing
all divergence or synchronizing it indiscriminately. The target is preserved
functional distinctions under composition, not identical microscopic trajectories.

The independent firing audit initially rejected its own float64 reconstruction
of a float32 update on empty-current ticks. Native I_t starts as Python 0.0,
which NumPy treats as a weak scalar; converting it to a strong np.float64 zero
promotes the reconstruction and changes rounding. The auditor now preserves
that native scalar case, with a regression fixture. The simulation was not
changed and no comparison tolerance was loosened. Integration, active threshold,
cooldown, firing and reset checks pass at every observed tick. The selected
eligibility update residual remains below 1.12e-16.

This experiment executes 4,668 neural ticks and stores its records in
`.live/research/20260909_threshold_release_paired_seed23`. The corresponding
`threshold_release_audit_paired_seed23` folder contains all per-tick effects
and threshold margins. The reusable probe and data-only auditor have tests for
passive-state equality, restoration on errors, exact event-only intervention,
RNG preservation, scalar precision and deliberate corruption. No shared neuron
implementation, live agent or embodied suite was changed or restarted.

Final verification: 78 focused tests pass; the latest five instrument tests
also pass after guarding nonfinite observations and correcting the driver's
returned queued-event count. Those final observation checks pass all 1,500
recorded target-neuron ticks. All experiment and analysis workers have exited.

## Cue completion exposes a conflict between acquisition and expression, 9 September 2026

The 144-neuron synthetic preparation supplies a useful association, but not
an autonomous recurrent memory. New probes replay its acquisition exactly on
seeds 11, 23, 44 and 77 under both opposite audiovisual assignments. There are
24 physical cases per assignment: clean cues, omissions of 25%, 50% and 75%,
replacement of four of sixteen receptors by receptors from the other cue,
balanced eight-plus-eight mixtures, and silence. Each nontrivial corruption
level uses two subsets per cue. These subsets are not independent graph seeds.
Initial-state and selected-association-weight-reset controls accompany each
trained-state probe. All learning rates remain positive.

Every clean or omission probe activates all sixteen associated neural consumer
coordinates and none of the opposite coordinates. Every one of these cells
fires at probe ticks 5, 13, 21 and 29. No consumer spikes occur afterward in
the 64-tick probe. Four supporting receptors therefore suffice for complete
cross-sensory reinstatement, but this is cue-driven expression, not sustained
post-cue activity. All 32 replacement probes instead activate both complete
outputs at those same ticks. Initial and selected-weight-reset controls are
silent. Balanced mixtures activate both outputs without an assigned correct
class. Silence remains quiet.

The distinction is architectural. Learning changes which visual inputs can
drive the auditory population. It does not need an upper population to do
so. At the present operating point, both a weak and a strong matching cue
produce the same spike raster. Another population receiving only that raster
cannot infer the missing strength distinction from it. This is a limitation
of this pathway and probe, not a claim that all internal state or other possible
neural routes carry no additional information.

### A label-blind inhibitory population and its failure

The next preparation has 176 neurons. Thirty-two added PAULA cells receive
all visual receptors with identical unit input weights. Their thresholds span
0.5 to 31.5, so progressively more cells respond as more synchronous receptors
fire. Each projects an initial inhibitory weight of -0.08 to every auditory
cell. Auditory thresholds change from r=0.65, b=0.90 to r=0.25, b=0.50.
There is no cue-specific inhibitory wiring, Python comparison, decoded class,
or learning freeze. The two-hop inhibitory route uses zero dendritic delays;
its current reaches the auditory hillock at the same tick as the original
one-hop visual route with its one-tick dendritic delay. Impulse tests and
recorded inputs verify this alignment before any firing decision.

The gain choices were informed by the preceding learned weights, about 0.177
for paired inputs and 0.02 for unpaired inputs. They are not blind parameter
selection. The intended operation is sensory-dependent subtractive inhibition,
not divisive normalization or a reconstruction of a biological cell type.
[Assisi et al., 2007](https://www.bazhlab.ucsd.edu/wp-content/uploads/2014/04/NNeurosci2007.pdf)
motivate testing feedforward inhibition and inhibitory heterogeneity. Their
conductance-based model adjusts temporal integration windows through changing
inhibitory timing. This preparation does not reproduce that mechanism.

Fresh acquisition with the added circuit retains the learned visual-to-auditory
weights but strengthens active inhibitory inputs to roughly -0.122 to -0.124.
All 1,024 selected association weights are exactly equal to the corresponding
144-cell acquisition in each of the eight seed/assignment comparisons.
The existing cell-wide neuromodulatory learning boost also applies to their
bounded native plasticity. Birth-gain arithmetic therefore does not describe
the trained circuit. In seed 11, paired cue 0, the first four-receptor response
has auditory pre-spike potentials about 0.196 to 0.204, below r=0.25. The
12-plus-4 cue has matching-cell potentials about 0.172 to 0.198. The independent
auditor reconstructs these decisions from actual arrivals, pre-update input
weights, dendritic propagation, float32 integration and cooldown state.

| Trained condition | Clean | Retain 8/16 | Retain 4/16 | Replace 4/16 |
| --- | --- | --- | --- | --- |
| Original 144 cells | 16/16 complete selective | 32/32 complete selective | 32/32 complete selective | 32/32 mixed |
| 176 cells, learned inhibitory gain | 16/16 complete selective | 32/32 complete selective | 32/32 silent | 16 silent, 16 partial selective |
| Same acquired state, only inhibitory q restored | 16/16 complete selective | 32/32 complete selective | 32/32 complete selective | 32/32 complete selective |

Counts describe physical probes across four graph seeds, two assignments and
two cues, with two subsets for each corruption. Full rasters, rather than this
table, establish the result. After inhibitory restoration, all correct consumer
cells fire at ticks 5, 13, 21 and 29, with no opposite spikes in clean, omission
or replacement probes. Balanced mixtures activate both outputs; silence stays
quiet. A replacement case is judged as sensor corruption of one intended cue.
If the physical world really contains two causes, suppressing one is not an
automatically correct behavior.

Removing inhibition entirely makes every clean cue activate both outputs at
the lowered threshold. Restoring the original high threshold while retaining
learned inhibition silences all 50% and 75% omission probes and every replacement
probe. Resetting only the learned visual association weights silences recall.
The improvement therefore requires learned association, suitable excitability
and suitable inhibitory gain together. Neither stronger inhibition nor a lower
threshold alone fixes the preparation.

The inhibitory restoration is an explicitly recorded host intervention. It
restores the 1,024 inhibitory incoming q values to their birth values, retains the learned
association and all other recorded state, then permits ordinary positive
plasticity during the probe. It establishes a causal expression failure, not
an autonomous solution or long-term stability. The next experiment should
test pathway-specific plasticity regulation or a local inhibitory balance rule
through acquisition and continued experience. Simply freezing these weights
or resetting them between experiences would evade the problem.

There is a relevant biological distinction to investigate before the next
extension. [Mitsushima, Sano and Takahashi, 2013](https://www.nature.com/articles/ncomms3760)
report that contextual learning strengthens hippocampal excitatory and inhibitory
synapses through different acetylcholine receptor families. This supports
examining pathway-specific susceptibility to modulation. It does not establish
that inhibitory learning should always be slower, or that removing its
modulatory sensitivity will stabilize this artificial network. The article's
abstract and initial results were checked here; a mechanism-level implementation
still requires reviewing the relevant inhibitory experiments and methods.

### Evidence and limits

There are 24 bounded experiments in these three steps, totaling 202,752 neural
ticks including exact replays and passive-observer controls. Records are under
`.live/research/20260909_association_cues_*`,
`20260909_association_balance_{paired,swapped}_seed*`, and
`20260909_association_balance_rescue_*`. Corresponding audit folders retain
per-tick consumer outputs; balance audits also retain every auditory cell's
pre-spike potential, active threshold and excitatory/inhibitory currents.
All 24 experiment audits pass. The maximum selected-learning residual is
2.23e-16. The balance auditor independently reproduces all auditory soma
decisions and verifies inhibitory delivery, but does not independently
reconstruct every inhibitory neuron's native plasticity update.

The first balance audit implementation repeatedly decompressed NPZ arrays
inside its inner loop. Those eight analysis processes were interrupted and
replaced by a bounded per-trial materialization. No simulation or recorded
evidence changed. Final audits run with eight workers and about 90 MB per
process, rather than accumulating histories across experiments.

This preparation still uses synchronous synthetic receptor masks and a
coordinate-preserving neural consumer. It does not establish learned semantic
readout, real-media recognition, arbitrary temporal tolerance, recurrent
hierarchical states, embodied competence or consciousness. It does identify
a concrete compositional requirement: learning a useful projection must not
silently destroy the excitation/inhibition relationship that lets other
populations use it. No shared neuron implementation, live UI or embodied
agent was changed. The original research objective remains unachieved.

Final verification: all 82 focused tests pass. All 24 experiment audits pass;
the complete physical-case grids were also checked separately after tightening
the balance auditor's case validation. All experiment and analysis workers
have exited. Recorded evidence remains on disk; no live simulation restarted.

## Port-specific modulation removes the manual rescue, 9 September 2026

The next experiment replaces the manual inhibitory-weight restoration with a
local neuron extension present throughout acquisition. `PortModulationNeuron`
inherits the existing eligibility and bounded-plasticity mechanisms. For a
declared native-rule input port, its incoming learning rate is
`eta_basal * (1 + sensitivity * (cell_gain - 1))`. The cell gain still comes
from its existing local neuromodulator concentration. The 1,024 auditory
inhibitory input ports use sensitivity 0.25; the comparison uses 1.0. Basal
adaptation remains positive and unchanged. This is quarter sensitivity to
modulation, not an exact quartering of the entire learning rate.

No thresholds, weights, wiring, inputs or timing parameters are changed from
the 176-cell preparation. There is no runtime reset in the acquisition parent.
The extension only reevaluates the existing bounded incoming weight update at
the port's local rate. Forward propagation, somatic integration, eligibility
learning and outgoing retrograde adaptation keep their inherited rules. Empty
declarations and unit sensitivity follow the parent class exactly. Overriding
an eligibility-rule port is rejected. The new extension adds static receptor
susceptibilities, not a supervisor or an additional dynamical state.

This hypothesis has a biological motivation, with important limits.
[Mitsushima et al., 2013](https://www.nature.com/articles/ncomms3760) distinguish
muscarinic and nicotinic receptor contributions to excitatory and inhibitory
plasticity. The inhibitory results, discussion and recording/drug methods were
read for this follow-up. Inhibitory plasticity also supports learning in that
study, and some effects cross the excitatory/inhibitory distinction. The paper
does not imply that inhibitory learning should be universally weak or absent.
Neither the rate equation nor sensitivity 0.25 is derived from that experiment.

### Acquisition and fourfold continued exposure

Sixteen runs compare the two sensitivities on seeds 11, 23, 44 and 77 under
both opposite learned assignments. Each receives 64 paired presentations per
cue, with diagnostics after 16 and 64 presentations. The initial 32 training
trials of every unit-sensitivity run reproduce all arrays from the preceding
176-cell experiment exactly. The acquisition parent continues through both
checkpoints without any modification from its diagnostic clones.

At both checkpoints, all reduced-sensitivity clean, 25%, 50% and 75% omission,
and 25% replacement probes produce complete selective consumer activity.
Each checkpoint has 16 clean probes and 32 probes at each corruption level,
across the four graph seeds and two assignments. All sixteen corresponding
consumer cells fire at ticks 5, 13, 21 and 29, with no opposite-cell spikes.
Selected-association-weight-reset clones are silent in every case. Silence
controls remain quiet. Balanced mixtures change from silence at the early
checkpoint to both outputs at the later checkpoint; they have no assigned
correct class, but this change shows that the response function is still evolving.

The unit-sensitivity control repeats its early failures and, after extended
exposure, is silent for every visual-only recall case, including clean cues.
At both checkpoints, all 1,024 selected association weights are exactly equal
between the two sensitivities in all eight matched seed/assignment pairs.
The difference therefore does not require better associative acquisition.
It concerns expression of the same learned projection under different
co-adapting inhibitory gain.

In seed 11, paired training, active inhibitory weights have magnitudes
0.08890 to 0.08924 with reduced sensitivity after the first checkpoint and
0.12502 to 0.12550 after the second. They have changed substantially, not
remained frozen. The unit-sensitivity weights reach magnitudes 0.12219 to
0.12407 and then 0.56820 to 0.57993. These differences are independently checked
against every local inhibitory weight update, not inferred from final weights.

### A working interval, not lifelong stability

The extended-exposure result is positive but has a warning in the full-tick
record. Across all eight reduced-sensitivity networks and the replacement
probes, the rejected output's distance below threshold at the four auditory
response ticks shrinks from 0.70931–0.76519 to 0.02841–0.07881. Meanwhile the
matching output's suprathreshold margin increases. The current binary response
passes, but the operating range is changing. There is no proof of stability
under further exposure, temporal jitter, changed input density or additional
associations. Suppressing this warning behind a pass percentage would repeat
the earlier acceptance errors.

This extension supplies an autonomously running correction over the tested
interval. It is not yet a self-adjusting homeostatic controller. The next
experiment must disturb the sensory timing/density and continue learning before
using the mechanism in the larger real-media brain. If a fixed susceptibility
only postpones loss of discrimination, the next intervention should regulate
the excitation/inhibition relationship itself. Do not solve that by freezing
weights or relaxing a meaningful behavioral requirement. Do not add more
hierarchical populations merely because this isolated interval passes.

The 16 experiments execute 294,912 neural ticks. Records are under
`.live/research/20260909_association_continual_{quarter,unit}_{paired,swapped}_seed*`.
The corresponding `association_continual_audit_*` folders retain full consumer
rasters and reconstructed per-neuron preactivation currents and thresholds.
Independent audits check each selected association update, each auditory
inhibitory incoming update, every auditory soma decision, sensory drive,
inhibitory delivery, checkpoint continuity and selected-weight interventions.
All 16 pass. Maximum selected-update residual is 4.45e-16; inhibitory update
residual is zero at recorded precision. Other neurons' native updates are not
independently reconstructed.

Verification passes 84 focused simulation tests and 22 neuron-extension tests.
The four new extension tests cover exact default/unit behavior, positive basal
adaptation without modulation, fractional sensitivity, clone continuity and
invalid declarations. The independent inhibitory ledger test rejects a
deliberately corrupted weight. No shared base neuron or network implementation,
live brain page, V1–V4 agent, or other organism was changed. This remains an
isolated synthetic association, not real-media understanding, recurrent
hierarchy, embodiment or consciousness. The full research objective remains active.

## Temporal composition exposes an information loss, 9 September 2026

The quarter-susceptibility preparation has now been tested with the same
physical receptor identities and doses delivered at different times. Eight
runs cover seeds 11, 23, 44 and 77 under both learned assignments. Each exactly
replays its preceding 128 acquisition trials and both full checkpoint states.
At each checkpoint, 24 physical cue cases receive six schedules: synchronous,
a shared three-tick delay, fixed receptor offsets of zero to three ticks,
offsets resampled each volley, minority-first and majority-first delivery.
Every selected receptor receives four pulses. Receptor spike records verify
that none are lost to cooldown. No weights are frozen or reset in these probes.

The eight runs execute 245,760 ticks, including acquisition, and retain 2,304
probe traces. All synchronous controls reproduce the previous records exactly.
All eight independent timing audits pass. They reconstruct every auditory soma
decision, the selected associative updates and auditory inhibitory updates.
Maximum selected-update residual is 5.56e-16, inhibitory residual zero at
recorded precision. This is audit validity, not an assertion of behavioral
success. Other neurons' native plasticity is not independently reconstructed.

### Do not mistake delayed or early recall for failure

Shared delay preserves selective recall. Spreading a clean cue also preserves
whole-probe complete selective recall at both checkpoints. At the early
checkpoint, most such responses finish before the last receptor's signal
could reach the consumer. A rule requiring a new response after that last
arrival would incorrectly call these probes failures. At the later checkpoint,
all 75%-omitted cues also recall completely over the probe under both spread
schedules, although most have no output after the final receptor's arrival.
The new audit reports the prefix and completed-input intervals separately.
Neither interval alone defines successful temporal perception.

There are genuine changes in the corrupted-cue trajectories. With fixed
spread, 14 of 32 early 25%-replacement probes and all 32 late probes produce
both candidate patterns somewhere in the trace. Resampled spread produces
both in all 32 probes at each checkpoint. Some late probes settle on the
majority pattern after earlier competing activity; others retain both or
produce the minority pattern last. Thus even a whole-probe mixed label loses
important distinctions. The individual timelines remain the primary evidence.

### Opposite evidence, identical output history

A stronger result does not depend on choosing an arbitrary response deadline.
Compare two branches of the exact same acquired state. In the first, twelve
receptors for cue A fire at tick 0 and four for cue B at tick 3. In the second,
four A receptors fire at tick 0 and twelve B receptors at tick 3. Repeat each
physical volley at ticks 8, 16 and 24. The sensory majority is opposite, but
category order is unchanged.

Both produce precisely the same consumer trajectory: all sixteen A-associated
output cells fire at ticks 5, 13, 21 and 29; all sixteen B-associated cells
fire at ticks 8, 16, 24 and 32. This is equality of every coordinate at every
tick for the entire 64-tick probe, not equal totals or similar plots. It holds
for both category orders, two receptor subsets, both checkpoints, all four
seeds and both learned assignments. All 64 matched comparisons have identical
auditory spike outputs and identical eight-field consumer traces. None have
identical eight-field auditory traces. Full internal dynamic states, including
weights, eligibility and queues, are not claimed equal.

The independent `association_channel_audit.py` saves these matched witnesses,
actual receptor schedules, consumer/auditory traces and reconstructed auditory
currents. It rejects mismatched physical doses, changed trace files, and a
disagreement between audited and raw consumer activity. Tests ensure that a
single shifted spike cannot disappear behind equal counts and that identical
outputs do not imply identical internal states.

### Where the distinction disappears

For paired seed 11, early checkpoint, auditory neuron 65 receives net current
1.012354612 at tick 3 in the twelve-A branch and 0.337006390 in the four-A
branch. Its threshold is 0.25. Both inputs therefore produce the same spike
and reset its somatic potential. At the later checkpoint the respective
currents are 4.386104584 and 1.461799622, again producing identical spikes.
The consumer receives the same completed pattern in either case. This example
is supported by the full-neuron reconstruction, not an average current.

The somatic setting here is lambda=1 with dt=1. Between spike decisions its
discrete update reduces to the current arriving on that tick, so this soma
does not accumulate earlier receptor evidence. Its other state variables and
synapses still adapt. Input-dependent inhibition is also driven by the current
receptor volley. Separating the cue therefore creates separately completed
candidates rather than a neural comparison of the accumulated evidence.

For an otherwise identical observer receiving only this fixed consumer output
history, opposite majorities are indistinguishable. Increasing that observer's
size cannot recover a distinction absent from its input. This limited
statement does not apply to an upper population with additional sensory or
inhibitory projections, different release signals, or feedback that changes
the lower computation. No such upper population was simulated in this test.

Nor does the result prove that the brain ought to choose the majority in every
stream. If these are two successive events, recalling A then B is appropriate.
If they are noisy fragments of one event, the original corruption task requires
combining them. Event grouping is therefore part of the missing problem, not
a clock the evaluator may silently supply. Pattern completion usefully removes
some input variation; the failure concerns variation a later task still needs.

### Consequence for the hierarchical architecture

The next construction should separate candidate completion from evidence
integration and test the signals connecting them. A candidate may activate
quickly while a population retains recent supporting and conflicting evidence.
That evidence must reach the integrating population before the distinction is
lost, or through another neural projection. A fixed Python voting window,
decoded confidence scalar, or externally assigned winner would bypass the
research question. Simply making the present consumer persistent could preserve
both saturated candidates without recovering their relative support.

[Chaudhuri et al., 2015](https://www.cns.nyu.edu/wanglab/publications/pdf/chaudhuri_neuron2015.pdf)
provides a relevant structural hypothesis. Its results and model equations
describe temporal hierarchies arising jointly from local excitation gradients
and specific inter-area coupling. The primary model uses threshold-linear
population rates; a nonlinear extension includes NMDA gating. This is not a
PAULA implementation and its physical time constants cannot be assigned to
simulator ticks without calibration. It motivates testing temporal integration
through connected populations, not treating a layer number or a longer soma
constant as proof of a higher-level dynamical regime.

Next distinguish noisy fragments of one event from real A-to-B transitions,
include reversed order and varied spacing, and require a neural consumer to
use the surviving evidence. Compare additional evidence projections against
the saturated-output-only condition before scaling. Keep local learning active
and retain the already working synchronous recall as a regression control.

Records are `20260909_association_timing_*`, `association_timing_audit_*` and
`association_channel_audit_*` under `.live/research/`. The timing workers all
exited successfully. The focused simulation suite passes 87 tests, and three
new channel-audit tests pass separately. A fresh execution of the refactored
continual audit exactly matches the archived seed-11 quarter-sensitivity result.
No neuron-model code, live simulations, V1–V4 agents or other organisms changed
in this follow-up. This establishes a specific temporal interface limitation,
not a repaired hierarchy or progress to mammal-level consciousness.

## A parallel learned population preserves evidence, 9 September 2026

The first structural response to the temporal aliasing result is now running
in one PAULA network with the original association circuit. It adds 192 cells,
six for each of the 32 auditory coordinates, for 368 neurons and 16,192
connections in total. Each added cell has its own initially weak visual
synapses and learns from the same paired sensory experience. There is no copy
of acquired weights, decoded cue identity, externally supplied confidence, or
host decision rule inside the neural loop.

The six cells per coordinate have thresholds 0.04, 0.08, 0.16, 0.32, 0.64 and
1.28. Their somatic integration constant is four ticks rather than one, and
their cooldown threshold is twice their ordinary threshold. They use the
existing PAULA spiking/reset dynamics and the existing positive-basal,
neuromodulated plasticity extensions. No neuron-model code was changed.
This is a deliberately constructed population hypothesis, not a reproduction
of a biological microcircuit or an emergent hierarchy of intrinsic timescales.

Visual and inhibitory input conductance per added cell matches the original
association cell. Population conductance therefore increases sixfold for this
parallel route; it is not a conductance-normalized scale experiment. The audio
afferent weight is eight, rather than 1.5, so its isolated initial somatic
increment is approximately 1.98 at lambda=4 and can recruit every threshold
band during acquisition. This increase is explicit, not attributed to population
size. A new terminal on each contributing source separates the added projection's
native retrograde adaptation from the original projection. Both remain plastic.

The original 176-cell trace and all its recorded association/input arrays match
the previous experiment exactly throughout acquisition. Its synchronous recall
probes also match exactly. Thus the new route has not silently repaired or
damaged the original route. They operate together as parallel populations.

### Completed diverse-population cohort

Eight diverse-population runs, four seeds and both opposite learned assignments,
have completed 158,720 ticks. Each includes 128 acquisition trials and 112
diagnostic probes at the early and late checkpoints. The probes cover clean,
75%-omitted and 25%-replaced cues, silence, four temporal schedules, and genuine
event transitions in both orders with zero, eight or 24 empty ticks between
events. The acquisition parent is never reset by its diagnostic clones.

All eight independent audits pass. The new data-only auditor reconstructs
every added-cell soma decision and all 67 incoming synaptic weight updates,
including the eligibility and native bounded rules. It checks local rate
modulation, neural delivery, temporal continuity and the checkpoint states.
The maximum local weight-update residual is 3.56e-15. Soma reconstruction
matches the actual float32 arithmetic and heap ordering exactly. The eight
recorded cellular fields do not stand in for a complete intracellular state.

All 64 previously ambiguous paired histories now have different evidence-bank
spike trajectories, while their original consumer histories remain identical.
This distinction is more than an arbitrary changed spike: in every one of the
512 ordered-fragment volleys, the stronger cue evokes more bank activity over
the aligned eight-tick volley. Its excess is 16 to 48 spikes at the early
checkpoint and 16 to 32 at the late checkpoint. Those counts describe the
recorded population signal; no neural downstream population has yet been shown
to compute or act on that comparison.

For paired seed 11 at the late checkpoint, twelve A receptors followed by four
B receptors recruit five A threshold bands at ticks 3, 11, 19 and 27, then
three B bands at ticks 6, 14, 22 and 30. Reversing the evidence strengths while
keeping A first recruits fewer A bands and five B bands. Each fully recruited
band contains the sixteen corresponding auditory-coordinate cells. This
preserves a difference the old all-or-none completion output discarded.

The six threshold bands acquired exactly equal selected association weights
within each coordinate at both checkpoints in every seed/assignment. Different
band responses therefore do not require different learned memories. Resetting
only the new population's visual association weights to birth values makes it
silent in all 32 reset probes, with the old memory still present. All 96
event-transition probes activate the new cue and show no old-cue bank firing
after the new cue's first possible neural response. This checks switching,
not learned event segmentation. No boundary signal is sent to the brain.

### Equal-size comparison completed

All eight homogeneous controls have completed. The complete sixteen-run
experiment contains 317,440 ticks and all sixteen independent audits pass.
Both architectures have the same connections, initial weights, teaching
strength and integration constant. In the homogeneous control all six
thresholds equal the geometric midpoint of the diverse range.

At checkpoint 32 neither architecture has an exact output alias in the 32
matched opposite-majority comparisons. At checkpoint 128 the homogeneous bank
aliases all 32 pairs, while the diverse bank aliases none. Homogeneous activity
ties between the opposite cue populations in all 256 late ordered volleys.
The diverse bank retains positive stronger-cue activity margins in all 256.
All visual association weights, not only a selected subset, are exactly equal
between architectures at both checkpoints in every seed and assignment.
This isolates threshold diversity as the changed operating range for this
comparison. It does not establish unlimited range or lifelong stability.

[Wimmer et al., 2015](https://www.nature.com/articles/ncomms7177) is relevant
to the next step. Its results and methods distinguish a relatively linear
sensory circuit from a competing integration circuit, coupled in both
directions. Feedback can stabilize categorization while reducing sensitivity
to later evidence. Their model uses LIF neurons and AMPA/NMDA/GABA conductances;
neither that model nor its hardwired task populations are implemented here.
The useful research question is whether a PAULA consumer can use the new
evidence while remaining able to revise a candidate and recognize a real
transition. An offline spike-count comparison is not that consumer.

Source is `association_evidence_population.py`; independent reconstruction is
`association_evidence_audit.py`. Records are
`20260909_association_evidence_{diverse,homogeneous}_{paired,swapped}_seed*`
and corresponding `association_evidence_audit_*` directories under
`.live/research/`. All sixteen runs and audits are now complete. The focused
regression at that stage passed 95 tests, including five observer, topology and
audit-corruption tests. No live/body simulations or V1–V4 agent configurations
were changed. The full embodied research goal remains active.

## Neural use of learned evidence, 9 September 2026

The next experiment tests an actual neural consumer rather than an offline
count decoder. Each auditory coordinate has a graded PAULA integrator receiving
six individual evidence-cell spike channels through six equal input weights
of 1/6. All 32 integrators project to a graded mean-inhibition cell. Each of
32 ordinary spiking contrast cells receives its own integrator positively and
the common mean negatively. Dendritic delays align the two paths. No cue
identity, majority count or trial clock enters this network.

The integrators use lambda=16. Two controls change either this to lambda=1 or
replace the evidence input with the original saturated completion output,
replicated across six normalized ports. The latter is explicitly a
unit-amplitude spike-channel control. It is not a replay of learned source
terminal amplitudes. Every consumer learns continuously through 128 training
records with positive basal native adaptation. Diagnostic branches preserve
the acquisition parent's state and do not reset it.

This is an existing phenomenological graded extension, not a reconstruction
of cortical conductances. In particular, its non-spiking cells never advance
the native last-spike timestamp, so their inherited timing rule depresses
active inputs. Weak adaptation permits the measured interval; indefinite
maintenance has not been established. The upper wiring preserves auditory
coordinates. It has not learned an abstract association or semantic category.

### What the trajectories show

Eight runs across four seeds and opposite assignments record 476,160 ticks
over the three conditions. All eight independent arithmetic audits pass,
including a repeat with strengthened threshold and source-grid checks. They
reconstruct incoming delivery, graded integration, contrast spiking, native
incoming plasticity and consumer-terminal updates. The maximum weight-update
residual is 1.78e-15. Six corruption checks plus a threshold corruption check
exercise the auditor independently of the simulation's success flags.

The distinction between a preliminary candidate and completed sensory evidence
matters. A receptor input can first affect the contrast output six ticks later.
For each eight-tick volley, the observer reports every response from the last
receptor arrival plus six through the next volley's earliest possible effect.
These intervals describe this protocol. They are not a universal deadline or
an event-boundary signal supplied to the neurons.

The slow evidence consumer has a positive stronger-cue response margin in all
1,024 completed-volley intervals for the corrupted-cue tests. These span two
checkpoints, four timing schedules, four subsets per schedule, four seeds,
opposite assignments and four volleys. Every interval retains its individual
spike counts, raw trace path and alignment. This is not a whole-trial accuracy
score. A fast consumer can respond correctly before these intervals and then
fall silent, which must not be relabeled as failure to recall.

At the late checkpoint, the slow evidence consumer favors the stronger cue in
all 128 majority-first and all 128 minority-first completed intervals. The
fast consumer favors the weaker cue in all 128 majority-first intervals and
the stronger cue in all 128 minority-first intervals. The slow saturated-input
control has the same order dependence. This is a neural demonstration that
both preserved evidence strength and temporal integration contribute here.
Increasing integration time after categorical saturation is insufficient.

There are limits visible in the same traces. In late minority-first trials,
the slow evidence consumer activates the minority at ticks 6, 14, 22 and 30,
before the later majority fragment reaches it. It favors the majority again
after that fragment. At the early checkpoint the corresponding minority
response occurs only at tick 6, or at ticks 6 and 14. The return of repeated
prefix responses after more learning exposes remaining saturation pressure.
This is not an unchanging decision latch or learned event segmentation.

All 128 clean and all 256 omission probes produce the expected population
without competing output. All 128 silence probes and all 32 evidence-weight
reset probes remain silent. The saturated-input control still recalls in its
32 bank-reset probes because it receives the old population, which that
intervention deliberately does not erase.

### Retention and switching are different questions

All 96 real-transition probes switch to the new cue. For back-to-back events,
the slow consumer first responds to the new cue 14 ticks after its first
receptor stimulus, compared with the earliest causal response at six ticks.
The old response continues during that extra interval. An eight- or 24-tick
gap reduces new-response latency to six ticks. No old response recurs after
the first new response in these tests. The fast control switches at six ticks
in all cases but loses the temporal evidence advantage described above.

For example, paired seed 11 at checkpoint 128 changes receptor identity at
tick 32. The old contrast cells still fire at ticks 39, 42 and 45. The new
cells begin at 46 and continue thereafter. This is the cost of this circuit's
retention, not a delayed host command. Several traces end while the new
response is still active; they do not establish its eventual extinction time.

The resulting architectural question is narrower than "add more memory."
What neural evidence should shorten retention when a new event begins, while
allowing temporally fragmented evidence about one event to accumulate? A
supervisor cannot solve this by receiving a hidden experimenter boundary.
The task must make relevant context available through sensory history, action
or other neural projections. If two worlds supply exactly the same accessible
history but require opposite event groupings, no deterministic reader of that
history alone can distinguish them. More population layers do not remove the
missing information.

`evidence_consumer_comparison.py` supplies the trace-linked observer. Its
`20260909_evidence_consumer_comparison/observations.json` retains every probe's
spike times, completed-volley intervals and transition carryover separately.
The full focused regression now passes 110 tests. Direct connection inside one
433-cell PAULA network is tested separately below. Isolated replay alone does
not establish that result.

### Direct neural connection, completed pilot

`evidence_embedded.py` constructs one 433-cell PAULA network. The original
368 cells retain their classes and incoming wiring. The additional 65 cells
use the existing graded extension, with its ordinary spiking path for contrast
cells. There are 288 additional connections. Actual source terminals now feed
the integrators and receive their native retrograde signals. Only the original
visual and auditory receptor ports remain external. No new neuron-model code
or host neural controller was introduced.

The paired seed-11 pilot completed 19,840 ticks in 535.33 seconds. Every recorded
lower-population field and incoming weight update matches the independently
recorded source experiment exactly, including all 128 training trials and 112
probes. This comparison deliberately excludes the newly connected outgoing
terminals, which now adapt. At the end of training their amplitudes range from
0.99254805 to 0.99256945 instead of the isolated bank's unit release.

That return adaptation changes the consumer's continuous state. Across the
56 early probes, the largest integrator-potential difference from unit-amplitude
replay is 0.00023844. Across the 56 late probes it is 0.00144552. Nevertheless,
every contrast-cell spike bit matches the isolated consumer at both checkpoints.
All earlier retention and switching observations therefore also hold for this
one directly connected pilot. The source-terminal change is neither absent nor
large enough to change these observed decisions.

`evidence_embedded_audit.py` independently reconstructs source-terminal return
adaptation, the released boundary amplitudes and the consumer's dynamics and
plasticity. It checks every lower record against the previously audited source.
The complete pilot audit passes. Recorder-passivity and corrupted-terminal
tests pass as well. This is a measured example of composition surviving a
plastic boundary, not evidence that arbitrary PAULA modules compose safely.

The seven further replications have completed, covering four seeds and both opposite
assignments with at most four simulation workers. All eight independent audits
pass. Across all 896 probes, every contrast-cell spike bit matches its isolated
slow-consumer counterpart. The largest integrator-potential difference across
the cohort is 0.00150212. Source terminal amplitudes after acquisition range
from approximately 0.99254626 to 0.99257124 across the cohort. The trace-linked
comparison is `20260909_evidence_embedded_comparison/comparison.json`. This
preparation has no body, no learned upper association and no descending
somatic regulation. Native retrograde adaptation should not be conflated with
the proposed hierarchical supervisor.

## Returning to real sensory input, 9 September 2026

`media_drive_audit.py` now reconstructs the actual visual and auditory receptor
dynamics in all four earlier eligibility-media runs, covering seeds 11 and 23
and both assignments. It independently applies the recorded physical features,
four-phase sensor sampling, one-tick dendritic delay and attenuation, somatic
threshold/reset and local bounded weight update. Recorded incoming weights are
checked before and after every episode. Each output retains the physical
feature, applied input, threshold margin and individual receptor spike at
every tick, alongside all downstream population rasters.

The neutral sound-only probe for recording 0 produces 134 receptor spikes,
while recording 1 produces 3,037. Visual-only probes produce 5,631 and 6,321.
These counts identify unequal drive for inspection; they do not define memory
content or establish a failure. The original recordings include background and
speech, and should not be treated as clean semantic dog/bark and cat/chirp
labels. The source media and transfer functions remain unchanged.

### An exact sensory-output alias under a physical perturbation

`sensory_population_probe.py` tests the first sensory population on those real
inputs. Thirty-six conditions combine the two recordings, separate vision and
sound, gains of 0.5, 1 and 2, and three population constructions. The baseline
has 192 receptors. The homogeneous and diverse banks have six cells per
physical receptor, or 1,152 cells. Homogeneous thresholds are all 0.6; diverse
thresholds are 0.1, 0.2, 0.4, 0.8, 1.6 and 3.2. Incoming parameters, physical
sampling phase and positive local adaptation are otherwise unchanged. Both
expanded populations receive six times the baseline total afferent conductance.
The diversity comparison is equal-size and equal-conductance. No downstream
consumer receives an externally computed population count.

Visual gain changes pixel luminance before the existing darkness transfer.
The first three unavailable frames remain unavailable. Audio gain changes
amplitude through an additive dB shift before the fixed band thresholds.
Neither operation normalizes against clip statistics or uses future samples.
These are physical perturbations, not reassignments of the learned category.

At half luminance, both videos drive the original visual receptors above
threshold at every available sampling opportunity. Across the full 300 ticks,
the two videos produce exactly equal spike histories and all eight recorded
cellular fields. The six equal-threshold copies produce the same equality.
Their incoming weights differ, so this is not equality of complete neural
state. It is an exact loss of distinction in the measured transmitted channel
over this interval.

The diverse bank first distinguishes the two half-luminance videos at tick 4
and differs at 3,756 cell-ticks. This is a finite dynamic-range result, not
semantic recognition or invariance to lighting. It does not require a new
neuron equation, and duplication alone does not reproduce it. The explicit
unchanged baseline and equal-threshold controls reproduce the original neutral
receptor traces exactly.

All 10,800 experimental ticks are complete. The data-only auditor reconstructs
every cell and incoming weight update in all 36 conditions, including each
physical gain transformation. Its largest weight residual is 9.77e-15. The
comparison retains every differing-cell count by tick plus the full binary
difference arrays. Evidence is in `20260909_sensory_population_media`,
`20260909_sensory_population_media_audit` and
`20260909_sensory_population_media_comparison`. Seven-threshold unit tests
also reject shifted physical drive and corrupted soma traces.

[Herzog et al., 2026](https://pubmed.ncbi.nlm.nih.gov/41611678/) reports
heterogeneous luminance, contrast and timing responses at cone outputs, with
horizontal-cell feedback contributing to that diversity. Its abstract and
indexed introduction were checked here; the complete methods were not
accessible in this pass. That biological result motivates investigating early
sensory dynamic range, but does not validate these PAULA thresholds or imply
that this circuit implements horizontal-cell feedback.

### Full-graph graded sensory release, completed transfer screen

The next transfer preserves the 1,152-cell audiovisual graph and its existing
learning pathways. Two runs compare unchanged spiking receptors with the
existing graded-release mechanism on the 192 sensory neurons. A fixed release
gain of 0.25 maps the initial physical transfer to approximately 0.99 times
intensity. No clip-dependent normalization or learned semantic representation
is inserted. This is an alternative to threshold recruitment for exposing
unsaturated evidence, not a claim that the diverse bank has already been
integrated into that graph.

`GradedEligibilityNeuron` composes two existing extension paths without adding
an equation. It preserves ordinary eligibility behavior when grading is off,
keeps bounded native adaptation active when grading is on, and rejects
spike-based eligibility ports on graded cells. Its non-spiking cells retain
the negative native timing direction. This limitation is explicit, not hidden
behind a neuron name. All 28 focused neuron-extension tests pass.

`graded_media_probe.py` records full acquisition and before/after single-sense
probes at all three physical gains. The spiking control must reproduce the
earlier acquisition and neutral probes exactly. Both runs completed under
`20260909_graded_media_{spiking,graded}_paired_seed11`, each recording 10,368
ticks. The spiking control reproduces every prior acquisition array and every
neutral probe array exactly. Both independent `graded_media_audit` analyses
pass, with maximum selected-learning residual 1.11e-16. They reconstruct sensory
somata, release before terminal scaling, sensory incoming adaptation and the
selected cross-sensory updates. They do not reconstruct all outgoing terminal
amplitudes or downstream currents because those histories were not recorded.

The full focused simulation regression passes 121 tests in 122.33 seconds.
No live or embodied simulations were started during this continuation.

### Better sensory distinction does not yet transfer into auditory recall

With graded receptors, the two half-luminance videos produce different visual
receptor outputs from tick 4, visual-core outputs from tick 6 and initial upper
population outputs from tick 12. The distinction survives the full graph. These
are differing neural trajectories, not evidence of semantic recognition or a
learned higher-level abstraction.

At original brightness, both trained silent-video probes produce zero auditory
spikes throughout all 300 ticks. Clip 0 comes closest to threshold at tick 20,
auditory coordinate 121, with soma minus threshold -0.06448384. Clip 1 comes
closest at tick 23, coordinate 94, with margin -0.02237003. Coordinates and probe
ticks are zero-based. The threshold ranges from 0.65 to 0.6501462, so a large
threshold-regulation excursion does not explain this silence. Actual auditory
input still drives these cells. At half luminance, a few auditory spikes appear,
6 and 11 over the two trained probes. That brightness dependence is not a
robust memory result.

The graded variant changes both the timing and magnitude of sensory drive.
Matching its peak release scale to the old spiking output does not match mean
drive or total current. Its selected learned weights are weaker, with median
0.31342 versus 0.33918 in the spiking control. That difference alone cannot
identify whether recall fails during acquisition, expression or their coupling.
The receptors also remain non-spiking during native adaptation, so this is not
a pure amplitude-only intervention.

The next experiment, `graded_recall_factors.py`, reconstructs the trained parent
through exact acquisition replay, then makes twelve cloned probes. It crosses
three selected-weight states, graded-trained, spiking-trained and initial, with
two visual release rules and two clips. It preserves all other graded-trained
state, including eligibility and signals in flight. Plasticity remains positive.
No-op branches must reproduce their recorded probes exactly. The independent
selected-update equation and prior-modulator learning-rate check run on every
branch. Donor weights and release switches are explicit experimental
interventions, not autonomous neural operations. Even a restoration of spikes
will require an experience-content test before it can count as associative
recall.

The storage/retrieval distinction has a biological experimental precedent.
[Roy et al., 2017](https://pubmed.ncbi.nlm.nih.gov/29078397/) reports that
optogenetic stimulation could retrieve memories inaccessible to natural cues
in a mouse amnesia preparation. Disrupting downstream engram connectivity
prevented that experimentally driven recall. The abstract and PubMed figure
legends were checked; the full PMC methods were inaccessible in this pass.
This motivates separating content retention, cue access and downstream
expression. It does not establish that silent PAULA cells contain an engram.
Our preparation still needs evidence that its induced activity depends on
which sound was associated with the visual input.

### Completed causal factor test, 6,768 ticks

The twelve branches completed in 401.80 seconds after exact acquisition replay.
Every no-op probe array matches the preceding graded recording, and the trained
parent remains unchanged. The selected-learning reconstruction residual is at
most 1.11e-16. The following spike totals are an index into the saved per-tick
records, not an acceptance score. The two clips each last 300 ticks.

| Selected association weights | Visual release during probe | Auditory spikes, clips 0 / 1 | First auditory spike ticks, clips 0 / 1 |
| --- | --- | --- | --- |
| Graded-trained | Graded | 0 / 0 | None / None |
| Graded-trained | Spiking | 20 / 34 | 18 / 20 |
| Spiking-trained donor | Graded | 1 / 5 | 20 / 18 |
| Spiking-trained donor | Spiking | 135 / 197 | 15 / 17 |
| Initial selected weights | Graded | 0 / 0 | None / None |
| Initial selected weights | Spiking | 5 / 1 | 18 / 24 |

The graded-trained network is not incapable of expressing auditory activity.
Changing visual release alone restores some activity, and resetting the
selected association weights removes much of that response. Donor weights
alone restore only isolated onset spikes. Together, donor weights and spiking
cue release produce a much larger response. Thus the weights and cue delivery
jointly control access to auditory activity in this preparation. No added upper
population was necessary to demonstrate that effect.

The actual trajectories prevent a stronger recall claim. With graded-trained
weights and spiking cues, clip 0 produces spikes on ticks 18 through 26, 28,
38, 239 and 251. Clip 1 has an onset burst from ticks 20 through 29 and sparse
later events. With donor weights but graded release, every auditory spike is
before tick 24, with none thereafter. These are not sustained reinstatements.
Against the fixed graded-parent auditory reference, the post-tick-32 centered
projection is 0.00425 for graded-trained weights plus spiking cues and 0.01922
for donor weights plus spiking cues. These numbers are normalized projection
coefficients, not percentages or accuracy. Their 32-tick windows change sign.
They do not establish that the associated sound, rather than general excitation,
is being reconstructed.

Evidence is in `20260909_graded_recall_factors_paired_seed11` and
`20260909_graded_recall_factors_comparison_paired_seed11`. The independent
comparison checks declared recorded-state changes and retains full per-cell
S-r trajectories and per-tick sound-reference projections. It cannot infer
pre-reset firing margins from post-spike S, or certify hidden fields absent
from snapshots. The actual simulation branches use deep copies and exact replay,
not JSON restoration. Weak positive adaptation remains active throughout.

The next discriminating test is an opposite-assignment and matched-strength
intervention. It must ask whether changing the learned relationship changes
which auditory state the same visual cue evokes. A larger spike count or a
new oscillation would not answer that question. This one-seed result supports
a causal acquisition/expression diagnosis, not associative-memory acceptance,
embodiment or a consciousness claim.

### Counterbalanced learning histories expose a temporal interpretation problem

`media_order_control.py` crosses audiovisual assignment with two repeated
presentation orders. All four histories share one graph seed, 11, and the same
configuration and physical recordings. Each recording occurs four times, with
96 ticks of withdrawal after every 300-tick exposure. Across the two orders,
paired and swapped assignments have identical visual and auditory sequence
multisets. This controls those marginal sequences and final-item identity, but
does not remove nonlinear interactions between the joint histories. It is not
a four-seed replication.

Each trained state is probed with native graded visual input, diagnostic spiking
visual release, spiking visual release after resetting only selected association
weights, and sound-only input. Initial visual and auditory controls are also
recorded. Every probe starts from a separate full-state copy. The four runs
complete 27,072 ticks and all parents remain unchanged. The independent audit
reconstructs sensory acquisition and selected learning, with maximum selected
update residual 1.11e-16. It also checks branch starts and exact initial outputs.

| Learning history | Trained spiking-cue auditory spikes, clips 0 / 1 | Selected-weight reset, clips 0 / 1 |
| --- | --- | --- |
| Paired, order 01 | 28 / 56 | 2 / 2 |
| Paired, order 10 | 24 / 34 | 6 / 1 |
| Swapped, order 01 | 15 / 15 | 7 / 3 |
| Swapped, order 10 | 21 / 14 | 9 / 2 |

These totals index the trajectories, not accuracy. Most auditory activity occurs
in the first 32 ticks. The native graded probes remain silent except for two
onset spikes in one clip under one history. Learning changes recruitment, and
the selected pathway contributes, but the spatial sound-reference projection
does not reliably reverse when the learned pairing reverses. The auditor now
separates raw, common-mode and nonuniform spatial projections. Common-mode
recruitment could carry intensity information; removing it from analysis would
silently impose a particular code. Its presence alone does not establish an
associated sound ensemble either.

After counterbalancing order, the auditory spatial assignment-effect trace has
30 positive, 14 negative and 256 zero ticks. Its onset mean is -0.01340, post-32
mean is +0.001712, and whole-probe mean is +0.00009984. These are projection
coefficients, not probabilities. Selecting only the post-onset interval would
give an optimistic interpretation of an effect dominated by sparse events and
opposing phases. The upper population has 145 positive and 144 negative ticks,
with 11 zeros. The result supports neither stable sound reconstruction nor an
upper-layer memory interpretation.

Records are `20260909_media_order_{paired,swapped}_order{0,1}_seed11`. The
full-timing audit is `20260909_media_order_audit_full_timing_seed11`; the earlier
audit summary lacks explicit onset/full fields but retains the underlying
traces. Both complete. A new test rejects an interpretation that hides a
negative onset behind a positive post-onset average. No neural learning rule
was changed by these controls.

### A downstream weight changes upstream transmission before its neuron fires

The trained-versus-reset comparison revealed a return-path effect. Under all
four histories, selected incoming amplitudes diverge at tick 11 or 12, before
auditory spike outputs diverge at ticks 18 through 22. Visual receptor soma and
output histories remain identical. This prompted a targeted causal test rather
than treating the component graph as a sequence of spike-only transformations.

`media_backchannel_probe.py` exactly replays the paired/order01 acquisition and
branches it four ways. Learned versus initial selected input weights are crossed
with intact versus interrupted retrograde events from those same synapses.
The lesion removes only those return events after local computation. It does
not remove forward connections, zero learning rates or remove unrelated feedback.
Each branch records a 64-tick cue prefix, every cell's usual fields, every output
terminal's information and modulation, and the selected return events. The
observer's intact prefixes match the original records exactly.

| First consequence of changing selected weights | Intact returns | Selected returns cut |
| --- | --- | --- |
| Auditory soma S differs | 8 | 8 |
| Visual terminal information differs | 10 | 28 |
| Selected incoming amplitudes differ | 12 | 27 |
| Visual soma S differs | 14 | 21 |
| Auditory spike output differs | 18 | 18 |

These are first divergences in this comparison, not universal conduction times.
The lesion confirms that native return signaling carries the early effect back
to visual transmission before the changed auditory spike response. Later
forward and recurrent routes remain. This does not show that the early effect
is beneficial, harmful, or sufficient for memory, and it is not a proposal to
disable it.

The graph gives each of its 190 contributing visual source neurons a single
output terminal shared by 9 through 29 downstream connections. A terminal change
therefore affects more than the dendrite that sent the error. That is an
explicit architectural coupling to investigate, not evidence that independently
tested populations retain their former transfer functions when composed.

The experiment completes 3,424 ticks in 185.09 seconds. Its records and complete
difference trajectories are `20260909_media_backchannel_paired_order0_seed11`
and `20260909_media_backchannel_audit_paired_order0_seed11`. The audit validates
the selected routing and observed consequences; it does not independently
reconstruct each return-event equation. The recorder's lesion and exception-safe
restoration have tests. The latest focused suite passes 16 tests in 0.95 seconds.
The preceding broader regression passed 127 tests before these last additions.

### Literature implications for the next architecture

[Garner and Keller, 2022](https://www.nature.com/articles/s41593-021-00974-7)
reports learned, stimulus-specific suppression through auditory-to-visual cortical
interactions. Its main results and experimental design support testing predictive
effects on later sensory processing rather than requiring every association to
replay a sound-only pattern. That is a different hypothesis, requiring matched
and conflicting input probes and a functional consequence.

[Audio-visual experience strengthens multisensory assemblies, 2019](https://www.nature.com/articles/s41467-019-13607-2)
reports strengthened functional multisensory associations. The computational
methods, equations 1 through 6, were reviewed here. Their model uses rate units,
dense recurrent connectivity, assigned stimulus preferences and a developmental
phase. It freezes inhibitory plasticity and potentiation thresholds during
pairing. Those assumptions cannot be silently transferred into this project as
proof of raw-media representation learning or continuously adapting PAULA.

[Tsukano et al., 2026](https://www.nature.com/articles/s41593-026-02217-z)
reports that orbitofrontal input suppresses anticipated auditory responses via
inhibitory neurons, and that inactivation reverses habituation. Its abstract and
initial results were reviewed, not the complete methods. This supports a
functional, intervention-based test for a proposed supervisor. Naming a PAULA
population a supervisor, or observing a decrease in activity, does not establish
such predictive control.

### Connection-specific native returns do not repair this real-media transfer

The existing `projection_terminals` library now also exposes `contact_terminals`
and `mean_pooled_return_rates`. These are build-time alternatives. No neuron
equations, media inputs, connection weights or behavioral controller change.
The contact alternative gives every outgoing connection of all 192 visual-core
neurons its own native output terminal. The network still has 1,152 neurons and
21,760 connections; terminals increase from 1,152 to 4,248. Initial release and
distance on each edge remain equal. Somatic dynamics, modulation and recurrent
paths remain shared. This is connection-specific adaptation, not independent
neurons or isolated modules.

The second alternative retains shared terminals but divides each selected
source's positive return-learning rate by its outgoing connection count. Under
equal error streams and unclipped linear updates, this approximates the mean
contact-terminal update. Future closed-loop activity and rates can diverge, so
it is not an exact dynamic dose match. It tests whether slower shared adaptation
can explain an apparent benefit from terminal splitting.

This is a finer version of the earlier family-terminal intervention, which did
not reliably repair recall. [Reyes et al., 1998](https://pubmed.ncbi.nlm.nih.gov/10195160/)
reports target-specific facilitation and depression along a single axon. Its
abstract was checked again. That motivates local terminal differences; it does
not establish the current long-term PAULA update rule or perfectly independent
biological terminals.

`contact_terminal_transfer.py` prepares explicit configuration inputs, not
fabricated acquisition records. It reuses `media_order_control.py` unchanged.
Eight fresh runs cross the two architectures with paired/swapped experience and
orders 01/10, all on graph seed 11. Each completes 6,768 ticks, for 54,144 new
ticks. Four workers execute at a time. Contact runs take 416.43 through 421.35
seconds; mean-pooled runs take 409.27 through 413.82 seconds. Concurrent observer
work differs, so these durations are not a controlled performance benchmark.

Both full-tick cohort audits pass, with maximum selected-update residual
1.11e-16. `contact_terminal_audit.py` independently checks actual configurations
against the original graph without invoking the builders. All twelve histories,
including the preceding shared-terminal controls, have identical recorded birth
state except the declared terminal copies. Initial per-edge terminal state
also matches. Initial sound-only auditory and upper spike histories are equal
across architectures, so the sound-reference axes used here are equal. Full
cross-architecture output differences and recruitment trajectories are retained.

| Architecture | Graded-cue auditory spikes across eight cue probes | Spatial assignment projection, onset / post-32 / full mean | Positive / negative / zero ticks |
| --- | --- | --- | --- |
| Original shared | 2, both in one probe's onset | -0.013404 / +0.001712 / +0.000100 | 30 / 14 / 256 |
| One terminal per connection | 1, in one probe's onset | -0.002707 / +0.000047 / -0.000247 | 10 / 17 / 273 |
| Slower mean-pooled returns | 2, both in one probe's onset | +0.004165 / +0.000386 / +0.000789 | 21 / 17 / 262 |

The projections refer to diagnostic spiking-cue probes, not the almost-silent
graded probes in the second column. They are observer coefficients, not accuracy.
All individual auditory whole-probe spatial cue contrasts remain positive even
when training assignment reverses. The upper population remains active, but its
assignment-effect projections change sign across time. Contact and mean-pooled
upper traces each have 141 positive, 148 negative and 11 zero ticks. A favorable
mean does not establish a stable learned state. Raw and common-mode observers,
selected-weight resets, all cell outputs and all windows remain in the audits.

Thus neither tested alternative repairs the demonstrated recall limitation.
This does not exclude associative information in another code, and it does not
prove that terminal separation is unnecessary elsewhere. It rules out adopting
this particular change as a sufficient repair. These are one-seed architecture
screens, not accepted four-seed memory components or embodied results.

Records are `20260909_{contact,mean_pooled}_media_order_{paired,swapped}_order{0,1}_seed11`.
Cohort audits are `20260909_{contact,mean_pooled}_media_order_audit_seed11`, and
the cross-architecture record is `20260909_terminal_architecture_comparison_seed11`.
Prepared inputs are `20260909_{contact,mean_pooled}_terminal_source_seed11`.
All are under `.live/research/`. Worker session 69121 has exited successfully.

### A local learning clock, distinct from elapsed simulation time

`eligibility_exposure_audit.py` decomposes the actual selected weight trajectory
as `q(t) = A(t) q(0) + B(t)`. For the observed local potentiating and depressing
coefficients, `A(t) = exp(-sum eta*(Lplus+Lminus))`. The existing rule's affine
form was already documented; this audit measures its cumulative exposure in the
real-media runs and saves A and B for every selected synapse at every tick.
It reconstructs all twelve histories within 1.50e-15 of recorded weights.

The final median A is 0.972 through 0.987 in the original shared runs, 0.973
through 0.984 with contact terminals, and 0.971 through 0.985 with mean-pooled
returns. This is a conditional initial-weight coefficient, not a percentage of
memory or an assertion that the brain has learned almost nothing. Some synapses
experience much stronger changes, and small changes can alter neural trajectories.
The decomposition holds the observed activity and rate histories as data. It
is not a closed-loop sensitivity calculation: a weight intervention can change
those very histories.

Equal presentation times are not equal learning exposures. Summing the local
`eta*(Lplus+Lminus)` over selected ports and audiovisual exposure ticks shows
that audio clip 1 accounts for 92.8 through 98.4 percent across the twelve
histories. Withdrawal ticks are excluded from that percentage but retained in
the records and decomposition. The original four shared histories span 93.9
through 98.1 percent. This does not isolate loudness as the cause; sensory
encoding, postsynaptic recruitment, timing, modulation and recurrent feedback
all contribute to the measured local opportunities.

Consequently, four presentations cannot be treated as equal effective training
for both relationships, or as evidence of a settled learning regime. This also
does not prove that more exposure will repair recall. The next acquisition test
should use a declared exposure-duration series and local learning trajectories,
with assignment/order controls. A separate cue-plus-sound test should ask whether
learned context changes subsequent processing when the receiving population is
driven. These distinguish incomplete acquisition, cue access and predictive use
without demanding perfect replay of a sound-only raster or adding an untested
upper layer.

Decompositions are `20260909_media_exposure_{paired,swapped}_order{0,1}_seed11`
and corresponding `20260909_{contact,mean_pooled}_media_exposure_*` directories.
Positive adaptation remains active in every original simulation; the audit is
data-only. Its tests cover absent eligible events despite a positive rate, small
updates and varying local drive.

### Native return information updates are now independently reconstructed

`native_return_audit.py` closes part of the earlier backchannel audit gap. It
reconstructs every delivered selected output-information update in the two
intact 64-tick branches, 11,259 learned-state and 11,257 reset-state events.
All 22,516 match exactly. The calculation uses the preceding completed cell
modulation and float32 native error/terminal arithmetic. `Network.run_tick`
delivers returns before ticking neurons. Using current-tick modulation instead
changes 4,697 and 4,615 predictions respectively, with maximum discrepancies
around 1.2e-5. Using float64 throughout also fails bitwise reconstruction.

The audit takes recorded error vectors as inputs. Their generation and per-event
terminal modulation updates remain outside its independent coverage. It does
not establish useful memory. The reconstruction is
`20260909_native_return_update_audit_seed11`; no new neural simulation was needed.

The broader association/media/population regression passes 136 tests in 150.90
seconds. A subsequent focused suite passes seven tests, including the newly
added birth-state fixture. The implementation adds reusable configuration and
data-audit modules; it does not change the shared base neuron, V1-V4, their live
servers or the visualization page.
