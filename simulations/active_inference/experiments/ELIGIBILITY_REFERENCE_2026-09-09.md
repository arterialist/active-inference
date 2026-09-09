# Neural reference for plasticity, separate from somatic inhibition

## Question and model change

The completed local-competition courses reduced shared prediction bias while
also weakening the joint audiovisual component. They did not establish retained
association. This candidate asks whether a population can influence which
synapses are rewritten without directly inhibiting the representation being
expressed. It is not another inhibition-strength sweep or an accepted memory
repair.

`neuron/extensions/experimental/contrast_eligibility.py` adds an explicit
subclass of the current magnitude-feedback predictor. The inherited local
context trace is `x_i <- d*x_i + (1-d)*arrival_i`. A separate neural reference
trace is `z_r <- d*z_r + (1-d)*reference_arrival_r`. For the selected prediction
ports only, the update becomes:

```
q_i <- clip(q_i + eta(e_previous) * e_previous * (x_i - g*z_r), 0, cap)
```

The error, positive basal rate, bounded error-dependent amplification and
context timescale remain as before. The change allows negative eligibility:
activity below the neural reference can reverse a synapse's update even when
the neuron-wide error has the same sign. It does not require a negative release
amplitude, a host-computed error or a negative information weight.

The references arrive on declared zero-throughput receptor ports. They do not
directly contribute to membrane integration. Forward potentials and returns
use pre-update weights in the inherited order; only selected information
weights are replaced after that tick. With no reference mapping, the inherited
path is exact. With a mapping but strength zero, the same added neural circuit
remains and the old learning rule is used. Existing classes and `neuron.py`
are unchanged.

This is a phenomenological eligibility extension, not a faithful OLM cell or a
claim that the original PAULA equations already implemented it. The primary
abstract of [Sejnowski, 1977](https://pubmed.ncbi.nlm.nih.gov/925522/) motivates
distinguishing covariance-dependent storage from raw activation. Its nonlinear
model and proposed plasticity are not reproduced here. The current spatial
reference is not an estimate of statistical covariance across experiences.

## Neural construction and limits

`components/learning/eligibility_reference.py` adds one graded reference cell
for each 192-cell mixed bank, bringing this preparation from 590 to 592 cells.
Each reference receives all members of its bank. Birth input weights total
`1/.99`, compensating the declared one-tick dendritic attenuation of `.99`.
Its membrane time constant is two ticks. It projects to both predictive cells
through zero-throughput reference receptors; each selected input uses the
reference belonging to its actual source bank. Source membership is anatomy,
not a stimulus label. No external input is added.

This is a delayed reference, not instantaneous normalization. The extra
synaptic path and integration remain in the simulation. Unequal feature
statistics, reference lag, lower weight clipping and subsequent adaptation can
all prevent the intended cancellation. Subtracting one spatial reference does
not mathematically remove the factorial shared-response component measured
across four different audiovisual episodes.

Native return paths remain active. The added reference cells can therefore
alter upstream terminals even without direct somatic inhibition. The wired
control has exactly the same extra paths. Equality of instantaneous forward
output at fixed weights is not equality of future coupled trajectories.

## Verification completed before interpretation

Fifty-one focused tests pass, including eight reference-pathway tests and three
sensory-substitution/trajectory tests.
They establish exact inherited trajectories with contrast disabled, opposite
weight changes for above/below-reference inputs under a common error, and
unchanged current somatic output for the matched fixed-weight intervention.
Host-driven reference receptors, direct somatic throughput and invalid bank
construction are rejected.

The embodied recorder reproduces the unobserved run exactly. A separate
auditor reconstructs reference traces and selected weight updates from actual
arrivals, rather than trusting reported eligibility. It also checks reference
arrivals against source-cell release and terminal values. Deliberate corruption
of weights, rates, reference arrivals, reference traces and eligibility is
detected. An executable checkpoint reproduces continuation with reference
traces and signals in flight exactly.

These tests establish the implemented mechanism and observation path, not
learned regulation, retained association, sensory semantics or consciousness.

## Completed bounded embodied screen

`eligibility_reference_acquisition.py` uses the previous magnitude-feedback
birth graphs and the same real audiovisual feature recordings. Every seed must
first reproduce a full old 364-tick body episode exactly with the extension
disabled. Each of three conditions then runs a continuous brain/body history:

| Condition | Reference contrast strength | Basal selected rate |
| --- | --- | --- |
| Wired control | 0 | 1e-5 |
| Slow control | 0 | 2.5e-6 |
| Neural contrast | 1 | 1e-5 |

The quarter-rate control is fixed in advance, not fitted to match observed
learning exposure. It tests one slower-learning alternative, not every possible
rate. All conditions retain error-dependent amplification and positive basal
adaptation. Nonselected and outgoing learning remain active.

The initial screen has one balanced four-pair acquisition block, then one
balanced block with the physical load relation reversed. Reversal changes the
world's force, not neural configuration or sensory labels. Body state, delayed
afferents, neural queues and learning traces continue across the change.
Weight-transfer resting probes and matched acquired-state learned/reset probes
run in isolated branches after each block; none writes back to acquisition.

Seeds 11, 23, 44 and 77 completed in
`.live/research/20260909_eligibility_reference_screen_seed{seed}`. Each worker
exited normally after 16,012 executed ticks, including its exact 364-tick old
episode replay. The independent completed-course audit covers 62,592 new
recorded ticks and 288 probe cases across the four graph seeds.

Full cell states, selected weights, actual reference arrivals, every reference
cell's incoming weights, terminal releases, body integration state and local
learning variables are retained. `eligibility_reference_analysis.py` is the
independent whole-course auditor. It now reconstructs reference-cell incoming
weights from local tick 1 and membrane/output from tick 2, including float32
arithmetic and heap accumulation order. Earlier in-flight history is explicitly
unchecked, not represented as a successful zero-residual check. Corruption
tests cover pool weights, membrane state and output.

The reference changes actual write allocation. The four contrast courses
contain 166,291, 157,044, 123,720 and 129,923 positive-input weight updates
opposed to the current signed error. Both native controls contain zero such
updates. Full per-port masks are retained. For example, seed 11's first normal
training pair, local tick 69, predictor 583, source 371, has error
−0.006774917512541292, positive arrival 0.006222407333552837, effective
eligibility −0.015692244809312805 and positive weight change
2.153192639990753e−7. No positive scalar of the native rule can produce that
update at that fixed state. This does not rule out useful future trajectories
under a differently regulated native rate.

It does **not repair joint recall** in this short screen. For every resting
probe tick 16 through 63, all three conditions and all four seeds predict the
correct load direction for pairs 00 and 11, and the wrong direction for 01 and
10. This pattern occurs after both normal and reversed acquisition. Because
the world's force assignment reverses, the common prediction direction also
changes. Adaptation occurs, but the retained response is still dominated by a
shared direction rather than the required audiovisual conjunction. Smaller
amplitudes alone are not improvement: they weaken correct responses too.

One block per relation cannot establish learning capacity, equilibrium or
long-term retention. The reference is spatial, not a feature-specific temporal
mean; unequal feature responses, delayed error and clipping remain unresolved.
No larger acquisition sweep is justified solely by its valid local equations.

## Does the regulator improve the same acquired organism?

`eligibility_reference_intervention.py` restores the contrast-acquired normal
checkpoint, retaining the same brain, body, weights, delay queues and random
state. Four pair probes compare intact contrast, strength-zero native learning,
and quarter-rate native learning. Each branch keeps positive plasticity.
Intact probes reproduce every old recorded field exactly.

All four workers completed 4,608 ticks, including 1,536 exact intact replay
ticks. The independent intervention audit verifies physical integration,
actuator work balance, actual afferents, neural motor commands, learning,
reference releases and the covered intracellular pool equations.

For the strength-zero intervention, prediction output first differs at local
tick 6 or 7, muscles and physical motion at 8 or 9, raw bodily sensors at 9 or
10, and supplied delayed afferents at 73 or 74. Learning errors differ at 74 or
75. These are observed divergence times, not proof of a serial causal chain.
Seed 77 pair 11 also changes terminal information at tick 18 and reference
arrivals at 58; the other native comparisons show neither within this probe.

Disabling contrast yields less absolute displacement on every tick from the
first physical difference through tick 95 in all 16 cases. Final differences
range from −0.0007473443 to −0.0003471000 radians. Quarter-rate native learning
instead gives greater final displacement, by 0.0002852373 to 0.0006001911
radians. These small, acquired-state effects do not establish a retention
tradeoff or a universally preferable rate. They reject calling this regulator
an improved bodily controller on the evidence presently available.

## The consequential result: body feedback and learning are separable here

The error divergence appears close to the arrival of changed body signals.
That coincidence could misleadingly suggest that learning has responded to the
consequences of its own changed action. `eligibility_reference_yoke.py` tests
that interpretation directly.

The native-learning branch receives the intact branch's exact recorded sensory
input while its real body continues moving under its own neural muscles. Only
delivered bodily afferents are substituted; actual sensors and their physical
delay queue are recorded separately. Weights, neural returns and adaptation
remain live. This is a diagnostic intervention, never an agent controller.
An unchanged native branch must first reproduce its earlier record exactly.

Four workers completed another 3,072 ticks, including 1,536 exact native replays.
The independent analyzer verifies all 16 pair/seed comparisons and reconstructs
the actual physical sensors separately from the substituted input. The following
interval facts hold in every case:

| Comparison | What changes | What stays exactly equal over all 96 ticks |
| --- | --- | --- |
| Native/yoked versus intact contrast | Selected weights at tick 0; prediction at 6–7; body at 8–9; learning error at 74–75 | All delivered sensory input |
| Native/closed versus native/yoked | Supplied afferents at 73–74; joint output at 78–82; muscle output and body at 81–85 | Predictive outputs, selected weights, errors, rates, mixed-bank and reference outputs, and all recorded terminal information |

Thus the later error change does not require changed bodily feedback. The
body's return signal affects the restoring reflex, but does not affect the
learner during these probes. This is not evidence against PAULA composition.
It identifies a missing functional dependency in this particular composition.

The code explains why that separation is plausible beyond the measured prefix.
Joint receptors project only to muscles. Muscle outputs have no forward neural
targets. The comparison population senses the imposed external load, which
does not depend on the arm's position, velocity or action. Prediction also
reaches the comparison population through delayed neural paths. Native return
paths do exist, but these graded muscles never spike, have no rate modulation,
and their return on a prediction input depends on that input and its local
weight, not on the joint contribution to the muscle membrane. A drawn return
arrow therefore does not establish cross-input credit assignment. The current
intervention demonstrates 96-tick independence, not a universal absence theorem
for other PAULA equations, states or architectures.

The next structural experiment should make the learned consequence depend on
action and body state. For example, an encountered mechanical resistance
depends on movement, and a neural prediction must combine sensory context,
proprioception and motor information. The task must require useful movement:
standing still cannot count as mastering resistance. Compare active coupling
with matched sensory replay and motor-path interventions. Preserve the present
external-load learner as a control. Merely improving its audiovisual XOR score
would not supply the missing sensorimotor dependency.

## Evidence and reproduction

All raw files below are local under `.live/research/`; generated data are not
silently treated as published repository fixtures.

- `20260909_eligibility_reference_screen_analysis`: complete acquisition,
  reversal, reference pathways, opposed-write masks and all recall intervals.
- `20260909_eligibility_reference_intervention_seed{seed}` and the matching
  `_analysis`: acquired-state regulatory interventions and exact intact replays.
- `20260909_eligibility_reference_yoke_seed{seed}` and the matching `_analysis`:
  actual versus substituted body signals, exact native replays, complete neural
  and physical records, and per-tick effects of the separated paths.
- `20260909_eligibility_reference_yoke_figures_readable`: four figures, one per seed,
  showing every pair's actual motion, magnified body-feedback effect and
  learning-error contrast. No averaging or smoothing; shared scales across seeds.

The experiment entrypoints accept explicit parent and output directories and
refuse overwrite. Invoke the yoke module with `--analyze`, all four completed
seed directories and `--output` for independent analysis. Source, checkpoint,
record and parent hashes are checked. Local executable checkpoints are trusted
research state, not a safe format for untrusted uploads.

No live agent versions, C. elegans code, shared core neuron equations or media
encoders were changed by these diagnostic additions. No new long sweep is
running. The full organism, hierarchy and consciousness-research goal remains
unachieved; the new result changes which dependency needs to be built next.
