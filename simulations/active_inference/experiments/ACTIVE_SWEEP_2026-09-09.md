# Action-dependent physical learning: sweep preflight

## Why this experiment exists

The preceding acquired-state sensory replay isolated a gap: the imposed force
learner could influence the body, but changes in bodily reafference affected
only a restoring reflex during the tested interval. Its force target did not
depend on action. Better cue-to-load classification would not close that gap.

This new preparation retains that predictor, the real audiovisual receptors,
mixed populations and continuous plasticity. It adds the existing PAULA
four-phase motor rhythm, velocity receptors and optional motor/proprioceptive
projections into the mixed populations. The physical world supplies spring
and movement-dependent drag. Consequently, different actions can generate
different force observations and different neural teaching events.

This is a structural and task-necessity preflight, not an accepted new agent
version, an acquired action policy or a general sensorimotor model.

## Physical constraint and arithmetic before running

The original single-hinge MuJoCo geometry, motor conversion of 0.2 Nm per
command unit, 4 ms step and native damping of 0.12 Nm s/rad are preserved.
A sampled environmental torque is applied before each physical step:

```
tau_environment = -0.15 * angle - drag * angular_velocity
```

Drag is zero in the free world and 0.8 Nm s/rad in the resistive world. Both
terms oppose displacement or motion respectively; their coefficients do not
depend on a neural label, expected output or performance. MuJoCo integrates
native damping implicitly; the additional environment is explicitly sampled.
This discretization is declared and audited, not mistaken for an exact
continuous-time energy law.

The task counts alternating crossings of +0.008 and -0.008 radians, starting
with the positive gate. Waiting beyond a gate earns no extra crossings. Two
crossings complete one full sweep. The count is an observer and never enters
the brain. There is no metabolic/reward organ in this preparation yet.

Before observing the new full-size traces, a rough impulse calculation set
the gate scale. The inherited eight-unit CPG-to-muscle input gives roughly
0.0064 Nm s of motor impulse per stroke. With the original damping and a
164-tick rhythm, the expected alternating excursion is of order 0.02 radians
peak-to-peak. Adding drag reduces the corresponding impulse/displacement
scale toward 0.007 radians. Gates separated by 0.016 radians should distinguish
the worlds. This ignores coupled learning, spring transients and neural return
adaptation; actual free-world completion must be checked, not assumed.

The new body sensors encode applied torque/0.2, angle/0.05 and velocity/0.2 as
six nonnegative opponent channels, without clipping. They have a continuous
64-tick afferent delay. To preserve the previous restoring reflex's initial
forward physical gain, its joint-input weights are multiplied by 0.05.
Otherwise the new sensor units would accidentally amplify that reflex
twentyfold. This compensation does not preserve subsequent plasticity or
retrograde errors under changed input units; no full-trajectory equivalence
with the previous body experiment is claimed.

## Neural composition and controls

The 596-cell full preparation consists of the previous 590-cell magnitude-
feedback learner, four ordinary spiking CPG cells and two graded velocity
receptors. The CPG graph and parameters come from the reference rower builder.
Reserved empty ports make its learning-window bounds valid. A single birth
input initiates it; no host clock sustains or selects movement. CPG phases
zero and two each add an eight-unit projection to the corresponding existing
antagonist muscle. This intentionally adds motor authority.

In fused context, each mixed cell also receives both actual muscle outputs,
both joint channels and both velocity channels through heterogeneous signed
weights of magnitude 0.125. The paired contextual banks retain matched
heterogeneity. These anatomical source identities are not stimulus labels.
This adds conductance; it is not a current-matched gain control.

| Condition | Extra drag | Sensorimotor context weights | Actuator transmission |
| --- | --- | --- | --- |
| Free/fused | 0 | Signed 0.125 | Intact |
| Loaded/fused | 0.8 | Signed 0.125 | Intact |
| Loaded/sensory | 0.8 | Zero, same edges and returns | Intact |
| Actuator cut | 0.8 | Signed 0.125 | Zero, diagnostic only |

All neurons retain positive basal plasticity. Prediction uses the previously
declared local signed neural-error extension; no new cellular equation is
introduced. The unsuccessful spatial-reference contrast is not installed.
Motor output reaches the predictor indirectly through the mixed populations
only in the fused version. The initial predictor weights remain zero.

Each branch starts at birth and runs 1,024 uninterrupted neural/physical ticks.
One real audiovisual recording loops every 300 ticks. Its identity does not
set the mechanical resistance. This is not yet a multimodal discrimination or
latent-context experiment. Four graph seeds are 11, 23, 44 and 77.

## Evidence required before interpretation

The free world must support repeated physical sweeps. The loaded world must
actually impede them. The actuator cut must abolish motion and motion-derived
force while leaving the neural oscillator and positive adaptation available.
If the baseline cannot sweep or the new resistance does not impede it, this
world does not yet justify an adaptive upgrade.

Retain every neuron, selected weight, error, local rate, terminal coefficient,
actual physical sensor, delivered sensor, actuator command and MuJoCo state
at every tick. The auditor reconstructs environmental torque, receptor units,
delay history, muscle-to-actuator conversion, physical integration and gate
crossings. Tests require exact observer/no-observer agreement and reject
corruption of physical, neural-learning and task records.

Before claiming retained learning benefits, branch the executable acquired
state and compare intact versus reset selected weights, including their early
and late trajectories with learning active. Improvement during training alone
is insufficient. Compare motor omission, changed physical conditions and
matched sensory replay before calling a pattern an action model. No perfect
trajectory or millimeter-level precision is required; useful task-dependent
adaptation is the target.

## Literature used and boundaries

[Shadmehr and Mussa-Ivaldi, 1994](https://reprints.shadmehrlab.org/jneurosci94.pdf)
used movement-dependent force fields and tested adaptation, aftereffects and
transfer. The abstract and experimental methods on pages 1–5 informed the
physical manipulation and the need to distinguish acquisition from aftereffects.
Their fitted controller and multidimensional field are not implemented here.

[Tseng et al., 2007](https://www.diedrichsenlab.org/pubs/Tseng_JNeurophys_2007.pdf)
separated sensory prediction error from online motor correction using reaching
conditions that differed in correction opportunities. Its abstract, introduction
and task methods informed keeping prediction, reflex correction and learned
control as separate claims. This preparation is not a cerebellar reconstruction
or a reproduction of their human visuomotor experiment.

## Completed four-seed result

All four workers exited normally after all four conditions. The independent
audit verifies 16,384 recorded embodied ticks. It reconstructs actual selected
input arrivals from source-neuron release and terminal values from tick 1,
as well as physical, sensory, task, learning and return-path checks. Tick-zero
in-flight source history is not inferred. The mechanical audit also checks
the compiled mass, force balance and per-step work balance, including implicit
integration loss. Mechanical work is not a metabolic or ALERM energy measure.

| Graph seed | Free/fused crossings | Loaded/fused | Loaded/sensory | Actuator cut |
| --- | --- | --- | --- | --- |
| 11 | 10 | 0 | 0 | 0 |
| 23 | 10 | 0 | 0 | 0 |
| 44 | 6 | 0 | 0 | 0 |
| 77 | 6 | 0 | 0 | 0 |

These counts index complete trajectories, not a substitute for them. For
example, seed 11's free crossings occur at ticks 61, 369, 450, 508, 614, 683,
764, 858, 924 and 1021. The other seeds and every unfavorable interval remain
in the audit. The loaded/fused angle spans approximately −0.00265 to +0.00668
radians across the four runs, never reaching either gate. The free world
supports repeated sweeps; the added physical resistance defeats them.

The motor rhythm itself does not fail. All four seeds retain the same CPG
spike times: phase zero at 40, 204, 368, 532, 696 and 860, with successive
phases offset by 41 ticks. All conditions retain neural motor output. With
actuator transmission cut, the body remains still, force sensors remain zero
and selected predictive weights remain zero, although learning rates stay
positive. No motion-derived teaching opportunity is the cause, not a freeze.

### The new causal dependency is present

Comparing loaded/fused against actuator cut starts with the same neural graph,
weights and body. In every seed the first differences follow this sequence:

| Recorded local tick | Changed quantity |
| --- | --- |
| 42 | Physical motion |
| 43 | Actual environmental force and bodily sensors |
| 107 | Delivered delayed bodily input |
| 108 | Force, joint and velocity receptor output |
| 110 | Mixed-population and error-comparator output |
| 111 | Predictor error state |
| 112 | Selected predictive weights and local learning rate |
| 114 | Predictive output |

Unlike the prior externally imposed-load preparation, eliminating action's
physical consequence eliminates these learning events while retaining the
neural rhythm and audiovisual stimulus. The physical intervention and exact
per-tick sensor audit establish a movement-to-learning dependency. This is
not yet learned causal inference or autonomous discovery of the mechanism.

The fused versus sensory-only comparison supplies a different intervention.
Motor information changes mixed output at tick 44, selected weights at 112,
prediction at 114 and subsequent body motion at 116. It also changes upstream
terminal values at ticks 46–50. Thus extra context is actually used and reaches
action, but the zero-throughput control retains return paths and is not a
matched-current or return-isolated test of representational capacity.

### Why this is not a successful adaptive controller

Both loaded brains fail the task. The fused predictor reaches only about
−0.00672 to +0.00669 output units while the physical load reaches approximately
±0.3465 at the force-receptor scale. These are trajectory ranges, not a
pointwise gain ratio. More importantly, many complete intervals have a
prediction sign opposite to the current environmental force, which sends
compensatory muscle current in the wrong direction. For seed 11, the first
such interval is ticks 125 through 199; further intervals recur throughout
the course. The raw records retain every tick, and the figures expose the
weak estimate alongside the current physical load.

The present experiment does not separate inadequate exposure, temporal credit
misalignment and deficient phase/state representation. The 64-tick context
trace and delayed verification were inherited from the earlier cue task, not
derived for this 164-tick movement cycle. There is no reason to assume a broad
exponential eligibility trace assigns delayed errors to the right movement
phase. Conversely, a poor 1,024-tick acquisition does not establish an
asymptotic capacity limit. Neither more training nor a faster rate is accepted
as a repair without a discriminating test.

Next, use the retained executable states for matched learned/reset prediction
branches and inspect the resulting motor effects. Then test temporal/state
representation against the actual physical error history. Existing graded
temporal-basis populations are available; do not add another cell equation
before determining whether the missing information is in the forward
representation, the learning history, or both. A delayed-arrival manipulation
must state its effect on prediction timing, credit and physical latency rather
than treating those as interchangeable clocks. No new sweep was launched
after this preflight.

## Retained evidence and implementation

`components/body/loaded_hinge.py` contains the physical environment and
transducer. `components/motor/active_sweep.py` builds the compositional neural
delta. `active_sweep_probe.py` records acquisition and executable neural/body
states; `active_sweep_analysis.py` independently checks complete case families;
`active_sweep_figures.py` plots all four seeds without temporal averaging.

Local raw runs are `.live/research/20260909_active_sweep_seed{seed}`. The
independent output is `20260909_active_sweep_analysis_complete`; the physical figure is
`20260909_active_sweep_figures/active-sweep-all-seeds.png`. Approximately 50 MiB
per seed was retained, with a 3 GiB free-space reserve. These generated files
are local evidence, not silently published fixtures.

Seven dedicated tests verify construction, actual motion necessity, alternating
gate counting, six-channel delay, exact read-only observation, deliberate
corruption detection and exact executable checkpoint continuation. The focused
regression set passes 58 tests. No shared
core neuron equations, live agent versions or other organisms were changed.
The result is a physically consequential learning loop and a replicated
adaptive-control failure, not completion of the artificial-organism goal.

## Acquired-weight intervention: memory reaches action but distorts the sweep

The next experiment is complete. Each of the four loaded/fused brains resumes
at global tick 1024 for another 512 uninterrupted ticks, just over three motor
cycles. A paired branch resets only the selected predictive information weights
to their zero birth values. Both branches retain the acquired body, oscillator
phase, sensory delay, pending neural signals, terminal coefficients, context and
error traces, and positive adaptation. This is not a whole-brain memory reset.
The reset branch can relearn immediately, and later differences include the
coupled consequences of that reacquisition.

An intact checkpoint is also serialized again and replayed for 96 ticks. Every
recorded field matches the intact continuation exactly, including delivered
return events and delay history. Four seeds therefore produce 4,480 audited
ticks, of which 384 are exact replay controls. The separate analyzer reconstructs
physical integration, transduction, sensory delays, learning updates, selected
source arrivals, terminal returns and mechanical balance. All eight substantive
branches still complete zero gate crossings.

In all four seeds, the selected-weight intervention changes predictive output
at local tick 1, muscle output and actual body motion at tick 3, raw bodily
afferents at tick 4, and mixed-population output at tick 5. The CPG output stays
exactly the same for all 512 ticks. The learned weights therefore have an
expressed downstream motor consequence; this is not a silent memory pathway.

Error-comparator outputs first differ at tick 67, predictor error state at 68
and local rate at 69. Delivered sensory input differs at 68, but force/joint/
velocity receptor output differs only at 69. The chronology must not be
described as exclusively body-mediated learning feedback: the changed internal
prediction-to-comparator path is already active. The earlier actuator-omission
experiment establishes the physical dependency separately.

### Full stroke trajectories, including contrary intervals

The motor rhythm starts alternate positive and negative half-cycles at local
ticks 0, 82, 164, 246, 328, 410 and 492. For each branch, stroke advance is the
angle change since just before that half-cycle, multiplied by its neural motor
direction. The paired difference uses each branch's own prestroke position.
This observer is not a neural input, a gate score, or a claim that more movement
is always preferable. The last half-cycle is incomplete and labeled as such.

| Completed half-cycle | Direction | Intact minus reset advance across seeds, radians |
| --- | --- | --- |
| 0–81 | Positive | -0.0000615 to -0.0000194 |
| 82–163 | Negative | +0.0000392 to +0.0000928 |
| 164–245 | Positive | -0.0001564 to -0.0001007 |
| 246–327 | Negative | +0.00000785 to +0.0000615 |
| 328–409 | Positive | -0.0001339 to -0.0000885 |
| 410–491 | Negative | -0.0000129 to +0.0000312 |

These endpoints alone miss a real transient benefit. In the first positive
stroke, intact weights increase advance from tick 3 until ticks 52, 59, 33 and
46 respectively for seeds 11, 23, 44 and 77. The sign then reverses before the
stroke ends. Every later complete positive stroke has less advance throughout
its recorded interval. Negative strokes generally advance further, but the last
one has a smaller endpoint advance in seed 11. All favorable and unfavorable
intervals, not just these endpoints, are retained in the independent analysis.

The effect is small relative to the approximately 0.0066–0.0070-radian stroke
excursions. It does not approach the 0.016-radian gate separation. The retained
prediction changes movement and its directional balance, but does not restore
useful alternating sweeps. There is no justification here for increasing its
gain and declaring the architecture repaired.

### What this changes about the next experiment

The missing capability is not simply storage or an absent link from storage to
action. Selected acquired weights are expressed and affect movement before the
two branches receive different sensory evidence. What remains unproven is that
the learned relation is appropriate for the ongoing sensorimotor phase.

The inherited 64-tick physical delay, delayed prediction verification and broad
64-tick exponential eligibility are different mechanisms. Matching the first
two does not guarantee that synaptic credit is assigned to the context that
produced the verified prediction. A useful next intervention should change
credit timing while preserving forward representation and the physical world,
then compare it against a representation-only change. The present traces do not
establish which is the limiting cause, nor that longer acquisition cannot help.
No new cellular rule, gain tuning or enlarged population was introduced in this
matched-state experiment.

### Reproduction and evidence

`active_sweep_memory.py` runs the acquired-state branches; the separate
`active_sweep_memory_analysis.py` checks them and retains full difference masks
and stroke trajectories. `active_sweep_figures.py --memory` displays the actual
angle, per-stroke effect and neural prediction for every seed. Source and state
hashes tie each continuation to its acquisition. Nine dedicated sweep tests and
the 60-test focused regression set pass, including reset scope, exact replay,
active reacquisition, corruption rejection and direction-aware stroke analysis.

```sh
uv run --offline --no-sync --with cloudpickle==3.1.2 python -m \
  simulations.active_inference.experiments.active_sweep_memory \
  .live/research/20260909_active_sweep_seed11 NEW_OUTPUT_DIRECTORY
```

Repeat for graph seeds 23, 44 and 77. Pass those four output directories to
`active_sweep_memory_analysis --output NEW_ANALYSIS_DIRECTORY`. Existing
directories are never overwritten. The completed local outputs are
`.live/research/20260909_active_sweep_memory_seed{seed}`, with independent
analysis in `20260909_active_sweep_memory_analysis` and the full-trajectory
figure in `20260909_active_sweep_memory_figures/active-sweep-memory.png`.
The new records and checkpoints occupy approximately 55 MiB; raw evidence
remains local. All four workers have exited. No live simulation or large suite
was started. This result narrows a consequential failure; it is not a new
accepted agent version or completion of the organism research goal.

## Eligibility-kernel experiment: hypothesis and declared comparison

The next experiment changes local learning history without changing forward
representation. An experimental `CascadeEligibilityNeuron` subclass composes
the existing magnitude-feedback predictor with a cascade of decaying local
states. Only selected predictive weight updates use the final cascade state.
The inherited exponential trace remains observable; all forward integration,
unselected learning and retrograde messages keep their inherited order.

For each selected input and each unit tick:

```
z[0,new] = d*z[0,old] + (1-d)*actual_input_arrival
z[k,new] = d*z[k,old] + (1-d)*z[k-1,new]
q[new] = clip(q[old] + eta(previous_error)*previous_error*z[last,new], 0, 1)
```

The cascade has no hold counter or behavioral clock. It is a phenomenological
intracellular model, not a claim about a particular molecular pathway. Its
impulse kernel has unit mass and mean age `stages*d/(1-d)`. The stages are
additional dynamical state, preserved in executable checkpoints. Default
configuration follows the parent exactly. Core `neuron.py` is unchanged.

[Suvrathan, Payne and Raymond, 2016](https://pubmed.ncbi.nlm.nih.gov/27839999/)
report region-specific timing of cerebellar plasticity, including sensitivity
to delayed instruction in the flocculus. The abstract, introduction and figure
descriptions motivate testing a delay-sensitive window, not this cascade or its
parameters. The [2018 correction](https://pmc.ncbi.nlm.nih.gov/articles/PMC5777216/)
concerns an axis label in Figure 1, not changed data. No Purkinje-cell mechanism
is claimed to have been reproduced.

Four conditions cross kernel shape with mean age, using the same 596-cell
graph, initial weights, sensors, body, motor rhythm and positive basal rates:

| Condition | Local stages | Mean eligibility age, ticks |
| --- | --- | --- |
| Original exponential | 1 | 63.5013 |
| Longer exponential | 1 | 82.5208 |
| Cascade, original mean | 8 | 63.5013 |
| Cascade, verification mean | 8 | 82.5208 |

The original mean is the discrete exponential mean for tau=64. The second
is calculated from the configured forward verification path: context dendrite,
predictor membrane, prediction-to-comparator transmission and dendrite,
comparator membrane, error-receptor transmission and trace, then the convention
of using the previous completed error. This gives 69 fixed ticks plus mean
filter ages 3, 7 and 3.5208. It is a held-linear-path calculation, not the
measured conduction latency of the whole nonlinear, adapting organism.

At the 164-tick motor fundamental, before seeing this experiment's outcomes,
the calculated kernel phases are -66.72, -71.45, -134.55 and -171.29 degrees
in table order. The nominal linear verification path is -180.60 degrees.
Thus a longer exponential barely changes the phase discrepancy. The cascade
changes it substantially. This predicts a shape effect beyond extending a
single time constant. It is not a stability proof: rectification, opposite
prediction branches, changing release, nonstationary inputs and body feedback
are omitted from that calculation.

Equal kernel mass does not imply equal learning exposure. Their gains at the
motor fundamental are approximately 0.378, 0.300, 0.673 and 0.531, and their
temporal widths differ. Error-dependent learning rates and subsequent physical
trajectories may also diverge. These differences are part of the intervention,
not hidden behind a claim of equal effective learning rate.

Each condition has 1024 acquisition ticks followed by matched 512-tick intact
and selected-weight-reset continuations. The original condition must reproduce
the already retained acquisition and both continuations exactly on every
pre-existing recorder field. All added intracellular stages are recorded at
every tick. The independent auditor checks their recurrence, the actual neural
teaching arrivals, selected source release, physical equations and gate events.
Four graph seeds are 11, 23, 44 and 77. This is a bounded mechanism comparison,
not an asymptotic learning or agent-version acceptance claim.

### Completed kernel result

All four workers completed all four conditions and their two acquired-state
branches. The independent audit checks 32,768 embodied ticks. Within that total,
8,192 baseline ticks reproduce the previously retained acquisition and both
memory probes exactly. The default subclass therefore does not covertly change
the baseline. Six cascade-specific tests and the 66-test focused regression
set pass. An additional corruption check confirms that a changed comparator
output is rejected even when recorded learning-error values are left untouched.

The matched cascade changes the *use* of stored weights, not merely their
magnitude or the existence of a motor effect. In all four seeds, resetting
selected weights changes prediction at tick 1 and muscle/body motion at tick 3.
The neural clock remains identical between intact and reset branches. The
completed half-cycle effects below use the same direction-aware, prestroke
reference as the preceding memory experiment. Units are microradians, so
100 in this table means 0.0001 radians, not a large recovery.

| Seed | Positive 0–81 | Negative 82–163 | Positive 164–245 | Negative 246–327 | Positive 328–409 | Negative 410–491 |
| --- | --- | --- | --- | --- | --- | --- |
| 11 | +66.7 | +230.6 | -4.6 | +215.5 | +26.0 | +185.4 |
| 23 | +89.0 | +145.6 | +23.5 | +131.2 | +39.1 | +117.3 |
| 44 | +191.8 | +49.5 | +142.2 | +48.3 | +133.6 | +61.0 |
| 77 | +114.6 | +98.8 | +69.9 | +87.6 | +70.1 | +88.5 |

By contrast, the original and longer exponential conditions have negative
retained-weight effects at every completed positive stroke endpoint. Changing
shape while keeping the original mean improves this pattern, but seeds 11 and
23 retain some negative positive-stroke effects. The verification-matched
cascade has a positive endpoint effect in both directions across three seeds;
seed 11 retains one small negative effect. This supports the hypothesis that
the inherited broad learning window was consequential. It does not prove that
phase alone explains the change, since kernel shape also changes frequency gain
and effective local exposure.

Endpoints still conceal transient failures. For the matched cascade, the exact
intervals with *less* stroke advance than the reset branch are:

| Seed | Local tick intervals, end exclusive |
| --- | --- |
| 11 | 164–184, 242–246, 328–344, 492–505 |
| 23 | 164–181, 328–342, 492–505 |
| 44 | 82–94, 246–259, 410–416 |
| 77 | None; greater advance throughout 3–511 |

The complete traces retain the favorable intervals too, all other conditions
and the final incomplete half-cycle. They show a small learned contribution,
not uniform superiority at every moment. Even with the matched cascade,
completed intact stroke advances are only about 0.00674–0.00718 radians.
Every acquisition, intact continuation and reset branch still has zero gate
crossings. Current-force prediction residuals remain large around rapid motor
events. The physical challenge is not solved.

The next justified experiment is continued acquisition of the intact matched
cascade alongside the original baseline, starting from the saved full states.
This asks whether the useful contribution grows, stabilizes or degrades over
more sensorimotor cycles. Keep the same resistance and gates, and keep learning
available. Preserve intermediate executable states and repeat acquired-weight
interventions. Do not install a host compensation term or change the task to
turn the present effect into a pass. A longer course is now testing a measured
beneficial contribution, rather than assuming additional exposure will fix the
earlier wrong-direction learning. Forward representation remains a separate
candidate limitation if the contribution saturates or loses specificity.

More acquisition in this one world cannot establish a transferable body model.
The organism may have learned a useful correction tied to its imposed rhythm.
Once the contribution warrants a transfer probe, branch acquired state into a
changed resistance or rhythm without hand-retiming eligibility, retain positive
adaptation, and compare intact/reset onset and subsequent trajectories. Keep
this a bounded discriminating test, not another large sweep. The single-hinge
preparation is an instrument for understanding coupled adaptive populations,
not the endpoint of the organism project.

### Kernel experiment evidence

The reusable extension is
`neuron-model/neuron/extensions/experimental/cascade_eligibility.py` in the
sibling neuron repository. `active_sweep_credit.py` constructs and records the
factorial comparison; `active_sweep_credit_analysis.py` independently audits
complete case families. `active_sweep_figures.py --credit` plots continuous
acquisition/continuation, current-force residual and acquired-weight effects
for all four seeds without averaging. The visualization uses common seed
scales and retains contrary intervals rather than selecting favorable strokes.

Local raw records are `.live/research/20260909_active_sweep_credit_seed{seed}`.
The audit is in `20260909_active_sweep_credit_analysis`, and the figure is
`20260909_active_sweep_credit_figures/active-sweep-credit.png`. Added cascade
states increase raw size: the four runs occupy approximately 710 MiB in total,
with about 11 GiB disk space still available. Raw data remain local. All workers
have exited; no live agent server or large version-acceptance suite was started.

An exploratory read encountered an archive that was still being written. That
was an observation race, not a neural or simulation failure. No worker was
restarted. The completed independent audit waited for all case manifests and
verified the finalized files. Future consumers must likewise use completion
evidence rather than treating file existence as a completed recording.

### Continuing acquisition, declared before execution

Continue the original exponential and verification-matched cascade from their
intact tick-1536 states, for seeds 11, 23, 44 and 77. The fixed endpoint is
tick 4304, twenty complete 164-tick motor cycles after the earlier tick-1024
intervention. Record every tick and save executable neural state, physical
integration state, sensor-delay history and gate history at ticks 2560, 3584
and 4304. Do not change the graph, learning rates, resistance or gates.

At 4304, compare 512 continuing ticks with retained versus birth-reset selected
predictive weights. All other state is retained, including intracellular credit
stages. Both branches keep adaptation active. Compare a 64-tick checkpoint
reload against the uninterrupted intact trajectory on every recorder field.
Audit local learning, actual neural releases, physical integration and gate
events independently, using the same audit as the preceding kernel experiment.

The hypothesis is that the small useful acquired-weight contribution can grow
with experience. Saturation, wrong-direction contribution, drift, or loss of
usefulness are equally valid outcomes. Retain full per-stroke trajectories;
do not substitute favorable endpoint counts for task completion. Motor phase
matches the earlier intervention, but audiovisual phase and the rest of acquired
state do not. This compares two ages of a continuing organism, not duration
with all other variables held fixed. No transfer or general body-model claim
follows from success in this preparation alone.

The producer/analyzer is `active_sweep_acquisition.py`. Four workers run two
conditions each, with a fixed artifact budget of 384 MiB per seed, 1.5 GiB
combined, and a 3 GiB free-space reserve. This is 30,848 additional recorded
ticks including 512 replay ticks, not a restart of the version-acceptance matrix.

### Completed acquisition: useful memory grows, but control remains inadequate

All four workers completed both conditions. The independent auditor checked
30,848 additional ticks, including 512 exact checkpoint-replay ticks. The
67-test focused regression set passed. No neuron equation, synaptic graph,
plasticity rate, physical parameter or external input schedule changed during
this continuation. The intervention at tick 4304 has the same observed motor
phase as the earlier intervention: positive pulses start at local ticks 0, 164,
328 and 492, with negative pulses at 82, 246 and 410.

The original exponential learner deteriorates. Its retained weights reduce
advance at every completed stroke endpoint in both directions and every seed.
At tick 1024, this learner had usually helped negative strokes while impairing
positive ones. At tick 4304 its selected memory hurts both. For seeds 11, 23
and 77, the intact branch has less stroke advance throughout local ticks
3 through 511. Seed 44 has brief favorable intervals 82–88 and 246–249,
end exclusive, but all its completed endpoints are negative too.

The matched cascade has the opposite age comparison. Its acquired-weight
contribution grows at all six corresponding completed endpoints in each seed.
Every late completed endpoint is positive. The table gives the late effect in
microradians, retained minus reset; the earlier table above uses the same
physical reference convention. These are measured contributions to movement,
not weights interpreted as memories merely because they changed.

| Seed | Positive 0–81 | Negative 82–163 | Positive 164–245 | Negative 246–327 | Positive 328–409 | Negative 410–491 |
| --- | --- | --- | --- | --- | --- | --- |
| 11 | +484.2 | +716.2 | +457.7 | +742.2 | +499.6 | +686.4 |
| 23 | +371.1 | +472.4 | +320.0 | +453.2 | +339.4 | +449.1 |
| 44 | +639.0 | +289.6 | +660.8 | +286.4 | +622.0 | +348.7 |
| 77 | +365.2 | +430.7 | +395.3 | +383.9 | +382.5 | +419.8 |

The full trajectories still contain harmful onset intervals for the cascade:

| Seed | Less stroke advance, local ticks, end exclusive |
| --- | --- |
| 11 | 164–171, 328–334, 492–497 |
| 23 | 164–173, 328–335, 492–500 |
| 44 | 82–93, 246–256, 410–419 |
| 77 | None; greater advance throughout 3–511 |

In all eight condition/seed pairs, selected weights differ immediately,
recorded cellular state after one tick, body motion after three, and delivered
physical input and neural teaching after 68. The reset branches continue
learning, and retain other acquired state. The experiment therefore measures
selected memory's contribution with online reacquisition allowed; it neither
erases all memory nor measures performance with learning disabled.

No continuation or probe crosses a gate. Completed cascade stroke advances
are 0.007228–0.007497 radians, still far short of the 0.016-radian gate gap.
The continuous physical curves also expose drifting sweep centers, including
opposite drift directions across seeds. Current-force residuals remain large
near motor events. A growing useful contribution does not establish accurate
force prediction, sufficient compensation, stable lifetime behavior, or a
transferable model of the body. Two acquisition ages also do not establish a
monotonic learning curve or an asymptotic limit.

This does rule out the simple explanation that these selected weights never
become useful under actual movement. Under the same forward anatomy, changing
local credit produces experience that increasingly helps rather than increasingly
hinders movement. Kernel gain, timing and the resulting experience histories
remain coupled; the result is not an attribution to timing alone.

The next discriminating step should be a bounded change in resistance from
the saved acquired state, retaining sensory delay history and adaptation, with
matched selected-weight interventions. Do not hand-retime the cascade. Check
prediction and physical action separately: a correction can assist movement in
several worlds without accurately representing their different loads. A more
ambitious body should reuse useful adaptive organization, but neither stronger
innate drive nor an easier world is evidence of learned compensation. The
present single-joint test is not the definition of the organism goal.

### A consequential literature distinction

[Herreros-Alonso, Arsiwalla and Verschure, 2016](https://proceedings.neurips.cc/paper_files/paper/2016/file/65699726a3c601b9f31bf04019c8593c-Paper.pdf)
derive eligibility from the closed-loop response, rather than transport delay
alone. Their derivation assumes linear time-invariant plant/controller dynamics;
the example uses a supplied reference and predefined temporal bases. This is
not a convergence theorem for our adapting, rectified PAULA network. The useful
hypothesis is that appropriate credit depends on what downstream components do
to a signal, not only when it arrives. Before adopting that mechanism, measure
the relevant response in the coupled preparation. Do not put a known physical
model or externally fitted control policy inside a neuron and call it acquired.

A separate consolidation lead is [Bae et al., 2025](https://www.nature.com/articles/s41467-025-60511-z).
Its abstract and introduction describe task-dependent transfer between cerebellar
cortex and nuclei, investigated with a dual-system model and mouse experiments.
Only those sections were checked here; methods and supplements need review
before any implementation. Their explicit optimization objective does not supply
a PAULA-local transfer mechanism. This is a possible later question about a
corrective population teaching another pathway, not a reason to replace the
present resistance-transfer experiment or assume consolidation is necessary.

The saved evidence is `.live/research/20260909_active_sweep_acquisition_seed{seed}`,
with independent analysis in `20260909_active_sweep_acquisition_analysis`.
`active_sweep_figures.py --acquisition <earlier-kernel-analysis>` joins the entire
0–4815 physical course and compares both ages of the weight intervention. The
figure is `20260909_active_sweep_acquisition_figures/active-sweep-acquisition.png`.
It retains every sample, common seed scales, all unfavorable intervals and
the final incomplete stroke. The new raw records occupy approximately 655 MiB;
9.9 GiB remains free. All workers have exited. No live agent or version matrix
was started, and no new agent version is accepted by this result.
