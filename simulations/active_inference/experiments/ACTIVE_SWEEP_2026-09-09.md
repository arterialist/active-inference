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
