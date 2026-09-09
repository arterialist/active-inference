# Memory use depends on the body state and the control objective

## Result

A correct force association is not automatically a useful posture correction.
At the magnitude-feedback course's block-12 checkpoint, learned prediction for
audiovisual pair 01 has the correct sign in every seed from probe tick 9 onward.
Yet retaining those weights leaves the already-displaced arm farther from zero
than resetting them. Moving only the physical arm to rest reverses that effect.
The learned neural contribution now reduces displacement. The brain, its queued
events and delayed afferent history begin in the same acquired state.

The four-seed factorial experiment completed 12,288 ticks. These include 3,072
exact full-field replays of the earlier acquired probes. Selected plasticity,
physical replay, afferent delay, neural actuator commands and context-return
events pass independent audits. Every condition continues learning; diagnostic
branches never alter the acquisition course.

| Seed | Learned-minus-reset absolute displacement at tick 191, acquired body | Body moved to rest |
| --- | --- | --- |
| 11 | +0.112044 rad | -0.112044 rad |
| 23 | +0.064307 rad | -0.064307 rad |
| 44 | +0.087644 rad | -0.087644 rad |
| 77 | +0.123374 rad | -0.123374 rad |

Positive means farther from the zero-angle reference, not universally worse
behavior. The corresponding sign difference holds from tick 9 through 191,
not only at the endpoint. The probe contains all four audiovisual pairings,
including seed 44's wrong prediction for pair 10 and early sign transients.

For each memory condition, the recorded neural fields, selected weights,
incoming contexts, error traces, rates, terminals and supplied sensory inputs
are exactly equal between the two body conditions for ticks 0 through 63.
The body intervention first changes a recorded neural field at tick 65 in all
32 comparisons. Physical outcomes can differ before the brain receives that
change. This is a measured consequence of the afferent delay, not a requirement
for instantaneous or perfect biological compensation.

## What the intervention does and does not identify

The source checkpoint was selected retrospectively because three seeds briefly
expressed all four associations there. Each branch either retains or resets
only the selected predictive weights. All other acquired neural state remains.
The second intervention constructs a resting physical arm at the same simulator
time, while preserving the actual previous afferent samples. This is a physical
relocation experiment, not an autonomous action or a claim that relocation has
no energetic cost. The reported mechanical accounting begins after relocation.

The current network learns an association between media and imposed load.
Its innate joint-to-muscle pathway supplies posture correction separately.
Cancelling a load that was helping move a displaced arm toward zero can slow
that return even when the load prediction is correct. At rest, the same learned
contribution can reduce the displacement caused by the load. Learning more
precisely cannot by itself resolve which physical objective matters.

The nonlinear neural branches need not implement linear superposition. In this
one-joint preparation, however, the nearly equal and opposite pose effects are
consistent with its simple mechanical response and the initially identical
neural commands. They are not evidence of general body-independent memory.

## Corrected physical accounting

Earlier derived columns called `net_torque` contain actuator plus imposed load,
but omit passive joint damping. Those source records are preserved unchanged.
They must not be interpreted as the total torque producing acceleration.
`body_state_memory_analysis.py` distinguishes all three terms and verifies
`motor + load - damping * velocity_after = inertia * delta_velocity / dt`
against the compiled MuJoCo arm at every tick.

It also records actuator work, load work, damping loss, kinetic-energy change
and the implicit integration step's numerical dissipation. The per-step work
balance includes that dissipation rather than labelling it unexplained physical
or neural energy. These are mechanical quantities, not ATP expenditure,
variational free energy, or a tested ALERM metabolic budget. Three new tests
cover mechanical balance, deliberately corrupted velocity, event intervals and
the boundary handling of extended replay. The combined focused suite passes
36 tests.

## Implication for the next composition experiment

ALERM's claim that local regulation composes into organism-level control needs
this distinction. A local prediction objective is not automatically the
organism's objective. The existing assay's imposed load does not itself depend
on the organism's action, even though joint feedback closes a separate motor
loop. It tests a useful association but not a complete learned action–perception
model. No changes to the user's ALERM paper were made.

The next environmental intervention should make load or contact depend on body
motion, then test whether proprioceptive and motor-copy pathways into the learned
representation improve consequential behavior. Compare against the preserved
brain before adding those pathways. Define the environmental success condition
explicitly; do not silently swap prediction accuracy, small torque, low work and
small displacement. This need not wait for flawless recall in the current task.

[Todorov and Jordan, 2002](https://www.nature.com/articles/nn963) motivate judging
motor corrections by task-relevant outcomes rather than exact trajectories.
Their optimal-control model is a conceptual comparison here, not a controller
installed in PAULA or a derivation of these neural circuits.

The separate feedback-enabled acquisition courses remain running. The first
four blocks audit 32,512 recorded ticks; every gate remains functional, while
joint recall still shows a shared-direction bias. A contrast improvement in
the shorter screen has not yet established retained learning.

## Retained evidence

Under `active-inference/.live/research/`:

- `20260909_body_memory_seed{11,23,44,77}`: all 16 branches per seed, full tick
  records and source/checkpoint identities.
- `20260909_body_memory_analysis`: 16 pairing cases, every physical/neuronal
  trajectory and learned-minus-reset effect, with interval annotations.
- `20260909_body_memory_visual_evidence`: one complete figure per seed for pair
  01. Each shows the planar arm trajectory, angle, neural prediction and the
  change in absolute displacement. It is a kinematic projection, not a rendered
  MuJoCo camera. All 192 samples are retained, with shared scales across bodies.
- `20260909_competition_learning_b4_analysis`: the completed-prefix audit of
  the still-running feedback learning courses.

This result explains a coupled failure and improves its observability. It does
not establish learned body models, mammal-level breadth or consciousness.
