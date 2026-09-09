# Magnitude feedback for graded inhibitory connections

## Result and scope

The opt-in `MagnitudeRetrogradeNeuron` prevents the measured inhibitory-terminal
failure in a completed 193-cell, 16,000-tick preparation. The unchanged comparison
also ran 16,000 ticks. Four 590-cell embodied acquisition courses have now
completed the same sixteen-block schedule as the previous crossed audiovisual
tests. All 130,048 recorded training/probe ticks pass independent physical,
sensory, learning and ordered-context-return audits. No suppressed-bank cell
becomes active during acquisition in any seed. The gate repair survives this
body experiment, but does **not** provide reliably retained joint association.
The four courses executed 131,504 ticks including exact reference replays.

## What changes

For a negative incoming information weight `w`, the extension replaces only the
information coordinate of the native return error, `a - w`, with `a - abs(w)`.
The inhibitory forward weight keeps its sign. Native event timing, all other
return coordinates, presynaptic update equations and positive adaptation remain.
The default path is unchanged. Enabling this experiment requires bounded graded
cells and zero plastic throughput on the affected inhibitory ports.

For a non-spiking target, the inherited timing direction stays negative. With
held incoming magnitude `q`, source activity `h` and arrival approximately `h*u`,
the proposed terminal update is `u_next = u + eta*(q-h*u)`. This has a positive
fixed point for positive `q,h` in the corresponding held-coefficient system.
The native negative-weight comparison instead pushes toward a negative value.
This argument is not a positivity or stability proof for the complete adaptive,
delayed, multiple-target network. The full embodied variant changes return
errors at **all** negative incoming ports, including sensory and comparator
connections. It is not an intervention confined to the context gate.

The base neuron, existing subclasses, shared C. elegans circuitry and live agent
versions are not modified. The new class lives under
`neuron/extensions/experimental/magnitude_retrograde.py` in `neuron-model`.

## Actual isolated trajectory

One source feeds 192 inhibitory targets through one shared terminal, preserving
the measured return-event fanout. Each target also receives a varying positive
input with a distinct phase. The gate receives constant drive until tick 12,000,
withdrawal until 13,000, then varying drive. Both local learning rates are `1e-7`.
No feedback is frozen, no terminal is reset and no target is removed.

| Observation | Native return | Magnitude return |
| --- | --- | --- |
| First target leakage in sustained phase, after initial settling | Tick 10,395 | None |
| First negative source coefficient | Tick 11,621 | None |
| Final source coefficient | -0.00012888303899671882 | 1.7836227416992188 |
| Target activity on withdrawal | Already leaking | Begins at tick 12,009 |
| Target activity after gate returns at 13,000 | All remaining 3,000 ticks | Ticks 13,000–13,002 only |
| Final inhibitory incoming weight | -3.97920071405951 | -3.9668971902473054 |

The corrected targets remain responsive during withdrawal and stop again after
the expected finite propagation/integration delay. In the native case, the
negative release is ignored by the positive-arrival input mask; input-triggered
updates then stop. Its apparently stable final negative coefficient is a failed
interface, not useful homeostasis. The isolated and previous embodied failures
reach exactly the same negative coefficient in these records.

The incoming inhibitory magnitude continues decreasing even in the corrected
condition. This extension has not repaired that issue. For these never-spiking
cells the inherited bounded incoming rule remains depressive; weak plasticity
does not establish lifetime preservation. Long-term regulation must address
incoming adaptation as well as outgoing release, without disabling learning.

## Evidence and tests

Local data under `active-inference/.live/research/`:

- `20260909_magnitude_gate_native` and `20260909_magnitude_gate_enabled` contain
  all cell fields, every incoming weight, every terminal, inputs and ordered
  context-return events in 512-tick chunks.
- `20260909_magnitude_gate_analysis` contains the checked full trajectory and
  references to the hashed raw records. Both workers exited successfully.
- `20260909_magnitude_learning_seed{11,23,44,77}` contain four independent
  graph-seed courses. Each completed block has full neural and physical/delay
  checkpoints. Do not interpret a progress file as completion of the course.
- `20260909_magnitude_learning_b1_analysis` verifies the first block across all
  four seeds. The context bank stays suppressed. At resting probe ticks 16–63,
  same-index audiovisual pairs predict the correct positive direction; crossed
  pairs predict the wrong direction. All ticks, including onset, are retained.

The body protocol is 16 balanced blocks, four 364-tick episodes each. Four
weight-only resting probes follow each block. Eight acquired learned/reset
probes follow blocks 4, 8, 12 and 16. All probes last 96 ticks and keep adaptation
active. Each seed first reproduces an original 364-tick acquisition episode
exactly with the extension disabled and the new observer enabled. The complete
planned course is 32,876 executed ticks per seed, including that replay.

Twenty-three focused tests pass. They cover exact default body behavior,
checkpoint continuation with queued returns, local error replacement, invalid
options, read-only observation and deliberate corruption of retained events.
The first test attempt incorrectly assumed every corrected return would be
positive. Small inhibitory weights correctly produce a negative return when
arrival exceeds their magnitude. The assertion now tests that actual equation.
Another test caught lost arithmetic type information: serializing an evolving
float32 coefficient into a float64 table is insufficient for exact recurrence.
The recorder now stores its pre-update scalar type explicitly. No tolerance
was widened to hide either failure.

## Biological interpretation

[Dudok et al., 2024](https://pubmed.ncbi.nlm.nih.gov/38422134/) report
activity-dependent suppression of inhibitory synapses through retrograde
endocannabinoid signaling in behaving mice. The abstract and indexed text of
[Barti et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11135421/) report
target-dependent release regulation associated with presynaptic molecular
organization. These studies support examining local, target-dependent release
regulation. Neither derives the magnitude-error equation used here. This is a
PAULA consistency experiment, not a molecular endocannabinoid model or evidence
that biological inhibition must remain permanently on. Full-text PMC access
was blocked during this continuation; the literature claim is limited to the
retrieved abstracts and indexed passages.

## Full-course result and interference mechanism

The subsequent `20260909_magnitude_learning_b3_analysis` audits all first three
blocks: 22,080 recorded ticks and 48 expression probes. No context-terminal
sign failure or active suppressed-bank cell occurs in that acquisition prefix.
At the third block's resting probe ticks 16–63, the opposite bias now dominates:
crossed pairs have the correct negative prediction, while same-index pairs are
wrong. The subsequent full sixteen-block audit is retained in
`20260909_magnitude_learning_b16_analysis`, including 384 probe cases. All four
workers exited successfully. The final context-terminal coefficient is
2.0760421752929688 in every seed; none became negative during acquisition.

At block 12, seeds 11, 23 and 77 express the correct relation for all four
pairings at every resting probe tick 16 through 63, before new somatic evidence.
Seed 44 still gets pair 10 wrong throughout that interval. At block 16, all
seeds again predict the same negative direction for all pairs: 01/10 are right
and 00/11 are wrong throughout that interval. The block-12 result is a
retrospectively identified transient, not a stopping rule or general retention
claim. Each probe continues adapting, and onset and post-feedback ticks remain
in the evidence.

At the final probe tick 63, the shared prediction component is -0.331 through
-0.293 across seeds, while the joint component is positive, 0.084 through 0.122.
This is not simply erased joint content. The four-block per-update audit,
`20260909_magnitude_credit_b4`, projects all 23,296 actual acquisition updates
onto one declared four-pair reference. Joint contributions largely accumulate,
while much larger changes rewrite the response shared by all pairings. Absolute
shared effects at reference tick 63 sum to 9.20 through 9.57 across seeds, versus
0.025 through 0.045 for joint effects. These are conditional projections, not
derivatives or counterfactual trajectories of the coupled brain. Every
acquisition-tick by reference-tick effect and its source identities are saved.
Native reference reconstruction has zero residual in all four seeds.

Thus preserving the inhibitory interface repairs one consequential failure,
but does not remove learning interference. The incoming-plasticity issue also
remains. A neuron-wide rate multiplier changes the magnitude, not the direction,
of one unclipped update at a fixed state. Temporal regulation, clipping and
feedback can still change the integrated trajectory. The next experiment asks
whether neural competition changes that trajectory usefully.

## Local competition screen

`components/learning/feedback_competition.py` adds four inhibitory pools
per mixed bank, each represented here by one graded cell reading 48 principals.
Each reads only its disjoint territory with total birth input conductance one
and returns weight -1 to each member. Eight additional cells bring the complete
brain from 590 to 598 cells. No external average, pair label or desired action
enters these circuits. Existing positive adaptation and return pathways remain;
the default disabled configuration is an exact copy of the original.

The fixed gain was chosen before the screen. The corresponding held-coefficient,
all-active common-mode delay polynomial has largest root magnitude 0.909. That
calculation is not a stability proof for the adaptive network. The isolated
256-tick test suppresses without silencing its driven principals.

Four completed embodied screens compare no added cells, wired cells with zero
output weight, and functional feedback. Each condition receives four fresh
364-tick crossed-media episodes. All 17,472 ticks pass independent selected
learning, physical, afferent, neural-command and context-return checks. Added
cells are fully observed, not independently equation-reconstructed. Original
episodes replay exactly. The zero-output circuit retains extra retrograde paths
and is not assumed to be identical to absent wiring.

| Seed | Joint/shared norm ratio, no feedback | With feedback | Conditional margin, no feedback | With feedback |
| --- | --- | --- | --- | --- |
| 11 | 0.0803 | 0.1024 | 0.3451 | 0.3440 |
| 23 | 0.0813 | 0.0975 | 0.3109 | 0.3122 |
| 44 | 0.0734 | 0.0807 | 0.2949 | 0.2490 |
| 77 | 0.0714 | 0.0914 | 0.2799 | 0.2960 |

Norm ratios in this table use mean norms over ticks 16 through 63. The ratio
advantage also holds at every individual tick in that interval for every seed.
All four active-bank inhibitory cells operate throughout the interval. The
first mixed-cell divergence from the wired-zero control occurs at tick 8 for
audio 0 and tick 9 for audio 1, for both videos in every seed.

Absolute joint norm decreases in three seeds on that window average, and seed
44 loses conditional margin. The margin uses fixed factual incoming histories
and legal constant weights, with primal/dual numerical checks; no fitted weights
are stored or installed. Neither contrast nor conditional capacity establishes
learned behavior. The screen supports a continued-acquisition comparison, not
acceptance of the circuit as a memory repair. Data and full member traces are
in `20260909_feedback_competition_seed{11,23,44,77}` and
`20260909_feedback_competition_analysis`.

[Agnes and Vogels, 2024](https://www.nature.com/articles/s41593-024-01597-4)
model excitatory and inhibitory plasticity as co-dependent through neighboring
currents, obtaining stable learned organization in their simulations. This
motivates testing compatibility between local adaptation processes. The present
PAULA feedback circuit neither implements their conductance/STDP equations nor
demonstrates their results. Its existing graded learning remains a separate
model hypothesis. Biology informs the experiment without prescribing its form.

Thirty-three focused tests pass across local extensions, exact runtime replay,
ordered-event corruption checks, update attribution and feedback construction.
`competition_acquisition.py` reuses screened graphs for the same continuous
course. Its completed unchanged-condition validation reproduced the original
first block exactly, including every field of all four diagnostic branches:
1,840 recorded ticks, 2,204 executed including the source-screen replay.
The validation is retained in `20260909_competition_acquisition_none_preflight`.
Four feedback-enabled sixteen-block courses have started in
`20260909_competition_learning_seed{11,23,44,77}`. They are in progress, not
completed results; their own source-screen preflights must also replay exactly.
These experiments address composition and retained neural use of experience,
not semantic recognition, mammal-level breadth or subjective experience.
