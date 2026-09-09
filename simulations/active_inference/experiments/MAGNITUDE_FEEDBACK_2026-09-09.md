# Magnitude feedback for graded inhibitory connections

## Result and scope

The opt-in `MagnitudeRetrogradeNeuron` prevents the measured inhibitory-terminal
failure in a completed 193-cell, 16,000-tick preparation. The unchanged comparison
also ran 16,000 ticks. This is an isolated gate result, not accepted memory or a
new agent version. Four 590-cell embodied acquisition courses are running from
the same birth graphs and schedule as the previous crossed audiovisual tests.
The first completed block, 7,360 recorded training/probe ticks across four seeds,
passes independent physical, sensory, learning and ordered-return audits. It
does **not** solve the joint association.

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

## Next decision

Finish and audit the four existing courses before changing their source or
starting replacements. Compare the preserved gate, selected weight trajectories,
common versus joint prediction components, and acquired/reset physical behavior
with the original sixteen-block failures. If inhibition stays functional while
recall still overwrites, that separates two failures instead of declaring the
whole system repaired. The longer-term incoming-plasticity issue also remains.
This preparation is one experiment toward the full PAULA/ALERM objective, not
semantic recognition, a learned hierarchy, mammal-level cognition or consciousness.
