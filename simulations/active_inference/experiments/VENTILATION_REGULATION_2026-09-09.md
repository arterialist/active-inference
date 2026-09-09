# Organ feedback and a construction-induced motor wave

## Research question

Can the acquired, continuously plastic sensorimotor preparation recruit useful
movement in response to a bodily maintenance need? This continues the actual
596-cell brain at tick 6352, including learned prediction weights, neural and
physical state, queued events and delayed bodily afferents. It does not change
V1–V4 or claim a new validated organism version.

The physical chamber and energy budget are specified in
[the preceding necessity screen](VENTILATION_2026-09-09.md). They create an
operating range where insufficient movement loses oxygen and excessive
activation spends energy. The units are illustrative. Ideal extraction,
negligible chamber load, absent CO2/perfusion dynamics and uncoupled oxygen and
energy metabolism remain limitations. Resource debt is a declared task metric,
not a calibrated biological death threshold.

## Neural addition

Seven existing-form PAULA cells are appended. Three implement the existing
energy afferent/deficit/alarm route. An oxygen afferent and signed comparator
drive two phase-coincidence relays, which project to the existing antagonist
muscles. The resulting brain has 603 cells. There is no new neuron equation,
host motor multiplier, external phase decoder or host action selector.

Organ signals have a 64-tick delay. The oxygen comparator integrates tonic
reference minus twice the oxygen afferent. Each relay rectifies rhythm plus
deficit minus one, with gain three. Its muscle projection has weight eight.
These are specified circuit weights, not acquired interoceptive goals. The
reference comes from an already-adapted terminal, so a nominal half-reserve
comparison is not an invariant half-reserve set point.

Every old neuron remains the acquired object. Old muscle fan-in grows from
three to five, increasing its plasticity-window upper bound. New cells use
positive basal postsynaptic and retrograde rates of 1e-7. Existing predictive
plasticity stays active. Native return pathways remain present even in zero-q
forward controls; these are part of the coupled system, not assumed noise.

## Why the earlier apparent success was not acceptable evidence

The first expanded networks maintained oxygen even when their oxygen output
weights were zero. The unmodified acquired brain failed. Direct source and
state inspection found a stale bootstrap value:

```text
At tick 6352, original CPG cell 591 / port 0:
  authoritative vectorized external input = 0
  backing external-input dictionary       = 5
  input-to-hillock delay                   = 40 ticks
```

The shared runtime consumes and clears the vectorized input arrays. The
dictionary can retain the original birth pulse. The additive graph installer
invalidates the vector cache; its next reconstruction imports the stale pulse.
The new graph therefore received a second motor startup event that the
experiment did not declare. Earlier checks of the recorder's `birth_input`
field missed it because that field described the intended driver, not the
actual input buffer.

The `20260909_ventilation_regulation_v2_*` recordings are retained as
confounded results. Here `v2` names a recorder revision, not agent V2. They
must not be cited as proof of organ regulation. Their earlier exact checkpoint
replays showed repeatability of the constructed state, not validity of the
construction intervention.

The repair is a construction-only adapter in `core/external_input_state.py`.
It validates the external interface, rejects real pending drives and synchronizes
the backing representation before the new installer rebuilds it. It does not
change `neuron.py`, the shared runtime or old checkpoint source requirements.
Other callers of the older installation helper and runtime unknown-key cache
rebuilds remain audit targets; this is not a claim of a global runtime repair.

## Causal test of the hidden pulse

`ventilation_reseed_probe.py` branches the original 596-cell checkpoint without
adding any neurons. It compares unchanged continuation, stale cache rebuild,
an explicitly declared amplitude-five pulse, and synchronized cache rebuild.
Each condition records 256 consecutive neural and physical ticks. The intended
comparisons require equality in all 38 recorded fields, not only the motor
raster. A re-kick is a diagnostic intervention, never an organism policy.

The checker also records actual CPG input buffers. Its initial implementation
incorrectly treated port zero as exclusive to the birth pulse; the recurrent
edge shares that port. Full acquired runs exposed the mistake. The corrected
check sums external injection and neural release in runtime order. The
regression now covers a complete recurrent cycle. Failed check outputs were
retained rather than weakening numeric tolerances.

## Corrected feedback comparison

The matched conditions are full feedback, oxygen-output zero-q, energy-output
zero-q and tonic neural recruitment. They retain the same cell and edge counts.
The tonic condition removes oxygen inhibition and uses one third of the tonic
reference. It is an approximate additional rhythm drive, not exact doubling of
total muscle output. Every run starts from the same per-seed acquired state
and lasts 1024 ticks, or 4.096 seconds. Seeds are 11, 23, 44 and 77.

All sixteen corrected courses finished. Full input, release, learning, muscle,
physical and resource records are primary evidence. The added-path audit independently reconstructs
inputs, delayed potentials, membrane dynamics, bounded updates and native
return events for the seven added cells and two original muscles, together
with existing predictive-learning and body checks. It does not claim to
independently rederive every equation of all 603 neurons.

### Completed results

| Seed | Feedback minimum oxygen, mL | Oxygen-cut first debt index | Feedback energy demand, J | Tonic energy demand, J |
| --- | ---: | ---: | ---: | ---: |
| 11 | 0.047641 | 388 | 0.373793 | 0.389832 |
| 23 | 0.042407 | 381 | 0.397829 | 0.378577 |
| 44 | 0.047838 | 386 | 0.356246 | 0.382170 |
| 77 | 0.045652 | 382 | 0.391310 | 0.378932 |

Indices are zero-based within the 1024 recorded steps. Every full-feedback,
energy-cut and tonic course has zero accrued oxygen and energy debt. Every
oxygen-output-cut course accrues oxygen debt, despite ending with positive
oxygen again. Its first-deficit indices match the unmodified original brain.
The corrected full-feedback CPG retains its original single-wave, 164-tick
per-cell rhythm. Feedback does not require the unintended extra startup event.

The energy alarm has exactly zero output throughout every full-feedback run.
Removing its output weight changes only the two recorded q-array fields;
all other fields, including neural activity and body/resource trajectories,
are identical. Energy-regulatory use has not been demonstrated.

Full feedback inspires 5.517–5.559 mL over the course, compared with
9.973–10.326 mL under tonic recruitment. It avoids oxygen spill, whereas tonic
recruitment spills 0.792–0.864 mL. Lower gas throughput is not lower energy
cost: feedback spends less energy in two seeds and more in two. Tonic
recruitment also passes the stated resource task. This environment therefore
does not yet establish the necessity of condition-responsive regulation.

The construction diagnostic completed all sixteen 256-tick branches. In each
seed, the stale rebuild and explicit pulse are identical in all 38 recorded
fields. Synchronized rebuild and unchanged continuation are likewise identical.
Their first between-pair differences are actual input at local tick 0, neural
state at tick 40, and muscles and physical state at tick 42. At 4 ms per tick,
these correspond to 160 and 168 ms after intervention. The unchanged branches
also match the first 256 ticks of the older unmodified baseline in all 36
shared fields. This establishes a particular construction cause, not a general
claim that adding populations cannot usefully alter a motor regime.

There are 20,736 new retained research ticks in the completed comparisons:
16,384 corrected regulation ticks, 4,096 construction-diagnostic ticks, and
256 exact initial-checkpoint replay ticks. The regulation and replay courses
pass the independent checks with maximum added-path neural residual 0.0.
The construction branches establish full-record pairwise identities; they
are not counted as independently rederived model-equation ticks. Thirty-seven
focused tests pass. Every launched research and test worker is terminal.

### Reproduction and retained evidence

Run from `active-inference/`, with the original acquired checkpoints available:

```sh
uv run --offline --no-sync --with cloudpickle==3.1.2 python -m simulations.active_inference.experiments.ventilation_regulation .live/research/20260909_active_sweep_return_seed11 NEW_OUTPUT --mode feedback --ticks 1024
uv run --offline --no-sync --with cloudpickle==3.1.2 python -m simulations.active_inference.experiments.ventilation_reseed_probe .live/research/20260909_active_sweep_return_seed11 NEW_RESEED_OUTPUT
uv run --offline --no-sync --with cloudpickle==3.1.2 --with pytest python -m pytest tests/test_ventilation_feedback.py tests/test_ventilation.py tests/test_metabolic_population.py tests/test_active_sweep.py tests/test_cascade_eligibility.py -q
```

Completed data live under `.live/research/`:

- `20260909_ventilation_verified_{mode}_seed{seed}` contains config, protocol,
  complete ticks, initial/final executable neural and bodily state, and manifest.
- `20260909_ventilation_reseed_verified_seed{seed}` contains the four full
  construction branches, per-cell rasters and exact-comparison manifest.
- `20260909_ventilation_verified_replay_seed{seed}` records the 64-tick exact
  replay and checked field list for each full-feedback initial checkpoint.
- `20260909_ventilation_regulation_figures/` contains `resources.png`,
  `reseed.png` and a manifest identifying every plotted input. No sample is
  downsampled or removed from these plots.

Historical `regulation_v2_*` courses remain explicitly confounded. The later
`ventilation_clean_feedback_*` records have the construction repair but failed
the first, incorrectly exclusive-port checker; they are not the accepted
courses above. Producer changes can invalidate their old checkpoint source
checks. Do not bypass those checks or overwrite old manifests to relabel them.

## Next discriminating question

A forward-output cut asks whether a projection contributes. It does not alone
prove that changing oxygen information is necessary, because appending a
plastic population can also change shared upstream terminals. A same-topology
oxygen-input clamp should distinguish changing bodily information from a
beneficial fixed recruitment regime. All actual organ measurements must still
be recorded separately from the experimentally substituted neural input.

The energy alarm must become active before its useful regulation can be
claimed. A later course should require the coupled organism to adjust to
changing demand or resource supply and compare it with tonic recruitment.
Preserve imperfect learned prediction; do not tune an isolated respiratory
circuit to perfection before testing the coupled organization.

This remains a small maintenance task in a much larger artificial-life goal.
It does not demonstrate mammal-level breadth or subjective experience.

## 10 September: sensory substitution exposes two-resource coordination

The same-graph clamp comparison is complete. Each seed branches the verified
603-cell initial checkpoint into intact oxygen input or a reading held at .25,
.50 or .75 of reserve. The actual body, oxygen and energy continue evolving.
Substitution occurs after the real 64-tick organ queue; both raw and delivered
signals are retained. No anatomy, motor command or learning rule changes.
Every intact 1024-tick record exactly reproduces the prior full-feedback course.

Holding the reading high produces oxygen debt in all four seeds, first at
local indices 392, 383, 390 and 385. Holding it low maintains oxygen but spends
enough actual energy to recruit the energy alarm. Holding it at its initial
half-reserve value maintains both resources and costs less energy than normal
oxygen feedback in every seed. Thus the afferent influences recruitment, but
varying oxygen information is not necessary or best in this fixed environment.

The low-reading condition permits a causal test of the previously unused
energy pathway. A matched branch zeros only its two incoming muscle weights,
from -2 to 0. All cells, other weights, sensory histories and native return
edges remain. Adaptation stays active, including in the predictor.

| Seed | First energy-alarm output | First muscle/body difference | First energy debt with output cut | Minimum energy intact, J |
| --- | ---: | ---: | ---: | ---: |
| 11 | 429 | 431 | 751 | .077046 |
| 23 | 495 | 497 | 830 | .079282 |
| 44 | 494 | 496 | 828 | .078372 |
| 77 | 495 | 497 | 829 | .079346 |

Indices are zero-based within each 1024-tick course. Before alarm recruitment,
the matched branches have identical recorded activity, body, sensory and
predictive-weight histories. The intact pathway prevents energy debt while
preserving oxygen. Cutting its muscle projections produces energy debt in
all four seeds, although final energy has recovered above zero. This is a
bounded demonstration of bodily energy feedback restraining expenditure
driven by another pathway. It is not learned arbitration or evidence of
indefinite maintenance with a finite gut supply.

The actual motor traces also expose a temporal imbalance. Under natural
oxygen feedback, retraction activation costs roughly three times protraction
activation; the fixed half-reserve input produces much more balanced costs.
The full records retain each phase source event, actual and delayed oxygen,
the arriving deficit signal and relay output. A delay/phase interaction is a
hypothesis, not yet a demonstrated cause. First predictive-weight differences
occur before the changed physical afferents return, so those early changes
must not be credited to newly delivered bodily feedback.

The analyzer checked 20,480 recorded ticks across the sixteen clamp and four
energy-output-cut courses. Independent checks cover physical integration,
resource accounting, the actual sensory substitution, muscle-output identity
and predictive learning. The four intact courses also pass the existing full
added-path equation audit. Clamped courses do not claim that auditor's complete
nine-cell reconstruction, which assumes unmodified organ input. No claim here
independently rederives all 603 cells.

Retained artifacts under `.live/research/`:

- `20260910_ventilation_input_clamp_seed{11,23,44,77}` holds every full course,
  protocol and final neural/physical checkpoint.
- `20260910_ventilation_energy_conflict_seed{11,23,44,77}` holds the matched
  two-weight interventions, including raw ticks and continuing state.
- `20260910_ventilation_clamp_analysis/` holds the checked per-tick contrasts,
  phase events, explicit neuron IDs and first-divergence records.
- `20260910_ventilation_clamp_figures/` contains `energy-conflict.png` and
  `oxygen-clamps.png`, with all samples and hashed input provenance.

Reproduce with the original verified checkpoints available:

```sh
uv run --offline --no-sync --with cloudpickle==3.1.2 python -m simulations.active_inference.experiments.ventilation_input_clamp .live/research/20260909_ventilation_verified_feedback_seed11 NEW_CLAMP_OUTPUT
uv run --offline --no-sync --with cloudpickle==3.1.2 python -m simulations.active_inference.experiments.ventilation_energy_conflict .live/research/20260909_ventilation_verified_feedback_seed11 NEW_CONFLICT_OUTPUT
```

Next change the physical oxygen supply over time, leaving both neural
regulators and acquired sensorimotor learning intact. Compare actual oxygen
feedback with the fixed half-reserve input that performed better here. The
world must reveal whether their differences help during both challenge and
recovery; a passing unchanged world cannot answer that question.

## 10 September: changing air breaks the composed organization

The challenge/recovery experiment is complete. Each of four seeds branches
the same acquired initial state into actual delayed oxygen feedback or the
half-reserve sensory clamp. Environmental oxygen fraction is .21 for ticks
0–511, .105 for 512–1279, then .21 for 1280–2047. Total duration is 8.192
seconds. No environmental-change flag, brain parameter change, weight reset
or host motor intervention occurs. Only the gas concentration changes.

All eight opening 512-tick prefixes exactly match their previous fixed-world
records, including inputs, returns and weights. The new resource accountant
and existing physical/predictive checks cover all 16,384 ticks. The analyzer
also reconstructs both phase relays from recorded local potentials, including
their actual .99 dendritic attenuation. Its first guard incorrectly assumed
unit attenuation and rejected the record; the correction uses the existing
configuration without changing the brain or tolerances. Forty-one focused
tests pass across the existing comparison and new world adapter.

| Seed | Actual feedback: first oxygen debt | First release by both relays without phase input | First energy debt | First energy-alarm output | First predictor membrane bound | Fixed reading: first oxygen debt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 719 | 769 | 803 | 875 | 1157 | none |
| 23 | 719 | 769 | 803 | 874 | 1206 | 1233 |
| 44 | 721 | 770 | 805 | 877 | 1228 | 1389 |
| 77 | 719 | 769 | 804 | 875 | 1181 | 1233 |

Indices are local and zero-based. Every actual-feedback branch fails both
resource budgets. None recovers viable operation after normal air returns.
The fixed-reading branches retain energy and avoid the recorded membrane
bounds, but three incur oxygen debt. Seed 44's first deficit occurs during
recovery, after gas concentration is restored. A final-reserve test would
conceal all three fixed-reading failures. This challenge does not establish
that feedback is universally necessary: the fixed control passes in seed 11.

### The first identified change in circuit function

The supposed phase-coincidence relay is an additive threshold circuit. With
unit membrane time constant and one-tick dendritic delay, its output is three
times the positive part of the attenuated weighted input sum minus one. If
the deficit contribution alone crosses that threshold, the rhythm input is
no longer necessary. Both antagonist relays then release together. The exact
recorded potential histories reproduce this output change in all four seeds.
The off-phase check uses the preceding tick's actual input because dendritic
propagation takes one tick; it does not mistake a delayed phase pulse for
spontaneous release.

In the fixed-low clamp experiment, the deficit stayed below this range and
the energy pathway successfully restrained expenditure. Real oxygen loss
now drives the circuit outside that operating range. The alarm first responds
71–72 ticks after the first actual energy deficit, consistent with its delayed
measurement and neural integration. This is useful feedback arriving too late
for the new expenditure history, not an unwired alarm.

The energy ledger distinguishes activation from net torque. By the first
energy deficit, common antagonist activation has spent .258–.274 J across
seeds. This is the accumulated term `2 * .1 * .004 * m0 * m1` in the identity
`m0² + m1² = (m0-m1)² + 2*m0*m1`. It is actual paid activation that does not
increase net torque. Removing it mathematically is an accounting comparison,
not evidence that a neural intervention could preserve the same trajectory.

Later the predictive populations reach the base model's membrane bound of
1000, followed by some velocity and joint cells. Predictive weights remain
inside their declared 0–1 interval while muscle outputs approach 1000 and
motion escalates. Restoring normal air does not undo this state. These late
trajectories are failure diagnostics, not viable physiology: the energy model
continues recording expenditure after debt instead of imposing an undeclared
host exhaustion brake. Numerical clipping is not evidence of neural stability.

The matched sensory control establishes an effect of actual oxygen feedback
on this failure. It does not yet isolate the cause of the later predictive
escalation. A forward-path intervention is needed to separate excessive
off-phase recruitment from the predictor's subsequent contribution. Freezing
learning or replacing real feedback with a permanent fixed signal would not
solve the research problem.

### What this changes next

The next intervention should preserve the full learned brain while making
phase authorization independent of deficit amplitude, then repeat the same
challenge and recovery. Compare a diagnostic predictor-to-muscle output cut
to determine whether the later instability is a second failure. Keep positive
adaptation, organ signals and physical consequences active in the candidate
organism. The target is useful composition across conditions, not a perfect
isolated respiratory component. A literature-grounded neural design and
explicit latency/range accounting must precede implementing that change.

Reproduction:

```sh
uv run --offline --no-sync --with cloudpickle==3.1.2 python -m simulations.active_inference.experiments.ventilation_changing_air .live/research/20260910_ventilation_input_clamp_seed11 NEW_CHANGING_AIR_OUTPUT
uv run --offline --no-sync --with cloudpickle==3.1.2 --with pytest python -m pytest tests/test_ventilation_changing_air.py -q
```

The four `20260910_ventilation_changing_air_seed{11,23,44,77}` folders retain
both complete courses, protocols and final executable neural/physical states,
including the environmental clock and sensory-clamp setting. The
`20260910_ventilation_changing_air_analysis/` folder retains checked per-tick
contrasts, all off-phase and membrane-bound events, stage-specific debt
increments and `challenge-recovery.png`. Every sample is retained. All four
simulation workers and the completed analyzer exited successfully; the
behavioral result is failure despite those successful executions.

## 10 September: phase-dependent recruitment separates two coupled failures

Sixteen new 2048-tick courses are complete, across seeds 11, 23, 44 and 77.
All branch the acquired neural state at tick 6352, with pending signals,
learned weights, physical state and sensory history preserved. All use the
same expanded 608-cell graph and the same changing-air course. The declared
interventions select the old or new motor route, with or without the two
predictor-to-muscle projections. Diagnostic zero weights remain zero under
the existing multiplicative rule; neurons, neural teaching, other learning
and native return pathways remain active in every condition.

The new five-cell circuit receives deficit D and the two existing rhythm
channels R. A matched excitatory relay P and inhibitory comparator Q supply
each output G. With held unit weights and without attenuation, its local
transformation is P=relu(D), Q=relu(D-R), G=3*relu(P-Q), or 3*min(D,R) for
nonnegative inputs. These are neural operations, not a Python online rule.
The implementation uses existing graded PAULA cells, distance-one dendrites,
positive adaptation and separate P terminals for the two consumers. No neuron
equation changed. The extra layer adds two ticks relative to the old relay.

[Yang, Murray & Wang (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5034308/)
motivates disinhibitory routing. This cancellation population does not reproduce
that paper's NMDA/GABA compartment model or establish a respiratory anatomy.
Its gating is imperfect. In the full predictor-connected new-route courses,
peak off-phase output is .000781–.000861, versus phase peaks near 2.94.
The audit uses actual arrivals with the measured three-tick Q-input-to-G-output
delay, rather than comparing simultaneous source and output samples.

| Motor route | Predictive muscle projections | Energy debt | Any recorded membrane bound | New oxygen debt during normal-air recovery |
| --- | --- | --- | --- | --- |
| Old additive relay | Connected | 4/4 seeds | 4/4 seeds | 4/4 seeds |
| Old additive relay | Diagnostic cut | 4/4 seeds | 0/4 seeds | 4/4 seeds |
| New phase-dependent route | Connected | 0/4 seeds | 0/4 seeds | 4/4 seeds |
| New phase-dependent route | Diagnostic cut | 0/4 seeds | 0/4 seeds | 0/4 seeds |

Every condition nevertheless incurs oxygen debt during the reduced-air phase.
This is not a fully viable organism. The connected new route maintains energy
above .0473–.0679 J across seeds, while its first oxygen deficit still occurs
at indices 720, 720, 721 and 720. It has 108, 98, 7 and 111 further oxygen-debt
ticks after normal air returns. Cutting its predictive motor projections removes
those recovery deficits, but not the earlier challenge deficit. A candidate
that disconnects acquired motor influence is not accepted as the solution.

The old route first incurs energy debt at 803, 803, 805 and 804. Cutting its
predictive projections moves these onsets to 817, 817, 815 and 817 but does not
prevent depletion. It does eliminate the later recorded membrane-bound events.
Thus predictive motor influence contributes to the large late escalation,
while the earlier energy failure can occur without it. This is a whole-course
intervention, not proof that a cut made at the instant of failure would recover
the same trajectory. Switching routes also changes latency and the earlier
body history; the outcome is not attributed to cancellation alone.

The added graph is not dynamically inert even when its muscle weights are zero.
Compared with the original 603-cell course, the expanded legacy control first
changes an old terminal coefficient at tick 3, muscles/body at 88 and selected
predictive weights at 91, in every seed. Its first oxygen and energy failures
remain at the earlier indices. The factorial comparisons therefore use this
same expanded graph, not an assumption of exact equivalence to the old brain.
The recorded CPG output trains remain identical across all four conditions.

All 32,768 ticks pass independent physical, resource, sensory-delivery,
predictive-learning and new-path input/integration checks. The new-path
residual is zero in these recordings. This does not independently rederive
every update in all 608 cells. Forty-seven focused tests pass, including local
range probes, preservation of acquired state and deliberate trace corruption.
All four simulation workers and the first analyzer exited successfully.

The raw courses and initial/final executable states remain in
`.live/research/20260910_ventilation_phase_composition_seed{11,23,44,77}`.
The checked full-tick contrasts and shared-axis figure are in
`.live/research/20260910_ventilation_phase_analysis_v2/`. The earlier analysis
is retained; v2 shortens overlapping axis labels and includes explicit group
identities, without changing the underlying recordings or numerical analysis.

This bounds the respiratory line rather than setting up another automatic
relay or gain repair. The new route is a useful, imperfect compositional part;
its improved behavior was designed, not acquired. The next research question
is whether the interacting learned populations can change their coordination
when consequences change, without the experimenter selecting a new operating
regime for each condition. Existing imperfect mechanisms and observation tools
should support that test. A remaining respiratory defect warrants more work
only if it demonstrably prevents that broader experiment.
