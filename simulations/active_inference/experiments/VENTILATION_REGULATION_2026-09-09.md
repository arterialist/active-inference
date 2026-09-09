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
