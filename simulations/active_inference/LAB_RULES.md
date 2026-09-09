# LAB RULES — the standing protocol (extracted from the paused challenge crons)

The 10-minute and 29-minute crons are **stopped**. Their rules are preserved here because they were
earned: nearly every rule below exists because breaking it produced a confident, wrong result.

---

## RULE -1 — READ `neuron.py` IN FULL BEFORE ANY NEURON-LEVEL REASONING

**919 lines, 41 KB. Read ALL of it — not a grep, not a recalled snippet — before designing, tuning,
explaining or debugging anything at the neuron/synapse level.** Also `network.py` when timing,
delays or external inputs are involved.

**WHY: my LIF priors are wrong here and they fire automatically.** Every mechanism error this
session came from substituting textbook LIF for what this model actually does. I do not notice the
substitution happening — it feels like knowledge, not assumption.

**PAULA IS NOT LIF. Confirmed divergences (verify each one in the source, do not trust this list):**

| textbook LIF | PAULA | what the wrong prior cost |
|---|---|---|
| `t_ref` = absolute refractory | `t_ref` = **plasticity LTP/LTD window**; refractory is **`c`** | "t_ref rate ceiling" -> built a whole recruitment population on a premise that does not exist |
| max rate = `1/t_ref` | max rate = `1/c` | same |
| adaptation reduces excitability | `t_ref` adaptation is **metaplasticity** (BCM sliding threshold), moves the LTP/LTD boundary, NOT excitability | called it "positive feedback, not homeostatic" — backwards twice over |
| leaky integrator + self-excitation = graded persistence | **spiking RESETS `S`**, so self-excitation is BISTABLE: forget (<=6.0) or latch (>=7.0), no graded regime | months of sweeps hunting a middle that cannot exist |
| threshold is fixed | `r = r_base + w_r·M`, `b = b_base + w_b·M` — **neuromodulated, and this DOES change firing** | missed the one real excitability lever while drafting a `neuron.py` change request |
| one threshold | `r` normally, **`b` during cooldown** (`<= c` ticks since last spike) | — |
| plasticity is per-synapse | reward gate + direction are **neuron-global**; only `info_val` is per-synapse | — |

**The tell:** if an explanation sounds like something I already knew before reading this codebase,
it is probably imported and probably wrong. Stop and open the file.

## RULE 0 — PER-TICK OR IT DOES NOT COUNT

**No summary statistic may be the primary evidence for any claim.** Ever.

Violated repeatedly, each time producing a false mechanism:

| what I did | what I concluded | what was true |
|---|---|---|
| summed 1200 ticks of EMD/HS spikes into one "selectivity %" | translation *inverts* the rotation estimate | seed-11 noise; 4 seeds straddle zero |
| replaced that with another run-average | HS has a low-angular-velocity **detection floor** | no floor — a **DC offset** shifting the whole tuning curve |
| read the arbiter per **step** not per **tick** | "the arbiter never fires" | it fired 1578/3200 |
| averaged bump position with a **wrapped** endpoint | bump displacement −0.13 | bump was running away at **+5.92** (laps the ring ~6×) |
| read a **binned profile** of \|h\| vs distance | recruitment restored graded distance coding | per-tick r within time windows ≈ 0 |
| read **overall** r(dist,\|h\|) | both L3 criteria met, 8 seeds | all of it was the Q1 charging transient |

**The procedure.** Every experiment logs a per-tick array: `t`, the driving variable (true ω / speed /
heading from world state), and every population's instantaneous spike flag. Then read the
**trajectory**, not its mean. Before claiming anything, answer *from the trace*:

- What is the **latency**? Does it **saturate**? Is the failure a steady **floor** or a **transient**?
- Does it **drift** over the run? Is **variance > mean**?
- Does the effect survive **within** time windows, or only **between** them?

A run-percentage is a hypothesis **generator** only. Confirm on the trajectory or drop it.

**Constant-drive tests throw away the regression.** Drive a *varying* ω and regress per-tick response
against per-tick true ω. A fixed `W.act(turn=…)` collapses the tuning curve to a single point.

---

## RULE 1 — CHECK THE HARNESS BEFORE BLAMING THE CIRCUIT

Five separate false results this session came from the test rig, not the brain:

1. **Walled path** — constant speed pins the agent at the arena radius (`dist` = 11.00), so the true
   home azimuth sweeps continuously and the L3 criterion is untestable. **Happened five times**,
   including in a "fixed" version where `turn=0.9` at `kyaw=0.36` rotates only **7.8°** in 24 ticks.
   *Now guarded:* the harness asserts `peak < 10.5 AND end < 0.6 × peak` and refuses interpretation.
2. **Small-n medians** — "4.9°, 100% under 30" rested on **88 of 3200 ticks (2.8%)**.
   *Now guarded:* n is always printed next to the median.
3. **Trace filename collisions** — `w_acc_self` wasn't in the filename, so runs overwrote the very
   traces being compared. *Now guarded:* every varied parameter appears in the name.
4. **Stale glob** — `fx_*.log` matched an unrelated earlier experiment's `DONE` markers and exited a
   wait loop before the real jobs had started.
5. **Protocol that separates the variables you need correlated** — a square path with
   `speed=0.9,turn=0` on legs and `speed=0,turn=1.5` on corners means the agent is **never**
   translating and rotating on the same tick, so the CD-vs-yaw regression had n=0..4 samples.

**Geometry that actually works** (arena 11.0, `kv=0.06`, `kyaw=0.36`, speed 0.9):
- travel = **0.054 units/tick** → a 100-tick leg covers 5.4 units
- rotation = **0.36 × turn deg/tick** → a 180° reversal needs `180/(0.36·TV)` ticks; TV=1.5 → **333 ticks**
- closed circle radius = `v/ω` = **8.6/turn** → turn=2.2 gives radius 3.9, diameter 7.8, never walls

---

## RULE 2 — ARITHMETIC BEFORE WEIGHTS

**Compute `(weight / lam)` against threshold BEFORE choosing any weight.** Seven errors, in both
directions — silence *and* saturation:

| circuit | setting | per-tick increment | threshold | result |
|---|---|---|---|---|
| `pi_accum` CD→ACC | `w_cd_acc=1.2, lam=6` | 0.20 | `r_acc=0.9` | **silent** |
| `pi_accum` CD→ACC | `w_cd_acc=9.0, lam=6` | 1.50 | 0.9 | **relay** (1 spike in → 1 out) |
| PG speed gate | `w_pg=w_pg_s=1.4, lam=2` | 0.70 + 0.63 = 1.33 | `r_pg=1.7` | **AND gate cannot fire** |
| ring self-excitation | `w_self=1.4, lam=2` | 0.70 | `r_ring=0.9` | sub-threshold; survives only ≥3 cells wide |
| accumulator→AGI | `w_acc_to_agi=0.22` | 0.66 | `r_agi=1.6` | AGI fired **0** times |

The integrating window is **narrow**: below threshold the cell is silent, above it the cell relays.
Sub-threshold-per-spike **plus** self-excitation is the only regime that sums over time.

---

## RULE 3 — VERIFY IT SPIKES **IN THE BODY**

`pi_accum` was validated in an isolated net (graded cosine lobes, |h|=175) and wired into the agent
**dead** (ACC alive 0.0%). Isolated validation proves nothing about the embodied build.

**Every circuit gets measured twice: in isolation AND in the body.**

---

## RULE 4 — WIRED IS NOT DRIVEN

Two occurrences of *a synapse that exists, reads as configured, and delivers nothing*:

- **`SEEDSYN`** — computed as `rj[RING[0]]-1`, but the tonic synapse is created *after* the seed, so
  the index pointed at the tonic. Every birth seed was delivered at amplitude 4.0 through a 0.12
  weight → 0.48 against threshold 0.9. Sub-threshold. Fixed via per-cell `SEEDMAP`; that one fix took
  `cx_navigator` from 156° to 18° home-vector error.
- **Ring tonic** — `cc.parts` wires a tonic synapse per ring cell as an **external input**, and
  external inputs deliver nothing unless `set_external_input` is called **every tick**. Nothing ever
  called it, so `w_tonic` was multiplied by zero at every value.

**Detection signal for both: an inert knob.** Which gives:

## RULE 5 — IDENTICAL RESULTS ACROSS A PARAMETER CHANGE = THE KNOB IS DEAD

Check the kwargs allowlists (`_cck`, `_mbk`, `_nvk`, `_cpk`, `_vck`, `_ak`) — several silently
dropped every parameter they were supposed to forward. Then check whether the target is an external
input that nothing drives. Then check whether the parameter is inert *by mechanism*
(`w_hs_opp` is post-threshold inhibition — too late to matter, identical from 0.0 to 6.0).

## RULE 6 — NEVER HARDCODE A SYNAPSE INDEX

Use a running counter. Five collisions: `SEEDSYN`, DOP/TEACH/GATE, Δ7→P-EN, vAC ids landing on
CPGP/RLY (would have wired walking rhythm into the aversive MBON with nothing crashing), and HS
`NAZ+i2` overwriting interleaved anti-preferred synapses.

**Check id budgets before growing a population.** `CRid = 300 + NR·NP + i·NP + p`, and `PG` starts at
1000, so `NP ≤ 9`. At NP=10 the CR block reaches 1019 and silently overwrites PG.

## RULE 7 — MEASURE THE RIGHT QUANTITY

- **Latch/ladder fill** = cells **spiking** in a ≥8-tick window. **Never membrane `S`** — spiking
  resets S to 0, so latched active cells read as empty.
- **Bump displacement** = cumulative **unwrapped** per-tick sum. Never a wrapped endpoint.
- **Bump sharpness** ≠ bump width. A 130° bump had **0.26°** centroid jitter and 24× discriminability.
  The config I called "sharpest" was a pinned attractor decoding one state regardless of seeding.
  (Rat head-direction tuning is ~90° wide; precision comes from population averaging.)
- **Hold test**: total spikes held at 100% while the *spatial profile* diffused to uniform. Measuring
  total activity passed a circuit that had already lost all its information.

## RULE 8 — REPLICATE ON ≥4 SEEDS, REPORT SPREAD

Single-seed effects died on replication twice. Report the spread, not just the mean. Seeds used:
11, 23, 44, 77, and for 8-seed runs add 5, 13, 91, 7.

## RULE 9 — LITERATURE FIRST, AND NOT FLY-ONLY

Check the actual papers before designing a circuit, and check whether the mechanism generalises
beyond *Drosophila* (this is not a fly). Errors caught this way: an **antipodal drain** I invented
that appears in no organism (Stone 2017 uses a population code; bidirectionality comes from the code,
not from subtraction), and `sh_bank`, which *is* the paper's real mechanism (Turner-Evans 2017:
P-EN/E-PG phase offset varies linearly with rotational velocity) and which I wrongly killed on a
metric bug.

## RULE 10 — EXECUTION HYGIENE

- Parallelise 8 configs per sweep.
- **zsh does not word-split unquoted vars** — `set -- $cfg` silently fails; killed 6 jobs twice.
- **Never block on `until` in the foreground** — a tool timeout SIGTERMs the process group and kills
  the nohup children. Launch, then wait in a separate background call. (`setsid` doesn't exist on macOS.)
- Editing a `def` signature: **a multi-line comment inserted mid-line swallows every parameter after
  it** (this broke `w_eff`/`w_rly_eff`). Put comments *above* the `def`, then `ast.parse` to verify.
- **Every wait loop needs a bounded lifetime.** An `until [ $(... | grep -c DONE) -ge N ]` loop whose
  condition can never become true spins forever once its tool call returns. Three such loops survived
  for hours because the sweeps they waited on had finished under *different log filenames*. Always cap
  the loop with an iteration counter, and make the waiter watch the **exact filenames** the launcher
  wrote — never a glob, which can also match an unrelated older experiment's `DONE` markers.

```zsh
i=0; until [ $(cat exact_1.log exact_2.log 2>/dev/null | grep -c DONE) -ge 2 ] || [ $i -ge 40 ]; do
  sleep 45; i=$((i+1)); done
```

---

## RULE 11 — `neuron.py` IS NOT MINE TO EDIT

Shared with the C. elegans work. No changes without explicit approval **and** a quantified proposal
including a regression showing nothing else moves. Reading it is always allowed — and reading it
corrected two load-bearing misconceptions:

- **`t_ref` is not the firing refractory.** The refractory is `c`:
  `will_fire = S >= active_threshold and (tick - t_last_fire) >= c`.
- **`t_ref` is the LTP/LTD BOUNDARY and drives BOTH**: `direction = +1 if delta_t <= t_ref else -1`
  — inputs inside the window potentiate, outside they depress.
- **Its adaptation is HOMEOSTATIC** (the code names it `t_ref_homeostatic`):
  `t_ref = upper - (upper-lower)·normalized_F_avg`, so more activity gives a *shorter causal window*,
  meaning an over-active neuron depresses more of its inputs. A BCM-style sliding threshold that
  limits runaway potentiation. Neuromodulators shift it further via `w_tref · M_vector`.

A whole design (the recruitment population) was justified by a "rate ceiling from `t_ref`" that does
not exist. **Read the model before reasoning about its dynamics.**

## RULE 12 — SCALE A CIRCUIT AS A STRUCTURAL HYPOTHESIS, NOT AS A MAGIC FIX

When a PAULA circuit is too coarse, too jittery, lacks dynamic range, or is
trying to make one cell do a population's job, **grow the relevant neural
population before inventing a new neuron behavior or moving computation into
Python.**  Scaling is a legitimate circuit intervention: add cells with a
declared neural role, signed wiring, and controlled heterogeneity.  It is not
permission to duplicate a broken loop until one endpoint happens to look good.

Choose the scaling axis from the predicted failure:

| Observed limitation | Structural scaling candidate | Required causal prediction |
| --- | --- | --- |
| Angular/spatial quantisation | More ring columns or more phase bins | Smaller physical changes alter the decoded state without inducing a second bump or losing liveness. |
| One threshold creates a dead zone or saturation | Threshold/gain/time-constant population | Recruitment changes smoothly with the driving current; each member's trace has a distinct operating range. |
| Fast alternating sensory drive cancels | Opponent sensory or interneuron population with distinct time constants | The signed population residual predicts physical rotation **within strokes**, not merely its run average. |
| Weak/noisy motor output | Distributed relay/motor population with graded readout | Population activity changes muscle force and body motion under a connection/output ablation. |
| One state represents incompatible functions | Separate maintenance, shift, readout, and inhibitory populations | Ablating one role removes only its predicted causal contribution. |

### Scaling preflight

1. State the biological/circuit reason for the population and what varies
   between members (threshold, time constant, dendritic target, spatial phase,
   or signed projection).  Do not add identical copies merely to increase
   total conductance.
2. State whether total incoming/outgoing conductance is held constant or is
   intentionally increased.  Normalise when testing representation/resolution;
   increase it only when testing capacity and report that explicitly.
3. Recompute the ID budget, per-neuron fan-in, `t_ref` bounds, delay
   attenuation, and terminal/synapse numbering before building.  The existing
   `CRid = 300 + NR·NP + i·NP + p` layout reaches the `PG=1000` block at
   `NP=10` for `NR=36`; a larger population needs a new, non-overlapping ID
   allocation rather than a silent collision.
4. Preserve the default build and expose the scale through an explicit,
   rebuild-gated configuration.  A population index must never select an
   external target, body coordinate, heading, or behavior in Python.
5. Log every member or a lossless per-column/per-population raster.  A mean
   can hide common-mode firing, cancellation, a single overactive cell, or a
   dead added population.
6. Compare baseline, scale-only, and the intended wiring/heterogeneity
   condition in both the isolated harness and the embodied body.  Keep the
   same seed/world course where that isolates the structural change; then use
   the normal multi-seed replication before accepting a result.

### What this project has actually shown

Population construction has already solved several *local, causal* problems.
These are reusable patterns, not evidence that the unverified compass/homing
stack works:

| Reusable PAULA pattern | What worked | Evidence boundary |
| --- | --- | --- |
| CPG → staggered relay bank → graded muscle population | Distributed relays and graded muscles produced forward motion and opposite physical turns; `w_cpg=0` or zero muscle gain removed actuator drive and displacement. | Accepted motor/steering primitive; not navigation. See `experiments/paula_motor_causal.py`. |
| Antennal-lobe / sparse Kenyon-cell / APL-inhibition / MBON / LH population route | Toxin teaching in the presence of food odour prevented later food collection; removing the LH avoidance output restored collection while learned MB activity remained. | Accepted local counterconditioning, not general valence learning. See `experiments/embodied_mb_food_avoidance_causal.py`. |
| Bilateral toxin sensors → temporal trend population → STEER | The `TXL/TXR→TPOOL→TRISE→STEER` population removes the head-on toxin blind spot; `w_trise=0` restores a physical contact. | Accepted local hazard response, not lifetime survival. See `experiments/embodied_headon_toxin_causal.py`. |
| Hunger population + PAULA WTA arbiter + background search population | Food contact drains hunger and produces the measured FORAGE→EXPLORE/SEARCH transition without a Python action branch. | Accepted narrow interoceptive transition, not general active inference. See `experiments/embodied_arbiter_explore_causal.py`. |
| Full tick trace plus a matched raw-sensory replay | Replaying the recorded physical yaw through the same PAULA route reproduced the live compass phase exactly in the current long-turn diagnostic. | A diagnostic method: it rules out a host/physics handoff explanation; it does not make the compass accurate. |

### Compass-specific negative guardrails

The following are evidence against treating population size as an automatic
compass repair:

- A literal 72-column ring/P-EN expansion and an eight-pair heterogeneous
  vestibular afferent bank both remained inaccurate in full embodiment.  More
  cells did not repair the missing stable P-EN→ring travelling-wave dynamics.
- The experimental `PhaseLockedGradedNeuron` sample/reset/hold subclass is
  PAULA-internal but phenomenological, not a faithful P-EN/central-complex
  cell model.  Its isolated clock behavior and its embodied failures belong in
  diagnostics only, never in an accepted biological claim.
- An ordinary recurrent CPG plus signed graded integrators and inhibitory
  reset *does* operate in the body, but its best current long turn reaches
  ring −59.2° for body −122.0°.  Increasing the P-EN→ring delay to 44 ticks
  or adding reciprocal integrator inhibition makes it worse.  The next
  compass change must address the spatial P-EN/E-PG topology, not add another
  untraced smoothing scalar.
- An explicit 44-cell ordinary PAULA CPG-phase population plus 72 PB
  coincidence gates was a useful **scaling diagnosis**, not a repair. With
  full-tick body input, every selective gate remained below threshold (peak
  1.98 versus 2.10), so the ring received no update. A one-time,
  evidence-derived increase of only the local E-PG dendrite restored neither
  precision nor robustness: it made both directions fire broadly (20,390 PB
  spikes in the fixed-current harness) and eliminated signed motion. Before
  scaling a gate population, measure active-column and background dendritic
  margins; if they overlap, add an explicit contrast/competition population
  rather than duplicating or amplifying the same source.
- A structural relay changes temporal causality. The PB bridge initially
  acquired an unaccounted PAULA update tick; an impulse harness caught it
  before topology conclusions were drawn. Every added neural layer therefore
  needs a one-impulse latency check, followed by full-tick isolated,
  recorded-replay, and embodied traces. Never infer timing from period means.

Therefore: **scale populations first when the trace identifies a population
capacity problem; rebuild the circuit topology first when the trace identifies
a wrong causal transformation.**  Do not claim either route works until its
full-tick, isolated-plus-embodied, multi-seed evidence passes.


---

## RULE 13 — VERIFY THE TEMPORAL INTERFACE BEFORE SCALING ITS CONSUMER

Before adding a larger downstream population, compare its actual incoming
signal histories for conditions requiring different responses. Match initial
state and inspect every channel at every tick. Equal totals are insufficient;
exactly identical histories are a stronger counterexample. State explicitly
whether the observer receives spikes alone, release amplitudes, modulation,
additional projections or feedback. Equality of recorded soma fields does not
establish equality of weights, terminals, eligibility traces or in-flight signals.

The 9 September association timing experiment found opposite physical cue
majorities with identical 64-tick consumer histories in 64 matched comparisons.
This rules out resolving those majorities from that fixed output channel alone,
not from the entire PAULA state or a feedback-coupled architecture. Candidate
completion can discard evidence strength that a later integrator needs.

Do not require a new response after the final receptor arrival if useful recall
already occurred. Do not label an early competing candidate wrong before its
evidence is complete. Distinguish noisy fragments of one event from actual
successive events; the world/task must justify that distinction. Test temporal
interfaces on full trajectories before interpreting a whole-probe pass/fail.

## RULE 14 — THE COMPONENT INTERFACE INCLUDES PLASTIC RETURN PATHS

Do not infer a component's influence from its spike raster alone. Record actual
presynaptic release, terminal modulation and relevant retrograde events when
comparing composed circuits. A silent postsynaptic cell may still send native
plasticity errors that alter an upstream terminal. Check whether that terminal
is shared across several projections before interpreting an intervention as
local to one forward connection.

In the 9 September real-media experiment, changing selected auditory input
weights altered visual terminal information at tick 10 and incoming amplitudes
at tick 12, before auditory spike outputs diverged at tick 18. A selective
return-event lesion delayed the first two effects to ticks 28 and 27, while
the direct auditory soma and spike differences remained at ticks 8 and 18.
Both intact observer prefixes reproduced the original full-field records.
See `media_backchannel_probe.py` and `media_order_audit.compare_backchannel`.

This is a diagnostic rule, not permission to eliminate plasticity or feedback
to make a component easier to validate. Keep positive adaptation in the brain.
Use declared temporary lesions to identify paths, then test the whole coupled
system and its functional use. First divergences are local experimental facts,
not general conduction constants, and an observed return-path effect alone
does not establish useful memory, self-regulation or consciousness.

## RULE 15 — ELAPSED TIME IS NOT LOCAL LEARNING EXPOSURE

Do not infer sufficient acquisition from equal presentation durations or a
fixed number of simulator ticks. Measure the actual local learning coefficients,
neuromodulatory rates and resulting weight trajectories. Preserve weak positive
adaptation; absence of eligible events is different from a frozen learning rate.

For the current selected eligibility rule and observed activity history,
`q(t) = A(t) q(0) + B(t)`, with
`A(t) = exp(-sum eta*(Lplus+Lminus))`. Save the full per-port trajectory when
using this decomposition. A is a conditional initial-weight coefficient, not a
fraction of semantic memory or the derivative of the closed-loop neural system.
Changing a weight may change subsequent spikes, rates and return signals.

The 9 September three-architecture real-media comparison found one clip supplied
92.8 through 98.4 percent of the selected pathway's summed local exposure during
audiovisual presentation despite equal clip durations. Final median A ranged
from .971 to .987 across those histories. This prevents an equilibrium claim,
but does not prove that longer training will repair recall. Use declared
duration comparisons and distinguish acquisition from expression and neural use.
See `eligibility_exposure_audit.py` and the terminal-comparison findings.

## RULE 16 — BRANCH THE EXECUTABLE STATE, NOT ITS DISPLAY SNAPSHOT

Counterfactual probes must start with the same actual neural state except for
their declared intervention. Equal displayed fields or JSON values do not
guarantee equal numeric types, queued signals, cache aliases or random state.
Use a full object clone or an exact trusted-local runtime checkpoint. Verify an
uninterrupted continuation against reload with active inputs and signals in flight.

`core/runtime_checkpoint.py` preserves those dependencies for the tested PAULA
preparation. Use its branch stepping interface for isolated random streams and
separate processes for concurrent branches. This is executable local data, not
an upload format. It does not include a body or world; see its usage document.
Do not erase plasticity to make replay easier. A weight intervention changes
one part of a coupled dynamical state and may immediately alter return pathways.

## RULE 17 — A MEMORY OBSERVER CAN DRIFT OR HIDE A LEARNED CONTRIBUTION

Report which physical experience and neural state define a reference pattern.
Compare pre-experience and current references when representations change.
Do not choose whichever observer makes a candidate pass. In the 9 September
weight-cycle experiment, all six reversed-pairing cycles had the expected late
sign under the original sound reference but the opposite sign under the trained
reference. Neither alone establishes what a neural consumer can use.

Conversely, requiring a learned effect to reverse the total response preference
can reject a real contribution hidden beneath a stronger sensory bias. Separate
intact, reset and structure-preserving interventions on full trajectories. Report
their limits: a final-weight shuffle changes birth placement too, and a
history-dependent gain can produce an assignment interaction. Functional use by
the neural system, across controls and independent seeds, remains the target.

## RULE 18 — DISTINGUISH AN INPUT CLOCK FROM AN INTERNAL RHYTHM

Inspect the actual sensory scheduling protocol before interpreting oscillatory
neural activity. Test a uniform or otherwise noninformative stimulus through
the real encoder, not only simultaneous constant-current fixtures. Compare a
declared alternative schedule with matched external dose and retain every
channel's time course. Equal external dose does not imply equal local learning
exposure, so that difference must remain part of the interpretation.

The four-tick media encoder produced periodic contrast responses to a spatially
uniform image in the 1,441-cell preparation. Mean-dose-matched continuous input
greatly reduced that response. Both added-population equation audits were exact.
See `population_input_phase_probe.py` and the corresponding full recordings.
This identifies external pacing, not an invalid sensory protocol or proof that
all neural oscillations are externally imposed. A wave display is not evidence
of an autonomous regime, a learned attractor, or workspace synchronization.

## RULE 19 — RECORDED ARRAYS NEED EXPLICIT COLUMN IDENTITY

Never reconstruct a recorded array's column order from a serialized dictionary's
iteration order. Record neuron IDs, port IDs and field names in the same order
as the array. A JSON encoder may sort dictionary keys while leaving the values
and neural state unchanged. Verify reloaded recordings against an exact replay
prefix before using them for a causal comparison.

The contrast weight-transplant control caught precisely this error: population
allocation order differed from alphabetized JSON keys. Recovering the known
neuron-ID allocation order restored exact agreement. New transplant manifests
declare that ID sequence; the adapter has a JSON round-trip test. The failed
attempts are not accepted results and no old recording was silently rewritten.

## RULE 20 — AN ERROR SIGNAL IS NOT A FAMILIARITY LABEL

Compare learned histories on the same physical input before calling a response
expectation-dependent. Retain the actual opponent channels, sensory arrivals,
prediction arrivals and local learning-rate traces. A louder stimulus, a silent
population or a positive endpoint is not sufficient evidence of useful error
computation. Include weight-placement controls and every unfavorable interval.

Conversely, do not demand lower error for the familiar history on every tick
without a task that justifies that criterion. A contextual statistical prediction
can be worse on one observation. The 9 September audiovisual mismatch assay
found a brief burst where the contrary-history prediction fit better, despite
favorable surrounding and late contrasts. Its full neural audit passed. This
distinguishes imperfect temporal prediction from a recording or wiring failure.
Check whether the available sensory/action history actually predicts the event
before adding circuitry to anticipate it.

Supervisory inputs need their own causal account. A published learning rule
that assumes a global error estimate does not establish how PAULA generates
that estimate. Specify which neural signals reach a regulator, what it changes,
and whether the resulting dynamics remain useful under ongoing adaptation.

## RULE 21 — CHECK WHAT THE BODY CAN TRANSMIT AND THE CONTEXT CAN SUSTAIN

Inspect compiled physical units, actuator conversion and sensor saturation,
not only the XML text or animation. Record raw physical quantities alongside
their encoded inputs. A bounded afferent that stays at its rail cannot expose
movement precision to any downstream population. The 9 September rower probe
found degree/radian ambiguity and excessive drive before studying learning.

Before extending acquisition, ask whether the actual contextual release
history and allowed synaptic strengths can generate the target's temporal
profile. A body can remain displaced after motor-copy current has faded.
Learning traces need not be forward memory. Conditional capacity bounds must
state their input histories, initial-state bound, equations and numeric slack;
they are not universal claims about PAULA or requirements of perfect prediction.
Grow temporal/state representation when the evidence identifies that failure.

Auditors must reproduce dtype conversions at physical and neural boundaries.
Do not widen tolerances to hide an arithmetic-order discrepancy. Preserve
failed checks, validate the corrected checker against deliberate corruption,
and recover missing checkpoints only through verified executable continuation.

## RULE 22 — A PLASTIC RECALL PROBE IS ALSO A LEARNING EPISODE

With ongoing adaptation, an initially poor memory can be repaired during the
same probe used to score it. Retain the onset, every selected update, actual
neural teaching arrivals and the physical trajectory. Compare matched executable
states with the declared learned weights reset. Do not assume late convergence
means the earlier memory was unnecessary, or that good final behavior proves
the desired response was already stored.

When needed, use an explicit diagnostic lesion of neural teaching pathways to
separate expression from rapid reacquisition. Keep positive basal rates and
record residual state, continuing updates and any affected retrograde paths.
This is an intervention, not a mechanism to install in the agent. Verify an
unchanged replay before interpreting the lesion; do not hide its broader effects.

In the 9 September context-organization experiment, teaching-path interruption
preserved compensation for one recording but exposed persistent wrong responses
for the other across four seeds. Ordinary probes had repaired those responses
online. Both training blocks ended with the first recording, so recency remains
unresolved. The finding is not proof that zero latency or error-free recall is
biologically necessary. Also check task necessity: a continuously available
direct load signal can let feedback sidestep the memory capability being tested.

## RULE 23 — A RELEASE EXTENSION CHANGES THE PLASTICITY EVENTS IT INHERITS

When a subclass suppresses somatic spikes, inspect every inherited use of
`t_last_fire`, not only its forward output. In the current graded preparation,
native retrograde timing direction stays negative. At a negative incoming
information weight, the native error compares positive arrival amplitude with
that negative weight. Returning this error can push a presynaptic information
coefficient through zero even when all incoming weights keep their signs.

The sixteen-exposure audiovisual course found precisely this at a context
terminal shared by 192 inhibitory targets. Its coefficient crossed below zero
by block eight in the retained checkpoint series. Negative incoming information
is ignored by the base input mask, so a source with active soma and intact
inhibitory wiring ceased to suppress its target bank. Four full-state terminal
restorations recovered suppression but did not fix the joint association.
See `crossed_av_gate_probe.py` and the continuation findings.

Record terminal coefficients, actual arriving release and return-event counts
alongside soma output and postsynaptic weights. Check both sign conventions and
feedback fanout. Dividing fanout or lowering a rate can delay a failure without
repairing it. A temporary restoration is a causal diagnostic, not an autonomous
solution; keep plasticity active and verify any repair in the coupled brain.

## RULE 24 — A BODY LOOP DOES NOT AUTOMATICALLY CLOSE THE LEARNING LOOP

When a learning change alters action and later neural error, do not infer that
the error changed because of the action's physical consequences. A delayed
internal prediction path can produce the same chronology. Branch the acquired
state, replay identical bodily afferents while letting the actual body move,
and retain the real sensors separately from delivered inputs. Verify unchanged
replay first. This sensory substitution is a diagnostic, never a host policy.

The 9 September eligibility-reference experiment found that bodily feedback
changed joint neurons, muscles and motion in all sixteen seed/pair cases, but
did not change predictor output, selected weights or learning error within
96 ticks. Error changes persisted with identical sensory input. The external
load was independent of action, and the joint pathway fed only a restoring
reflex. This identified separable learning and body-control loops, despite
their being drawn as one embodied network.

Inspect return equations as well as forward edges. An input-local return need
not transmit a different input's contribution to the same neuron's membrane.
State the observed time horizon and intervention limits. Make consequential
sensorimotor coupling a task requirement before calling improved exogenous
prediction an acquired model of the organism's own action.

## RULE 25 — CONSTRUCTION MUST PRESERVE THE ACTUAL INPUT STATE

When attaching a population to an acquired network, inspect consumed inputs,
pending events and cache reconstruction as part of the intervention. An old
bootstrap value in a backing dictionary can differ from the authoritative
vectorized input state. Rebuilding that cache can replay a startup pulse even
when the driver reports zero new input and every old neuron object survives.

The 9 September organ-feedback experiment demonstrated this across four seeds.
A stale cache rebuild and a declared second pulse produced identical full
records; synchronizing before rebuilding reproduced unchanged continuation.
The extra wave appeared after 40 ticks and changed the body two ticks later.
Record actual receiving-port values, including the sum of external and recurrent
inputs when they share a port. An exact replay of an already-confounded initial
state proves repeatability, not validity of the original construction.

This does not make native plastic return pathways a nuisance to remove. Once
construction is controlled, their contribution remains a legitimate causal
question. A zero forward weight and a clamped sensory signal test different
relationships. Keep both the physical measurements and any experimentally
substituted inputs in the record. See `ventilation_reseed_probe.py` and
`VENTILATION_REGULATION_2026-09-09.md` for evidence and unresolved scope.

## RULE 26 — A CIRCUIT'S FUNCTION CAN CHANGE OUTSIDE ITS TESTED INPUT RANGE

A name such as coincidence gate or regulator is a hypothesis about a circuit's
actual dynamics. Test it on the signal ranges created by the coupled organism,
including challenge and recovery. Bounded individual weights do not establish
stability of the brain–body loop. Preserve latency, native return paths and
the actual input-to-output transformation when interpreting an intervention.

In the 10 September changing-air course, deficit alone crossed an additive
phase relay's threshold. Both antagonists then received output without rhythm
input. Energy depletion preceded the delayed alarm, and later predictive
activity reached the numerical membrane bound. The four-seed fixed-low-input
success had never exercised that regime. Full relay-potential reconstruction
identifies the off-phase release; the later predictor escalation still needs
its own causal intervention. A completed run or bounded endpoint does not
turn those failure dynamics into self-regulation.

## WHERE TO LOOK

- **`ROADMAP.md`** — status, the 8 numbered failures (F1-F8), and the ordered Phase 0-4 plan
- **`ARCHITECTURE.md`** — every population, measured evidence, confidence grades, structural traps
- **`LAB_RULES.md`** — the measurement protocol. Rule 0: per-tick or it does not count
- **`RESEARCH_LOG.md`** — what was tried, what it cost, what was retracted
- **`live_brain.py --port 8770`** — live UI: structure switches, parameter sliders, rebuild
