# Research log

## 2026-07-29 (late) — READING neuron.py IN FULL OVERTURNS TWO "STRUCTURAL" CONCLUSIONS

Prompted by the observation that I keep substituting textbook LIF for this model. Read all 919 lines
instead of grepping. Two findings, both of which invalidate a conclusion I had called structural.

### 1. DENDRITIC DISTANCE ATTENUATES. Every delayed synapse is scaled by 0.95^dist.

`neuron.py:621  V_arriving = V_initial * (delta_decay ** distance)`, `delta_decay = 0.95`, and
ckit's `syn(nid, sid, w, dist)` maps `dist` straight onto `distance_to_hillock`. So dendritic
distance is BOTH a delay and an exponential attenuation. I had been using it as a pure delay -- my
own note calls it "a precise functional primitive (S peaks at tx+distance)", which is true about the
TIMING and silent about the AMPLITUDE.

| circuit | w | dist | arrives | threshold | % of thr |
|---|---|---|---|---|---|
| **CPU4 ladder advance (`w_prev`, `d_adv=90`)** | 2.40 | 90 | **0.0237** | 1.60 | **1.5%** |
| CPU4 ladder drive (`w_drive`) | 1.00 | 1 | 0.9500 | 1.60 | 59% |
| CPU4 ladder latch (`w_self`) | 5.00 | 1 | 4.7500 | 1.60 | 297% |
| compass shift push (`w_push`, d=8) | 1.10 | 8 | 0.7298 | 0.90 | 81% |
| **RISE/TRISE delay bank (dist 35..110)** | 3.40 | 110 | **0.0121** | 0.60 | **2.0%** |

**The CPU4 ladder was never a ladder.** Its rung-to-rung advance arrived at 1.5% of threshold, so
every rung saw only drive (59%) plus latch (297%) -- which IS a bistable latch. I measured that
bistability, called it "structurally binary, no graded regime exists", and built `pi_accum` to
replace it. The defining mechanism had been attenuated away by a factor of 100 and I never checked.

**The RISE trend detectors are compromised the same way.** The delay bank spans 35..110 ticks, i.e.
0.95^35 = 0.166 down to 0.95^110 = 0.0036, so the "delayed" arm of each rise detector arrives at
17% to 0.4%. A rise detector whose delayed arm is missing is not computing a difference, it is
relaying its input. This affects RISEP (food) and TRISE (toxin, arm C of the run in flight).

### 2. THE SUBSTRATE HAS A SLOW ANALOG STATE VARIABLE. I claimed it did not.

Each neuron carries `M_vector`, an EMA of neuromodulatory input with `gamma = [0.99, 0.995]`
(tau ~ 100-200 ticks). **Spiking resets `S` but NEVER touches `M_vector`**, and M drives excitability
directly: `r = r_base + w_r . M`, `b = b_base + w_b . M`, plus the learning window via `w_tref . M`.

My "no graded regime: forget below 6.0, latch above 7.0" result was about SYNAPTIC self-excitation,
which is bistable precisely BECAUSE spiking resets S. Neuromodulatory feedback is a different
pathway with its own persistent analog state that spiking does not clear. I generalised a result
about one pathway into a claim about the substrate, and was drafting a `neuron.py` change request on
that basis.

The channel is documented in ckit and armed on the receive side:
- `syn(...)` sets `u_i.adapt = [0.3]*nm` -- postsynaptic receptors ON by default
- `term(...)` sets `u_o.mod = [0.0]*nm` -- presynaptic senders OFF by default, with the comment
  "Set it to drive the TARGET's M_vector ... That is the substrate's gain-control channel."
- `nm_internal = 1.0` in NeuronParameters -- neuron-to-neuron neuromodulation enabled

So the mechanism is one parameter away and has never been used anywhere in the agent.

### Consequences
- **F2 (no graded integrator) is NOT established.** It was measured on synaptic self-excitation only.
  Neuromodulatory self-feedback (population -> interneuron -> `mod` -> target's M -> target's `r`)
  is an untested route with a ~200-tick time constant, and it is what neuromodulation does in
  biology rather than a trick.
- **The CPU4 ladder deserves a re-test at a sane `d_adv`** before `pi_accum` is treated as its
  necessary replacement.
- **Any circuit using dendritic delay >20 ticks needs its amplitude re-checked.** 0.95^20 = 0.36.
- No measurement is retracted; the ATTRIBUTIONS built on top of them are.

---

## 2026-07-29 (late) — L3 resolved into TWO channels; L2 velocity encoding confirmed structural

### THE L3 RESULT: both criteria are achievable, at DIFFERENT operating points, differing in ONE knob
Measured in the body, curved path, 4 seeds, per-tick, sample sizes reported:
  cd_neg=1.0 (SIGNED cosine CD)    -> direction 17.7-30.0 deg  |  r(dist,total) +0.056..+0.100
  cd_neg=0.0 (RECTIFIED cosine CD) -> direction 79-118 deg     |  r(dist,total) +0.342..**+0.830**
Mechanically coherent: RECTIFIED CD only ever charges, so total activity accumulates with distance;
SIGNED CD cancels on reversal, so the vector ANGLE stays clean but nothing accumulates.
No readout transform bridges them -- power sharpening p=2/4/8 and hard WTA were all tested on the
broad traces and plateau at 55-63 deg, never under 30.
=> DESIGN (measurement-determined, not guessed): TWO ACC populations from the SAME CD, one signed
   (feeds the DIRECTION readout), one rectified (feeds the MAGNITUDE readout). Parallel channels with
   different tuning reading a common input is what the mushroom body and visual system here already do.

### L2: ring persistence FIXED; velocity encoding is STRUCTURAL
**FIXED - the ring tonic was never injected.** cc.parts wires a tonic synapse per ring cell as an
EXTERNAL input; external inputs deliver nothing unless set every tick, and nothing ever set it, so
w_tonic was multiplied by zero at every value (0.12 and 1.2 gave byte-identical traces -- the inert-knob
signature). The ring was running on w_self/lam = 1.4/2 = 0.70 against r_ring=0.9 plus two neighbours:
a bump >=3 cells wide survives, narrower collapses irreversibly. Over 8 seeds at ZERO angular velocity
the bump died at tick 36/36/36/51 on 4 of 8 and never recovered. With the tonic injected: **alive 8/8,
every gain, every NP**. Same class as SEEDSYN -- a synapse that exists, reads as configured, delivers
nothing. That is now TWO occurrences; both were caught by the inert-knob rule.

**A recorded NEGATIVE is superseded**: "lower thresholds kill the ring attractor" was true when the ring
had no floor. With the tonic, r_conj_lo=0.90 keeps the ring 100% alive AND makes slow turns visible for
the first time (registered 0.0 deg -> 43-56 deg).

**STRUCTURAL, 11 routes falsified**: the P-EN shift is a threshold-gated RESONANT wave, not a velocity
integrator. At fixed gain, registered/actual rotation = 0.00 (TV=0.5), 0.01 (TV=1.0), 1.00 (TV=1.5),
0.80 (TV=2.5) -- zero response below a critical drive, over-response above. Falsified this session:
threshold level, threshold spread (0.90-1.75 sends slow turns back to 0.00-0.06), population size
(NP 4->9 gives ratios up to 2.23 and one SIGN INVERSION -- extra shift cells add drive to the SAME wave,
they do not add independent range the way extra ACC cells did), and gain (0.42->1.00 but 0.44->0.23,
non-monotonic). Plus the eight already on record.

**Tonic trade-off, single-variable, 4 seeds**: tonic ON degrades direction 20.6-23.7 -> 38.6-81.9 deg
while roughly tripling r. Velocity-GATING the tonic recovers direction (21.7-28.9) but loses r
(-0.087..+0.004). The two criteria are coupled through BUMP WIDTH; gating does not separate them.
This is the same coupling the two-population design resolves.

### Harness bugs that produced FALSE results this session (all now guarded)
* walled path (dist pinned at arena radius 11.0) -- THREE times, incl. a "fixed" version where
  turn=0.9 at kyaw=0.36 rotates only 7.8 deg in 24 ticks. Now asserted: peak<10.5 AND end<0.6*peak.
* small-n medians: "4.9 deg, 100% under 30" rested on 88 of 3200 ticks (2.8%). n now always reported.
* trace filename collisions (w_acc_self not in the name) overwrote the traces being compared.
* stale-glob: fx_*.log matched an unrelated 05:15 experiment's DONE markers, exiting a wait loop early.
* square path separated rotation and translation completely -> the CD-vs-yaw regression had n=0..4.

### Retractions
"translation inverts the rotation estimate" (seed-11 noise; 4 seeds straddle zero) · "low-omega
detection floor" (no floor -- a DC offset) · "recruitment restored graded distance coding" (binned
profile rose, per-tick r within time windows ~0) · "3/4 seeds cancel" (2 of the 3 had DEAD rings;
only seed 44, ring 74.5%, ratio 1.07, -79.1%, was genuine) · "no bidirectional integration, needs a
neuron.py decision" (wrong -- it was my own w_acc_nbr diffusion plus signed-vs-rectified CD, both
circuit-level) · "cd_neg falsified" (artifact of the square path).

---

## 2026-07-29 (cont.) — L3 chain traced end to end: PG gate, harness, and an integrator that RELAYS

### PG speed gate had invalid AND-gate arithmetic (SEVENTH weight/lam error)
`r_pg=1.7, w_pg=w_pg_s=1.4, lam=2`. Per-tick increment 1.4/2=0.70 (ring) + 1.4*0.9/2=0.63 (speed)
= **1.33, BELOW the 1.7 threshold**: the gate could not fire even with both inputs present. That
starved CD (CD/tick 0.35) and therefore the accumulator. With `w_pg=2.4, w_pg_s=2.6` the gate is
clean: **PG/tick 12.60 moving vs 0.03 stopped (420:1)** and **CD/tick 0.35 -> 1.97**.

### Two harness bugs of my own, both of which produced "circuit" results
1. Constant speed pinned the agent at the arena wall (`dist` 11.00 = radius) for ~15/16 of every run,
   so the true home azimuth swept continuously and the |h|-vs-distance criterion was untestable.
2. The "fix" still walled it: `kyaw=0.36` means `turn=0.9` yields **0.324 deg/tick**, so a 24-tick
   turn phase rotates the body 7.8 deg, not 90. The agent walked nearly straight into the wall.
   Correct geometry: a 90 deg corner needs ~90/(0.36*TV) ticks; TV=1.5 -> 167 ticks, and
   TV*k_ang(0.65)=0.975 stays under w_cap=1.2 so the compass drive does not saturate.
   With that, `dist` spans **0.04-9.15** and the criterion is finally measurable.

### THE ACCUMULATOR IS A RELAY, NOT AN INTEGRATOR
Per-column spike counts are IDENTICAL between CD and ACC:
  CD  524 559 519 516 534 573 515 514 515 524 503 521
  ACC 523 558 519 516 534 573 515 514 515 523 502 520
Every CD spike fires exactly one ACC spike. `w_cd_acc=9.0, lam_acc=6` -> **1.5 per single spike vs
r_acc=0.9**. I moved it from SILENT (1.2/6 = 0.2, below threshold) straight past the integrating
window to SATURATED RELAY, and never tested below 6.0. Sub-threshold integration needs
**w_cd_acc/6 < 0.9, i.e. w_cd_acc < 5.4**, so several CD spikes must SUM over time, with
self-excitation holding charge between them. Same weight/lam family as the silence bug, opposite
direction. **This is why |h| does not grow with distance** (r(dist,|h|) -0.18..-0.45): a relay carries
instantaneous heading, not an integral.

### CD tuning is real but far too shallow
Peak column DOES track heading (yaw 0-45 -> col 0; 90-135 -> col 1; 180-225 -> col 5), but modulation
is only 0.44 vs 0.33 -- ~30% on a large pedestal -- because PG fires 12.35 of 36 columns. Population
spread across ACC columns is 1.14 (1.0 = no tuning at all), so the population vector direction is
noise: home-vector error scatter 57 deg with circular concentration 0.27 (a constant offset would
give ~1.0; removing the best offset only moves 59.1 -> 56.8, so it is NOT a convention error).

### Knobs measured INERT this session (identical outputs across changes)
`w_hs_opp` 0.0-6.0 · `w_acc_self` 5.4 vs 6.0 · `w_acc_agi` 0.5 vs 1.5 at w_acc_self=6.0 ·
`w_acc_nbr` 0.0 vs 0.4. Narrowing CD tuning (`cd_cut` 0.05->0.7, `cd_pow` 1->3) made error WORSE
(47.8 -> 82.2), so tuning width is not the lever.

### Compass follow-ups
The +9/-60 deg CCW/CW offset flip was measured on the OLD k_ang=0.45 trace; at k_ang=0.65 it is
**3.8 deg on seed 11** but 37.4/23.4 on seeds 23/44. It is NOT a latency -- applying the best lag
makes the flip WORSE (3.8->29.9, 37.4->52.6, 23.4->45.7). My claim that this flip propagates into L3
was wrong; L3's error has a different cause (the relay above).

### Open, in order
1. `w_cd_acc` into the integrating window (test 2.0-5.0) with `w_acc_self` holding charge -- the
   single highest-value experiment; it is what makes |h| grow with distance.
2. Deepen CD modulation (narrow the PG bump, not the CD cosine).
3. Seed-dependent compass offset (3.8 vs 37.4 deg).

---

## 2026-07-29 (overnight, to 10:00) — L2 COMPASS FIXED IN THE BODY (70-112 deg -> 13-16 deg)

**Rule 0 adopted after a third violation: NO SUMMARY STATISTIC AS PRIMARY EVIDENCE.** Every claim now
requires a per-tick trace against a TIME-VARYING drive. This session it overturned two of my own
diagnoses in a row and then found the real fault, which no run-average could have shown.

### Two retractions, both from averaging
1. "Translation INVERTS the rotation estimate" (HS -39.8% at turn +0.3) -- that was **seed 11 alone**.
   Seeds 11/23/44/77: -5.8, +0.6, -1.5, +1.1. Straddles zero. There is no inversion.
2. "HS has a low-angular-velocity DETECTION FLOOR" -- there is no floor. The per-tick tuning curve
   shows discriminative signal at EVERY omega down to 0.05 rad/s. What existed was a **DC OFFSET**
   (zero-omega bias -0.080) shifting the whole curve down, so HS_CW exceeded HS_CCW during CCW
   rotation purely from the bias.

### The actual fixes (all measured per-tick, replicated on 4 seeds)
* **d_emd 4 -> 16.** The Reichardt delay sets velocity tuning (peak ~ delta_phi/d_emd); 4 was tuned to
  fast flow. Per-tick r(omega,HS): 0.347+-0.072 -> **0.587+-0.031**, better on ALL 4 seeds.
  d' at |omega| 0.05-0.15: 0.04 (one seed NEGATIVE) -> **0.41, positive on all 4**. Floors 6 -> 1.
  d_emd 24/32 score higher r but open holes in the mid range -- rejected.
* **k_ang 0.45 -> 0.65.** Vestibular gain is sharply NONLINEAR: 0.45 -> tracking slope 0.085 (bump
  barely moves), 0.9 -> 2.43 (overshoot). Slope~1.0 crossing is 0.55-0.65.
* **w_hs_shift 1.3 -> 0.0.** Visual shift drive DEGRADES tracking (per-tick r 0.609 -> 0.204), and at
  k_ang 0.45 collapses ring liveness to 20% and inverts the slope. Visual anchoring needs a LEARNED
  cue->heading map, not a raw HS->P-EN shift.
* **w_hs_opp is INERT** -- identical bias/r/d' from 0.0 to 6.0. Post-threshold inhibition is too late,
  exactly as the source comment predicted. Dead parameter.
* **Retina rate is NOT the bottleneck**: RT=1 vs RT=4 gives r 0.552 vs 0.575. RT=4 is cheaper AND
  marginally better.

### L2 RESULT — TARGET MET
Median heading error in the body, per-tick, d_emd=16 k_ang=0.65:
  seed 11 **15.8 deg** (59.8% of ticks <20) · seed 23 **13.1 deg** (62.9%) · seed 44 **15.3 deg** (76.7%)
Baseline was 70-112 deg. Target <20 deg met on 3/3 seeds.

### L3 — accumulator was SILENT, now alive; home vector 37-39 deg (target <30, NOT met)
`pi_accum` defaults were `w_cd_acc=1.2, lam_acc=6` -> **0.2 against threshold 0.9: structurally
silent**. ACC alive 0.0% in the body. I had read graded lobes from the ISOLATED net and wired it in
without confirming it spikes inside the agent -- the exact check my own rule demands. The weight/lam
family of errors is now at SIX. With w_cd_acc>=6.0 ACC is 100% alive, home-vector median 37-39 deg.

### THE REMAINING FAULT — a CW/CCW asymmetry, visible ONLY per-tick
Compass offset by octile: **+6.9, -60.4, +9.2, -60.2, +9.2, -60.2, +9.2, -60.2** -- not drift, a
PERFECTLY PERIODIC two-state flip locked to the turn direction. The bump lags one way and leads the
other, a ~70 deg swing. It propagates straight into L3: home-vector error by octile
**19.1, 11.9, 78.6, 52.6, 22.2, 4.7, 73.0, 46.6** -- excellent (4.7-11.9 deg) on half the sweep,
~75 deg wrong on the other half. A median reports 37.5 and describes neither regime.
**This single asymmetry is now the blocker for both L2 residual error and all of L3.**

### Harness flaw to fix
Constant speed drives the agent into the arena wall (dist pins at 11.00 = arena radius), so
`r(dist,|h|)` -- the "|h| grows with distance" criterion -- is untestable under this protocol.

---

## 2026-07-29 (overnight) — drain -> global inhibition; Layer 3 partially restored

**Confirmed the drain is the killer, IN THE BODY, single variable:** cell0 spikes 13 (w_drain=1.6) ->
4000 (w_drain=0), a 300x change, while antipodal DRIVE was unchanged (903 vs 1306). But removing the
drain alone SATURATES the ladder (fill 16/16 all columns, |h|=0) -- it was the ONLY bound.

**Replaced it with LGI (ladder global inhibition)**, per the comparative PI literature. Result in the
body across 6 worlds: fill 11-45 (was 0), OPP 1171-5455 (was 0), CPU1 L-R -163..+6 (was -6).
Home-vector error remains 33-172 deg and HOME still never fires -- magnitude works, direction does not.

**The ladder is structurally binary.** w_self<=3.2 -> fill 0 everywhere; w_self=5.0 -> permanent latch
(cell0 200/200 every bin). No graded regime exists, so LGI can only do nothing or annihilate.

**Three arithmetic errors of one family in my own new circuit**, all "weight/lam vs threshold":
w_cd_acc/lam = 0.2 vs 0.9; w_acc_self swept 0.85-1.25 when the critical point was 2.7; w_acc_to_agi
giving 0.66 vs r_agi 1.6 (AGI fired 0 times -- a dead parameter I only found by measuring).
RULE: compute (weight / lam) against threshold BEFORE choosing any weight in this substrate.

Companion to `ARCHITECTURE.md` (state) — this file is the **narrative**: what was tried, what it cost,
what was retracted. Append newest entries at the top.

---

## 2026-07-28/29 — The session of five bugs and three bad metrics

**Net result: one-line bug fixes restored a system I had spent the session declaring architecturally
broken. Almost every "X is broken" verdict I issued was wrong.**

### The bugs (each invisible to summary-level metrics)

1. **`SEEDSYN` off-by-one** — computed as `rj[RING[0]]-1`, but the TONIC synapse is created *after*
   the seed, so it pointed at the tonic. Every birth seed was delivered at amplitude 4.0 through a
   0.12 weight → 0.48 vs threshold 0.9, sub-threshold. Also a single global index where per-cell
   layouts differ (Δ7 coverage). Fixed via per-cell `SEEDMAP`.
   **Impact: with this one fix and nothing else, `cx_navigator.py` at its own untouched defaults went
   from a 156° home-vector error to 18°, and homing from "test crashes" to closing 62% (historical
   record: 67%, orbits). The original design was never broken.**
2. **US never reached the brain** — `_consume()` clears `world.event` on every world step, but
   `run_episode` sampled it once per agent step (16 sub-steps). 7 toxin contacts → **0** STG_T spikes.
   Fixed with a latched `take_event()` → 112 spikes. Before this, no associative learning of any kind
   could occur in the body; this is very likely the real source of the "<20 step MB retention".
3. **Three kwargs allowlist gaps** — `mb_parts`, `navcore_parts` and `nv.parts` silently dropped
   parameters (the latter two accepted none at all). Any knob not in the list was inert.
4. **ID collisions** — my vAC ids first landed on `CPGP`/`RLY`, which would have wired walking rhythm
   into the aversive MBON with nothing crashing. NR=144 still collides (CR overlaps PG).
5. **Old self-test didn't run** — `steer()` returns 3 values, the test unpacked 2. So the navigator's
   recorded result had not been re-run since the OPP/MUS_F work.

### The metric errors (worse than the bugs — they made me confident)

1. **Wrapped endpoint for bump displacement.** Reported **−0.13** for a bump actually running away at
   **+5.92** (it laps the ring ~6×). Invalidated the d_push sweep, the ND7 2×2, the r_conj_lo and
   w_push sweeps — and wrongly killed `sh_bank`, which is the paper's actual mechanism
   (Turner-Evans 2017: P-EN/E-PG phase offset varies linearly with rotational velocity).
2. **Bump WIDTH as "sharpness".** Called a 130° bump unacceptable while its decoded centroid precision
   was **0.26° with 24× discriminability**. The config I called "sharpest" was a PINNED attractor that
   decodes one state regardless of seeding. Biology agrees: rat HD tuning is ~90°, precision comes
   from population averaging.
3. **Membrane `S>0.8` as ladder fill.** Spiking RESETS S, so latched active cells read as empty.
   Produced "fill never exceeds 1 in any world"; correct windowed metric gives **6–7**.

### Retractions issued this session
"the original navigator is unreproducible" · "the drain is the root cause" · "the ladder is
structurally unable to recover" · "sharpness is the blocker" · "retention is monotonic in kernel angle"
· "the arbiter never fires" · "OPP is silent" · "learning does not change behaviour" · "the agent has
no toxin repulsion" · "velocity encoding is missing" · "ladder ∝ speed".

Pattern: **six consecutive negatives traced to my protocol, not the substrate** (US latch, food-odour
background, inter-trial interval, probe geometry, missing control cell, probe distance). Then three
more to metrics. The substrate was right nearly every time.

### What was genuinely established
- Compass: centroid jitter 0.26°, heading hold drift +0.0°/−0.9° over 1200 ticks, tracking
  1.05±0.20 at ω=0.5 over 9 birth positions.
- `sh_bank` makes bump speed scale with velocity (1.97× vs body 2×); without it 1.18× — a fixed-speed
  wave where ω merely gates motion.
- Δ7 is what keeps the ring alive (`d7=False` → 0 spikes/400 ticks); tonic alone does not.
- US→MBON verified in the body (`STG 16 → MBON 7603`), needs ≥30k ticks to catch a contact.
- Lateral toxin avoidance causally verified by ablation.
- Learning discriminates only at a realistic inter-trial interval (~5% US duty, not 25%).
- **Layer 3**: LAD/CD/HL/PG/GI are byte-identical between navigator and agent; the ladder fills to 6
  in the agent on a scripted L-path but dies during free foraging. **The circuit is fine; the regime
  starves it.** Seed 44 briefly reached fill 16 with a **0.1° home-vector error**.

### Process rules adopted
Every circuit measured **isolated AND in the body** (added to the challenge cron) · per-tick sampling
via `tick_hook`, never per-step · run every cell of a factorial, including the "obviously unnecessary"
control · verify the stimulus actually reaches the target population before interpreting a probe ·
when claiming a mechanism, log the quantity that mechanism predicts in the same run.

### Open, in priority order
1. `w_drive × w_drain` 2×2 in the body — the ladder arithmetic (0.5/CD spike vs 1.6 threshold vs −1.6
   drain) is still unresolved.
2. Why free foraging starves the ladder when a scripted L-path does not. **This is the Layer-3 gap.**
3. Compass in the body — never measured post-fix.
4. Hunger saturation (2400/bin constant) → FORAGE saturation (3178–3200 every world) → arbiter cannot
   select; HOME has fired 0 times in 384,000 ticks.
5. Consolidate agent / navigator / pi_agent compass configs into one vetted default, then make it the
   default so `live_brain.py` (which passes no kwargs) actually runs the verified work.


---

## WHERE TO LOOK

- **`ROADMAP.md`** — status, the 8 numbered failures (F1-F8), and the ordered Phase 0-4 plan
- **`ARCHITECTURE.md`** — every population, measured evidence, confidence grades, structural traps
- **`LAB_RULES.md`** — the measurement protocol. Rule 0: per-tick or it does not count
- **`RESEARCH_LOG.md`** — what was tried, what it cost, what was retracted
- **`live_brain.py --port 8770`** — live UI: structure switches, parameter sliders, rebuild
