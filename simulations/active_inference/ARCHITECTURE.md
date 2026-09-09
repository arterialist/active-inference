# PAULA EMBODIED AGENT — COMPLETE ARCHITECTURE

**Status date:** 2026-08-03 · **Live build:** strict version profiles in `aif_agent3d.py`
(95 neurons for V1, 299 for V2, 345 for V3, 373 for V4) in the modular
`components/body/world.py` MuJoCo body

The historical population measurements below describe the pre-migration core.  The maintained
versioned entry points are now `agents/reactive_v1/`, `agents/memory_v2/`,
`agents/interoceptive_v3/`, and `agents/obstacle_v4/`; current evidence and the extraction boundary are recorded in
`docs/ACTIVE_INFERENCE_MIGRATION_MANIFEST.md`.

Full reference for the embodied insect-style mini-brain: every population, what it does, what is
*measured* to work, what is measured to fail, and why. Companions:

- **`LAB_RULES.md`** — the standing measurement protocol (Rule 0: per-tick or it doesn't count)
- **`RESEARCH_LOG.md`** — the narrative: what was tried, what it cost, what was retracted
- **`../../docs/ACTIVE_INFERENCE_ARCHITECTURE_ARCHIVE_2026-07-29.md`** — the archived previous version

## Live control plane (2026-08-03)

The live brain is a negotiated client/server system rather than a baked
topology artifact. Run `uv run aif-live versions`, then start V1/V2/V3/V4 with
separate `aif-live start --version ... --port ...` processes (V4 defaults to
the `obstacle_detour` world; the live server also exposes the corner, chicane,
and maze variants). Each server
exposes `/api/session`, `/api/schema`, `/api/topology`, `/api/state`,
`/api/neuron/{id}`, and `/api/command`, plus a WebSocket at `<http-port+1>/ws`.
The browser obtains neuron IDs, graded-cell IDs, topology, and binary-frame
dimensions from the hello contract; it never assumes the static V1 count.
The neuron endpoint is a read-only display adapter for intracellular variables
and bounded incoming/outgoing synapses.

The harness web lab is `uv run aif-live lab --port 8850` and lives under
`lab/`. It exposes only the maintained embodied component harnesses and stores
full-trace runs under ignored `.live/lab-runs/`.

---

## 0. THE PAULA-ONLY RULE

Behaviour lives in neural wiring, not Python. The only sanctioned non-neural elements:

1. **Transducers in** — a physical field becomes a neuron's external input current.
2. **Transducers out** — a graded muscle's membrane `S` becomes an actuator force.
3. **One-time birth seeds** — an initial current injection at t=0 to break symmetry.

Nothing else on the control path. Measurement code may compute anything, provided it never feeds back.

**A violation currently in the build** is documented in §6.3: visual→compass anchoring maps azimuth to
heading through `world3d.SUN_AZIMUTH`, a ground-truth world constant.

---

## 1. CONFIDENCE SCALE

Calibrated against my own error rate. **Eight retractions** so far; grades reflect that.

| grade | meaning |
|---|---|
| **A** | Per-tick **in the body**, ≥4 seeds, with a control that would have caught the opposite result. Survived an attempt to break it. |
| **B** | Per-tick in the body, ≥4 seeds. No adversarial control yet. |
| **C** | Isolated only, **or** <4 seeds, **or** from a summary statistic. A hypothesis. |
| **D** | From code reading or a single run. Historically ~50% wrong. |

Anything C or D that matters must be re-measured before being built upon.

---

## 2. BODY AND WORLD

`world3d.py` — MuJoCo, planar-kinematic body (x, y, yaw; gravity off, contacts disabled). Arena
radius **11.0**. Food/toxin sources emit odour fields; toxin contact latches a US event.

**Motion constants every experiment must respect** (`LAB_RULES.md` Rule 1):

```
W.act(turn, speed, kyaw=0.36, kv=0.06)
travel        = speed * 0.06        -> 0.054 units/tick at speed 0.9
rotation      = 0.36 * turn deg/tick
circle radius = v/omega = 8.6/turn  -> turn=2.2 gives radius 3.9, never walls
```

`W.act` is a **kinematic pose setter** for scripted probes. In free behaviour the agent moves through
its muscles.

### 2.1 Transducers (grade A)

| dir | signal | target |
|---|---|---|
| in | odour concentration L/R | bilateral chemosensors |
| in | toxin contact | `STG_T`/`STG_F` latches via `world.take_event()` |
| in | angular velocity | `CL`/`CR` shift cells, gain `k_ang` |
| in | forward speed | `PG` speed gate |
| in | retinal image | photoreceptors (`vc.drive_from_image`) |
| out | `MUS_L`,`MUS_R`,`MUS_F` membrane `S` | turn and thrust forces |

**`take_event()` — load-bearing fix.** `_consume()` cleared `world.event` every *world* step while
`run_episode` read it once per *agent* step (16 sub-steps): **7 toxin contacts → 0 `STG_T` spikes**.
No associative learning was possible in the body before this. After: 112 spikes.

---

## 3. LAYER MAP

```
L0  transducers ......................... A
L1  substrate: ring attractor, CPG ...... A (persistence)      §5
L2  compass: heading estimate ........... FAILS                §6
L3  path integration: home vector ....... FAILS                §7
L4  action selection: arbiter + HOME ..... blocked on L3        §8
L5  learning: mushroom body ............. works                §9
L6  interoception: hunger ............... works but saturates  §10
L7  motor: CPG + graded muscles ......... A                    §11
```

**Vision is a frontier task and is set aside.** Parked: compass visual anchoring, homing, chroma-based
toxin recognition, EMD/HS refinement. Active priorities in §13 are deliberately vision-free.

---

## 4. NEURON MODEL FACTS THAT GOVERN EVERY CIRCUIT

From reading `neuron/neuron.py` (reading allowed; editing is not — Rule 11):

```
membrane    dS = (dt/lam) * (-S + I_t)
spike       will_fire = S >= active_threshold and (tick - t_last_fire) >= c
on spike    O = p ; S = 0                       # SPIKING RESETS S
threshold   active_threshold = b during cooldown (<= c ticks), else r
t_ref       = clip(upper - (upper-lower)*F_avg + w_tref·M, lower, upper)
            upper = c * num_inputs ; lower = 2c
plasticity  direction = +1 if (tick - t_last_fire) <= t_ref else -1
```

**Four consequences, each of which has cost a wrong design:**

1. **The refractory is `c`, not `t_ref`.** `t_ref` is the plasticity causal/acausal window. An entire
   recruitment population was built to escape a "`t_ref` rate ceiling" that does not exist.
2. **`t_ref` is the LTP/LTD boundary and its adaptation is HOMEOSTATIC.** `direction = +1 if
   delta_t <= t_ref else -1`, so it drives BOTH potentiation and depression. Higher `F_avg` shrinks
   the causal window (`t_ref_homeostatic`), so an over-active neuron depresses more of its inputs —
   a BCM-style sliding threshold. It IS self-limiting.
3. **Spiking resets `S`.** Any metric reading `S` as "accumulated level" reads latched active cells
   as empty.
4. **A cell that never spikes is a true analog integrator.** `r` huge + `lam` huge → the membrane
   integrates near-losslessly. This is the sanctioned pattern behind graded muscles and `XACC`/`YACC`.

**Plasticity (`reward_hebb`):** `Δw = eta_post·(nm·direction·info_val − rh_decay·w)`. Reward gate and
direction are **neuron-global**; `info_val` is **per-synapse instantaneous**. Hebbian specificity is
therefore available (only synapses whose presynaptic cell was active move), but there is no
eligibility trace and no per-synapse credit assignment.

---

## 5. L1 — SUBSTRATE

### 5.1 Ring attractor (`central_complex.py`)

`NR=36` heading columns (10°/col), ids 0–35; global inhibitor `GI=200`; Δ7 structured inhibition;
P-EN shift cells `CL[i][p]`/`CR[i][p]`, `NP=4` per column per direction.

**Ring persistence — grade A, fixed this session.** The tonic is wired as an *external input* and
nothing ever set it, so `w_tonic` was multiplied by zero at every value (0.12 and 1.2 gave
byte-identical traces — Rule 5). The ring survived only on `w_self/lam = 1.4/2 = 0.70` against
`r_ring = 0.9` plus two neighbours: a bump ≥3 cells wide persists, anything narrower collapses.

Measured over 8 seeds at **zero angular velocity** (straight travel):

| before | after |
|---|---|
| died at tick **36, 36, 36, 51** on 4/8 seeds, never recovered | **alive 8/8**, every gain, every `NP` |

Fix injects the tonic every tick in `AIFAgent3D.tick()` (`tonic_amp`, defaulted 0.0 pending §6.4).

**Δ7 keeps the ring alive** (`d7=False` → 0 spikes/400 ticks); tonic alone does not. The agent
overrides module defaults: `d7=True, r_d7=0.3, w_ring_d7=0.9, w_tonic=0.12`.

**Superseded negative:** "lower shift thresholds kill the ring" was true *before* the tonic had a
floor. With it, `r_conj_lo=0.90` keeps the ring 100% alive **and** makes slow turns visible for the
first time (registered 0.0° → 43–56°).

### 5.2 Bump quality (grade B)

Centroid jitter **0.26°**, 24× discriminability; heading-hold drift +0.0°/−0.9° over 1200 ticks.
Bump *width* ~90–130° is fine — precision comes from population averaging, as in rat HD cells. Never
confuse width with sharpness (Rule 7).

---

## 6. L2 — COMPASS (**FAILS**)

### 6.1 Vestibular path
`ω → k_ang → CL/CR external input → P-EN shift → bump moves`

### 6.2 Structural failure (grade A — a solid negative)

The P-EN shift is a **threshold-gated resonant travelling wave**, not a velocity integrator. Fixed
gain, registered/actual over a 179° body turn:

| turn rate | 0.5 | 1.0 | 1.5 | 2.5 |
|---|---|---|---|---|
| ratio | **0.00** | **0.01** | 1.00 | 0.80 |

**Slow turns register exactly zero** — not attenuated, zero, on 3 seeds. Foraging is dominated by slow
turns. Also non-monotonic in gain (`k_ang` 0.40→0.84, 0.42→**1.00**, 0.44→0.23, 0.46→1.35) and
seed-dependent at fixed gain.

**Eleven routes falsified:** threshold level; threshold spread (0.90–1.75 sends slow turns back to
0.00–0.06); population size (`NP` 4→9 → ratios to 2.23 and a **sign inversion**: extra shift cells add
drive to the *same* wave, they do not add independent range); gain; delay stagger; lower thresholds;
stronger GI; push-pull; spatial bank; `sh_bank` in the body; visual shift drive.

### 6.3 Visual anchoring — **EXISTS AND VIOLATES THE PAULA RULE**

```python
implied = w3.SUN_AZIMUTH - vc.az_of(a, vc.NV1AZ)
src_of[int(round((implied % 2π)/2π * cc.NR))].append(a)
```

Every visual azimuth is bound to a ring column through a **ground-truth world constant**. The agent
does not learn that a scene means a heading — it is compiled in. This feeds `w_anchor`/`w_veto` into
the ring, with the `PE` self-model on top.

**This may have propped up every good compass number ever measured**, including the L2 "MET" result.
Removing it may look like a regression before a learned map takes over.

**Replacement (designed, parked with vision):** all-to-all VR→RING inhibitory connectivity at small
uniform weight, carved by depression. Grounded in **Fisher 2019** and **Kim / Hermundstad 2019**
(ER→E-PG is GABAergic and **plastic**; the synapse onto the *currently active* heading is weakened,
so a familiar scene inhibits every heading except the remembered one) and **Seelig & Jayaraman 2015**
(re-anchoring after disorientation). Feasible here because `info_val` is per-synapse.
**Acceptance test must include a scene-rotated world** — a learned map re-anchors, a hardwired one
cannot.

### 6.4 Tonic trade-off (grade A, single-variable, 4 seeds)

| | direction error | r(dist,\|h\|) |
|---|---|---|
| tonic OFF | **20.6–23.7°** | +0.04 … +0.16 |
| tonic ON | 38.6–81.9° | **+0.16 … +0.33** |
| velocity-gated | 21.7–28.9° | −0.09 … +0.00 |

The tonic is background excitation: it prevents collapse during straight travel *and* broadens the
bump, blurring heading. Gating on angular velocity recovers direction but loses the magnitude benefit
— the criteria are coupled through **bump width**, which gating does not separate.

### 6.5 Vision inputs (parked)

EMD `d_emd` 4→16 (per-tick r(ω,HS) **0.347±0.072 → 0.587±0.031**, better on all 4 seeds; low-ω d'
0.04 → 0.41; detection floors 6 → 1). HS: uniform pooling of *signed* flow is already the correct
rotation matched filter — rotation is spatially uniform, forward translation spatially antisymmetric,
both measured per-azimuth; `hs_prof` (|cos| weighting) tested **wrong**, stays 0.0. `w_hs_shift=0.0`
because visual shift drive **degrades** tracking (per-tick r 0.609 → 0.204). `w_hs_opp` is **inert**
(identical 0.0–6.0; post-threshold inhibition is too late).

---

## 7. L3 — PATH INTEGRATION (**FAILS**)

### 7.1 What works

- **PG speed gate (grade A, fixed).** `r_pg=1.7, w_pg=w_pg_s=1.4, lam=2` → 0.70 + 0.63 = **1.33
  against a 1.7 threshold**: the AND gate could not fire even with both inputs. With `w_pg=2.4,
  w_pg_s=2.6`: **PG/tick 12.60 moving vs 0.03 stopped (420:1)**; **CD/tick 0.35 → 1.97**.
- **CD signed cosine (grade B).** `cd_neg` wires the negative lobe as inhibition (Stone 2017 CPU4):
  modulation depth **0.25 → 1.00**, ACC per-column spread **1.14 → 10+**.
- **Opponent coding (grade B).** The substrate cannot decrement a latched cell and doesn't need to:
  with rectified CD the return leg charges the *opposite* column and the population vector cancels at
  readout. Every cell integrates up only.

### 7.2 Blocking failure — no graded integrator (grade A)

| `w_acc_self` | retention 900 ticks after drive stops |
|---|---|
| ≤ 6.0 | **0%** — relay; encodes current speed |
| ≥ 7.0 | **100%** — permanent latch |

**No graded regime exists.** A self-exciting PAULA cell is a bistable switch. The old CPU4 ladder hit
the identical wall from an opposite design (`w_self ≤ 3.2` dead, `5.0` permanent latch). Displacement
needs the opponent *difference* to stay graded; once both columns latch it freezes — `|h|` measured
**flat to 4%** across the whole distance range, 8 seeds.

### 7.3 The analog accumulator already in the codebase

`central_complex.py`: `XACC=1800, YACC=1801`, `r=1e9` (never spikes), `lam=50000` (near-lossless),
driven by `PG` with cos/sin weights, **read from membrane `S`**. The correct primitive — `pi_accum`
was built from scratch into the bistability wall while this sat in an already-read file.

**It does not currently work either.** Bounded circular path (geometry VALID, peak 7.81, min 0.00),
8 seeds: direction **40.6–67.8°**, `|S|` **flat at 0.016** across distance bins 2.08→7.65. Charge
reaches only 0.006–0.016 because `lam=50000` makes the per-tick increment `I/50000`. **`k_pi` needs
raising by orders of magnitude** — untested, and the obvious next L3 experiment.

### 7.4 Retracted L3 claims

Every "L3 direction met" figure (19.9–24.9°, 20.6–23.7°) came from a **walled path**, where the agent
is pinned at the arena edge so the true home azimuth barely moves and a near-static estimate scores
well. On a bounded circle the same config gives 40.6–67.8°. **L3 direction is not met.**

---

## 8. L4 — ACTION SELECTION (blocked)

`aif_arbiter.py` — spiking EFE mode-WTA over FORAGE / HOME / EXPLORE with an interoceptive hunger
ladder. **3/3 selection and switching verified in isolation (grade B).**

**In the body HOME has fired 0 times in 384,000 ticks.** Two causes, both vision-independent:

1. L3 supplies no usable home vector (§7).
2. **Hunger saturates** (2400/bin) → FORAGE saturates (3178–3200 every world) → the WTA has no
   dynamic range. **Non-vision bug, on the active list.**

**Recorded lesson:** a self-latching mode is an unbeatable mode. Persistence must come from the
population, never from self-excitation.

---

## 9. L5 — LEARNING / MUSHROOM BODY (works)

`mushroom_body.py` — PN → KC (sparse) → APL (Lin 2014) → MBON → AVOID; `STG_T`/`STG_F` sting latches
as US; vAC population (Vogt 2016) at ids 87000+.

**Measured (grade B):** KC sparse coding with **0 shared cells** between odours; US→MBON in the body
(`STG 16 → MBON 7603`); discrimination T−F **+0.73/+0.84** (needs `w_km0=0.4` for a baseline-active
MBON, per Hige 2015); learning changes CS-evoked MBON/AVOID on **6/6 seeds**.

**Open (grade C):** the behavioural standoff replicates on only **4/6 seeds** (+0.22±0.15), and the
relation is *inverse* — the largest AVOID (89 spikes) gives the smallest effect (−0.02). "Paralysis"
was falsified (failing seeds have the longest paths and largest displacement). Leading hypothesis:
**the avoidance turn does not scale with approach speed.** Non-vision; on the active list.

**Protocol:** learning discriminates only at a realistic inter-trial interval (~5% US duty, not 25%).
Verify the CS reaches the KCs before interpreting any probe (at D=3.0, ORN_T=1).

---

## 10. L6 — INTEROCEPTION

Hunger ladder fills with time, drains on eating. **Saturates at 2400/bin**, collapsing the arbiter's
dynamic range (§8). Non-vision; on the active list.

---

## 11. L7 — MOTOR (grade A)

Coprime-pacemaker CPG (`CPGP` 85000–85003) → relay gating (`RLY` 85100+) → graded muscles
`MUS_L/MUS_R/MUS_F`, whose membrane `S` becomes force. Locomotion and steering **emerge from the
stroke** — nothing sets a velocity or a heading. Verified across worlds.

---

## 12. ID MAP (check before adding any population)

| range | population |
|---|---|
| 0–35 | `RING` |
| 200 | `GI` · 250+ `D7` |
| 300 – 300+72·NP | `CL` / `CR` shift cells — **`NP ≤ 9`**, at NP=10 the CR block reaches 1019 |
| 1000+ | `PG` · 1400+ `PEG` |
| 1800, 1801 | `XACC`, `YACC` (analog, never spike) |
| 9000–9011 | `CD` (signed) · 9020–9031 `CDM` (rectified) |
| 9100+ | `LAD` · 9500/9600 `HL`/`HR` · 9690 `LGI` · 9700+ `OPP` |
| 9800–9802 | `MUS_L`, `MUS_R`, `MUS_F` |
| 9900 + c·12 + n | `ACC_G` (≤10037) · 10100 `AGI` · 10200+ `ACCM_G` |
| 85000–85003 | `CPGP` · 85100+ `RLY` |
| 87000+ | vAC (`VPN`, `VKC`, `VAPL`) · 87500 `TPOOL` · 87510+ `TRISE` |

`tref_upper` sets the **plasticity window** (`upper_t_ref_bound`, i.e. how fast a population
forgets — `tref_upper=8` was the verified sweet spot for "forget when danger is gone"), **not**
excitability. Add a population only if it is PLASTIC and you intend to tune its forgetting;
for non-plastic populations it is inert. The firing-rate cap is the refractory `c`, not `t_ref`.

---

## 13. ACTIVE PRIORITY LIST — CORE SYSTEMS, NO VISION

Vision is a frontier task. **Assume vision is fine; work everything else.** Parked: visual anchoring
(§6.3), chroma→toxin recognition, homing (compass → anchoring), EMD/HS refinement.

**1. Head-on toxin blindness.** Innate avoidance is verified for *lateral* toxins (turn sign 4/4,
repulsion +1.04) but head-on toxins are **invisible** (±0.05 despite ~6000 TXL/TXR spikes) because TL
and TR cancel. **Documented fix, never built:** sum TXL+TXR → a STEER population (klinokinesis), per
Gomez-Marin & Louis — lateral gradient sets turn *rate*, temporal change terminates runs. Pure
chemosensation.

**2. Hunger saturation → arbiter dynamic range.** Hunger pins at 2400/bin, FORAGE at 3178–3200, so
the WTA cannot select and HOME has never fired. Fix the ladder's range, re-test mode switching in the
body per-tick.

**3. Learned avoidance does not scale with approach speed.** Standoff replicates 4/6 seeds with an
*inverse* relation to AVOID magnitude. Test speed-scaling of the avoidance turn; if absent, gate it
on the forward-speed signal PG already carries.

**4. Re-verify the arbiter in the body, per-tick.** It works in isolation and has never been shown to
select in the body.

**5. `k_pi` sweep on `XACC`/`YACC`.** Charges to only 0.016; needs orders of magnitude more gain. This
is the one L3 item that does **not** depend on the compass — it can be driven from ground-truth
heading to isolate the accumulator.

---

## 14. STRUCTURAL TRAPS (read before designing anything)

1. **Forget-or-latch.** No graded regime from self-excitation. Use an analog cell (`r` huge, `lam`
   huge, read `S`) when you need to accumulate.
2. **The shift is a resonance**, not an integrator. Do not expect proportional velocity encoding from
   any arrangement of P-EN parameters.
3. **Opponent differences saturate** once both sides latch.
4. **Bump width couples direction precision to magnitude range.** They cannot be optimised
   independently within one population.
5. **More cells help recruitment, not wave speed.** Extra ACC cells removed a ceiling; extra shift
   cells amplified the same wave and inverted its sign.
6. **Global inhibition normalises away the quantity you are reading.** A constant-total accumulator
   cannot encode magnitude.
7. **Neighbour excitation diffuses a latched profile to uniform** — total spikes hold at 100% while
   all spatial information is lost.
8. **A rectified accumulator measures path length, not displacement.** Both are real; only one is a
   home vector.


---

## WHERE TO LOOK

- **`ROADMAP.md`** — status, the 8 numbered failures (F1-F8), and the ordered Phase 0-4 plan
- **`ARCHITECTURE.md`** — every population, measured evidence, confidence grades, structural traps
- **`LAB_RULES.md`** — the measurement protocol. Rule 0: per-tick or it does not count
- **`RESEARCH_LOG.md`** — what was tried, what it cost, what was retracted
- **`live_brain.py --port 8770`** — live UI: structure switches, parameter sliders, rebuild
