# PAULA embodied agent — master architecture & status

This is the preserved 29 July 2026 architecture assessment. Its confidence
scores and status statements belong to the recorded configurations and tests
of that period. They are not acceptance results for the current model. The
historical text below is retained without reconciling it to later research.

**Last updated 2026-07-29.** Confidence scores are grounded in this session's error record: I made
**three metric errors and four false "broken" verdicts**, so anything not measured with a
state-appropriate metric, in the body, over a long span, is marked low.

## Confidence scale

| score | meaning |
|---|---|
| **A** | measured IN THE BODY, correct metric, ≥2 seeds, long span, survived a falsification attempt |
| **B** | measured in the body OR isolated with correct metric, but not both / not falsified |
| **C** | measured once, or with a metric I have not validated against a known-good and known-broken case |
| **D** | asserted from code reading or a short run; historically ~50% of these were wrong |

---

## THE READOUT RULES (read before measuring anything)

These caused every false conclusion this session. They are not style notes.

1. **Ladder / latch fill → count cells SPIKING in a window (≥8 ticks), never membrane `S`.**
   Spiking resets S to 0, so a latched, actively-firing cell reads as EMPTY. Using `S>0.8` produced
   "fill never exceeds 1 in any world" — completely false; correct metric gives 6–7.
   Use `fillwin()`; `nav.fills(win=8)` is the reference implementation.
2. **Ring / bump displacement → cumulative UNWRAPPED per-tick sum, never a wrapped endpoint.**
   `((b1-b0+180)%360-180)/dy` returns small/negative numbers once the bump laps the ring. It reported
   **−0.13 for a bump actually running away at +5.92**. That single error invalidated the d_push sweep,
   the ND7 2×2, the r_conj_lo and w_push sweeps, and wrongly killed `sh_bank`.
3. **Heading quality → decoded CENTROID precision, not bump WIDTH.**
   A 130°-wide bump resolves 10° steps at 24× its noise floor (jitter 0.26°). Width said
   "unacceptable" while precision was excellent. The "sharpest" config (NR=72/NB=3) turned out to be a
   PINNED attractor decoding 340.0/340.1/340.1/340.2 regardless of seeding — sharp but immobile.
4. **Populations firing mid-step → sample in a `tick_hook`, never after `run_episode`.**
   Per-step sampling reported the arbiter as `[0,0,0]`; per-tick showed FORAGE firing 1578/3200.
5. **Before trusting a metric, ask what it reports for a known-good and a known-broken circuit.**

## THE STRUCTURAL TRAPS (each cost hours)

1. **kwargs allowlists.** `parts()` filters kwargs by explicit name lists. A parameter missing from the
   list is SILENTLY INERT. Found in **three** places: `_mbk` (mb_parts), `_nvk` (navcore_parts, which
   took none at all), `_cpk` (nv.parts, which took none at all). If a knob does nothing, check the
   allowlist before concluding anything about the circuit.
2. **`t_ref = c × num_inputs`.** Adding synapses silently gates a population off. `AIFAgent3D` applies
   `tref_upper`; the standalone Compass/Navigator do NOT. GI pools all NR ring cells and Δ7 pools a
   sector, so both scale fan-in with NR.
3. **Per-cell synapse indices.** Δ7 coverage differs per ring cell, so a SINGLE global synapse index is
   wrong for most cells. `TONICSYN` was fixed for this; `SEEDSYN` was not — it was `rj[RING[0]]-1`,
   off by one, pointing at the TONIC synapse, so every birth seed was delivered at `w_seed`=4.0
   through a `w_tonic`=0.12 weight → 0.48 vs threshold 0.9, **sub-threshold**. Fixed via `SEEDMAP`.
   **This one bug accounted for essentially the entire "the original design is broken" investigation.**
4. **ID collisions.** `CR` spans `300 + 2·NR·NP`; at NR=144 it overlaps `PG` at 1000+ (144 duplicates).
   The builder keeps the incumbent SILENTLY. My vAC ids first collided with `CPGP`/`RLY`, which would
   have wired walking rhythm into the aversive MBON with nothing crashing.
5. **zsh does not word-split unquoted vars.** `set -- $cfg` puts the whole string in `$1`. Killed 6 jobs
   twice.
6. **Blocking `until` loops SIGTERM the process group on tool timeout**, killing nohup'd children. Use
   a separate `run_in_background` waiter. `setsid` does not exist on macOS.

---

## LAYERED ARCHITECTURE (ordered by dependency)

### Layer 0 — Transducers
| Circuit | Status | Conf |
|---|---|---|
| Odour field → ORN_F/ORN_T/ALN, FLp/FRp/TXL/TXR | works; ablation `tox_gain=0` ≡ empty world | **A** |
| Retina (72×20) → PR | works; food (.95,.72,.15) vs toxin (.85,.15,.25) | **B** |
| Muscle membrane S → actuator (`GEAR=26`) | works; speed 0.22–0.74 | **A** |
| Contact → `STG_T`/`STG_F` (the US) | **FIXED**: `_consume()` cleared `event` every world step while `run_episode` read it once per agent step (16 sub-steps) → 7 contacts, 0 spikes. Latched `take_event()` → 112 | **A** |
| Proprioception → speed / yaw-rate | PG 19,950–21,467 spikes | **A** |

### Layer 1 — Core substrate
| Circuit | Status | Conf |
|---|---|---|
| Ring attractor E-PG + GI + Δ7 + tonic | alive 17k–20k spikes/3.2k ticks, longest silence 1 | **A** |
| Birth seed (`SEEDMAP`) | **FIXED** (see trap 3) | **A** |
| P-EN shift CL/CR + `sh_bank` | bump speed scales 1.97× vs body 2×; without bank only 1.18× (fixed-speed wave) | **B** |
| Engine/CPG → relays → graded muscles | locomotion works | **A** |
| **Δ7 keeps the ring ALIVE** | `d7=False` → 0 ring spikes in 400 ticks; tonic alone does NOT revive it | **B** |

### Layer 2 — Heading
| Property | Value | Conf |
|---|---|---|
| Centroid precision (jitter) | **0.26°**, 10° steps at 24× noise | **B** |
| Heading hold, no rotation | drift **+0.0° / −0.9° over 1200 ticks** | **B** |
| Tracking ω 0.5–1.0 | 1.05 ± 0.20 / 0.82 ± 0.19 over 9 birth positions | **B** |
| Tracking ω ≤ 0.3 | **fails**, ring phase-locked `X.X.X.` (t_ref=2, whole ring synchronised) | **B** |
| Blur onset | NR=36 gradual; NR=72 cliff ω 1.0→1.25 (R 0.66→0.20) | **C** |
| Re-sharpen after spin | NR=36 R 0.42→0.76 heading KEPT (−4.3°); NR=72 R 0.00→0.93 heading LOST (+170°) | **C** |
| Re-anchor to injected truth | restores bump (R 0→0.66) but lands **33–41° off** — rescue, not correction | **C** |
| **Compass IN THE BODY** | **NEVER MEASURED post-fix** | **D** |

Two research variants kept deliberately: **NR=36/NB=3** (30° kernel, robust, resolvable) and
**NR=72/NB=3** (15° kernel, sharper R but PINNED — decodes one state regardless of seeding).
Excitatory kernel angle = `NB × 360/NR` is the master knob; NR sets centroid resolution, not width.
Biology note (general, not fly-specific): rat anterodorsal-thalamus HD tuning is ~90° — BROAD.
Precision comes from population-vector averaging (~width/√N). **Do not chase a narrow bump.**

### Layer 3 — Path integration  ← **THE BLOCKER** (partial fix landed 2026-07-29)

**WORKING CONFIG (opt-in, defaults unchanged):** `w_drain=0.0, w_lgi=0.8, w_lad_lgi=0.04`
Replaces the antipodal drain with LADDER GLOBAL INHIBITION (`LGI`, id 9690), which is what the
comparative PI literature actually specifies (local excitation + global inhibition over a circular
array; ants, bees AND mammals). Measured in the BODY, 900 steps, 6 worlds/seeds:

| seed | world | fill sum | \|h\| | OPP | CPU1 L-R |
|---|---|---|---|---|---|
| 11 | 9f/3t | 41 | 16.50 | 4214 | -120 |
| 22 | 9f/3t | 18 | 6.00 | 1171 | -58 |
| 33 | 16f/0t | 11 | 5.57 | 2089 | -111 |
| 44 | 10f/20t | 41 | 5.89 | 3601 | -47 |
| 55 | 5f/3t | 22 | 18.43 | 4466 | +6 |
| 66 | 14f/6t | 45 | 20.47 | 5455 | -163 |
| **SHIPPED s11** | 9f/3t | **0** | **0.00** | **0** | -6 |

Ladder fills in ALL worlds (was 0), OPP fires (was 0), CPU1 carries a real L-R asymmetry (was -6).
### READOUT VALIDATED AGAINST GROUND TRUTH (do not re-litigate this)
Isolated, pure straight travel to (+500,0): TRUE home 180 deg, `vec()` returns **165 deg -> 15 deg
error**. The convention is CORRECT (lobe points along TRAVEL, vec() flips it to point home).
The residual 15 deg is a SYSTEMATIC HALF-COLUMN OFFSET: the lobe is [5,5,0,...,0,5,5] = columns
{0, +30, -60, -30} deg, centroid -15 deg -- it includes column 10 (-60) but not column 2 (+60).
Fixed offset, not noise; correctable with a half-column phase shift in PHI_C.
FALSIFIED en route: the birth-seed "pedestal" (cell0 latched in EVERY column, giving fill 1 everywhere)
does NOT bias the bearing -- a UNIFORM component contributes exactly zero to a vector sum. Excluding
cell0 gives the identical 165 deg. Arithmetic would have shown this without a run.
Also: `cd_cut` 0.05 (15 deg err) beats 0.70 (30 deg) on a straight path, so the CD-sharpening direction
I pursued in the body was backwards.

**So straight-line PI is accurate to ~15 deg. The 167-177 deg body errors come from TURNING, not from
the readout, the pedestal, or CD tuning.**

**STILL BROKEN: home-vector DIRECTION is unreliable under turning, and HOME mode still never fires.**
`cd_cut` (CD tuning width) was swept 0.05/0.30/0.50/0.70/0.85 in the body across 6 worlds. It MOVES the
error a lot but not consistently: seed 11 gives 69.1 deg at 0.30, 119.0 at 0.50, 28.8 at 0.70, 114.3 at
0.85; seed 55 gives 57.0 / 22.1 / 23.7; seed 66 stays 167-177 in every setting. Best observed errors
22-30 deg (seeds 22/55 at cd_cut 0.70-0.85), worst 177 deg. NO setting is good across all worlds, so
direction is NOT solved -- treat |h| and the CPU1 steering asymmetry as live but the BEARING as
untrustworthy. The fill lobe is
FRAGMENTED (e.g. [0,0,6,5,5,5,5,5,5,5,0,0]) rather than a cosine lobe -- the columns that should be
the peak read ZERO. That fragmentation is the next thing to fix; the magnitude and steering signal are
live, the DIRECTION is not trustworthy.

**Why the ladder cannot be made graded:** measured, w_self<=3.2 gives fill 0 in every world and
w_self=5.0 latches permanently (cell0 fires 200/200 every bin with drain off and no LGI). There is no
middle -- the ladder is structurally BINARY, which is why LGI either does nothing or annihilates it.
A graded accumulator needs the `pi_accum.py` redesign (below), not more ladder tuning.

**`pi_accum.py` (redesigned accumulator, isolated only):** leaky integrators on a circular array with
local excitation + global inhibition, no drain. ISOLATED results: charges correctly, lobe MOVES ~90 deg
with a 90 deg turn, RETAINS through a 280-tick no-drive turn at `w_acc_self=5.0` (|h| 30.9 held), and
|h| GROWS with distance once AGI is live (`w_acc_to_agi=2.0, r_agi=0.8` -> 439 AGI spikes; at the
original 0.22/1.6 AGI fired **0** times and was a dead parameter). NOT yet wired into the agent.
Remaining isolated defect: ACC saturates at the 8-spike window ceiling, so it cannot SUM two legs --
leg2 overwrites leg1 instead of adding.
| Circuit | Status | Conf |
|---|---|---|
| PG speed-gated readout | ~20k spikes | **A** |
| CD cosine speed cells | 7,222–8,136 spikes | **A** |
| **CPU4 ladder — scripted L-path** | **fills to 6** (navigator 7), proper lobe | **B** |
| **CPU4 ladder — free foraging** | **fills transiently then dies** (seed 44: 16 → 0 → 0; most seeds 0) | **B** |
| OPP / home vector | fires when ladder is charged; seed 44 gave **0.1° home-vector error** | **C** |
| CPU1 HL/HR | fire but SYMMETRIC (184/183, 174/174) → no steering signal when ladder is empty | **B** |

**Neuron-level diff (navigator vs agent): LAD, CD, HL, PG, GI are IDENTICAL** in r, t_ref, fan-in and
every weight. Only RING differs (agent fan-in 30 vs 18: 8×−2.0 Δ7 + tonic; OPP t_ref 2.00 vs 64.00).
**So the circuit is not the problem — the foraging REGIME is.** A controlled L-path gives sustained
concentrated CD drive on one column; erratic foraging never sustains a column long enough.

Ladder arithmetic, unresolved: `w_drive=1.0` → ~0.5 per CD spike vs `r_lad=1.6`, while each antipodal
CD spike delivers `−w_drain = −1.6`. Cell 0 DOES have a CD drive synapse (no predecessor needed — I
wrongly claimed a structural dead-end), so recruitment is arithmetically possible only when the
antipode is quiet. **`w_drive × w_drain` 2×2 in the body is the next experiment.**

### Layer 4 — Action selection
| Circuit | Status | Conf |
|---|---|---|
| Hunger ladder | **SATURATED** at 2400/bin constant | **B** |
| Arbiter FORAGE | **SATURATED**: 3178–3200 in EVERY world/bin = ~50% duty = t_ref=2 ceiling. Not selecting | **B** |
| Arbiter HOME | **never fires**: 0 in 384,000 ticks | **B** |
| Arbiter EXPLORE | world-varying 0–1507 (so the WTA itself is alive) | **B** |
| Belief core BL/BR/UBEL/DL/DR, UNC | never isolated | **D** |

### Layer 5 — Learning (healthiest subsystem, independent of 2–4)
| Circuit | Status | Conf |
|---|---|---|
| PN → KC sparse + APL | KC 152–74,598/bin; **0 shared cells** between odour groups | **A** |
| **US → MBON in the body** | `STG_T=16 → MBON 7603` (s11) and `→5840` (s44), only after real contacts | **A** |
| KC→MBON plasticity | paired +0.33 vs unpaired −0.02 at ITI period 40 | **B** |
| Discrimination needs a REALISTIC ITI | US at 25% duty → none; at 5% → toxin +, food − | **B** |
| Baseline-active MBON (`w_km0`=0.4) | T−F +0.73/+0.84 vs +0.19/+0.35; non-predictive CS driven to floor | **B** |
| Extinction | activity-dependent: 0 spikes → 0.0%; 240 spikes → −72.5% | **B** |
| Behavioural effect | 4/6 seeds (+0.218±0.152); fails when the agent OUTRUNS its reflex | **C** |
| MBON→AVOID readout | 77% tonic pedestal; saturates on success | **C** |

Needs ≥30k ticks to catch a toxin contact — shorter windows produce false "MBON never fires".
Aso 2014: valence is a SKEW across opposing MBONs; Hige 2015: aversive learning DEPRESSES the
approach MBON. Current design potentiates a single aversive population — inverted vs biology.

### Layer 6 — Vision
| Circuit | Status | Conf |
|---|---|---|
| ON/OFF → V1 → V2 → AZ → SAL | retinotopic hierarchy | **C** |
| EMD → HS motion | CCW −84 / CW +88, compressive 0.4→3.0 rad/s | **B** |
| Chroma CH_RG | toxin adds 16.2 to R−G sum vs food 1.6 (10×) | **B** |
| CH_BY | **measured USELESS** — a sky detector (empty scene scores highest) | **B** |
| vAC: VPN→vKC→VAPL→MBON (`vac=True`, OFF by default) | 126 spikes toxin / 0 food / 0 empty | **B** |
| **Vision → behaviour** | **FALSIFIED**: odour-ablated ≡ no object present | **B** |
| Visual anchoring of compass | not built (task #23) | — |

### Layer 7 — Reflexes
| Circuit | Status | Conf |
|---|---|---|
| Lateral toxin avoidance TXL/TXR→TR/TL | +0.72/+1.04, ablation-confirmed causal | **A** |
| Head-on avoidance | tropotaxis is STRUCTURALLY BLIND (TL/TR cancel); `trise` fix gives +0.29 but is OFF by default | **B** |
| Klinokinesis RISEP→STEER | ~330 baseline, suppressed to ~2 by food odour | **B** |
| Sting latch / RFX | 160 spikes | **B** |

### Never measured
Self-modelling · 3D spatial understanding beyond heading · long-term retention past extinction ·
chroma→behaviour · P-EG loop (correctly unused — Hulse 2021 connectome REPLACED E-PG↔P-EG with
E-PG↔E-PG self-recurrence, so reviving `w_peg` would implement a superseded model)

---

## WHAT THE LIVE DEMO ACTUALLY RUNS

`live_brain.py:239` calls `AIFAgent3D(seed=…)` with **NO kwargs** → pure defaults. Therefore the live
agent has the ring-persistence, US-latch and birth-seed fixes (all in shared code) but **`vac=False`,
`trise=False`, `w_km0=0.02`, `w_drain=1.6`, ND7=8, d_push=8, sh_bank=False**. Almost everything
verified this session is OFF. Once Layer 3 is settled the defaults should BECOME the vetted config
rather than a pile of opt-in flags.

## PRIORITY QUEUE
1. `w_drive × w_drain` 2×2 in the body, corrected fill metric, long span, ≥4 worlds.
2. Why free foraging starves the ladder but a scripted L-path does not — that is the whole Layer-3 gap.
3. Compass in the body (never done post-fix).
4. Hunger saturation → FORAGE saturation (Layer 4 cannot select while its input is pinned).
5. Then: reconcile agent/navigator/pi_agent compass configs into ONE vetted default.
