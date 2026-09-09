# PAULA AGENT — STATUS AND ROADMAP TO FULL FUNCTION

**Recorded 2026-07-29.** The canonical ordered plan. Companions:

- **`ARCHITECTURE.md`** — every population, what it does, measured evidence, confidence grade
- **`LAB_RULES.md`** — the measurement protocol (Rule 0: per-tick or it does not count)
- **`RESEARCH_LOG.md`** — narrative: what was tried, what it cost, what was retracted

**1746 neurons** (shipped defaults) in a MuJoCo body. Live UI: `python live_brain.py --port 8770`.

---

## 1. WHAT WORKS — measured in the body

| subsystem | evidence |
|---|---|
| Transducers in/out | per-tick, all worlds |
| US reaches the brain | `take_event()` latch: 7 toxin contacts → **0 spikes**, fixed → **112**. Before this no associative learning was possible in the body |
| Ring persistence | tonic injection: **36%/0%/0% → 100%/100%/100%** alive; dead-at-tick-36 → never dies; 8/8 seeds |
| Odour tropotaxis | per-tick, ablation-controlled |
| Lateral toxin avoidance | turn sign 4/4, repulsion **+1.04**, ablation-confirmed causal |
| Mushroom body | 0 shared KCs between odours; US→MBON `STG 16 → MBON 7603`; discrimination **+0.73/+0.84**; learning changes MBON/AVOID 6/6 seeds |
| CPG + graded muscles | locomotion emerges from the stroke; nothing sets velocity or heading |
| PG speed gate | **12.60 moving vs 0.03 stopped** (420:1) after the arithmetic fix |
| EMD tuning | `d_emd` 4→16: per-tick r(ω,HS) **0.347±0.072 → 0.587±0.031**, better on all 4 seeds |

## 2. WHAT FAILS — with the measured mechanism

| # | failure | measurement |
|---|---|---|
| **F1** | Compass velocity encoding | Threshold-gated resonant wave. Registered/actual rotation = **0.00** (ω 0.5), **0.01** (ω 1.0), 1.00 (ω 1.5), 0.80 (ω 2.5). Slow turns register *exactly zero*. **11 routes falsified** |
| **F2** | No graded integrator | `w_acc_self` ≤6.0 → **0%** retention; ≥7.0 → **100%** forever. No middle. Opponent difference freezes once both sides latch → \|h\| flat to 4% across all distance |
| **F3** | Home vector direction | **40.6–67.8°** on a bounded circle. The 19.9–24.9° figures were **walled-path artifacts, retracted** |
| **F4** | Analog accumulator undercharged | XACC/YACC reach only **0.006–0.016** over 3200 ticks; `lam=50000` makes the increment `I/50000` |
| **F5** | HOME never fires | 0 times in 384,000 ticks. Hunger pins at 2400/bin → FORAGE pins at 3178–3200 → WTA has no dynamic range |
| **F6** | Head-on toxins invisible | ±0.05 turn response despite ~6000 TXL/TXR spikes — TL and TR cancel |
| **F7** | Avoidance doesn't change behaviour reliably | Standoff replicates 4/6 seeds, *inverse* to AVOID magnitude (89 spikes → −0.02 effect) |
| **F8** | `SUN_AZIMUTH` anchoring | Visual→compass map compiled from a ground-truth world constant. A PAULA-rule violation, and it may have propped up every compass number ever measured |

**Two verified fixes sat behind inert defaults:** `tonic_amp=0.0` (ring fix disabled) and `w_pg=1.4`
(PG fix never entered the defaults — it existed only as a test kwarg). Phase 0 addresses this.

---

## 3. ROADMAP

### Phase 0 — free wins, hours (no new science)  *[IN PROGRESS]*

1. **Ship the two verified fixes.** `tonic_amp=1.0`, `w_pg=2.4/w_pg_s=2.6` as defaults, then
   re-measure L1/L2/L3 per-tick. Both already measured; they just aren't reaching the agent.
2. **Measure `trise`** (task #26) → **F6**. Already built: TPOOL sums TXL+TXR, 6 TRISE cells detect
   the rise through a 35–110 tick delay bank, → STEER. +7 cells, never run in the body.
   Baseline to beat: ±0.05.

### Phase 1 — core systems, no vision

3. **Hunger dynamic range** (#27) → **F5**. Independent of the home vector.
4. **Arbiter in the body, per-tick** (#29, blocked by #27). Works isolated 3/3, never shown to select
   in the body.
5. **Speed-scale the avoidance turn** (#28) → **F7**. Gate on the forward-speed signal PG carries.
6. **`k_pi` sweep on XACC/YACC** (#30) → **F4**. The one L3 item testable *without* the compass —
   drive from ground-truth heading to isolate the accumulator.

### Phase 2 — the two structural walls

7. **F2, graded integrator.** The analog cell (`r`=1e9, `lam` huge, read membrane `S`) is the
   sanctioned escape and already exists. Step 6 tests whether gain alone fixes it. If not, this is
   where a `neuron.py` graded-neuron flag would be proposed, with a quantified regression.
8. **F1, velocity encoding.** Eleven parametric routes are dead. Needs a different mechanism, not
   another sweep.

### Phase 3 — vision (parked, frontier)

9. **Replace `SUN_AZIMUTH` with a learned map** (#23) → **F8**, and the real fix for **F1**: with
   re-anchoring you do not need to integrate ω accurately. Fisher 2019 / Kim 2019 (ER→E-PG GABAergic
   and plastic). Feasible because `info_val` is per-synapse. **Acceptance must include a
   scene-rotated world** — learned re-anchors, hardwired cannot.
10. **Wire the chroma channels** (#22). `vac` is built (+51 cells) and switchable; needs an
    odour-ablated validation run.

### Phase 4 — end-to-end

11. **L4 homing closed-loop**: HOME fires, agent closes distance. Blocked on 3, 6, 7.
12. **Long-horizon multi-world regression**, 8 seeds, geometry guard active.

---

## 4. THE HONEST SUMMARY

L1, L5, L7 work. L2, L3, L4 do not, and the two blockers behind them (**F1**, **F2**) are structural
rather than parametric — 11 and 3 approaches falsified respectively.

Phase 0 is the highest-value work available and costs hours, because it ships things already
measured. The unglamorous lesson: **three separate real capabilities (`SEEDSYN`, the ring tonic,
`trise`) were sitting disabled while I described them as broken or missing.** Hence Phase 0 before
Phase 2 — check what is already built and switched off before building anything new.

## 5. LIVE UI

`live_brain.py --port 8770` → **Params** button:

- **brain structure** — 9 on/off switches, one per population, with measured cell deltas
- **numeric groups** — 19 sliders (L1 ring, L2 compass, L2 vision, L3 path integration, world)
- **Shipped / Verified fixes** presets, **Apply & restart** rebuilds the network
- The wiring diagram regenerates to match (`topo_live.py`); optional circuits carry an **EXTRA**
  badge in the left panel, collapsed branches show **+N**
