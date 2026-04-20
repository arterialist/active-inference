# How the C. elegans sim learned to crawl — April 17–19, 2026

A deep walk-through of every change that moved the simulation from
"whole-body curl / whip / flung ballistic motion" to a clean, biologically
scaled **posterior-traveling S-wave with forward stride**.

All commits referenced live in `active-inference/` unless noted.

---

## 0. TL;DR

Two separate failures were stacked on top of each other and masked the
real problems:

1. **The neural command was wrong.** The "wave" was a synchronous
   whole-body contraction because (a) B-motor proprioception sensed
   the *wrong plane* (pitch instead of yaw), (b) it sensed the *same*
   segment — a local positive-feedback lock — (c) the drive was
   *unsigned*, so DB and VB could never alternate, and (d) the PAULA
   connectome has no intrinsic head oscillator to seed a wave.
2. **The body was floating.** MuJoCo's `opt.density = 4000 kg/m³` on
   a ~1000 kg/m³ worm produced ~4× buoyancy, so normal force on the
   floor was ~0 and *all* ground friction vanished no matter how
   anisotropic the friction tuple was. Every impulse from a muscle
   went into in-place rotation.

Fixing (1) produced a perfect neural wave over a worm that still
couldn't crawl. Fixing (2) finally let the wave translate into forward
motion. Supporting tuning (joint limits, muscle filter, tonic drive,
CPG amplitude, proprioception gain) brought the ratio of stride to
head-sweep into the biological regime.

Commits in order:

| SHA | Title | Core idea |
|-----|-------|-----------|
| `c5ea528` | fix food not being consumed sometimes | Post-body-step env hook; food check uses end-of-step pose and xy distance |
| `90caee1` | call hook | Call the new `post_body_step` from the C. elegans engine |
| `773f93f` | **fix locomotion and S-wave movement** | Signed non-local B-motor proprioception, head CPG, tapered gain, yaw plane, mass-weighted COM, hard joint-limit stops |
| `b1af168` | tune body settings | Density → 0, gravity → −9.8, forcerange ±3.5, `JOINT_ANGLE_MAX_RAD = 0.45`, `MUSCLE_FILTER_ALPHA = 0.16`, CPG/proprio re-tuned |
| `2ccf8e7` | update docs based on latest tuning | Documentation brought in line with the new constants and lab/demo parity |
| `be94282` | real_ms_per_neural_tick | Wall-clock pacing knob for interactive viewing |

---

## 1. The pipeline you need in your head

For every physics tick (2 ms MuJoCo step, `NEURAL_TICKS_PER_PHYSICS_STEP=1`):

```
body pose (joint yaws, contacts, head/COM)
      │
      ▼
SensorEncoder.encode()
  ├── chemical channels (AWC/ASE/…)
  ├── PROPRIO_NEURONS      → DVA, PVDL/R         (stretch receptors)
  └── MOTOR_PROPRIO        → _mpr_{DB*, VB*}     (B-motor reflex)
      │
      ▼
CElegansNervousSystem.tick()
  ├── _inject_sensory()                 normal synaptic path
  ├── _inject_motor_proprioception()    ⇦ stretch reflex → DB/VB.S
  ├── _inject_tonic_forward()           AVB tonic drive (+ optional B-motor bias)
  ├── _inject_off_cell_tonic()          AWC/ASER baseline firing
  ├── _inject_head_cpg()                ⇦ NEW: seed rhythm on DB1/VB1
  └── _volume_broadcast()               ALERM M0/M1
      │
      ▼
network.tick() — PAULA neurons settle
      │
      ▼
muscle decoder (low-pass α) → 48 actuator controls
      │
      ▼
CElegansBody.step()  → MuJoCo mj_step  → BodyState
      │
      ▼
environment.post_body_step()   ⇦ NEW: food consumption, end-of-step pose
```

Four layers changed in the last three days: **sensors**, **nervous
system**, **body/physics**, and **engine/environment**.

---

## 2. Sensor layer — `simulations/c_elegans/sensors.py`

### 2.1 Locomotion plane: pitch → yaw

Every inter-segment joint is a **two-axis hinge**:

- `j{i}{j}_pitch` (y-axis, dorsal/ventral)
- `j{i}{j}_yaw`   (z-axis, left/right — **the locomotion plane**)

All 48 actuators (`muscle_seg{N}_{D,V,L,R}`) drive **yaw**. The previous
`_encode_proprioception` / `_encode_motor_proprioception` filtered joint
angles by `"pitch" in jname`. Pitch is heavily damped and ~0 at all
times. **B-motor proprioception was reading a channel that never moved.**

Both encoders now filter by `"yaw" in jname`.

### 2.2 Non-local stretch receptor: offset −1 → −4

`MOTOR_PROPRIO` used to map each DB/VB neuron to the joint *immediately
anterior* of its soma (`_seg - 1`). That is still a local reflex: `DBₙ`
generates a dorsal bend at segment n, which bends joint n−1, which
drives `DBₙ` harder → deep curl, frozen segment.

```python
_PROPRIO_ANT_OFFSET: int = 4        # was 1
MOTOR_PROPRIO[name] = max(0, seg - _PROPRIO_ANT_OFFSET)
```

Reading a joint 4 segments anterior means `DBₙ` is driven by bends
that `DBₙ` **cannot directly amplify** (those come from segments
n−4/n−3). The loop now has head-to-tail transport delay and the wave
propagates instead of clamping. Offset = 2 fragmented the body into
~2 wavelengths; offset = 4 gives the ~1 body-length wavelength that
matches real crawling.

### 2.3 Signed, tanh-normalised drive

Old:

```python
curvature = abs(pitch_angles[joint_idx])
result[f"_mpr_{name}"] = clip(curvature / JOINT_ANGLE_MAX_RAD, 0, 1)
```

New:

```python
curvature = locomotion_angles[joint_idx]             # signed yaw
sign = -1.0 if prefix == "DB" else 1.0               # VB: +yaw, DB: −yaw
result[f"_mpr_{name}"] = float(np.tanh(sign * curvature / JOINT_ANGLE_MAX_RAD))
```

Sign convention (Wen et al. 2012, Fig. 3): when the anterior body
bends **dorsally** (+yaw), the ventral side is stretched → `VB` fires
more → posterior segment bends **ventrally**. Next segment, opposite.
Alternation is what creates an S-wave instead of a whole-body curl.

`tanh` (vs `clip`) gives a smooth saturating response and lets the
receiver (gain step) stay in `[-1, 1]` without hard kinks.

### 2.4 Stretch-receptor path also migrated to yaw

`DVA` (body-wide curvature integrator) and `PVDL/R` (segment-10 local)
now read yaw too. Before, they published ~0 every tick; they now see
the actual bend that ALERM gets driven by.

---

## 3. Nervous system — `simulations/c_elegans/neuron_mapping.py`

### 3.1 Head CPG (new)

The PAULA connectome is the Cook et al. 2019 connectome simplified —
specifically, it lacks an intrinsic head-rhythm circuit (RIM / RMD /
SMD / AVB oscillatory subnet). The chain reflex in §2 is *receptive* —
it amplifies and propagates an existing bend — but it cannot *start*
one. With a symmetric body at rest, no segment has a reason to fire
first, so nothing ever begins.

Verified empirically: with proprioception fixed but no CPG, every
adjacent segment's motor drive correlated at **r > 0.5 with 0 lag**
(`tuning/muscle_wave.py`). That is the signature of a synchronous
whole-body twitch.

```python
_HEAD_CPG_FREQ_HZ: float = 0.4     # final
_HEAD_CPG_AMP:     float = 0.08    # final
_HEAD_CPG_TARGETS = ("DB1", "VB1")
_NEURON_TICK_DT:   float = 0.002   # seconds/tick

def _inject_head_cpg(self, current_tick):
    phase = 2π · f · current_tick · dt
    drive = amp · sin(phase)
    DB1.S +=  drive
    VB1.S += -drive               # anti-phase
```

Two things to note:

1. **Anti-phase on DB1 vs VB1.** This is the "pacemaker" bit: it
   forces the head to alternate dorsal/ventral at a fixed frequency.
2. **Amplitude 0.08, not 0.35.** An early implementation used 0.35,
   which broke the neural wave free but also made every head sweep a
   violent impulse (the "whip" phase in `tuning/notes.md`). Once the
   body-medium coupling was fixed (§4) the CPG could back off to a
   small seed; the proprioceptive chain does the rest.

The frequency (0.4 Hz) interacts with observed gait: forward-speed
spectrum peaks at **0.76 Hz ≈ 2 × CPG frequency**, which is the
textbook signature of undulatory locomotion (each half-cycle of the
body wave delivers one forward thrust).

### 3.2 Tapered proprioceptive gain

Flat gain = every segment amplifies its local bend equally → the wave
reflects off the tail as a standing wave and locks. The head should
*drive*, the tail should *passively follow*.

```python
_PROPRIO_MOTOR_GAIN: float = 0.10     # was 0.08
_PROPRIO_TAIL_DECAY:  float = 0.5     # was 0.7; new field
```

```python
frac = MOTOR_NEURON_POSITIONS[motor_name]   # 0.0 head, 1.0 tail
gain = PROPRIO_MOTOR_GAIN * (1.0 - frac * PROPRIO_TAIL_DECAY)
n.S += val * gain
```

- Anterior DB1/VB1 get full 0.10 gain.
- Posterior DB7/VB11 get 0.10 · (1 − 0.5) = 0.05.

Decay 0.7 was slightly too aggressive — the tail stopped carrying the
wave — and 0.5 reproduces the biological asymmetry where head drives,
posterior follows. The tapered evol config key is now whitelisted in
the evolutionary optimizer.

### 3.3 Muscle low-pass filter reads instance attribute

The muscle decoder inside `motor_outputs_to_muscle_activations` used to
read the module-level constant `MUSCLE_FILTER_ALPHA` every step. The
lab's `/api/patch` writes `self._muscle_filter_alpha`, but the
decoder ignored it. Changed to:

```python
alpha = float(getattr(self, "_muscle_filter_alpha", MUSCLE_FILTER_ALPHA))
```

This is the bug that previously made lab UI knob twiddling invisible
for the muscle filter.

### 3.4 Forward-tonic nudge: 0.25 → 0.30

`_TONIC_FWD_CMD` (AVBL/AVBR depolarisation, proxy for the AVB↔B gap
junction bias) raised slightly so the forward command interneurons
keep the whole motor pool biased into a forward regime while the CPG
does rhythm. No structural change — just a tuning step.

---

## 4. Body / physics — `body.py`, `body_model.xml`, `config.py`

### 4.1 The buoyancy bug (root cause of "no crawling")

`body_model.xml`:

```xml
<!-- before -->
<option timestep="0.002" gravity="0 0 -5"   density="4000" viscosity="0.1"/>
<!-- after  -->
<option timestep="0.002" gravity="0 0 -9.8" density="0"    viscosity="0.3"/>
```

With `density = 4000 kg/m³` and worm mass-density ≈ 1000 kg/m³, buoyant
force per segment (ρ · V · g) was ~**4× the worm's weight**. The worm
floated. Normal force at the ground → 0 → `μ · N` → 0, no matter how
anisotropic `pair.friction = [0.01, 1.2, …]` was. Every muscle
contraction rotated the body in place; there was literally no surface
to crawl against.

`tuning/translation_check.py` made this legible: in 30 s the head
traced a 1.9 mm path while the COM moved 0.018 mm. Every segment
wiggled, nothing translated.

Crawling worms aren't in bulk fluid anyway — they're on an agar film —
so zero-density MuJoCo + a small viscosity (0.3) as numerical stability
damper is the right model. Gravity returned to physical −9.8 m/s² for
the same reason.

After the fix (30-s window, `tuning/gait_quality.py`):

| metric                         | before | after    |
|--------------------------------|--------|----------|
| forward speed (mean)           | 0.00 mm/s | +0.65 mm/s |
| motion-to-body-axis alignment  | +0.10  | +0.63    |
| lateral slip fraction          | 0.72   | 0.21     |
| heading drift (p95 \|dθ/dt\|)  | 132 °/s| 60 °/s   |
| forward-speed spectrum peak    | 4.35 Hz | **0.76 Hz = 2×CPG** |

### 4.2 Actuator force range: ±5.0 → ±3.5

With the wave now real and the body no longer floating, ±5.0 per
actuator overshot and made joints bang on their limits every cycle.
±3.5 is enough to drive a ±0.45 rad bend at 0.4 Hz without saturating.

### 4.3 Joint angle limit: 1.2 → 0.45 rad, hard-stop enforced

The comment in `config.py` used to say "~70° max bend". Real C. elegans
crawls at ~26° per inter-segment bend; 70° is more like a coiled
escape posture. With the old limit, the head could swing to ±70° under
strong drive and the tail, mechanically, would follow nonlinearly.

```python
JOINT_ANGLE_MAX_RAD = 0.45        # ~26°; was 1.2
```

`JOINT_ANGLE_MAX_RAD` is **also** the normaliser for all stretch
receptor and B-motor proprioception outputs, so tightening it
sharpens the proprioceptive gain curve at small bends, which is what
real animals use for gait control.

Because `body_model.xml` hardcodes `jnt_range` values and the lab
should be able to tune this live, `CElegansBody._apply_joint_limits()`
now overrides every hinge's range at `__init__` and on every `reset()`
using the current value of `config.JOINT_ANGLE_MAX_RAD`. It *also*
stiffens each hinge's `solref` / `solimp` to turn the joint-limit
constraint from a soft spring into a hard stop:

```python
solref = [0.005, 1.0]                         # 5 ms time constant
solimp = [0.995, 0.9999, 0.001, 0.5, 2.0]     # near-ideal hard-wall
```

Without the stiffening, strong proprioceptive feedback under the old
`_HEAD_CPG_AMP = 0.35` could push the joint 2–3× past its nominal
limit, which is precisely what "whip" looked like on the render.

### 4.4 Muscle filter α: 0.3 → 0.16

```python
MUSCLE_FILTER_ALPHA = 0.16        # was 0.3
```

α is the time constant of the low-pass on each muscle activation:
`a ← α · target + (1−α) · a`. 0.3 = fast tracking; every neural tick
the muscle snaps close to the new target. 0.16 puts the muscle's
effective time constant in the 10–20 ms range (per 2 ms tick), which
matches the biological electromechanical delay at the neuromuscular
junction and prevents the muscle from chattering at the rate the
neural state transitions. Visually this is what replaced the
"twitching" with a smooth sinusoidal drive per quadrant.

### 4.5 Mass-weighted COM

`CElegansBody.get_state().position` previously returned the head
(`seg0`) position. Head position swings side-to-side with every head
sweep. Any metric that used `BodyState.position` as "where the worm
is" got contaminated by head kinematics.

Now:

```python
xpos = self._data.xpos[seg_ids]          # (13, 3)
mass = self._model.body_mass[seg_ids]
com  = (xpos * mass[:, None]).sum(0) / mass.sum()
```

This is what `tuning/watch_com.py`, `gait_quality.py`, and the food
consumption code *thought* they were getting all along. The head is
still available on `BodyState.head_position`.

---

## 5. Engine / environment — `engine.py`, `environment.py`, `simulation.py`

### 5.1 `post_body_step` hook (`c5ea528`, `90caee1`)

Before: food consumption happened inside `environment.step()`, which
runs *before* the body steps. The head position used for the food
check was the **pre-step** pose, not the pose after MuJoCo integrated
motor outputs. On frames where the worm was moving fast, it could
enter and exit a food's consumption radius between `env.step()` and
the render, and the bite never registered.

After:

```python
# base_environment.py — new abstract slot
def post_body_step(self, body_state: dict) -> None: ...

# c_elegans/environment.py — override
def post_body_step(self, body_state):
    pos = body_state["head_position"]
    self._food_items = [
        (f, s) for f, s in self._food_items
        if np.linalg.norm(pos[:2] - f[:2]) > FOOD_CONSUMPTION_RADIUS_M
    ]
```

Two fixes in one: end-of-step pose *and* **xy-plane distance**. 3D
distance missed food when the nose pitched slightly — the worm would
crawl over the dot but register a large vertical gap and eat nothing.

`engine.py` and `c_elegans/simulation.py` call the hook after every
body step (with an `evol_trace` span so it's visible in profiles).

### 5.2 `real_ms_per_neural_tick` (`be94282`)

Pure ergonomics: optional wall-clock sleep after each neural sub-tick
so the interactive lab and demo can run at "real-time-ish" pace.
Default 0 (no sleep) — training and CI are unaffected.

---

## 6. The single-page tuned config

```python
# MuJoCo (body_model.xml)
opt.timestep     = 0.002
opt.integrator   = implicitfast
opt.gravity      = (0, 0, -9.8)
opt.density      = 0
opt.viscosity    = 0.3
geom.friction    = 0.01 0.8 0.001                 # default plane
pair.friction    = [0.01, 1.2, …]                 # worm ↔ floor (condim=6)
actuator.forcerange = -3.5 .. +3.5
joint.damping    = 0.05
joint.armature   = 0.0001
joint.range      = ±JOINT_ANGLE_MAX_RAD (applied at runtime, hard stop)

# Python constants (config.py)
JOINT_ANGLE_MAX_RAD       = 0.45     # ~26°
MUSCLE_FILTER_ALPHA       = 0.16
N_BODY_SEGMENTS           = 13
MUSCLE_QUADRANTS          = (D, V, L, R)

# Nervous system (neuron_mapping.py)
_TONIC_FWD_CMD            = 0.30
_TONIC_FWD_MOTOR          = 0.0
_TONIC_OFF_CELL           = 0.15
_PROPRIO_MOTOR_GAIN       = 0.10
_PROPRIO_TAIL_DECAY       = 0.5
_HEAD_CPG_FREQ_HZ         = 0.4
_HEAD_CPG_AMP             = 0.08
_HEAD_CPG_TARGETS         = (DB1, VB1)

# Sensors (sensors.py)
_PROPRIO_ANT_OFFSET       = 4       # B-motor reads joint N−4
motor proprio              = tanh(signed yaw / JOINT_ANGLE_MAX_RAD)
                              · sign(DB = −1, VB = +1)
stretch receptors          = |yaw| / JOINT_ANGLE_MAX_RAD
```

---

## 7. Why these changes compose into proper S-wave crawling

Walking forward through the loop with one neural tick:

1. **CPG seeds the head.** `DB1.S += +A·sin(ωt)`, `VB1.S += −A·sin(ωt)`.
   Head segment starts bending dorsal/ventral at 0.4 Hz.
2. **Head bend reaches the floor.** With `density = 0` and gravity
   −9.8, the head presses against the plane; anisotropic pair
   friction (transverse 1.2, longitudinal 0.01) converts the lateral
   push into a forward impulse instead of a slide.
3. **Joint yaw at segments 0–1 rises.** `_encode_motor_proprioception`
   publishes signed `_mpr_DB2`, `_mpr_VB2`, … values based on the yaw
   at joint `seg−4`. `DB2` reads joint ~0 (just clipped to head);
   `DB4` reads joint 0; `DB6` reads joint 2; etc.
4. **Signed stretch reflex alternates.** Because DB gets `sign=−1`
   and VB gets `sign=+1`, a dorsal head bend (+yaw) *excites* VB of
   segment N and *suppresses* DB — producing a ventral bend one
   offset downstream.
5. **Tapered gain lets the wave propagate without reflecting.**
   Posterior motor neurons get proportionally less proprioceptive
   drive, so the wave decays gently as it travels tail-ward instead
   of bouncing back.
6. **Muscle decoder low-passes at α=0.16.** Each actuator's control
   is a smooth sinusoid at ~0.4 Hz in the muscle's quadrant phase.
7. **Actuator force ≤ 3.5 over joint range ≤ 0.45 rad (hard stop).**
   Muscles produce bending within physical limits; the body does not
   over-rotate or hit its own limits, so the motion looks continuous.
8. **Each half-cycle delivers one forward thrust.** Ground friction
   anisotropy converts lateral motion into tangential drag asymmetry;
   because the wave travels posteriorly, net thrust is anterior.
9. **Forward-speed FFT peaks at 2 × CPG frequency = 0.76 Hz.** This
   is the invariant signature of genuine undulatory locomotion —
   each half-wave is one stride.

Before any of the April-17–19 changes, step 1 was missing (no CPG),
step 3 read pitch instead of yaw (zeros), step 4 was unsigned (same
sign of drive on DB and VB → co-contraction), step 5 was flat (wave
reflected), and steps 2/8 were broken because the worm was floating.
Every one of those had to be fixed for the stride to appear.

---

## 8. What's still wrong

From `tuning/notes.md`:

- **Acceleration peak/mean ratio ≈ 10** (target 3–5). The worm still
  has a slight lurch per stride; likely joint-limit bounce plus
  motor-neuron quantisation.
- The agar-film medium is still only modelled via anisotropic contact
  friction. A proper medium plugin (lateral viscous drag, no
  restitution) would close the remaining gap to in-vivo gait.

---

## 9. See also

- `tuning/notes.md` — engineering log with A/B sweeps and measurements
- `tuning/muscle_wave.py`, `wave_propagation.py` — neural-wave diagnostics
- `tuning/gait_quality.py`, `watch_com.py`, `translation_check.py` — body-motion diagnostics
- `docs/c-elegans/lab-parity.md` — why lab and demo servers can diverge
- `docs/c-elegans/sensor-encoder.md` — current sensor encoding reference
- `docs/c-elegans/mujoco-body.md` — current body physics reference
- `docs/neuromodulation/tonic-drives.md` — tonic + CPG + proprio references
