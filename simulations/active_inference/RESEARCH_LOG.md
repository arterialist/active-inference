# Research log

## 2026-07-31 — Cylinder replay passes two body-derived directions; closed-loop embodiment remains untested

The separate engineering cylinder is now more than a synthetic counter
harness. Its continuous raw-yaw transducer is fully PAULA: two fixed biased
opponent receptor currents feed graded PAULA receptors, signed PAULA
accumulators, PAULA refractory event doublets, then state-buffered quaternary
counter gates. Python replays only the measured yaw observation; it does not
threshold a turn, integrate yaw, decode for feedback, or write a state.

The baseline transducer is an explicit failure (TR body −80.80°, −57 expected
1.40625-degree bins, cylinder −7). A trace-calibrated parameterisation gives
−56 bins on that source. Held fixed, it gives +61 bins on an independently
body-generated opposite TL trace (+84.62°, +60 expected), with 100% turn
decode coverage in both records. A near-equal-count flat control (80 cells
versus cylinder 78) gets the sign but overshoots badly (TR −8 versus −4;
TL +9 versus +4 coarse bins). This is limited replay evidence, not held-out
seed generalisation (the seed-29 fixed protocol was byte-identical), biological
fidelity, a closed-loop body result, PI, or homing acceptance. The next step
is a separate embodied variant that audits every ring-connected pathway.

---

## 2026-07-31 — Explicit PB/P-EG stabilizer is active but initially degrades the corrected sensorimotor route

The separate V5 sensorimotor agent adds a visible maintenance geometry
E-PG→bilateral PB→P-EG→same-column E-PG to the V2 motor-prediction/raw-yaw
fusion. At PB→P-EG weight 0.8, maintenance PB release is nonzero but P-EG is
silent and the ring repeats V2's −20.26° response to the −80.80° physical
turn. At the isolated-harness active control weight 1.6, P-EG releases on
268/400 turn ticks (mean population release 2.47), but the ring falls to
−16.68° and contains reversal blocks. Thus the trace distinguishes absent
stabilisation from an active competing delayed loop. V5 remains experimental,
separate, and disconnected from PI/HOME/navigation; the next valid topology
test needs a spatially offset trailing P-ENb return rather than same-column
recurrence.

---

## 2026-07-31 — Separate sensorimotor-compass programme: V1 input-sign error found; V2 foundation correct but recurrent transfer remains negative

The working demonstration agent remains untouched. A separate opt-in series
now isolates motor prediction, physical self-motion, signed PAULA prediction
error, and local P-EN update (`sensorimotor_agent3d*.py`). The 1,400-tick
seed-11 embodied `TR` course records every tick of body yaw, raw yaw input,
four sensory and motor cells in both directions, both errors, both fused
updates, P-EN release/raster, and E-PG raster.

V1 is a useful negative control rather than a result to tune: it passed raw
*signed* yaw as PAULA external `info`. The PAULA input transport schedules
only `info > 0`, so a negative sensory value is absent rather than inhibitory.
The trace showed the impossible consequence directly: during a −80.80° body
turn, motor prediction was CW but V1 sensory release was CCW; ring motion was
only −7.71°. Prediction ablation gave 0.00°, while sensory ablation retained
−15.85° from relay prediction. This is a representation bug, not an embodied
failure.

V2 fixes sign explicitly: the body transducer supplies two non-negative
currents and ordinary positive/negative PAULA dendrites make the opponent
direction. During the same turn both mean signed sensory (CCW−CW = −0.0887)
and motor prediction (−0.1061) correctly favour CW. A 44-tick ordinary
membrane suppresses competing fusion, and a source-latency-matched P-EN ring
delay improves ring motion to −20.26°, but this remains far below body yaw.
The isolated recurrent transfer curve explains the constraint: direct P-EN
drive has a sharp nonlinearity (0.08→−0.76°, 0.20→−2.58°,
0.50→−60.52°, 0.75→−213.23° over the same 200-tick window). The embodied
fused source is 0.081 CW, and increasing only the final synaptic gain did not
reproduce the direct steady drive because it changes timing relative to the
bump.

Two explicit leading-update scale-ups are also recorded as negatives. V3
uses four real CPG phases × six ordinary PAULA coincidence cells per direction
and gives −3.57° ring motion: its pulses are too sparse. V4 uses only a
four-tick inherited leaky dendritic trace and four cells per phase; it becomes
too broad and counter-rotates (+13.62°). Neither uses the rejected
sample/hold subclass. The next valid biological structural experiment is the
separate P-EG→P-ENb-like trailing/stabilising loop, first in isolation. No
variant is wired into PI, HOME, homing, or the normal demo. Full architecture,
literature boundary, and artifacts are in `SENSORIMOTOR_COMPASS_DESIGN.md`.

---

## 2026-07-31 — Full-tick PB bridge and CPG-phase gate transfer results are both negative (X for compass promotion)

The explicit PB/EB bridge timing audit is now closed. The isolated,
latency-matched reset-gated bridge remains live and gives signed fixed-current
movement (+54.98° / −136.06°), but the sustained embodied turn is −80.80°
body yaw versus only −38.72° ring motion (two reversal blocks). Its raw-gyro
PAULA-only replay is phase-identical to the embodied circuit on every tick
(0.000° mean/max difference). The ordinary reset suppresses the accumulating
PB residual, but at the measured update phase it suppresses the useful drive
as well. This rejects that parameterization without treating the initial
one-tick relay error as a verdict on PB/EB anatomy.

The biologically motivated alternative was deliberately not the rejected
sample/reset/hold subclass. It adds 44 normal PAULA CPG-phase cells, a pair of
bounded normal graded vestibular afferents, and 72 normal PB coincidence cells
whose dendrites receive local E-PG, signed gyro, and selected gait phase. The
isolated fixed-current circuit is live and signed (+4.74° / −28.15°; 175 PB
gate spikes). Full-tick embodied/replay data show exactly why it fails under
composition: no PB phase gate reaches threshold (maximum 1.98 versus 2.10),
so both exact-replay and body ring motion are 0.00° during a −80.80° physical
turn. A measured local-ring scaling correction was run once, not swept: it
caused 20,390 broad PB spikes in both directions under the isolated harness
and eliminated signed motion. The present one-cell-per-column gate therefore
lacks a usable local-bump/background contrast margin; more copies or gain are
not a valid next move.

The replay inspector now exposes this evidence side by side with the body:
`uv run python simulations/active_inference/live_brain.py --compass-replay
simulations/active_inference/experiments/results/compass_replay_pb_phase_gated_selective_seed11.json`,
then open `/compass-replay`. It includes synchronized raw tick traces, 3-D
controls, and event markers. The current matrix and causal graph are in
`COMPASS_CAUSAL_BOUNDARY.md`; Rule 12 now makes phase-gate margin and relay
latency checks mandatory before a scaled circuit is promoted.

---

## 2026-07-31 — Initial PAULA PB/EB bridge transfer is negative, but a timing audit reopens the topology comparison

An opt-in, inspectable PB/EB topology now exists in the compass rather than a
relabeled direct P-EN→E-PG edge: bilateral ordinary PAULA PB update relays
carry P-EN CL/CR traffic to the offset E-PG target; the optional maintenance
path is E-PG→bilateral PB→P-EG→same-column E-PG.  It uses standard PAULA
synapses and graded membranes only.  The separate isolated whole-compass
harness (`python experiments/pb_eb_bridge_isolated.py --bridge
--w-pb-opponent 1 --pb-shift-lam 22`) showed the update relays active, ring
liveness 0.998, and signed fixed-current movement (+43.1° CCW, −95.9° CW).
That narrow prerequisite passed, so it was advanced rather than silently
discarded.

The full-tick evidence rejects promotion of the **initial timing
implementation**. In the sustained PAULA physical
turn, the 22-tick PB-opponent bridge produced −46.5° ring motion for −122.0°
body yaw.  The paired no-physics run received precisely the recorded raw gyro
before each corresponding PAULA tick and reproduced the ring phase exactly
(0.000° mean/max difference); this is a circuit transfer failure, not an
unobserved body/host handoff.  Its PB release trace shows why: the right-side
relay grows across stable 44-tick strokes (about 69→205→231 total release)
until it pins the ring and produces reversals.  Ordinary inhibitory reset
from the existing PAULA reset-clock was tested as the direct, biologically
ordinary repair.  Reset gain 8 still gives −39.07° ring for −80.80° body yaw
during the command, two reversal blocks, and an exact replay; gain 16 is
worse.  No PB candidate is wired into accepted PI or homing.

Crucially, `experiments/pb_eb_bridge_latency.py` then found a one-tick
comparison error: the ordinary PB relay receives on one PAULA update and
releases on the next, while the initial builder made its two dendritic delays
sum to the direct-edge delay. The bridge therefore arrived one tick later than
its direct `d_push=4` control. The builder now compensates for that relay tick
and the latency harness asserts matched arrival. The older traces retain their
evidence of slow-residual accumulation, but they are not a final rejection of
the latency-matched PB topology. It remains unaccepted and must rerun the
isolated → exact raw replay → body sequence before any PI/homing integration.
The full raw traces are served by `live_brain.py --compass-replay` and the
causal matrix / diagram is maintained in `COMPASS_CAUSAL_BOUNDARY.md`.

---

## 2026-07-31 — Conjunctive P-EN and rate-ring prototype resolves liveness, not heading fidelity (X for promotion)

Primary fly evidence requires a P-EN output to conjunctively encode the local
heading bump and angular velocity, and describes left/right shift populations
whose relative activity changes with turn direction ([Turner-Evans et al.,
2017](https://elifesciences.org/articles/23496); [Green et al.,
2017](https://doi.org/10.1038/nature22343)). The first continuous P-EN
prototype violated that condition: a velocity input alone produced graded
release from every P-EN column, so it drove the ring globally. It could remain
live in a short physical right turn but was asymmetric over sustained turns.

The experimental fix is strictly inherited from `GradedNeuron`, without a
change to `neuron.py`: `ConjunctiveGradedNeuron` releases only the local
minimum of its designated ring and velocity dendritic drives. The complete
gyro route retargets that gate from the unused direct-drive port to the actual
PAULA opponent port. A default-off rate-coded ring was then added using the
existing graded-neuron mechanism, which removes the previous all-or-nothing
spiking travelling-wave threshold. `compass_transfer_diagnostic.py` records
direct P-EN, raw-gyro-only, natural nonvisual food/toxin locomotion, and an
isolated replay of its exact raw gyro trace. Its frozen seed-11 trace at
`experiments/results/compass_transfer_conjunctive_20260731/` remains live in
all four conditions; the rough live course has 30 raw-yaw sign changes and
28.7 rad/s 95th-percentile raw yaw, while replay differs from live by only
3.38° mean circular ring phase. Thus irregular MuJoCo movement is not a
host/embodiment-only failure: the PAULA gyro representation itself produces
the same response in replay.

The new prototype does not yet meet the required tracking criterion. In the
sustained physical course, the candidate with P-EN gain 0.25 gives
right −48.9° ring versus −122.0° body and left +34.7° ring versus +56.9°
body; higher common gains over-rotate and can wrap the left course (for
example +422.8°). A divisively normalized PAULA opponent subclass was also
tested; it remains live but is under-gained at normalization 2 (−54.0°) and
over-gained at 4 (−287.8°) on the same right turn. These are retained as
explicit, trace-backed negatives. Recurrent compass PI, home-vector accuracy,
and homing remain unaccepted.

Focused nonvisual regression after the additions:
`20 passed` in 71.43 s (`test_embodied_configuration`, food gradient, toxin
gradient, and isolated PI tests). The new configuration regression verifies
that the P-EN gate reads the PAULA opponent synapse, not the unused direct
velocity port.

---

## 2026-07-31 — Long-turn recurrent-compass diagnostic remains negative (do not promote to PI)

The accepted continuous gyro route is a short physical-turn and ordinary-gait
component result, not a long-horizon heading claim. The persistent diagnostic
`python -m simulations.active_inference.experiments.embodied_compass_longturn_diagnostic`
now records the failure explicitly. In the declared sustained PAULA `TR`
course (ring burn-in tick 200; physical turn current 500--900), the direct
graded route ends with −122.0° physical yaw but +25.7° ring movement
(+147.7° tracking error). A graded opponent prototype was also tested without
changing the default or accepted route: a slow opponent version inverted
(+58.3° ring, +180.3° error), while the fast version could silence the ring
before the course ended. Increasing P-EN→ring delay also silenced the ring.

Consequently, recurrent-compass Stone PI and homing remain unaccepted. The
next valid change must establish a stable signed long-turn P-EN/ring regime
with the same raw-gyro, PAULA-only interface before PI parameters or HOME
behavior are retuned. The experimental opponent route is retained only as an
opt-in diagnostic representation, with a direct-unit regression; it is not a
default agent mechanism or evidence of navigation.

---

## 2026-07-30 — Continuous PAULA gyro route tracks physical gait in the recurrent compass (R)

**Component accepted, with bounded scope.** The earlier spiking vestibular
comparator route was sufficient for a prescribed-turn calibration but had a
dead band/runaway under normal CPG gait. The accepted opt-in replacement keeps
the same PAULA full-stroke delayed gyro afferents but sends their continuous
graded release directly to the local P-EN gates. Its complete causal path is
raw MuJoCo yaw rate → paired PAULA delay-line afferents → graded P-EN drive →
recurrent ring; the physical output path remains PAULA turn cells → relay /
graded muscles → MuJoCo. There is no injected pose/heading, host filter,
target, or scripted motor turn.

`python -m simulations.active_inference.experiments.embodied_compass_graded_causal`
produced the frozen four-seed artifact at
`experiments/results/embodied_compass_graded_causal_20260730T162806Z/`.
Seeds 11, 23, 44, and 77 each gave physical left/right turn responses of
+73.5°/+44.2° and −58.6°/−37.1° (ring/body gains 0.60/0.63). With no test
turn current, ordinary PAULA CPG locomotion gave +33.2° post-burn body yaw and
+24.1° ring movement (gain 0.72). The output-specific graded-afferent→P-EN
cut retained the physical turn (+77.2° in the turn control; +33.2° in ordinary
gait) while reducing ring movement to 0.0° in both contexts.

This accepts the physical vestibular-to-ring route, including normal gait
after its declared 200-neural-tick ring-settling burn-in. It remains opt-in
while the next integration question is tested. It does **not** establish
long-horizon heading error bounds, path integration, a home vector, HOME
selection, or return-to-origin homing.

---

## 2026-07-30 — Raw physical gyro now causally shifts the PAULA recurrent compass (R, isolated calibration)

**Component accepted, with a deliberately narrow scope.** The opt-in
full-stroke vestibular route now establishes a physical, nonvisual input path
to the recurrent ring: prescribed PAULA `TL`/`TR` current → existing PAULA
relay and graded muscle cells → MuJoCo body turn → raw signed MuJoCo yaw rate
→ PAULA delay-line notch/opponent vestibular events → P-EN shift → ring.
There is no injected pose or heading, host-side turn, visual cue, target, or
Python heading update.  The prescribed turn current is a component-calibration
stimulus into the shared PAULA turn neurons, not an autonomous action policy.

`python -m simulations.active_inference.experiments.embodied_compass_gyro_causal`
produced the frozen four-seed artifact at
`experiments/results/embodied_compass_gyro_causal_20260730T155556Z/`.  Across
seeds 11, 23, 44, and 77, the left neural turn rotated the body by +73.5° and
the ring by +41.2° during the declared stimulus window (gain +0.56); the right
turn rotated body/ring by −58.6°/−73.5° (gain +1.26).  In the output-specific
control, zeroing only notch/comparator→P-EN gain retained the PAULA turn
spikes, nonzero actuator output, +77.2° physical body turn, and physical
vestibular events, but reduced ring motion to 0.0°.  That separates the
recurrent compass response from a scripted pose update, missing motor course,
or missing gyro representation.

This does **not** accept free-running heading fidelity, stable integration of
natural locomotor stroke dynamics, path integration, HOME selection, or
return-to-origin homing.  The route remains experimental and off in the
default organism until it succeeds in that composed setting.

---

## 2026-07-30 — PAULA arbiter changes physical exploration after feeding (R, narrow action-selection route)

The default composed nonvisual agent now gives the hunger/arbiter state a
specific causal motor consequence without using a Python mode-to-action
branch.  The pre-existing background curved-search primitive was split from
the independent danger escape gate: `SEARCH` receives engine and food-trend
inputs and is inhibited by FORAGE; `STEER` receives only TRISE/AVOID danger
input (plus the existing HOME veto).  Both gates reach the same PAULA relay,
graded-muscle, and MuJoCo actuator route.  Therefore a hungry animal stops
background exploration, but can still reorient away from toxin or learned
danger.

`python -m simulations.active_inference.experiments.embodied_arbiter_explore_causal`
produced the frozen four-orientation artifact at
`experiments/results/embodied_arbiter_explore_causal_20260730T143340Z/`.
The experiment begins in a physically odour-free world.  At a fixed neural
tick it places food at the body; the normal MuJoCo radius/contact detector,
not an experiment-side neural input, then produces the usual food event and
hunger drain.  In all four seeded initial orientations, the intact circuit
had pre-meal hunger 4.2, FORAGE/EXPLORE spikes 288/0, and SEARCH 2; after
ordinary food contact it had hunger 0, EXPLORE/FORAGE 960/0, and SEARCH 112.
The output-specific control zeroed only FORAGE→SEARCH: it retained the same
food tick, hunger drain, WTA transition, and nonzero actuator path but
restored 34 pre-meal SEARCH spikes, 100.6 rather than 27.0 units of physical
left/right paddle-force asymmetry, and a larger local lateral excursion.

This accepts a limited physical consequence of the interoceptive FORAGE to
EXPLORE switch.  It is deliberately not evidence for autonomous food
seeking, HOME selection, general multi-drive arbitration, prediction-error
minimization, or a completed active-inference organism.  The scheduled meal
isolates the neural state transition; it does not substitute for a
goal-directed acquisition test.

---

## 2026-07-30 — PAULA mushroom-body counterconditioning changes embodied food approach (R, local learned avoidance)

**Component accepted, with a deliberately narrow scope.** The composed
nonvisual PAULA circuit now has a replicated learned physical outcome:
physical toxin contact while a neutral food odour is present trains the
`STG_T`/DAN → KC–MBON route; a later food-only physical probe recruits
`MBON`/`AVOID`, then the lateral-horn-like (`LH`) turn pathway and ordinary
PAULA motor circuit.  It avoids rather than collects that local food source.
The world supplies only sanctioned odour/contact currents and reads the
graded muscle membranes; it does not select the learned turn.

`python -m simulations.active_inference.experiments.embodied_mb_food_avoidance_causal`
produced the frozen four-seed artifact at
`experiments/results/embodied_mb_food_avoidance_causal_20260730T141500Z/`.
Eight short physical teaching trials were followed by a food-only MuJoCo
probe in two mirrored source placements.  The taught route collected no food
in all four seeds (minimum separation 0.758--0.830).  In contrast, both
controls collected the food in every seed: zeroing only `LH`'s learned-output
gain (`w_lh_avoid=0`) preserved substantial learned MBON/AVOID activity
(MBON 230--231, AVOID 51--52) but returned collection; removing the physical
teaching trigger (`w_trig=0`) removed MBON/AVOID activity and also returned
collection.  The output-specific ablation therefore separates the behavior
from a missing contact, missing odour, or an inert plasticity circuit.

`LH` retains a small innate food-sensory contribution, so the acceptance does
not claim that every `LH` spike is learned.  The accepted causal statement is
limited to this counterconditioned, nearby food source and the tested mirrored
placements.  It is not yet a general multi-odour, long-horizon, lifetime, or
arbiter-driven learning result.

---

## 2026-07-30 — Earlier MB action probe: no accepted effect (superseded)

The physical toxin-contact teaching experiment remains a valid neural learning
result: in its standard seed-11 trace, eight real toxin entries produce 48
`STG_T` spikes and a later non-contact toxin odour probe produces 8 MBON and
3 `AVOID` spikes, while the sting-trigger ablation produces 0 MBON/`AVOID`
spikes.  That establishes the physical teaching event and a toxin-selective
neural output; it does **not** establish learned avoidance behaviour.

An explicit 50-step MuJoCo probe with innate toxin turn and temporal-rise
routes disabled showed the problem directly.  The taught agent had MBON=10,
`AVOID`=3, and STEER=177--180 neural-tick spikes, essentially the same
trajectory as the teaching ablation (minimum distance 0.660 versus 0.661 in
the tested layout).  Raising the MBON/AVOID gains increased neural activity
but did not reliably increase clearance.  A provisional lateral
MB-identity×toxin-gradient circuit was also tested and then removed: it could
make turn populations fire, but its physical distance effect was small and
layout-dependent, so it is not retained as a claimed mechanism.

`w_avoid_steer` is now correctly forwarded through the builder allowlist for
future causal calibration; its default remains 3.0, so this does not alter the
accepted innate food/toxin routes.  Do not describe associative value as
behaviourally consequential until a learned output produces a replicated body
outcome against a teaching-path ablation.

---

## 2026-07-30 — Graded Stone PI memory now reaches embodied home output (R, expression path)

**Component accepted, with scope.** The optional `pi_stone` path is now a
working neural expression route in the composed MuJoCo organism:
physical-body speed → graded circular memory → opponent/home state →
heading-gated CPU1 comparator → home-turn neurons → ordinary relay/muscle
actuators.  It remains opt-in and does not alter the default organism.  This
is a causal output-path result, not yet a claim that the recurrent compass is
accurate or that the body reliably returns to its origin.

**Integration correction.** Selecting `pistone=True` built memory cells but
did not mark the agent to drive their speed transducer; direct `AIFAgent3D`
ticks also bypassed that drive.  The route previously added memory to `OPP`
but left CPU1 on the unrelated CPU4 ladder, allowing HOME to win without a
home turn.  The composed assembly now drives the memory on both stepping
surfaces, disables the legacy CPU4 sources only for `pistone_opp`, and wires
the same graded memory into both OPP and the existing heading-gated CPU1
comparators.  No Python angle/readout chooses the mode or turn.

**Causal result (R, 2026-07-30).**
`python -m simulations.active_inference.experiments.embodied_stone_home_causal`
recorded all four standard seeds under
`experiments/results/embodied_stone_home_causal_20260730T120152Z/`.  In an
empty, vision-free physical world, the PAULA CPG moves the MuJoCo body and its
measured speed is the only memory drive.  The `stone_outputs_zero` control
retains the same graded memory cells, body, speed transducer, CPG, relays, and
actuators but zeros only the Stone→OPP and Stone→CPU1 projections.

| Condition, every seed | Final memory spatial span | OPP / HOME / home-turn spikes | Peak actuator control |
|---|---:|---:|---:|
| Intact Stone output route | 0.440 | 1,943 / 2,322 / 152 | 10.68 |
| Stone output projections zeroed | 0.408 | 0 / 0 / 2 | 10.68 |

The two residual control home-turn spikes arise from CPU1's retained
heading-only input and are more than 75-fold below intact output; neither OPP
nor HOME fires in that control.  Every raw trace records proprioceptive speed,
the twelve graded memory states/releases, OPP, mode/CPU1/home-turn activity,
muscles, actuator control, and body pose.  The next spatial requirement is a
time-varying, physically valid recurrent compass plus an outbound/return
trajectory; do not infer return-to-origin success from this output-path test.

---

## 2026-07-30 — Lateral food collection now has embodied causal evidence (R)

**Component accepted, with scope.** The nonvisual PAULA food route now has a
physical collection result, rather than only a short sensor/turn trace.  The
result covers a deliberately lateral food source and the normal
food-sensor→opponent-turn→CPG/muscle/MuJoCo path.  It is not a long-life
foraging-rate, multi-source, learned-value, or arbitration claim.

**Causal result (R, 2026-07-30).**
`python -m simulations.active_inference.experiments.embodied_food_collection_causal`
recorded the standard four seeds under
`experiments/results/embodied_food_collection_causal_20260730T114104Z/`.
Vision and toxins are absent; the sole food source begins at `(-1.5,+1.0)`,
off the ablated agent's baseline path.  The initial source respawns after a
physical collection event, and the raw tick traces retain that event count.

| Condition | Food-sensor spikes | Collection tick | Food collected |
|---|---:|---:|---:|
| Intact food sensor→turn route | 5,267--5,335 | 707 | 1 / 1 / 1 / 1 |
| `w_sd=0` output ablation | 5,875 | none | 0 / 0 / 0 / 0 |

Both conditions retain the same normal peak actuator control (10.68), so the
control neither removes the stimulus nor freezes the body.  The ablation's
equal-or-higher sensor count but zero `TL−TR` confirms that collection requires
the neural food-sensor-to-turn output rather than a scripted path or passive
coasting through the food radius.

---

## 2026-07-30 — Symmetric head-on toxin escape now has embodied causal evidence (R)

**Component accepted, with scope.** The nonvisual agent now resolves the
opponent circuit's symmetric head-on blind spot through the neural temporal
rise route `TXL/TXR → TPOOL → TRISE → STEER → relay/muscle → MuJoCo`.  This
is a physical contact-avoidance result for one near, head-on hazard.  It does
not establish learned toxin value, a general maze policy, or long-horizon
survival across arbitrary hazard layouts.

**Causal result (R, 2026-07-30).**
`python -m simulations.active_inference.experiments.embodied_headon_toxin_causal`
recorded four standard seeds in
`experiments/results/embodied_headon_toxin_causal_20260730T113146Z/`.  Vision
and food are absent; the only source is a symmetric toxin at `(-0.75,0)`.  The
normal crossed toxin turn neurons have zero aggregate imbalance, as expected
for symmetric input, so the mechanism is demonstrated through TRISE and
STEER—not an invented signed turn.

| Condition, every seed | Toxin-sensor spikes | TRISE / STEER spikes | Contacts | Closest distance |
|---|---:|---:|---:|---:|
| Default TRISE route | 6,841 | 628 / 349 | 0 | 0.561 |
| `w_trise=0` output ablation | 6,831 | 620 / 279 | 1 | 0.516 |

The ablation preserves the temporal-rise population and normal actuator drive
(10.67 peak control) while severing only its output onto `STEER`; this rules
out a missing toxin stimulus, deleted circuit, or immobilized body as the
explanation.  `EmbodiedAgentConfig.trise` and the live-brain configuration now
default to `True`, so the accepted route is present in ordinary embodied and
live runs (1,753 neurons), rather than an experiment-only option.

---

## 2026-07-30 — Analog XACC/YACC path integration now has physical causal evidence (R, isolated heading route)

**Component accepted, with scope.** The analog `XACC/YACC` pair in
`central_complex.py` is now a supported displacement-memory primitive when
fed a calibrated head-direction sensory code and real MuJoCo speed.  The
evidence covers PAULA motor → moving MuJoCo body → physical heading/speed
transducers → `RING`/`PG` → non-spiking analog `XACC/YACC`.  It deliberately
holds recurrent-ring dynamics out of scope, so it does **not** accept the
vestibular compass or closed-loop homing.

**Causal result (R, 2026-07-30).**
`python -m simulations.active_inference.experiments.embodied_xacc_pi_causal`
recorded the standard four seeds in
`experiments/results/embodied_xacc_pi_causal_20260730T111019Z/`.  A PAULA
CPG/muscle rower drives the real MuJoCo body through an outward leg, a strong
left curve, a second leg, and a strong right turn that reverses the heading.
The runner derives heading from the body pose and speed from `qvel`; a
seven-cell external heading code is the explicitly scoped sensory transducer
that bypasses recurrent ring integration.  No Python path-integrator state or
body kinematics is used.

| Seed | Final vector angular error | Outbound magnitude / physical distance `r` | Zero-input retention | `k_pi=0` final magnitude |
|---:|---:|---:|---:|---:|
| 11 | 1.56° | 0.998 | 1.001 | 0 |
| 23 | 0.87° | 0.998 | 1.000 | 0 |
| 44 | 1.58° | 0.998 | 1.000 | 0 |
| 77 | 1.15° | 0.998 | 0.999 | 0 |

The final physical route turns at least 94.7° left and returns to within 1.5°
of its initial heading.  In every seed, the ten-tick outbound magnitude
windows are nondecreasing, the accumulator retains its vector across 240
zero-input ticks, and the ablation leaves the upstream PG count unchanged
(17,245--17,251 spikes) while removing the XACC/YACC vector entirely.  The
real-speed gain sweep selected `k_pi=64`: 16 and 64 are linear, 128 is already
compressed (0.846--0.917 of its low-gain prediction), and 256 is the measured
saturation control (0.427--0.543).  `AIFAgent3D.parts()` now forwards the
previously omitted `k_pi` and `lam_acc` arguments, without changing defaults.

The per-tick physical and replayed neural traces, source fingerprints, and
acceptance decision are all retained beside the manifest.  A compact focused
regression guards heading-vector orientation, retention, and the `k_pi=0`
control.  The next spatial step is to replace this calibrated heading-code
input with an accepted recurrent/vestibular compass output and then test
closed-loop homing.

---

## 2026-07-30 — Diagonal toxin avoidance now has short-horizon embodied causal evidence (R)

**Component accepted, with scope.** The full nonvisual `AIFAgent3D` now has a
current physical survival result for one near, diagonal toxin hazard.  This is
stronger than the previous lateral neuron-turn measurement because the normal
CPG, graded muscles, MuJoCo body, toxin-contact detector, and neural toxin
path stay in the loop.  It is not a symmetric head-on toxin result, a learned
valence result, or a long-horizon survival claim.

**Causal result (R, 2026-07-30).**
`python -m simulations.active_inference.experiments.embodied_toxin_escape_causal`
recorded the standard four seeds under
`experiments/results/embodied_toxin_escape_causal_20260730T111958Z/`.  Vision
is disabled, all food and other toxins are inert, and the only hazard is the
physical source at `(-1.2,+0.5)`.  Every result stores raw neural/physics ticks
including injected toxin current, left/right toxin-sensor spikes, turn spikes,
muscle state, actuator commands, pose, and source distance.

| Condition, every seed | Toxin-sensor spikes | Contacts | Closest source distance | Interpretation |
|---|---:|---:|---:|---|
| Intact toxin pathway | 6,500 | 0 | 1.194 | The moving PAULA/MuJoCo agent stays out of the physical contact zone. |
| `w_tox=0` toxin-sensor→turn ablation | 6,639 | 1 | 0.166 | The sensors and motor path remain active, but the agent enters the hazard. |

Both conditions retain substantial actuator drive (peak controls 10.67 and
10.68 respectively), so the effect is neither a sensory deletion nor a
freezing control.  The aggregate `TL−TR` total balances over the 800 neural
ticks and is therefore not cited as a signed-turn result; the causal macro
claim rests on physical contact versus no contact under the synaptic ablation.
The next toxin requirement remains the head-on blind-spot experiment.

---

## 2026-07-30 — Physical-contact mushroom-body valence learning now has causal evidence (R)

**Component accepted, with scope.** The integrated PAULA mushroom-body route now has current causal
evidence for physical toxin contact teaching a toxin-selective aversive neural response.  It covers
contact → sting latch/neuromodulation → KC-MBON plasticity → MBON/AVOID response to a later,
non-contact odour probe.  It does **not** yet establish that this learned signal improves a
long-horizon foraging trajectory.

**Causal result (R, 2026-07-30).**
`python -m simulations.active_inference.experiments.embodied_mb_valence_causal` recorded the
four standard seeds under
`experiments/results/embodied_mb_valence_causal_20260730T105130Z/`.  Each condition gives the
same persistent PAULA brain eight physical toxin-entry trials; only the body/environment is reset
between trials.  The subsequent food and toxin probes are physical, non-contact odour placements
through the normal antennal-lobe/KC route, with vision disabled.

| Condition, every seed | Toxin entries / STG-T spikes | Food / toxin MBON probe | Toxin AVOID probe |
|---|---:|---:|---:|
| Physical teaching | 8 / 48 | 5 / 8 | 3 |
| `w_trig=0` sting-trigger ablation | 8 / 0 | 6 / 0 | 0 |

The ablation preserves the same physical contacts and odour exposure but prevents the neural
teaching latch from firing.  It therefore removes the later toxin-selective MBON/AVOID response,
while the full circuit retains it without a new contact during probing.  This also exposed and
closed an allowlist defect: `w_av_mbon` (MBON→AVOID gain) had been silently dropped by the composed
agent.  Raw training/probe traces, synaptic-weight series, contact count, and source fingerprints
are retained in the evidence directory.

---

## 2026-07-30 — Integrated nonvisual lateral-toxin avoidance now has causal evidence (R)

**Component accepted, with scope.** The crossed lateral toxin circuit in the actual
`AIFAgent3D` is now a supported nonvisual survival primitive: a physical toxin gradient activates
the PAULA toxin sensors and selects the opponent turn population that turns away from its lateral
source.  It is **not** a head-on toxin solution—the symmetric-gradient blind spot remains open—and
does not establish learned toxin value or long-horizon survival performance.

**Causal result (R, 2026-07-30).**
`python -m simulations.active_inference.experiments.embodied_toxin_gradient_causal` recorded four
standard seeds in
`experiments/results/embodied_toxin_gradient_causal_20260730T104329Z/`.  Vision is disabled, food
is out of range, and the ordinary short-range toxin field plus normal MuJoCo body are retained.
Every raw trace records actual injected currents, toxin-sensor spikes, `TL`/`TR`, graded muscles,
actuator commands, and body pose.

| Condition | Toxin-sensor spikes | `TL − TR` | Interpretation |
|---|---:|---:|---|
| Toxin at `(-2,+1)` | 502 | +11 | Crossed toxin wiring selects the matching opponent turn. |
| Mirrored toxin at `(-2,-1)` | 504 | -9 | Mirroring the physical source reverses the selected turn population. |
| Same first field, `w_tox=0` | 505 | 0 | Sensor activity persists, but the toxin-sensor→turn pathway is gone. |

The table is identical for seeds 11, 23, 44, and 77; `acceptance.json` reports `passed: true`.
The usual CPG/graded-muscle/actuator route remains active in all three conditions, so the causal
control did not make the body still.  A focused regression now guards the signed output and the
synaptic ablation.

---

## 2026-07-30 — Integrated nonvisual food-gradient steering now has causal evidence (R)

**Component accepted, with scope.** The food sensory population and opponent-turn stage inside the
actual `AIFAgent3D` now has a current causal test in the MuJoCo body.  This is a genuine physical
odour → PAULA sensor → PAULA turn → normal graded-muscle/NMJ control-path result with vision disabled.
It establishes signed sensorimotor steering, **not** a long-horizon food-collection rate, learned
valence, toxin avoidance, or full behavioural arbitration.

**Causal result (R, 2026-07-30).**
`python -m simulations.active_inference.experiments.embodied_food_gradient_causal` recorded the
four standard seeds in
`experiments/results/embodied_food_gradient_causal_20260730T103120Z/`.  It keeps the real MuJoCo
body and physical odour transducer, places one food source at `(-1,+1)` or `(-1,-1)`, moves every
other food/toxin source out of field range, and disables visual-current injection.  Each 80-tick raw
trace records the currents actually injected, food sensor spikes, `TL`/`TR` turn spikes, graded
muscle state, actuator force, and body pose.

| Condition | Food-sensor spikes | `TL − TR` turn imbalance | Interpretation |
|---|---:|---:|---|
| Food at positive lateral `y` | 584 | -21 | The physical gradient selects TR. |
| Food at negative lateral `y` | 582 | +17 | Mirroring the same source flips the selected opponent turn population. |
| Same positive-`y` field, `w_sd=0` | 584 | 0 | Food sensors remain active, but the food-sensor→turn pathway is severed, so neither turn population fires. |

The pattern is identical in seeds 11, 23, 44, and 77; `acceptance.json` reports `passed: true`.
The normal CPG/muscle path remains active in every condition, proving that the ablation does not
freeze the body or erase sensory input.  `w_sd` is now explicitly forwarded to the navigation-core
constructor; it was previously a silently dropped experiment keyword, which would have made this
ablation invalid.

---

## 2026-07-30 — PAULA CPG, graded muscles, and MuJoCo locomotion now have causal evidence (R)

**Component accepted, with scope.** The compact PAULA/MuJoCo motor primitive is now a current,
version-pinned reusable component.  It is a motor-circuit acceptance result, not evidence that the
full `AIFAgent3D` food/toxin policy, spatial system, or arbiter is complete.  The direct motor probe
uses a fixed descending current solely to measure the last neural-to-body stage; it does not set body
position, velocity, yaw, or choose a behavioural action.

**Causal result (R, 2026-07-30).**
`python -m simulations.active_inference.experiments.paula_motor_causal` wrote a non-overwriting
manifest, SHA-256 source fingerprints, and raw per-body-step neural/body traces under
`experiments/results/paula_motor_causal_20260730T102544Z/`.  In each of the four standard seeds
(11, 23, 44, 77), the full circuit moved 2.51 units with a stable +0.4-degree heading change over
1,800 MuJoCo steps.  A small left/right descending current of 0.025 produced the expected opposite
turns (left: 1.90 units, -77.1 degrees; right: 1.92 units, +79.3 degrees).

| Neural/body condition | Outcome in every seed | Causal interpretation |
|---|---|---|
| Full PAULA CPG → relay → graded muscle → NMJ | Forward locomotion | Neural CPG spikes create graded muscle state, and that state alone drives MuJoCo actuator force. |
| Left/right descending current | Opposite signed turns while still moving | Differential neural muscle gating supplies a usable steering primitive. |
| `w_cpg=0` | CPG continues to spike; muscles, actuator force, and displacement are all zero | The CPG-to-muscle neural path is required. |
| `muscle_gain=0` | CPG and graded muscle state persist; actuator force and displacement are zero | The sanctioned muscle-state-to-actuator bridge is required. |

Every acceptance file reports `passed: true`; the raw records include CPG spikes, every graded
muscle state, every actuator command, pose, and descending-current input.  The regression suite
includes three focused tests covering forward propagation, signed turns, and both pathway ablations.

**Nonvisual evidence plumbing.** `aif_agent3d.run_episode()` now permits `vision=False` for a
controlled nonvisual test while preserving the existing live/default `vision=True` behavior.  The
now-accepted short-horizon food-gradient circuit above uses that path.  The live laboratory
observation of successful food seeking and toxin avoidance remains operationally valuable.  The
short-horizon food-gradient and lateral-toxin circuits are now R claims above; long-horizon food
collection and toxin survival, including head-on hazards, still need behavioral acceptance runs.

---

## 2026-07-30 — PAULA T-maze cue/belief/action circuit now has causal evidence (R)

**Component accepted, with scope.** The compact PAULA T-maze circuit now has a causal acceptance
test for its uncertainty-to-cue action pathway and cue-evidence-to-belief pathway.  This is a
discrete sensorimotor neural component, **not** evidence for the full MuJoCo organism, spatial memory,
or an embodied world model.

**Harness correction.** The old `w_epi=0` check was invalid because `max()` chose `ACUE` merely
because it was first in a Python dict when every action population had zero spikes.  That was a hidden
host-language policy.  The world bridge now maps all-zero action activity to physical immobility
(`ACENTER`); it still only maps the winning PAULA action population to the corresponding location
transition.  `Brain.run()` and `run_episode()` now retain per-neural-tick drivers, BL/BR/U/DL/DR/action
spikes, selected neural action, world location, and hidden context as observational evidence.

**Causal result (R, 2026-07-30).**
`python -m simulations.active_inference.experiments.paula_tmaze_causal --episodes 40 --seeds 11 23 44 77`
recorded a non-overwriting manifest, episode records, and raw tick traces under the ignored
`experiments/results/` directory.  The source/environment manifest records active-inference revision
`468497483569c2783c93e677856b4dae5387de95`, neuron-model revision
`4d88b106b7c7ce18bcb5c34b9b89d898a5c3a0e0`, Python 3.12.8, NumPy 2.3.5, plus SHA-256 fingerprints
for the uncommitted `paula_aif.py` and experiment runner sources.

| Circuit condition | Cue visits | Correct rewards | Causal interpretation |
|---|---:|---:|---|
| Full PAULA circuit | 40/40 per seed (160/160) | 40/40 per seed (160/160) | U drives the cue action when belief is unresolved; cue evidence subsequently drives BL/BR, DL/DR, and the context-correct arm action. |
| `w_epi=0` (U → ACUE ablated) | 0/40 per seed | 0/40 per seed | Without that neural output, every action population is silent and the agent remains still; Python does not choose the cue. |
| `w_ev=0` (cue → BL/BR evidence ablated) | 40/40 per seed | 0/40 per seed | The uncertainty pathway still reaches the cue, but no neural belief/value state selects a context-correct arm. |

All four-seed acceptance files report `passed: true`.  The raw traces, not these totals, are the
primary evidence: for example the full seed-11 center state has U/ACUE spikes, while the cue state
shows left or right BL/BR and DL/DR activity before the corresponding arm action.  The component is
therefore a supported PAULA cue-seeking, evidence-integration, and context-appropriate action primitive
for the discrete T-maze only.

**Negative L1 probe, not promoted.** A 480-tick stationary ring probe under the reconciled AIF
configuration remained localized, but `d7=False` was inert in that current build.  It has no valid
causal ablation and is not being promoted as a working Delta-7/ring component.  L1/L2 remains the next
embodied neural dependency after this discrete component.

---

## 2026-07-30 — Embodied configuration reconciliation is complete (software/instrumentation component)

**Problem.** The same nominal agent had three incompatible maintenance-current paths: the old
`run_episode()` loop hard-wired ring tonic current `1.0` every neural tick, while direct
`AIFAgent3D.tick()` and the live UI constructed `tonic_amp=0.0`.  Consequently a trace could not
identify which physical/neural configuration had actually been measured.

**Completed component.** `embodied_config.py` now defines one immutable, JSON-manifestable
`EmbodiedAgentConfig`.  Its explicit default preserves the former closed-loop episode behaviour
(`tonic_amp=1.0`, ungated; `w_tonic=0.12`), and `LEGACY_DIRECT_TICK_CONFIG` preserves the former
zero-tonic direct/live setting as an opt-in control.  `AIFAgent3D.tick()`, `run_episode()`, the live
laboratory, and the long-run recorder now use or emit this same object.  The live snapshot includes
the effective configuration, including tonic, PG threshold/weights, shift thresholds, and every
structural gate.  A misspelled shared setting now raises rather than disappearing through an
allowlist.

**Required paired traces (R, 2026-07-30).**
`python -m simulations.active_inference.experiments.config_reconciliation --ticks 80 --seeds
11 23 44 77` was recorded as separate non-overwriting seed directories under the ignored
`experiments/results/` tree.  For each seed and for both configurations it writes a manifest plus:

* an isolated, stationary ring trace; and
* a closed-loop MuJoCo trace, each sampled on every neural/physics tick.

The manifests pin active-inference revision `468497483569c2783c93e677856b4dae5387de95`, neuron-model
revision `4d88b106b7c7ce18bcb5c34b9b89d898a5c3a0e0`, Python 3.12.8, NumPy 2.3.5, MuJoCo 3.5.0, every
setting, and all four standard seeds.  They establish runner/configuration parity, **not** compass,
path-integration, homing, learning, or arbitration performance.

**Metric correction during this run.** A "one or more ring spikes on every tick" summary is invalid:
normal refractory/sparse dynamics produces individual zero-spike ticks.  The runner now reports an
8-tick windowed liveness field and the longest silent run while retaining raw per-tick spike counts as
primary evidence.  On the recorded 80-tick traces all 16 condition/context/seed traces had windowed
liveness 1.0; their longest silent run was 1--4 ticks.  Therefore these short traces do **not**
replicate or refute the older long-horizon ring-collapse result and must not be used as evidence that
the tonic has a behavioural benefit.

**Regression evidence.** `tests/test_embodied_configuration.py` passes four checks: the default
manifest is complete, direct tick and the closed episode receive the same `1.0` tonic, the old
zero-tonic setting is available only by explicit legacy configuration, and a package-imported config
is accepted by the legacy path-loaded agent.  `live_brain.py --verify 2` still produces identical
decoder/no-decoder state hashes (`deea58a39e724e4f`).

**Next dependency.** The configuration hazard is closed.  Repeat L1/L2 under this recorded default
with a time-varying angular-velocity trajectory over four seeds, actual injected currents, ring state,
true yaw, bounded geometry, and the stated offset treatment before changing a compass circuit.

---

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

## NEGATIVE: more drive does NOT lower the per-column cost -- the current gain is already OPTIMAL
Tested at the compass end (forced constant yaw == a body that turns harder), d_push=8, 2000 ticks:
    drive   ticks/col   bump deg/tick   width   CL cells/tick (of 144)   ring alive
    0.325     499.8        0.0200        15           2.24                91.4%
    0.650      44.2        0.2261        17          15.04                99.8%   <- CURRENT, the optimum
    1.200    2039.9        0.0049        19          41.92                99.8%
    1.950    5463.2       -0.0018        23          64.97                99.8%
    3.000    4255.7        0.0023        21          72.00                50.0%
Non-monotonic with a peak at the value the agent already uses. MECHANISM (mechanism logged beside
outcome): drive recruits more CL cells (15 -> 42 -> 65 -> 72 of 144), so MANY columns push at once and
the bump SMEARS instead of translating (width 17 -> 19 -> 23). A smeared bump has no defined position,
so displacement collapses. Same failure already on record as "those high slopes are a smear, not a bump".
=> k_ang=0.65 / w_cap=1.2 are at the optimum, not arbitrary. Raising muscle ggain (world3d.py:180, the
sanctioned NMJ transducer) to make the body turn harder would push drive PAST the optimum and make
tracking WORSE. The hypothesis is refuted at the compass end, so muscle scaling was not attempted.

## THE GAP IS ~30%, NOT ~40x -- a much more hopeful reframe
At the optimum, with SUSTAINED drive: bump 0.226 deg/tick vs body 0.36 deg/tick = 63% tracking, and
44.2 ticks/column vs the gait's 34-tick MAXIMUM sustained run. So the compass is ~30% too slow, not
orders of magnitude. Everything between 63% tracking and the -0.048 measured in the body is lost to the
OSCILLATION, not to weakness.
=> Aim at the GAIT PERIOD, not gait force: lengthen the sustained-run window from 34 toward ~45 ticks
(a CPG-frequency change). Stroke is currently 40 ticks = 160 ms = 6.25 Hz. Halving the CPG frequency
would roughly double the half-stroke window. Cost to check: slower strokes = slower locomotion, and the
foraging/thrust consequences must be measured, not assumed.
