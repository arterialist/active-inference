# Embodied PAULA agent handoff — 2026-08-02

This historical handoff includes a post-migration update dated 3 August 2026.
Its references to current blockers, supported routes, and next steps describe
that period. Preservation in Git does not resume its task list or establish
that later implementations have the same behavior. No research result was
rerun for this source-preservation commit.

## Purpose and authority

This is a **self-contained operational handoff** for an agent asked to decide
or execute the next research step in the embodied PAULA active-inference
programme. A reader needs no prior conversation, status document, or source
code orientation to understand the goal, constraints, supported results,
current blocker, rejected hypotheses, and recommended next decision. The
linked charter, logs, source files, and JSON artifacts are optional primary
records for verification—not prerequisites for using this handoff.

This document distinguishes measured facts from hypotheses and design advice.
It states what has actually been measured, what was rejected, and why spatial
navigation is the present blocker. If a source record appears to conflict with
this document, preserve the disagreement, inspect the named full-tick artifact,
and prefer the newer trace-backed interpretation over an older assertion.

Do not promote a build, visualization, isolated spike raster, ring-liveness
score, or final wrapped angle to a behavioural result. This programme has been
misled by each of those interpretations in the past.


## Conversation-driven decisions and actions

This programme was not merely handed a generic “make navigation work” request.
The user repeatedly corrected the investigation. Those corrections are part of
the scientific specification and must guide the next agent.

### What the user required

- **PAULA-only cognition, stated precisely.** The concern was not whether code
  names a mechanism “neural”; it was whether Python has secretly become the
  compass, smoothing filter, action selector, or home-vector solver. The
  acceptable boundary is physical measurement → declared PAULA current and
  PAULA graded-muscle state → physical force. Offline decoding and render
  controls are allowed only if they do not feed back.
- **Do not disguise a controller as a neuron.** The proposed 44-tick
  sample/accumulate/snapshot/hold subclass was challenged as biologically
  implausible. It was retained only as an experimental phenomenological
  diagnostic and rejected for accepted-agent use. The inherited
  `ConjunctiveGradedNeuron` route remains PAULA but must be honestly labelled
  experimental, not a faithful cell reconstruction.
- **Use full tick traces, not period summaries.** The user explicitly asked
  whether analysis was based on every neural tick. Subsequent compass work
  records raw pre-tick yaw, physical pose/rate, sensory and prediction
  populations, residuals, fused updates, P-EN release by column, ring raster,
  and relevant PB/P-EG/P-ENb state. Summary metrics are pointers into these
  traces, never primary proof.
- **Treat embodiment as the decisive test.** A mechanism can pass synthetic or
  replayed input and still fail when inserted into the whole brain/body. The
  user specifically suspected non-smooth locomotion. The resulting standard is
  “isolated harness, PAULA-only replay, embodied body” with a causal comparison
  table, rather than declaring an isolated mechanism complete.
- **Scaling means more independently functioning neurons, not a multiplied
  weight.** This led directly to V7: extra locally connected P-ENa-like cells
  with different time constants. Its small embodied improvement and the
  failure of doubled return conductance are the evidence for the current rule.
- **Use biology as a constraint and source of motifs, not a single-organism
  prison.** The user authorized a separate full architecture inspired by
  central-complex, sensorimotor, and other proven biological circuits, provided
  literature claims stay precise. It is valid to combine mechanisms across
  animals, but not to call an engineering convenience biologically faithful.
- **Do not over-optimize an impossible “perfect compass.”** Human and insect
  navigation has latency, drift, predictive action signals, sensory correction,
  and occasional failure. The acceptance bar is bounded, recoverable,
  directionally correct embodied behaviour with useful causal contribution—not
  exact physical yaw every tick or indefinite darkness accuracy.
- **Keep working demos safe.** Large new navigation architectures, the
  cylinder, and all V variants must remain separate from the food/toxin demo
  until evidence supports promotion.

### What was done in response

1. The original default/demo path was left intact. Separate V1--V7 agents,
   a cylinder prototype, CPG-period runner, and full-tick probe runners were
   added instead.
2. The invalid V1 signed-input representation was discovered from the actual
   PAULA transport semantics and replaced in V2 with two non-negative physical
   yaw ports plus local opponent dendrites.
3. The full P-EN-by-column raster was added because aggregate counts could not
   distinguish a local moving wave from a uniform population push.
4. The explicit PB/P-EG/P-ENb structural route, the leaky P-ENb variant, and
   heterogeneous P-ENa scale-up were built and tested. V6 established in-body
   activation without benefit; V7 established a small bounded benefit.
5. The user’s “could more neurons smooth it?” hypothesis was tested directly:
   V7 improved V2 by 4.37° on the named course; a twofold conductance increase
   worsened it. The rulebook was updated accordingly.
6. The user’s “why a ring; why not a cylinder?” hypothesis was pursued in
   parallel. The cylinder can hold/carry ideal quaternary state in isolation
   but is history-fragile under live gait yaw and remains quarantined.
7. The user asked whether shortening the stroke might eliminate the class of
   limitations. A process-local p=10/6/4 CPG sweep showed the opposite:
   faster recurrence increases yaw sign reversals, weakens/reverses the body
   turn, and worsens V2 gain. This closes that narrow repair hypothesis.
8. Corrected V2 source and output ablations were run after the user asked for
   causal distinction. They prove that update→P-EN is necessary, while current
   sensory and motor routes are individually sufficient to cross the same ring
   transition threshold; they do not prove useful sensorimotor fusion.
9. Superseded render/trace artifacts were cleaned only after exact targets were
   inspected; final distinct evidence traces were retained. No behavioural
   source was removed as a cleanup step.

### Inspectability the user requested

A seekable side-by-side replay page exists for the strict compass-boundary
bundles. It is intentionally observational:

```bash
cd active-inference
# Produce a compatible bundle first; choose an unused output name.
uv run python simulations/active_inference/experiments/compass_replay_bundle.py \
  --output experiments/results/<compass_replay_bundle>.json

uv run python simulations/active_inference/live_brain.py \
  --compass-replay experiments/results/<compass_replay_bundle>.json
# open http://localhost:8770/compass-replay
```

The page has synchronized embodied MuJoCo and no-physics PAULA-replay panes,
a ring/P-EN circuit display in each pane, linked 3-D pan/zoom controls,
per-tick seek/play and 44-tick stroke steps, a heading timeline, and causal
event/reversal overlays. The no-physics pane uses a visual kinematic path only;
it is not a second body or a controller. The renderer is
`compass_replay.html`; `experiments/compass_replay_bundle.py` creates its
`compass_replay_bundle.v1` input and `live_brain.py --compass-replay` serves
it. The earlier named bundle referenced by historical research notes is not
currently retained, so generate a fresh compatible bundle before opening this
page; do not point the page at a raw V2/V7 probe JSON.

Important limitation: this page currently targets the earlier
`compass_replay_bundle.v1` schema, not every newer V2/V7 probe JSON. The
newer probes preserve all required raw traces, but need a schema adapter/new
bundle builder before they can be inspected in the same split-pane UI. That is
a valid laboratory task; do not falsely claim V2/V7 visual replay is already
wired into the page.

### Explicit questions still open

The user wants a next agent to help decide, rather than blindly tune:

1. Is the first V8 intervention best framed as a spatially separate
   prediction/sensory P-EN architecture, or should the next action be pure
   transition analysis of the existing ring to identify the transfer barrier?
2. Which biologically grounded motifs should be imported for the leading,
   trailing, inhibitory, and sensory-correction roles without claiming one
   literal fly circuit?
3. What is the smallest meaningful acceptance envelope for heading that permits
   biological latency/drift yet is strong enough to support downstream PI and
   homing?
4. How should the split-pane replay be generalized to the V2/V7 full-tick
   schema so the user can directly inspect the next conflict/reversal tests?


## North star and research boundaries

The goal is a custom embodied organism that senses, learns, preserves internal
state, selects actions, and adapts in a MuJoCo world. It should eventually
forage, avoid toxins, learn valence, navigate, and use belief/world-model
mechanisms in an active-inference sense. Insect systems are sources for circuit
motifs, not a restriction to a literal reconstruction of one species.

The behavioural brain must be a PAULA network. Python may run physics,
transduce physical fields into declared neural currents, transduce graded
muscle state into actuators, render, record, and externally decode state. It
must not integrate heading, select a turn or mode, compute a home vector, or
smooth a neural signal into a behavioural result. Absolute heading,
`SUN_AZIMUTH`, object labels, and host-side navigation controllers are invalid
cognitive shortcuts.

All new spatial work is separate and opt-in. Do not replace the working demo
or the default `AIFAgent3D` with a speculative compass. The shared PAULA
substrate in sibling `neuron-model/` must not be casually rewritten.
`ConjunctiveGradedNeuron` inheritance is an explicitly experimental PAULA
extension with leaky dendrites, not a faithful real P-EN cell model. The
rejected `PhaseLockedGradedNeuron` sample/reset/hold design remains a
diagnostic mechanism only and must not be presented as biology.

## Repository and safety context

Run project commands from `active-inference/`, normally with `uv run python`.
Before neuron-mechanism work, read the complete active PAULA neuron
implementation and relevant transport/tick path. PAULA dendritic distance
attenuates as well as delays, and spikes reset membrane state; a configured
weight is not evidence that a downstream population is driven.

The `active-inference` worktree is intentionally dirty and has extensive
untracked research material. Do **not** modify the unrelated unresolved
conflict `simulations/c_elegans/neuron_mapping.py` (`UU`). Preserve user
changes. The simulation tree and experiment artifacts are largely untracked,
so `git diff` does not establish what exists. Inspect targeted files and retain
full-tick JSON traces; storage has previously been tight, but those traces are
primary evidence.

## System map and accepted foundations

```text
MuJoCo World3D
  physical odour/contact/speed/yaw/retina
        | sanctioned transducers
        v
PAULA sensory, value, heading, arbiter, and motor populations
        |
CPG -> relays -> graded PAULA muscle membranes -> MuJoCo actuators
```

`aif_agent3d.py` composes the exploratory embodied brain: motor plant,
central-complex candidates, mushroom-body/lateral-horn routes,
interoception/arbiter components, and World3D. It is not yet an end-to-end
active-inference organism. The live-brain decoder is observational and has
been hash-tested as non-feedback, but decoded labels are not neural state.

The following are genuine but bounded component results:

| Component | Supported result | Boundary |
| --- | --- | --- |
| Motor primitive | Four-seed causal PAULA CPG→relay→graded-muscle→actuator movement; removing CPG drive or muscle gain removes motion. | Motor/steering, not navigation. |
| Food/toxin behaviour | Physical lateral-food collection, toxin steering, head-on escape, and live food seeking/toxin avoidance; output ablations change the body outcome. | Local/horizon-limited scenarios, not lifetime foraging. |
| MB food avoidance | Toxin teaching with food odour creates an MB→AVOID→LH avoidance route; teaching/output controls restore collection. | One local counterconditioned source, not general value learning. |
| Interoceptive arbiter | Hunger-driven FORAGE→EXPLORE state change has a PAULA SEARCH-gate consequence and ablation. | Not a general multimotive policy. |
| Discrete inference | NumPy T-maze and a separate PAULA T-maze have cue/context/reward causal demonstrations. | Discrete demonstration, not embodied navigation. |
| Analog PI / home expression | XACC/YACC accepts physical speed with a known-good supplied heading code; Stone memory can express through OPP/HOME/motor populations. | No accepted recurrent compass or homing. |

The demo's food/toxin behaviour is real core sensorimotor success. It does not
imply working path integration, HOME, visual grounding, or global arbitration.


### Minimal terminology and data-flow glossary

- **PAULA:** the shared neural substrate in sibling `neuron-model/`. It is
  primarily a spiking, leaky membrane/synapse system; a spike resets membrane.
  Synaptic dendritic distance contributes both delay and attenuation. Some
  permitted subclassed cells release a graded value each tick rather than
  spiking; they are still PAULA units and must be identified in traces.
- **World3D / MuJoCo body:** the physical planar environment. It supplies
  odour, contact, speed, yaw-rate, and retinal measurements. It receives force
  only from the declared graded-muscle membrane states. A convenience scripted
  physics action method exists for probes but is never evidence of free agent
  behaviour.
- **CPG, relay, muscle:** the PAULA motor chain. The CPG generates gait phase;
  relays drive graded left/right muscles; muscles exert force on the body.
  “TR” and “TL” are existing PAULA turn-stimulus neurons used by diagnostics,
  not Python calls that turn the body directly.
- **E-PG-like ring / bump:** 36 recurrent heading columns. A localized activity
  bump is the candidate internal heading state. A live ring can be broad,
  static, or wrong; the measured phase, width, continuity, and relation to
  body yaw matter.
- **P-EN routes:** local cells that combine a heading-column signal with
  signed self-motion and return at a spatial offset. A clockwise/counter-
  clockwise asymmetry should move the bump; that P-EN→ring transfer is the
  current failure point.
- **PB, P-EG, P-ENb, Delta-7:** candidate central-complex bridge,
  maintenance/stabilization, trailing-update, and structured-inhibition
  populations. Their biological role is an inspiration and a testable
  topology—not automatically a faithful reconstruction.
- **Path integration (PI), HOME, OPP:** downstream candidate populations that
  use heading and speed to store displacement, express a home-oriented signal,
  and influence turn/arbitration. Some expression routes are measured, but they
  must not receive an unaccepted compass.
- **Exact raw-yaw replay:** record the raw physical yaw sample supplied before
  each live neural tick; run an otherwise identical PAULA circuit with no body,
  using only those samples. Equality localizes a failure inside the neural
  transduction/circuit rather than the physics-to-neuron handoff.
- **Full-tick trace:** one row per neural/physics tick, preserving physical
  drivers, external currents, population/ring states, motor state, configuration,
  and seed. Offline phase decoding is a measurement only; it never changes the
  brain.

### Evidence and promotion standard

A component is “working” only when a driving variable, relevant neural state,
physical output, comparison/control, and failure criterion are explicit. Every
substantive circuit must be tested in both an isolated harness and the real
embodied brain. A raw replay is a third diagnostic context when body input is
involved. Positive claims should normally use at least seeds 11, 23, 44, and
77 and enough horizon to expose drift, reversal, saturation, contact, or
boundary effects.

For heading work, the minimum trace has physical yaw/speed/pose; the exact
currents injected into PAULA; sensory, prediction, residual, and update
populations; P-EN release by column; ring phase/amplitude/width; and
relay/muscle state. A terminal angle, total spike count, correlation with a
monotonic trend, or ring “liveness” alone is explicitly insufficient. Every
parameter must be shown to change the built circuit; inert knobs are wiring or
measurement problems, not robustness.

A positive navigation claim is deliberately less strict than perfect tick-level
tracking but stricter than an attractive trajectory: it needs a persistent
single bump, correct average turn direction over seeds, bounded/recoverable
short-horizon error, no sustained counter-rotation under a sustained physical
turn, and a later demonstrated contribution to physical homing relative to an
ablation. Exact unity gain and infinite dark dead reckoning are not required.


## Navigation dependency chain

```text
real self-motion + motor prediction
  -> persistent, updateable heading state
  -> velocity × heading path integration
  -> home-vector readout
  -> PAULA mode/action arbitration
  -> actual return-to-origin behaviour
  -> later learned external-cue re-anchoring
```

Later stages cannot repair a broken heading update. Do not connect a candidate
compass to PI/HOME/navigation merely because it is live, has a decodable angle,
or works on synthetic input. The cylinder and V1--V7 routes are quarantined
from downstream control until their own acceptance gates pass.

Recurrence is necessary: without a persistent neural state, yaw can be sensed
but heading cannot be retained at rest or used for path integration. A ring is
not the only possible code, but the cylinder experiment shows that more angular
resolution does not repair unstable sensorimotor state updates.

## Earlier learning that remains relevant

- A default-preserving graded PAULA path closed a real substrate gap: a slow
  analog release can drive later populations where sparse spike trains cannot.
  That is a component advance, not a compass solution.
- Ring liveness is not tracking. One older body trace had a 14--18-column
  smear and only 35.3° bump movement while the body nearly completed a circle.
- A full-tick ordinary CPG/reset experiment retained correct directional P-EN
  bias yet created irregular and sometimes reverse ring motion. This localized
  failure downstream of the clock, at P-EN-to-ring transduction.
- The 44-tick sample/reset/hold proposal was rejected as phase-sensitive,
  phenomenological, and biologically unjustified.

Audit-era opt-in gyro-to-ring configurations achieved moderate short physical
left/right gains (about 0.60/0.63) under their own prescribed course. They are
useful references only. Long courses and the newer sensorimotor programme are
under-gained or unstable; do not conflate the configurations or the claims.


## Biological rationale and claim boundary

The next architecture is informed by, but not presented as a reconstruction of,
insect central-complex work:

- Heading activity can combine visual landmarks with self-rotation and persist
  when both are absent: [Seelig & Jayaraman, 2015](https://doi.org/10.1038/nature14446).
- Direction-specific PB/EB pathways have anatomy and physiology suitable for
  angular integration, supporting spatially offset rather than generic update
  routes: [Green et al., 2017](https://doi.org/10.1038/nature22343) and
  [Turner-Evans et al., 2017](https://doi.org/10.7554/eLife.23496).
- A second recurrent E-PG→P-EG→P-EN loop and rich parallel central-complex
  motifs support testing distinct leading and trailing/stabilizing pathways:
  [Turner-Evans et al., 2020](https://doi.org/10.1016/j.neuron.2020.08.006)
  and [Hulse et al., 2021](https://doi.org/10.7554/eLife.66039).
- Efference-copy-like signals exist in fly visuomotor circuits, but do not
  imply that planned action alone yields correct heading:
  [Kim, Fitzgerald & Maimon, 2015](https://doi.org/10.1038/nn.4083).
- Learned sensory re-anchoring is more defensible than an absolute-world cue;
  visual-to-heading associations can be experience dependent:
  [Fisher et al., 2019](https://doi.org/10.1038/s41586-019-1772-4).

These papers support parallel, recurrent, cue-corrected sensorimotor circuits.
They do **not** prove the exact PAULA topology, its cell counts, a 44-tick
clock, the cylinder code, or a literal “P-EN1 accelerator/P-EN2 brake” story.
Those remain hypotheses that must pass the stated causal tests.



## Current V1--V7 sensorimotor compass programme

Relevant code:

- `simulations/active_inference/aif_agent3d.py` — PAULA populations and input
  ports;
- `central_complex.py` — ring, P-EN, PB/P-EG/P-ENb, Delta-7, and scaled banks;
- `sensorimotor_agent3d_v2.py` through `sensorimotor_agent3d_v7.py` — separate
  opt-in agents;
- `experiments/sensorimotor_compass_probe.py` — 1,400-tick full-trace probe;
- `SENSORIMOTOR_COMPASS_DESIGN.md` — detailed ledger.

The standard probe settles through tick 199, starts normal CPG at tick 200,
drives PAULA TR on ticks 500--899, and runs to tick 1,400. The body path is
PAULA TR → CPG/relays → graded muscles → MuJoCo. Raw yaw is sampled before the
next neural tick and delivered only as two non-negative opponent currents.
Signed evidence is created by local PAULA excitatory/inhibitory dendrites.

### V1: retained negative control

V1 put signed raw-yaw values directly into a delay-line residual. PAULA's
external input path transports only positive `info`; a negative value vanished
rather than becoming inhibition. Its apparent directionality is invalid. V1
is retained as a trace-backed interface error, not tuned or promoted.

### V2: corrected sensorimotor architecture

V2 uses four PAULA time constants per direction for raw-yaw and final-relay
motor prediction. Opponent prediction-error and fused-update populations are
all PAULA. It has no decoded heading, body pose, host smoother, or Python turn
controller. The fused update gates local P-EN cells, which return at spatial
offset to the E-PG-like ring.

On the latency-matched V2 configuration (`lambda=44`, `d_push=3`, seed 11),
the body turns `-80.80°` and the ring turns `-20.26°`. Full P-EN rasters show
localized, correct-sign drive rather than a global all-column push. The ring
advances once and largely holds. Exact raw-yaw replay reproduces neural state
and raster, so this is not a body/host handoff problem.

Corrected V2 causal cuts:

| Cut, same body course | Body turn | Ring turn | Meaning |
| --- | ---: | ---: | --- |
| intact | `-80.80°` | `-20.26°` | Both sources and P-EN projection are active. |
| motor prediction = 0 | `-80.80°` | `-20.26°` | Sensory route alone launches this limited shift. |
| sensory input = 0 | `-80.80°` | `-19.71°` | Final-relay prediction alone also launches it on this stereotyped course. |
| update→P-EN gain = 0 | `-80.80°` | `0.00°` | The ring does not drift; this projection is necessary. |

This is not a robust-fusion success. Each source independently crosses the
same nonlinear state-transition threshold. The similar endpoint is evidence
that the P-EN-to-ring transfer is the bottleneck, not that the two sources
have been usefully combined.

V2 artifacts:

- `simulations/active_inference/experiments/results/sensorimotor_compass_v2_lam44_dpush3_seed11.json`
- `simulations/active_inference/experiments/results/sensorimotor_compass_v2_lam44_dpush3_seed11_ablation_prediction.json`
- `simulations/active_inference/experiments/results/sensorimotor_compass_v2_lam44_dpush3_seed11_ablation_sensory.json`
- `simulations/active_inference/experiments/results/sensorimotor_compass_v2_lam44_dpush3_seed11_ablation_pen_projection.json`

### V3/V4: phase-gated leading routes rejected

V3 used ordinary PAULA CPG coincidence gates, avoiding sample/hold, but its
sparse phase pulses made the bump jitter (`-3.57°`). V4 added leaky phase
overlap and counter-rotated (`+13.62°` during the same clockwise body course).
Exposing motor phase alone does not smooth the present ring update.

### V5/V6: PB/P-EG/P-ENb structural test

V5 built a visible PB→P-EG bridge. Low gain was a useful silent-path control;
high enough gain activated P-EG but introduced reversal blocks and degraded
tracking. V6 creates a stricter spatial route: local P-EG feeds distinct
P-ENb-like cells, with the same PAULA self-motion source and a trailing
opposite-offset return.

The first V6 threshold form was silent. The inherited leaky conjunctive form
is active on every turn tick (426.29 total release, peak 0.2194) but changes
neither body nor ring response: `-80.80°` body, `-20.26°` ring. This is a
successful in-body activation result and a negative behavioural result. Do not
make a trailing/braking route stronger while the leading route under-rotates.

### V7: modest population-scale positive result

V7 retains V2 and adds four P-ENa-like PAULA copies per column/direction (288
new cells). Each has local E-PG and fused-update dendrites, with velocity
traces 1/2/4/8 ticks. There is no phase counter, sample/hold, decoded heading,
or host smoothing.

On the same course, V7 turns the ring `-24.63°` instead of V2 `-20.26°`: a
real `4.37°` correct-sign improvement. Doubling only return conductance
degrades response to `-15.42°` without killing the ring. Rule: scale through
independently timed, locally connected neurons and measure their bounded
contribution; do not equate “more neurons” with scalar synaptic gain. V7 is
not an accepted compass.

Artifacts:

- `simulations/active_inference/experiments/results/sensorimotor_compass_v7_pena_scale_seed11.json`
- `simulations/active_inference/experiments/results/sensorimotor_compass_v7_pena_scale_w070_seed11.json`

## CPG acceleration was tested and rejected

`experiments/embodied_cpg_period_compass.py` changes `CPG_PERIOD`
process-locally and restores it; defaults/demo are untouched. The measured 44
ticks are the ordinary default p=10's double-wave cycle, not a mandatory
compass integration block and not simply `4p`. Same-phase intervals are 13/31
ticks at p=10, yielding a repeatable 44-tick double-wave. p=6 produces 28,
and p=4 produces 20.

| CPG period | Body turn | Raw-yaw sign changes | V2 outcome |
| --- | ---: | ---: | --- |
| p=10 | `-80.91°` | 18 | ring `-17.37°`, gain 0.215 |
| p=6 | `-46.08°` | 28 | ring `-7.51°`, gain 0.163 |
| p=4 | `+8.25°` | 40 | meaningful turn collapses; V2 skipped |

Exact raw-yaw replay is identical for viable p=10/p=6. Faster recurrence
increases body jitter and breaks CPG/muscle/body/sensory co-timing; it is not a
compass repair. Artifact:
`experiments/results/embodied_cpg_period_compass_seed11_final.json`.

## Cylinder: preserve the negative representation result

The four-level quaternary cylinder is a separate PAULA prototype, not a
biological claim and not connected to PI/HOME/motor. It passes isolated
counter/carry tests on well-separated synthetic pulses. In embodiment it has
exact raw-yaw replay but is history-fragile under alternating gait yaw:

- TR body `-80.91°` should yield `-58` bins; retained setting gives `-40`;
- TL body `+84.95°` should yield `+60` bins; retained setting gives `+46`;
- an extra ordinary graded sensory relay does not repair the TL shortfall.

Small pre-turn yaw-history differences change event/carry state before the
turn. The raw input becomes effectively identical after the turn starts, but
the persistent counter state has already diverged. More quantization does not
solve unstable embodied updates. Keep the cylinder parallel and quarantined.

Artifacts:

- `experiments/results/embodied_cylinder_compass_tr_seed11.json`
- `experiments/results/embodied_cylinder_compass_tl_seed11.json`
- `experiments/results/paula_cylinder_isolated_seed11.json`

## Current causal picture

```text
turn/CPG drive -> relay activity -> muscle force -> lurching body yaw
  -> raw-yaw PAULA receptors -> local P-EN update -> recurrent E-PG bump
  -> (only after acceptance) PI/home/arbiter/motor
```

Upstream influences are genuine body dynamics, muscle filtering, alternating
CPG phase, relay timing, and sensor transport. Exact replay excludes a
discrepancy after raw yaw reaches PAULA. Motor prediction still differs from
real yaw: relays encode the attempted motion, physical yaw reports the result.

At the current ring interface, localized direction-correct P-EN activity is
not turned proportionally into phase velocity. The attractor can hold a bump
but its transition dynamics are thresholded, delayed, or wrongly phased under
real gait waveforms. Sensory and motor routes currently act as redundant
triggers instead of a demonstrably useful estimator. V7 helps slightly through
heterogeneous timing; brute conductance makes the effect worse.

Downstream PI/home expression cannot repair this. If the bump is frozen or
under-updated, heading-dependent PI stays fixed and any home output reads a
frozen/ramping state. The legacy visual anchor is additionally invalid because
it uses `SUN_AZIMUTH`; replacement must be learned sensory association.

## Recommended next decision

Do **not** implement PI/HOME, CPG acceleration, a cylinder adapter, or another
global gain sweep next. First perform full-tick **transition analysis** of V2
and V7 under both directions, reversals, and deliberate motor-prediction versus
real-yaw mismatch. Offline phase decoding is allowed for analysis only.

At each tick/column, align:

1. E-PG bump phase, amplitude, width, and single-bump status;
2. per-column P-EN release, spatial offset, and arrival delay;
3. local inhibition and recurrence;
4. raw yaw, motor prediction, sensory residual, prediction error, and fused
   update;
5. next-tick and short-window change in ring phase.

Determine whether a stall/reversal arises from weak local drive, wrong-flank
arrival, self-inhibition/bump widening, an inter-column attractor barrier, or
temporal cancellation. Produce an empirical state-transition map, not a host
controller.

If that analysis supports the proposed mechanism, build a **separate V8**
rather than retuning V2 in place:

```text
final relays -> fast local prediction P-EN route ---------\
                                                        -> E-PG heading state
raw opponent yaw -> fast/medium/slow local sensory route-/
E-PG -> PB -> P-EG -> delayed trailing/stabilizing P-ENb route
prediction minus sensory -> PAULA disagreement cells -> gate prediction authority
```

Predicted and realised self-motion should remain distinct until each is
spatially gated by the local heading bump. Disagreement should reduce reliance
on prediction, not directly calculate heading. This is a biology-inspired
engineering hypothesis, not a literal single-cell claim; it is motivated by
direction-specific angular-update pathways and rich central-complex recurrence
(Green et al., 2017; Turner-Evans et al., 2017/2020; Hulse et al., 2021).

The first V8 gate is not exact unit gain. Require: persistent single bump,
correct average response to both directions across seeds, no sustained
counter-rotation during real turns, improvement over V2/V7/source ablations,
and a conflict result in which actual self-motion corrects wrong motor
prediction. Then, and only then, add learned landmarks. Give heading any
PI/HOME/navigation authority only after that route is accepted.

## Experimental rules and commands

- Begin each substantive circuit hypothesis with an isolated harness **and**
  the same circuit in the real body. Isolation is not promotion.
- Log full ticks: yaw/speed/pose, injected currents, source populations,
  P-EN-by-column output, ring state, relay/muscle state, seed, configuration,
  and source/environment manifest.
- Include known-good and known-broken controls. For compass work, update→P-EN
  and source-path ablations are baseline controls.
- Test left/right, reversal, and intended-versus-realised-yaw conflict. Positive
  claims normally need seeds 11, 23, 44, and 77 at minimum.
- Phase decoding is external analysis/visualisation only; it never feeds back.
- Keep candidates default-off and do not alter food/toxin/motor demo behaviour
  to improve a compass probe.

From `active-inference/`:

```bash
uv run python experiments/sensorimotor_compass_probe.py \
  --version v7 --output simulations/active_inference/experiments/results/<name>.json

uv run python experiments/embodied_cpg_period_compass.py \
  --output experiments/results/<name>.json

uv run pytest tests/test_embodied_configuration.py -q \
  -k 'sensorimotor or penb or pena or pb_eb_bridge'
```

Recent focused regression: `6 passed, 25 deselected`. Syntax checks pass for
`central_complex.py`, `aif_agent3d.py`, V6/V7, the probe, and the CPG runner.

## Bottom line

There is real embodied neural progress: motor control, local food/toxin
behaviour, learned local avoidance, an interoceptive action consequence, and
several PAULA transmission/heading components all have causal support.
Navigation is not yet working as a whole because no accepted recurrent compass
converts the real body's oscillatory self-motion into a stable, useful heading
state. The immediate task is not to hide this downstream or demand a perfect
animal; it is to make the heading-update transfer directional, bounded,
recoverable, and causally robust under the body the organism actually has.

## Post-migration delivery: the working V1/V2/V3 agent line (2026-08-03)

The navigation conclusion above remains unchanged, but the rest of the agent
has now been consolidated into a usable versioned library.  These versions are
packages, not standalone scripts, and they all build one PAULA network:

```text
agents/reactive_v1/       V1 = accepted olfactory reflexes + inherited CPG/muscle body
agents/memory_v2/         V2 = V1 + accepted Mushroom Body learned valence
agents/interoceptive_v3/ V3 = V2 + delayed metabolism + PAULA SLEEP arbitration
```

The former empty rhythmic V2 was merged into the already-proven V1 motor
foundation.  That is deliberate: a version boundary is justified by a named
environmental failure, not by disabling a working rhythm.  The new V3
challenge is concrete: a meal loads a gut-like store, digestion releases
usable energy gradually, and realized activity consumes energy.  The minimum
body delta is therefore `gut_load`, `energy_store`, digestion, and activity
cost.  The minimum neural delta is a fourth PAULA WTA population, SLEEP, which
inhibits search/motor drive while leaving the physical digestive dynamics
running.  The body never calls a mode selector; it exposes sensory afferents.

### V3 evidence

`simulations/active_inference/experiments/embodied_metabolic_rest_causal.py`
uses full per-neural-tick traces and four cases: unchanged V2, intact V3,
SLEEP-output ablation, and metabolic-afferent ablation.  The complete run
passed independently at seeds 11 and 23.  The values are identical across
those seeds because the harness deliberately places all sources at the same
controlled distance; this is a causal fixture, not a claim of broad random
world robustness:

| Case | Final energy | Post-meal SLEEP spikes | Post-meal actuator absolute sum |
| --- | ---: | ---: | ---: |
| unchanged V2 | 0.41839 | 0 | 1266.81 |
| intact V3 | **0.63931** | **768** | **2.37** |
| SLEEP output ablated | 0.41839 | 744 | 1266.81 |
| metabolic afferents ablated | 0.28684 | 0 | 1532.49 |

The harness rejects the build unless intact V3 has sustained SLEEP, preserves
more energy than V2, and loses that advantage under either causal ablation.
The full JSON bundles are retained under
`simulations/active_inference/experiments/results/`.

The prior aggregate-only acceptance was not sufficient for the live failure:
it could miss a body turning away from a reachable meal or a strict arbiter
with no EXPLORE input.  `experiments/embodied_v3_trajectory_causal.py` now
retains every neural tick and checks five physical cases: odour-free
exploration (and its energy-to-EXPLORE ablation), food approach, a
food/toxin conflict (and toxin-threshold ablation), and delayed meal/rest.
The retained seed-11 run passes with 73 EXPLORE-winner body steps in the
odour-free case, 0.592 minimum food distance in the reachable-food case,
0 toxin contacts and 1.085 minimum toxin distance in the conflict case, and
94 SLEEP-winner ticks after the meal.  These are fixture-level causal results,
not claims of broad random-world coverage.

The retained post-refactor two-seed fixture is
`embodied_metabolic_rest_causal_20260803T101248Z` (an identical rerun replaced
the earlier complete fixture after the final live-viewer integration).

### What was actually refactored

The high-leverage source boundaries are now real, not only labels:

- `components/body/world.py` owns the MuJoCo body and physiological state;
- `components/body/metabolism.py` defines the body-to-brain afferent contract;
- `components/arbitration/paula.py` owns the accepted arbiter and optional
  fourth SLEEP population;
- `components/arbitration/metabolic_parts.py` contributes V3 afferents and
  synapses;
- `core/brain_composer.py` validates explicit component selections and invokes
  the one whole-brain builder;
- `aif_agent3d.py` remains the compatibility builder for the still-large
  central-complex/visual/motor/MB core, with no duplicate controller path;
- root `world3d.py` and `aif_arbiter.py` are compatibility shims to the moved
  implementations, so old entry points continue to execute the same source.

The sibling PAULA repository also has a stable extension boundary:
`neuron/extensions/graded.py` and `neuron/extensions/conjunctive.py` are the
default-preserving homes for the reusable subclasses, while the phase-locked
sample/reset/hold experiment is isolated under
`neuron/extensions/experimental/`.

The PAULA-only boundary is preserved: Python performs physical sensing,
declared external-current injection, MuJoCo stepping, and observation.  The
SLEEP/forage/explore competition, inhibition, and motor veto are synaptic
operations inside the same PAULA topology.  The compass/cylinder/path-
integration variants remain explicitly experimental or quarantined and are
not silently enabled by V1, V2, or V3.
