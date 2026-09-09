# Research resumption record: PAULA embodied agents, active inference, and the live brain

**Audit date:** 2026-08-03.  **Scope:** the PAULA/MuJoCo embodied-agent and active-inference work in this directory, including the July Claude recovery material and the live-brain interface.  This is a reproducible hand-off, not a claim that every saved experiment succeeded.  It reconciles the maintained source, the recovered experiment conclusions, the contemporaneous research notes, and focused reruns made during this audit.

The two recovered Claude conversations were reviewed for this stream.  The temporary 3.1-GB
recovery corpus was audited and removed on 2026-08-03; durable embodied-agent material is now in
this directory, the sibling `neuron-model/`, and the retained `experiments/` source probes.  The
other large stream is the CIFAR/foveation mini-brain work in `neuron-model` and is intentionally
documented there rather than duplicated here.  Conversation transcripts are not a dependency of
this record.

The companion [`../../EMBODIED_AGENT_RESEARCH_CHARTER.md`](../../EMBODIED_AGENT_RESEARCH_CHARTER.md)
records the user's goal hierarchy, deliverables, and standing research
constraints.  This guide records the current evidence and blockers against
those goals.

## How to read the evidence labels

`R` means that the result was rerun against the current recovered workspace on the audit date.  `E` means that the result was previously measured with a named harness and retained trace/log evidence, but was not rerun in this audit; it is a strong lead, not a current regression test.  `D` means that code constructs or a design is present, but no accepted embodied measurement supports the behavioral claim.  `X` means a negative result, retraction, or a known invalid interpretation.  A successful build, syntax check, visualization, or isolated network is never treated as behavioral evidence.

The detailed chronological sources remain `ARCHITECTURE.md`, `RESEARCH_LOG.md`, `LAB_RULES.md`, and `ROADMAP.md`.  They are valuable lab notebooks but are not mutually consistent: the newest entry in `RESEARCH_LOG.md` corrects two broad conclusions stated as structural in the older architecture and roadmap.  For decisions after this date, this document is the entry point and those files are supporting primary records.

## Reproducible starting point

Run from the `active-inference/` repository root.  The package requires Python 3.11--3.13, MuJoCo, and the sibling `neuron-model/` checkout.  Several older modules still contain an absolute path to this particular workspace; this checkout therefore runs as-is, but a clone at another path will not yet be portable without a separate, reviewed path refactor.

```bash
cd /Users/arterialist/Projects/agi-research/active-inference

# Exact discrete generative-model demonstration.
uv run python -m simulations.active_inference.tmaze_agent

# PAULA implementation of the small T-maze sensorimotor loop.
uv run python simulations/active_inference/paula_aif.py

# Prove that the live display decoder is read-only.  Two steps are a focused smoke test.
uv run python simulations/active_inference/live_brain.py --verify 2

# Construct the full embodied brain and take two closed-loop agent steps.
uv run python -c 'from simulations.active_inference import aif_agent3d as m; a=m.AIFAgent3D(seed=11); a.birth(); h,modes=m.run_episode(a, steps=2, log_every=10**9); print(len(a.nb), a.t, modes)'

# Physical-speed, isolated analog path-integration acceptance (four seeds).
uv run python -m simulations.active_inference.experiments.embodied_xacc_pi_causal

# Local diagonal toxin-survival acceptance (four seeds).
uv run python -m simulations.active_inference.experiments.embodied_toxin_escape_causal

# Symmetric head-on toxin-escape acceptance (four seeds).
uv run python -m simulations.active_inference.experiments.embodied_headon_toxin_causal

# Lateral physical food-collection acceptance (four seeds).
uv run python -m simulations.active_inference.experiments.embodied_food_collection_causal

# Opt-in graded Stone memory-to-home-output acceptance (four seeds; not homing acceptance).
uv run python -m simulations.active_inference.experiments.embodied_stone_home_causal

# Local learned food-avoidance acceptance after physical toxin teaching (four seeds).
uv run python -m simulations.active_inference.experiments.embodied_mb_food_avoidance_causal

# Hunger-driven FORAGE-to-EXPLORE search gating through the PAULA arbiter
# (four physical initial orientations; a fixed food delivery isolates the
# interoceptive transition rather than testing food seeking).
uv run python -m simulations.active_inference.experiments.embodied_arbiter_explore_causal

# Opt-in physical raw-gyro-to-recurrent-ring calibration (four seeds).
# This is not a path-integration or homing acceptance.
uv run python -m simulations.active_inference.experiments.embodied_compass_gyro_causal

# Opt-in continuous gyro-to-ring route: prescribed turns, ordinary CPG gait,
# and output-specific ablations across four seeds. Still not PI/homing.
uv run python -m simulations.active_inference.experiments.embodied_compass_graded_causal
```

On 2026-07-30 the first command selected cue then the correct reward arm in both hidden contexts and reported full active inference at 100% reward over 40 episodes.  A historical fair-control run reproduced 100% reward for the full planner, 50% for a fixed-arm reward-only control, and 0% reward for curiosity-only planning; its temporary launcher is no longer in the cleaned workspace.  The PAULA T-maze reached the cue and reward in both contexts and reported 100% reward over 20 alternating-context episodes.  The live decoder check produced identical hashes (`acc4ae28d5a14896`) with and without decoding after two steps.  The earlier embodied smoke constructed 1,746 neurons and ran 32 neural ticks; the current default is 1,766 after the accepted TRISE, default MB-to-LH, and FORAGE-to-SEARCH route were enabled.  Neither count is a behavioral-success claim.

## System map

The work comprises three distinct claims that must not be conflated.  The discrete NumPy T-maze is an explicit generative-model and planning demonstration.  `paula_aif.py` is a compact PAULA spiking controller for the same task.  `aif_agent3d.py` is an exploratory embodied integration of vision, heading, path integration, associative learning, interoception, action selection, a motor plant, and a MuJoCo world.  Its components have uneven evidence and it is not yet an end-to-end active-inference organism.

```text
World3D (MuJoCo) ──retina / odour / contact / speed / yaw──> PAULA populations
                                                               │
  visual_cortex ──> visual ring ─┐                            │
  central_complex ─> heading ring ├─> PG/CD/PI candidates ────┤
  antennal lobe + mushroom body ──> learned AVOID ─> LH ──────┤
  hunger + uncertainty + beliefs ──> mode WTA ──> SEARCH gate ┤
                                                               v
 engine/CPG ─> relays ─> graded muscle membrane S ─> MuJoCo actuator force
```

`world3d.py` owns the planar MuJoCo body, food/toxin fields, eye rendering, and the sanctioned transducers.  It maps field concentrations and contact events to external neural currents; it maps graded muscle membranes to actuator force.  `AIFAgent3D.parts()` composes `central_complex.py`, `visual_cortex.py`, `cx_navigator.py`, `aif_arbiter.py`, the mushroom-body parts, and optional path-integration alternatives.  `run_episode()` advances the network at 16 neural ticks per agent step and advances physics on every neural tick.  It uses `World3D.take_event()` to deliver a latched food/toxin event exactly once, preventing a short physical contact from being lost between agent-level samples.

The motor route is intentionally non-kinematic in the embodied runner: coprime pacemakers and engine cells drive relays, relays gate graded left/right muscle membranes, and `World3D.act_muscles()` applies their values to MuJoCo actuators.  The separate `World3D.act(turn, speed)` method is a scripted-probe facility and must not be mistaken for free agent behavior.

## Current status by claim

| Area | Status | Exact evidence and interpretation |
|---|---:|---|
| NumPy T-maze, explicit generative model | R | `tmaze_agent.py` uses likelihoods `A`, transitions `B`, preferences `C`, prior `D`, exact Bayesian posterior updates, and expected-free-energy tree search.  Current run: full policy 100% reward over 40 alternating contexts. |
| Fair reward-only T-maze comparison | R | Recovered `scripts/aif_tmaze.py` uses `run_myopic_reward_baseline`, which commits to one arm without observing the cue.  Current run: 50% reward, as expected for equiprobable contexts. |
| Maintained T-maze “myopic” printout | X | `tmaze_agent.py` prints 0% for depth-one/no-epistemic planning.  It is not the fair reward-only comparison: all immediate options tie and the deterministic tie picks `CENTER`, so it repeatedly does nothing.  Do not cite 0% as a reward-greedy baseline.  Deep reward-only planning still gets 100% because the cue has instrumental value through the observation tree. |
| PAULA T-maze cue/belief/action circuit | R | Current four-seed causal experiment (`experiments/paula_tmaze_causal.py`, 40 balanced episodes/seed) records raw neural ticks. Full circuit: 40/40 cue visits and 40/40 rewards per seed. `w_epi=0` (U→ACUE) gives 0/40 cue visits and rewards because all-zero action activity now maps to stillness rather than a Python-selected cue. `w_ev=0` gives 40/40 cue visits but 0/40 rewards: cue seeking remains neural, but BL/BR evidence cannot select the context-correct arm. This is a supported discrete PAULA primitive, not embodied navigation or a world model. |
| Ring persistence and Delta-7 support | E | Eight-seed body measurements recorded that an externally injected tonic held the ring alive in 8/8 seeds; Delta-7 removal gave zero ring spikes over 400 ticks.  This needs a current, version-pinned rerun before publication. |
| Vestibular compass | R (short physical gyro-to-ring; long turns remain X) | Current four-seed causal experiment (`experiments/embodied_compass_graded_causal.py`) retains only raw MuJoCo yaw as compass feedback: paired PAULA delay-line afferents cancel the measured 44-tick gait stroke and their continuous graded release reaches the local P-EN gates. It produces physical left/right body/ring deltas of +73.5°/+44.2° and −58.6°/−37.1° (gains 0.60/0.63). The new `compass_transfer_diagnostic.py` additionally shows that the first continuous P-EN prototype's failure is not a host-only artifact: raw-gyro replay of irregular nonvisual MuJoCo locomotion matches the live ring within 3.38° mean circular phase but over-rotates too. A literature-motivated, default-off inherited `ConjunctiveGradedNeuron` now gates P-EN release on local ring × PAULA opponent velocity, and a rate-coded ring prevents the prior silence/threshold collapse. Both are liveness improvements, not tracking acceptance: its sustained right/left courses are −48.9°/−122.0° and +34.7°/+56.9° ring/body at the best non-wrapping tested gain; larger common gains over-rotate. This accepts the earlier opt-in short/gait route only, not long-horizon heading fidelity, path integration, HOME, or homing. |
| Visual motion | E | Increasing the Reichardt delay from 4 to 16 previously raised per-tick correlation between angular velocity and HS response from `0.347 ± 0.072` to `0.587 ± 0.031` across four seeds.  With the default `w_hs_shift=0`, this currently does not establish an effect on compass or behavior. |
| Analog XACC/YACC path integration | R (isolated heading route) | Current four-seed physical causal experiment (`experiments/embodied_xacc_pi_causal.py`) drives the existing `RING→PG→XACC/YACC` path from MuJoCo heading and `qvel` speed through an explicitly scoped seven-cell heading-code transducer, while recurrent ring dynamics are disabled. The curved/reversal body path gives 0.87--1.58° final vector error, 0.998 outbound magnitude/distance correlation, 0.999--1.001 retention across 240 zero-input ticks, and `k_pi=0` removes the vector without changing 17,245--17,251 PG spikes. `k_pi=64` is the measured linear gain; 128/256 compress. This accepts the analog memory, not the recurrent compass, CPU4 ladder, or closed-loop homing. |
| Stone PI home-output route | R (opt-in expression path) | Current four-seed physical causal experiment (`experiments/embodied_stone_home_causal.py`) runs an empty, vision-free MuJoCo world in which measured body speed drives the graded Stone memory. Its intact route reaches OPP/HOME/home-turn populations (1,943/2,322/152 spikes per seed) through fixed PAULA wiring; zeroing only Stone→OPP and Stone→CPU1 preserves the differentiated memory (final span 0.408) and normal 10.68 actuator peak but leaves OPP/HOME at zero and only 2 heading-only home-turn spikes. This accepts the memory-expression/motor route, not compass fidelity, vector accuracy, or homing. |
| MB counterconditioned food avoidance | R (local learned behavior) | Current four-seed physical causal experiment (`experiments/embodied_mb_food_avoidance_causal.py`) teaches one PAULA brain with physical toxin contact while neutral food odour is present, then makes a food-only MuJoCo probe in two mirrored placements with vision off. The taught route collects no food (minimum distance 0.758--0.830); `w_lh_avoid=0` preserves learned MBON/AVOID activity (MBON 230--231, AVOID 51--52) but restores collection in 4/4 seeds, while `w_trig=0` removes teaching and also restores collection. This accepts MB→AVOID→LH→turn counterconditioning of one local food source, not general valence learning, multi-odour behavior, or long-horizon foraging. |
| PAULA/MuJoCo motor primitive | R | Current four-seed causal experiment (`experiments/paula_motor_causal.py`, 1,800 body steps/seed) records every CPG spike, graded muscle state, actuator command, and body pose. Full CPG→relay→graded-muscle→NMJ movement was 2.51 units with +0.4° heading drift; a 0.025 left/right descending current yielded 1.90/-77.1° and 1.92/+79.3° respectively. `w_cpg=0` and `muscle_gain=0` both produced exactly zero actuator drive and displacement while retaining CPG spikes. This is an accepted motor/steering primitive, not an accepted food/toxin-navigation claim. |
| Integrated nonvisual food-gradient steering | R | Current four-seed causal experiment (`experiments/embodied_food_gradient_causal.py`, 80 neural/physics ticks/condition) disables vision and uses the normal `AIFAgent3D` MuJoCo body. The `(-1,+1)` and `(-1,-1)` physical food gradients give 584/582 food-sensor spikes and opposite `TL−TR` turn imbalances (-21/+17); `w_sd=0` preserves 584 sensor spikes but gives zero turn spikes. This is accepted short-horizon sensorimotor closure, not a long-horizon food-collection claim. |
| Integrated nonvisual lateral-toxin steering | R | Current four-seed causal experiment (`experiments/embodied_toxin_gradient_causal.py`, 80 neural/physics ticks/condition) disables vision and retains the real short-range toxin field plus MuJoCo motor path. Toxins at `(-2,+1)`/`(-2,-1)` give 502/504 toxin-sensor spikes and opposite `TL−TR` turn imbalances (+11/-9); `w_tox=0` retains 505 sensor spikes but no turn output. This supports signed lateral avoidance; symmetric head-on escape is separately accepted below. |
| Diagonal toxin survival | R (short horizon) | Current four-seed physical causal experiment (`experiments/embodied_toxin_escape_causal.py`) disables vision and places only one diagonal toxin at `(-1.2,+0.5)`. The intact agent has 6,500 toxin-sensor spikes yet 0 contacts (closest distance 1.194); `w_tox=0` preserves 6,639 sensor spikes and normal actuator drive but makes 1 contact (closest 0.166). This supports local diagonal survival, not symmetric head-on hazards, learned avoidance, or lifetime survival. |
| Lateral food collection | R (short horizon) | Current four-seed physical causal experiment (`experiments/embodied_food_collection_causal.py`) disables vision and places the only food source at `(-1.5,+1.0)`, deliberately off the `w_sd=0` baseline. The intact agent collects it at neural tick 707 in every seed (5,267--5,335 sensor spikes); the `w_sd=0` control retains 5,875 sensor spikes and normal actuator drive but collects none. This supports one-source lateral collection, not multi-source foraging, learned value, or arbiter-driven behavior. |
| Arbiter and interoception | R (FORAGE→EXPLORE search gate) | Current four-orientation physical causal experiment (`experiments/embodied_arbiter_explore_causal.py`) begins in an odour-free MuJoCo world and schedules one food item at the body at tick 160; ordinary physical contact then drains the neural hunger ladder. Before the meal, hunger is 4.2 and FORAGE/EXPLORE fires 288/0 while PAULA SEARCH fires 2 times; afterwards hunger is 0, EXPLORE/FORAGE fires 960/0, and SEARCH fires 112 times. Zeroing only the FORAGE→SEARCH gain preserves the food contact, hunger transition, WTA transition, and actuator path but restores 34 pre-meal SEARCH spikes and 100.6 versus 27.0 left/right paddle-force asymmetry. Hazard STEER receives no FORAGE input, so toxin escape remains separately available. This accepts one interoceptive FORAGE-to-EXPLORE motor consequence, not HOME behavior, food seeking, a general multi-drive policy, or broad active-inference claims. |
| Head-on toxin response | R (local hazard) | Current four-seed physical causal experiment (`experiments/embodied_headon_toxin_causal.py`) places one toxin at `(-0.75,0)` with vision and food absent. The default `TXL/TXR→TPOOL→TRISE→STEER` route has 6,841 toxin-sensor spikes, 628 TRISE spikes, 349 STEER spikes, zero contacts, and minimum distance 0.561. `w_trise=0` preserves 6,831 toxin-sensor spikes and 620 TRISE spikes but reduces STEER to 279 and makes one contact at 0.516. `trise` is now true in the shared default configuration and live brain. This resolves the local symmetric blind spot, not long-horizon or learned avoidance. |
| Visual anchoring | X | `aif_agent3d.py` maps visual azimuth to ring column using `world3d.SUN_AZIMUTH`, a ground-truth world constant.  This violates the PAULA-only interpretation and cannot support learned re-anchoring claims. |
| Full embodied organism | D | The 1,766-neuron V1/V2 default and 1,788-neuron V3 build are runnable, but no current acceptance test demonstrates robust cue grounding, path integration, value learning, mode selection, and goal-directed behavior together. |

## Important corrections to historical interpretations

The most consequential correction is about PAULA timing.  Dendritic distance is not a pure delay: an arriving signal is attenuated by `0.95 ** distance`.  Thus, for example, a weight 2.4 synapse at distance 90 arrives at approximately 0.0237, not 2.4.  The old CPU4 ladder “advance” path and the 35--110-tick RISE/TRISE delay arms were consequently much weaker than their diagrams implied.  A failure attributed to an intrinsic absence of integration may instead be a circuit whose delayed input was attenuated out of relevance.

The second correction is that a PAULA spike resets membrane `S`, but the neuromodulatory `M_vector` is a slower state that survives spikes and adjusts excitability and plasticity.  Prior sweeps established bistability for a particular synaptic self-excitation design; they did not establish that the substrate has no graded persistent state.  Any use of the older phrase “structural no graded integrator” must be replaced by the narrower, supported statement: the tested synaptic accumulator configurations either forgot or latched, while neuromodulatory feedback and correctly amplitude-compensated delayed circuits remain untested.

These corrections explain why the chronology contains apparent contradictions.  The current architecture and roadmap preserve useful measurements, but their broad F2 interpretation predates the late research-log analysis.  The correct scientific posture is to retain the raw finding, narrow its attribution, and rerun the decisive tests with measured arriving amplitudes.

## Live brain

`live_brain.py` serves the visualisation at `http://127.0.0.1:8770` by default and a WebSocket server at port 8771.  It starts a background `Sim` worker, builds an `AIFAgent3D`, drives it through `aif_agent3d.run_episode()`, and streams a binary spike frame on every neural tick plus a heavier JSON snapshot on agent-step boundaries.  `brain_live.html` is the interface; `topo_live.py`, `export_brain.py`, `brain_topology.json`, and `build_brain_page.py` provide the topology payload and the standalone 3-D page.

The decoder is deliberately observational.  `live_brain.py --verify N` runs identical seeded simulations with and without `Decoder.read()` and hashes every neuron membrane/spike state and postsynaptic information weight.  The audit's two-step run produced identical state hash, food count, toxin count, and home distance in both branches.  This proves decoder isolation for that focused trajectory; it does not validate the decoder's semantic labels or the organism's behavior.

Three live-brain cautions matter before a public demo or a scientific screenshot.  First, the former
tonic-default split was reconciled on 2026-07-30: `EmbodiedAgentConfig` is shared by direct ticks,
the closed-loop episode, the live laboratory, and the long-run recorder.  Its explicit default
preserves the old closed-loop value (`tonic_amp=1.0`, ungated); the old zero-tonic direct/live state is
available only as a named legacy control.  Every live snapshot exposes the effective configuration.
The paired 80-tick isolated-plus-body traces establish configuration parity only, not tonic benefit or
compass performance; see the newest `RESEARCH_LOG.md` entry.  Second, the wire format is dynamic:
the server advertises the neuron count and clients calculate `ceil(n/8)` for the spike bitmask
(1766 cells for V1/V2, 1788 for V3).  Third, the server binds only to loopback and an existing local
process may already occupy port 8770; do not terminate a running demo simply to run a test.

## Data, recovered artifacts, and what counts as publishable evidence

The maintained directory contains source-adjacent trajectories and media such as `nav_traj.json`, `forage_traj.json`, `multi_food_traj.json`, `dual_forage_traj.json`, `aif3d_episode.json`, `aif3d_final.json`, videos, and self-contained HTML views.  They are valuable demonstrations and replay inputs, but their filenames alone do not bind them to a source revision, parameter file, seed, or acceptance criterion.  Treat them as historical artifacts until a manifest is added.

The former recovery corpus contained 2,624 files and occupied 3.1 GB when audited.  It was a
temporary experiment notebook with many generated traces and stale paths, so it was removed after
the durable conclusions, source modules, and accepted fixtures were migrated.  The retained
`experiments/` directory contains the compass/cylinder and sensorimotor probes that remain useful;
Python bytecode caches are validation by-products and remain excluded from the publishable corpus.

The recovery has two immediately usable items.  `scripts/aif_tmaze.py` is the fair discrete baseline used above, and `scripts/aif_final.py` is a retained negative diagnostic: its prediction-error and arbitration claims did not reproduce under the recovered maintained implementation.  The latter should be used to prevent overclaiming, not promoted as working self-model behavior.  Probe scripts such as `compass_body.py`, `pi_validate.py`, `livestat.py`, `homing_full.py`, and `mb_minefield.py` are leads for rebuilding a clean experiment suite, but must first have paths, output naming, and current source imports reviewed.  Initial recovery validation generated Python bytecode caches; those non-primary by-products have been excluded from the curated corpus.

The entire `simulations/active_inference/` tree is currently untracked in the `active-inference` repository.  Separately, the repository has a pre-existing unresolved conflict in `simulations/c_elegans/neuron_mapping.py`.  Neither condition was modified by this audit.  This record concerns research quality and resumability, not a release or staging decision; keep the unrelated conflict separate when repository hygiene is addressed later.

## Experiment rules that are non-negotiable

Every resumed neural-circuit experiment must begin by reading the complete PAULA neuron implementation and the network timing path; `LAB_RULES.md` explains why.  In particular, `c` is the firing refractory, `t_ref` is the causal/acausal plasticity window, spiking resets `S`, and dendritic distance attenuates as well as delays.  The preflight arithmetic is the arriving per-tick input, including attenuation, divided by membrane leak relative to threshold; a configured weight is not evidence that a population is driven.

The primary output must be a per-tick trace containing true driving variables, neural output, timing, source revision, seed, parameter JSON, and a geometry guard.  Aggregate accuracy, total spike count, or a final wrapped angle can create a plausible but false mechanism.  For navigation and path integration, ensure the body has not remained at the arena boundary, report the count of valid time points, and keep speed and turning jointly varying when their relation is tested.  A circuit must be verified both in isolation and in the actual body.  At least four seeds are required for a positive claim; the established set is 11, 23, 44, and 77, with 5, 13, 91, and 7 added for eight-seed runs.

## Glossary

| Term | Meaning in this codebase |
|---|---|
| **PAULA** | The shared spiking-neuron substrate implemented in sibling `neuron-model`.  Its precise timing, membrane reset, synaptic distance, plasticity, and neuromodulation semantics govern every circuit here. |
| **Active inference (discrete demonstration)** | In `tmaze_agent.py`, Bayesian state inference plus expected-free-energy planning under an explicit `A`, `B`, `C`, `D` generative model. |
| **Expected free energy (EFE)** | In the discrete T-maze, the negative sum of expected preference satisfaction (pragmatic value) and expected information gain (epistemic value), recursively evaluated over future observations. |
| **T-maze context** | The unobserved fact that reward is on the left or right arm.  The cue reveals it without supplying reward. |
| **Fair reward-only baseline** | A policy that commits to an arm before seeing the cue; with two equally likely contexts, its expected reward is 50%.  It differs from a depth-one planner that can choose to stay still on a tie. |
| **World3D** | The planar MuJoCo environment in `world3d.py`, including the body, odour fields, food/toxin contact, retina, arena boundary, and transducers. |
| **Transducer** | The allowed interface between physical variables and neural current, or between a graded muscle membrane and a MuJoCo actuator.  It is not a behavioral controller. |
| **Ring attractor / RING** | Thirty-six heading columns that sustain a local activity bump representing a relative heading. |
| **Delta-7 / D7** | Structured inhibitory support for the ring.  Its presence is required for the historically measured persistent bump. |
| **P-EN shift cells (`CL`, `CR`)** | Clockwise/counter-clockwise shift populations intended to update the heading bump from angular velocity.  Their observed response is nonlinear and resonant. |
| **Tonic ring input** | An external maintenance input to each ring cell.  A synapse exists only if it is set every tick; its inconsistent default treatment is a current configuration hazard. |
| **EMD** | Reichardt-style elementary motion detector in `visual_cortex.py`; `d_emd` controls its temporal delay and velocity tuning. |
| **HS** | Wide-field horizontal-system-like motion population.  Its raw shift drive into the compass is disabled because it worsened prior tracking measurements. |
| **Visual anchoring** | Mapping a visual scene to the heading ring.  The present `SUN_AZIMUTH` mapping uses world ground truth and is not learned anchoring. |
| **PG gate** | A path-integration gate combining ring state and forward speed before CPU4-like direction cells.  Its weight/leak arithmetic is a critical preflight check. |
| **CD / CDM** | Directional cosine populations.  `CD` can be signed; `CDM` is rectified and measures path length rather than net displacement. |
| **CPU4 ladder** | The legacy spiking path-integration memory candidate.  It has not been accepted as a graded displacement accumulator. |
| **XACC/YACC** | Analog, non-spiking accumulators in `central_complex.py` whose membrane `S` is read as a vector.  They are accepted for the physical-speed, externally supplied heading-code route, but not as an accepted recurrent-compass or homing system. |
| **`pi_accum`** | Optional population accumulator attached to CD, with recruitment and global inhibition.  It is not part of the default build and its tested self-excitation regimes forgot or latched. |
| **`pi_stone`** | Optional graded circular memory path based on uniform speed excitation, heading-dependent inhibition, and leak.  Its opt-in OPP/CPU1/home-turn expression route has a four-seed physical causal test; it remains off by default and is not a homing claim. |
| **Mushroom body (MB) / LH** | Antennal lobe, projection neuron, sparse Kenyon cell, APL inhibitory, MBON, and AVOID route used for odour-valence learning.  The default LH-like turn populations receive learned AVOID and weak innate food-sensory input; the current physical result causally ablates only the learned AVOID output. |
| **US / sting latch (`STG_T`, `STG_F`)** | A toxin or food contact event.  `pending_event` plus `take_event()` prevents a brief physical contact from being cleared before neural consumption. |
| **RISE / TRISE** | Temporal trend detectors.  Long dendritic-delay versions must be re-evaluated with attenuation included; TRISE is the proposed non-cancelling head-on-toxin route. |
| **Hunger ladder** | Interoceptive population that raises FORAGE and drains after food contact.  Its causal FORAGE→EXPLORE transition has a narrow physical test; it is not yet a general metabolic model. |
| **Arbiter** | Spiking winner-take-all selection over FORAGE, HOME, and EXPLORE modes in `aif_arbiter.py`.  FORAGE now inhibits only the PAULA background SEARCH population; it does not veto the independent toxin/learned-danger STEER gate. |
| **Prediction error (`PE`)** | A population comparing a visual heading cue with the ring’s heading estimate.  Present diagnostics do not establish a working embodied self-model. |
| **Graded muscle** | A PAULA cell deliberately kept non-spiking; its membrane `S` controls an actuator.  A spike raster cannot represent its activity. |
| **Birth seed** | One-time external current used to break an initial symmetry, for example to create a heading bump or start a CPG. |
| **Live decoder** | The read-only presentation layer in `live_brain.py`.  It must not feed neural state back into the simulation. |

## Prioritized, falsifiable resumption plan

1. **Freeze an auditable baseline before tuning.**  Record the current commit hashes of `active-inference` and `neuron-model`, exact environment versions, output directory, parameter JSON, and seeds.  Add a small, versioned runner that emits a manifest beside each trace.  Acceptance is that the discrete T-maze, PAULA T-maze, live-decoder isolation, and two-step embodied build reproduce the audit outputs without relying on a Claude path.

2. **Make the two T-maze controls canonical.**  Promote or reproduce the fair 50% fixed-arm baseline in the maintained experiment suite and retain the depth-one/no-epistemic result only as a planner-tie behavior.  Add a PAULA control that actually distinguishes the epistemic route from an alternative route.  Acceptance is a table where every policy’s action sequence, cue exposure, and reward expectation are explicit.

3. **Reconcile embodied configuration before interpreting it.**  Define one explicit configuration object shared by `AIFAgent3D.tick()`, `run_episode()`, the live UI, and every benchmark.  Do not change defaults as part of a paper claim until the old and new configurations are measured side by side.  Acceptance is that all runners emit the same effective tonic, PG, threshold, and structural-gate settings.

4. **Repeat the L1/L2 measurements under the reconciled configuration.**  Run a time-varying angular-velocity trace over at least four seeds, logging ring state, true yaw, HS output, actual injected currents, and per-tick error after a clearly reported constant-offset treatment.  Acceptance is a bounded, non-wall-contact trajectory with liveness, tracking slope, error distribution, and seed spread; a median alone is insufficient.

5. **Close the sensory and trajectory gap before claiming spatial navigation.**  `embodied_xacc_pi_causal.py` accepts XACC/YACC under a calibrated physical-heading code, while `embodied_stone_home_causal.py` accepts the opt-in Stone memory→OPP/HOME/CPU1/home-turn expression chain.  Neither establishes a recurrent/vestibular compass, an accurate physical home vector, or return-to-origin behavior.  Replace the test heading code with an accepted recurrent compass output, drive a physical outbound/return course, and score vector error plus actual home-distance reduction with raw ticks and multiseed controls.  Do not promote the CPU4 ladder on the strength of either result.

6. **TRISE now resolves the local head-on blind spot; generalize it before promotion.**  `embodied_headon_toxin_causal.py` gives a four-seed physical contact-avoidance result against the `w_trise=0` output ablation, and `trise=True` is now the shared/default live configuration.  Next, sweep source distance, approach angle, and longer episodes while reporting false-positive turns in toxin-free worlds.  Keep the long-delay attenuation budget explicit; one narrow contact geometry is not lifetime survival.

7. **Restore action-selection dynamic range.**  Measure hunger, FORAGE, HOME, and EXPLORE on every neural tick in the body, first correcting saturation without path-integration claims.  The Stone experiment proves that an opt-in home input can express HOME; it does not establish switching against FORAGE/EXPLORE under ordinary conditions.  Then test mode switching under a preregistered stimulus schedule.  Acceptance is a non-saturated range and causal mode changes under mode-pathway ablations.

8. **Keep vision and learned anchoring separate until the above succeeds.**  Remove the `SUN_AZIMUTH` grounding only in a dedicated experiment with a scene-rotated control and a learned cue-to-ring map.  Acceptance is re-anchoring from sensory regularities that survives the rotation control; an azimuth lookup does not qualify.

9. **Promote only demonstrated components.**  Rerun the historical simple navigator, dual chemotaxis, motor, and mushroom-body learning references with the new manifest format.  A component becomes a supported reusable primitive only after its behavioral acceptance test and control pass on the current code.

## File guide

| Path | Role |
|---|---|
| `README.md` | Short map of stable reference agents and the two T-maze implementations. |
| `LAB_RULES.md` | Measurement protocol and PAULA-specific preflight rules. |
| `RESEARCH_LOG.md` | Chronological analysis, retractions, and the late attenuation/neuromodulation corrections. |
| `ARCHITECTURE.md` | Population-level architecture and historical confidence grading; read with the correction above. |
| `ROADMAP.md` | Earlier ordered work plan; useful but partly superseded by the resumption plan here. |
| `world3d.py` | MuJoCo world and transducers. |
| `aif_agent3d.py` | Composed embodied brain and closed-loop runner. |
| `central_complex.py`, `cx_navigator.py`, `pi_accum.py`, `pi_stone.py` | Heading and path-integration candidates. |
| `mushroom_body.py`, `aif_arbiter.py`, `visual_cortex.py` | Learning, mode selection, and vision subsystems. |
| `tmaze_agent.py`, `paula_aif.py` | Discrete and PAULA T-maze demonstrations. |
| `live_brain.py`, `brain_live.html`, `topo_live.py`, `export_brain.py` | Live instrumentation and topology export. |
| Former `experiments/claude_recovery_2026_07/` corpus | Removed during the 2026-08-03 cleanup; durable conclusions are in this record and the migration manifest. |
| `recovered_versions/snn_debug_2026_07/` | Non-overwriting historical versions retained for comparison. |

The next investigator should start with the four short reruns, read `LAB_RULES.md` and the newest `RESEARCH_LOG.md` entry in full, then perform the configuration-reconciliation step before chasing a new circuit hypothesis.  This order is deliberate: the most expensive errors in this stream came from interpreting a wired-but-undriven connection, a stale default, a wall-bound path, or an aggregate metric as a property of the neural substrate.
