# July 2026 Claude research provenance and resumption record

This is an archival account of the July sessions and their recovery. References
to the present workspace, current priorities, and verification describe that
audit, not the repository's state in October 2026. Recorded results were not
rerun for this source-preservation commit. Use the maintained research records
for later changes and retain the evidence distinctions below.

## Purpose and scope

This is the repository-local continuity record for the two Claude sessions titled
*Debug SNN classifier performance on CIFAR-10*. It records what was requested,
what was changed or generated, the evidentiary status of the principal research
claims, and the work that remained open when the sessions ended. It is intended
to let a researcher resume the work from repository material rather than from
an agent's conversational context.

This document was reconstructed from the session event logs, their task state,
file-history metadata, the recovered artifact corpus, and the present working
trees. It deliberately does **not** copy the conversations, chain-of-thought,
tool-result payloads, session task output streams, or Claude memory files into a
repository. Those are provenance sources, not research deliverables.

For the normalized user-directed objective hierarchy, deliverables, and
research standards, see the [embodied-agent research charter](EMBODIED_AGENT_RESEARCH_CHARTER.md).

The companion [recovery index](docs/CLAUDE_SESSION_RECOVERY_INDEX.md) gives
source-to-destination mappings and the retained file-history identifiers. For
the SNN/CIFAR implementation and results, start with the companion guide in the
sibling repository:
[`neuron-model/snn_classification_realtime/foveation/RESEARCH_RESUMPTION.md`](../neuron-model/snn_classification_realtime/foveation/RESEARCH_RESUMPTION.md).
Its linked `FOVEATION_RESEARCH.md` is the chronological laboratory notebook.
For the current embodied-agent state, start with
[`simulations/active_inference/RESEARCH_RESUMPTION.md`](simulations/active_inference/RESEARCH_RESUMPTION.md).
Its linked `ROADMAP.md`, `ARCHITECTURE.md`, and `RESEARCH_LOG.md` are the
supporting design plan and chronological laboratory records.

## Evidence conventions

The labels below prevent a transcript claim from being mistaken for a
reproduced result. **Verified now** means the recovered code or artifact was
checked in the present workspace. **Recorded result** means the result is in a
retained experiment log, result record, or a maintained research log, but was
not rerun during this provenance audit. **Transcript claim** means that the
statement was made during the session and should be independently reproduced
before it drives a design decision. **Retracted** means later evidence in the
same research record superseded it.

All times in this document are UTC. File-history versions are Claude's local
backup identifiers, not Git commits and not a substitute for a reproducible
environment lock.

## Session registry

| Session | Event-log interval | Log records | File-history events | Principal scope |
| --- | --- | ---: | ---: | --- |
| `be7dec7c-f6b9-4e48-b433-78e0ae71291c` | 2026-07-03T16:33:14.498Z to 2026-07-11T11:07:56.133Z | 7,649 | 168 snapshots | CIFAR-10 classifier audit, foveation/ALERM harness, mini-brain experiments, temporal readouts, attribution controls, and attractor analysis. |
| `4630c9cc-ae60-4a85-a5da-88544a269fbb` | 2026-07-08T20:24:53.066Z to 2026-07-30T05:44:51.709Z | 32,045 | 285 snapshots and 196 deltas | Continuation of the CIFAR work, standard-pipeline and circuit campaigns, then the PAULA embodied-agent, central-complex, live-brain, and Phase-0 validation work. |

The sessions overlap because the second is a continuation record that imported
the earlier session's compact summary and file-history state. The actual
provenance is therefore chronological, not a claim that the two sessions were
independent experiments.

## Chronological research record

### 1. Initial SNN/CIFAR diagnosis — 3–4 July

The work began with a report that the maintained
`neuron-model/snn_classification_realtime/` classifier performed substantially
worse on CIFAR-10 than on MNIST. The requested work was an implementation audit,
four concrete fixes, and calibration/oversaturation diagnostics. The session
then corrected a CIFAR dataset-name mismatch and established the initial
question that drove the rest of the work: whether the poor outcome was a
pipeline error, a representation problem, or an object-versus-background
attention problem.

The session modified the standard SNN input/dataset/building path and retained
the corresponding history versions. The relevant implementation group is
`snn_classification_realtime/{build_activity_dataset.py,network_builder.py,network_builder_direct.py,realtime_classification_fusion.py}`
together with `core/`, `activity_dataset_builder/`, `activity_preparer/`, and
`realtime_classifier/run.py`. These are direct repository files, not recovered
scratch artifacts; their exact history IDs are in the companion index.

### 2. Foveation and ALERM experimental harness — 4–7 July

The work pivoted to an active fovea as a testable explanation for CIFAR's
background/object confound. The agreed design used a fovea-sized perception
network so that inputs were not silently unused, activity-driven foveal
rollouts, and an ablation ladder. The user explicitly allowed a teacher signal
only as scalar `m0`/`m1` neuromodulation, not class-label leakage. The user also
required visual evidence: foveal position, individual and population activity,
and rendered dynamics.

The resulting foveation module contains the retina, fovea, saliency,
perception, signal, viewer, retinal-convolution, and experiment scripts. The
maintained [foveation research log](../neuron-model/snn_classification_realtime/foveation/FOVEATION_RESEARCH.md)
records the early outcomes. The notable recorded results were: local adaptation
did not make settling object-selective; the retinotopic glow map followed edges
in the graded regime; and contrast/edge energy was not an object detector.
Therefore any wording that treats glow as semantic object localisation is
**retracted**. The visuals remain useful as V1-like edge/activity diagnostics,
not evidence of object recognition.

Two durable methodological constraints were established here. First,
`neuron/neuron.py` is shared with C. elegans work: changes needed explicit
approval, default-off feature gating, and preservation of the legacy default
path. Second, an apparent mechanism required a per-tick measurement and a
control, not only an aggregate endpoint. The project later formalized the
latter rule in the active-inference lab rules.

### 3. Mini-brain, temporal-code, and CIFAR attribution experiments — 6–11 July

The mini-brain work compared random, retinotopic, and convolutional encoders;
linear/MLP readouts; foveal policies; native and external plasticity; and
temporal readouts. The locked high-level protocol used a 500-tick dwell and a
500-tick warm-up, with colour and grayscale CIFAR considered. The premise was
that a PAULA representation may be a dynamical regime rather than a tick mean;
therefore feature extraction had to avoid assuming that a static average is the
whole representation.

This phase made an important correction. The earlier broad conclusion that
PAULA could not produce beyond-linear structure was **retracted** after the
user clarified that the maintained default pipeline is a convolutional PAULA
network with an SNN classifier and had reached approximately 95% on MNIST. That
is a transcript/maintained-log datum, not a benchmark independently rerun in
this audit. The narrower, supported conclusion is that the failing reduced
mini-brain regime was not evidence for a universal PAULA limitation.

The first temporal experiment reported small, consistently positive pool-level
temporal-minus-mean differences. It was correctly marked provisional because
the test uncertainty was about one standard deviation. A larger user-run
powered experiment later yielded temporal-minus-mean changes of -0.019
(convolutional), -0.013 (random), and -0.001 (retinotopic). The earlier
"directional confirmation" is therefore **retracted**: for this
CIFAR-from-a-glimpse protocol, temporal-linear, GRU, and LSTM readouts did not
recover label information beyond the mean. This does not test every possible
temporally defined task.

The attribution ladder then separated the fixed retinal front end from the
spiking substrate. On the recorded 600-train/300-test colour-CIFAR control,
linear accuracy was 0.383 on retina features and roughly 0.217–0.230 after the
three PAULA substrate variants, while a dimension-matched random projection
retained roughly 0.383–0.393. The recorded conclusion is specific: a **frozen**
mini-brain substrate and its tested mean/dynamics descriptors were a lossy
bottleneck for this task, not the source of the front-end's class signal.
Neither a trained PAULA substrate nor a task whose target is intrinsically
temporal was established by this control.

The direct retinal-convolution-plus-MLP ceiling was recorded as 0.483 for a
6,000/2,000 colour-CIFAR split in the maintained log; a later task reminder
also mentions a 0.566 run without enough configuration detail in the retained
research log to treat it as the canonical number. The 0.483 result is the
auditable baseline. In both cases the substrate is bypassed, so neither number
is a PAULA-only result. The user cancelled the scaled end-to-end push because it
would not answer the attribution question.

The retino-RGB attractor-class experiment tested the attractive visual claim
that image classes might occupy distinct limit cycles. Its recorded result is
negative: 43/50 examined examples were classified as complex/weak-cycle, none
as clean limit cycles; the descriptor had a negative silhouette (-0.097); and
the full descriptor scored 0.241 versus 0.253 for its centroid-only component.
Thus the claim that those frozen retinotopic RGB attractors are class-clustered
limit cycles is **retracted**. The other planned arms (retino grayscale,
random RGB, random grayscale) were not run because the best-odds arm had
already produced a clear negative.

### 4. Standard-pipeline/CIFAR circuit campaign — 11–19 July

After the mini-brain attribution result, the user requested experiments using
the actual standard input path rather than a fovea replication. The session
recorded 600-, 3,000-, and selected 6,000-tick activity investigations,
per-class centroid checks, attractor/basin tests, and separate designs for
fixed versus learned features. The user then reframed the long-term target as a
PAULA circuit with an external activity decoder that should break 70% on RGB
CIFAR-10, while requiring frozen weights to remain only an isolated control.

The surviving source tree for this phase is
`neuron-model/snn_classification_realtime/{paula_vision/,run_basin_generalization.py,run_completion_probe.py,run_supervised_kappa.py}`,
`neuron-model/cifar_ceiling/`, and
`neuron-model/foveation_results/basin_generalization/`. It contains designs,
harnesses, and partial result material, but the two sessions do **not** provide
a verified, reproducible demonstration of the requested 70% RGB-CIFAR
PAULA-only outcome. That target remains open.

### 5. Circuit library and agent-design exploration — 18–20 July

The work expanded into PAULA circuit primitives, neuromodulation, dendritic
computation, and a request for a large visual circuit atlas rather than a
six-neuron toy. The durable implementation belongs in
`neuron-model/paula_agent/` and includes the circuit kit, candidate circuit
sets, architecture variants, agent-complex wiring, and runner. It was the
bridge from isolated circuit demonstrations to the embodied work below.

At this point the research goal was explicitly narrowed in a useful way:
ordinary Python must provide simulation, sensor injection, and motor decoding,
but it must not secretly select behaviour or hard-code an agent policy. Claims
of agent ability consequently require a wiring/ablation/per-tick trace, not a
visually plausible trajectory alone.

### 6. Embodied PAULA agent, memory, and navigation — 20–26 July

The agent work moved into
`active-inference/simulations/active_inference/`, first with foraging and
mushroom-body experiments and then with path integration, central-complex
navigation, an arbiter, a 3D MuJoCo world, visual-cortex work, recording, and
rendering. The session produced a reusable code group rather than a single
scratch demo: `aif_agent3d.py`, `central_complex.py`, `cx_navigator.py`,
`pi_accum.py`, `pi_agent.py`, `pi_stone.py`, `world3d.py`, `visual_cortex.py`,
the mushroom-body modules, and the associated record/render scripts.

The agent's early behavioural claims were repeatedly re-measured as the user
requested longer horizons, multiple worlds, and numerical traces. This caused
several welcome corrections: an isolated capability did not necessarily survive
in the embodied wiring; a mode or activity sampled once per physics step could
hide per-tick events; and aesthetically convincing foraging was not evidence of
planning or robust homing. The current agent documentation preserves those
retractions and should be treated as the authoritative status, rather than an
older visual artifact.

### 7. Live brain, audit, and central-complex correction cycle — 26–30 July

The final phase integrated the current agent with the live-brain server and
browser views. `live_brain.py`, `topo_live.py`, `build_brain_page.py`, and the
related HTML/JavaScript files provide a read-only human-interpretable status
decoder, topology rendering, parameter controls, and rebuild-aware optional
subsystem display. The recovered decoder isolation check found identical neural
state hashes with and without the decoder; the decoder is therefore external to
the neural state update (**verified during recovery**).

This phase also exposed the most consequential research-process errors. Reading
the whole neuron update path overturned assumptions about dendritic
distance-dependent attenuation and slow analog state. Per-tick checks then
retracted multiple aggregate-based conclusions: a wall-constrained homing number,
a claimed OPP-distance relation, a claim that a ladder path was dead, and several
compass/integrator interpretations. The maintained
[`RESEARCH_LOG.md`](simulations/active_inference/RESEARCH_LOG.md) records the
specifics and [`LAB_RULES.md`](simulations/active_inference/LAB_RULES.md) makes
per-tick evidence a mandatory rule.

The last session state does **not** show a finished autonomous active-inference
organism. It shows a substantially instrumented 1,746-neuron agent with some
validated subsystems and several structural navigation blockers. Treat the
current [roadmap](simulations/active_inference/ROADMAP.md) as the resume point.

## Durable findings and boundaries

The following conclusions are supported at the stated scope. A fixed
retinal-convolution front end has meaningful CIFAR signal, but the tested frozen
small substrates discarded class-aligned information. The same tests provide no
license to infer that all PAULA architectures, trained substrates, or
dynamics-defined tasks are incapable of useful computation. Rich sustained
dynamics and re-entry/savings were recorded, while class-uniform attractors and
a clean periodic limit cycle were not demonstrated for the retino-RGB control.

In the embodied work, the session verified that several mechanisms exist and
can be active in the body, including transducers, the US latch, ring liveness
under tonic injection, lateral toxin response, mushroom-body discrimination,
the CPG/graded muscle pathway, a corrected PG speed gate, and an EMD tuning
relationship. These are component-level findings. They do not establish
end-to-end homing, robust compass integration, learned visual anchoring, or a
planning world model.

The research process itself generated durable safeguards: inspect the actual
neuron mechanism before making mechanistic claims; preserve defaults through
feature gates; report uncertainty and do not elevate approximately one-sigma
effects; test a substrate against both its input and a matched random control;
and use trajectories/per-tick traces in the embodied agent. These safeguards
are part of the scientific output because several earlier conclusions changed
when they were applied.

## Exact resume state at session end

The logical task state was deliberately converted into repository documentation
rather than copied as Claude task files. Its current priorities are listed here
so that work can resume without the external task queue.

| Priority | State at session end | Resume action |
| --- | --- | --- |
| Phase 0 validation | In progress | Ship the already measured tonic-ring and PG arithmetic defaults, then run the A/B/C per-tick, multi-world, 100k-tick matrix. Read `ROADMAP.md` Phase 0 and the `phase0.py` recovery artifact before changing mechanisms. |
| Head-on toxin response | Pending | Enable and measure the existing default-off `trise` klinokinesis circuit against the approximately ±0.05 baseline response. |
| Compass/homing | Parked behind structural work | Do not repeat the eleven falsified parameter sweeps. The current evidence says the gait's sustained-yaw window is too short for the compass translation cost; test CPG-period/trajectory changes with unwrapped bump tracking. |
| Hunger, arbiter, avoidance, analog accumulator | Pending | Follow roadmap Phase 1: restore hunger dynamic range, show embodied arbiter selection per tick, test speed-scaled avoidance, and isolate `k_pi` against ground-truth heading. |
| Vision/chroma/anchoring | Parked | Chroma wiring and replacement of the hardwired `SUN_AZIMUTH` map require a separate visual-validation plan. The hardwired map is a stated PAULA-rule violation and cannot be used as learned-vision evidence. |
| CIFAR PAULA-only target | Open | Reproduce the maintained standard pipeline first, establish a configuration and seed manifest, then compare trained/adapting substrates with retina and matched-random controls. Do not reuse the frozen-mini-brain negative as a universal result. |

The persisted Claude task IDs 22–32, their final descriptions, and the source
locations for their experimental harnesses are summarized in the companion
index. There were no active session tasks at the final point that should be
silently resumed automatically.

## Migrated material and recovery boundaries

The 2,624-file, 3.1-GB non-conversational job corpus from session
`4630c9cc-ae60-4a85-a5da-88544a269fbb` was audited and removed on 2026-08-03.
It was organised as source/launch scripts, web visualisations, configurations,
notes, logs, arrays, media, and multi-file runs; its durable conclusions and
accepted source were migrated before removal.  The retained `experiments/`
directory contains the compass/cylinder and sensorimotor probes that remain
useful. Validation-created `__pycache__` files are not session artifacts.

The 31 recoverable SNN scratchpad tools from session
`be7dec7c-f6b9-4e48-b433-78e0ae71291c` are retained under
[`../neuron-model/snn_classification_realtime/foveation/recovered_snn_debug_tools/`](../neuron-model/snn_classification_realtime/foveation/recovered_snn_debug_tools/).
The temporary scratchpad directory is no longer present; these retained tools
and the local file-history references are therefore the available code
provenance. Four divergent historical implementation versions were kept under
the two repositories' `recovered_versions/snn_debug_2026_07/` directories
rather than overwriting current code.

Not migrated into either Git repository were the raw JSONL transcripts, task
queues/reminders, task-output streams, Claude memory files, and local
file-history blobs. They contain conversation/session infrastructure rather
than reproducible project assets. The source transcript files and backup store
remain outside the repositories at the time of this audit. No Git commit was
made as part of recovery or this documentation pass.

## Research-readiness handoff checklist

This audit intentionally leaves the working trees uncommitted and does not
make a release claim. In particular, the active-inference working tree had an
unresolved pre-existing conflict in `simulations/c_elegans/neuron_mapping.py`
during the July recovery audit. Keep that conflict separate from the recovered
research work and resolve it only with its own test evidence. The conflict is a
repository-management concern; it does not change the evidentiary status
recorded above.

For a new research run, record the Git revision, Python/uv environment, random
seeds, exact configuration, dataset split, horizon, output path, and the
per-tick measurement channels before launch. Store the run manifest with the
results. This makes the next result independently inspectable without either
Claude transcript.
