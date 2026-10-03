# Claude session recovery index — July 2026

This index records the historical recovery, including the corpus cleanup on
3 August 2026. Counts and verification statements describe that audit. Local
history identifiers are retained as provenance and are not Git revisions.

This index supports
[the provenance and resumption record](../RESEARCH_PROVENANCE_AND_RESUME.md).
It is an index of recoverable project material, not a copy of conversations.
The source session logs and Claude file-history store remain outside the Git
repositories.

## Source sessions

| Session ID | UTC interval | Source evidence read | Recovery relevance |
| --- | --- | --- | --- |
| `be7dec7c-f6b9-4e48-b433-78e0ae71291c` | 2026-07-03T16:33:14.498Z–2026-07-11T11:07:56.133Z | 7,649 event records; 168 file-history snapshots; task reminders through the retino-RGB attractor run | Original CIFAR/SNN and foveation research session. |
| `4630c9cc-ae60-4a85-a5da-88544a269fbb` | 2026-07-08T20:24:53.066Z–2026-07-30T05:44:51.709Z | 32,045 event records; 285 snapshots; 196 history deltas; final task IDs 22–32 | Continuation: standard-pipeline/CIFAR work, circuit library, embodied active-inference agent, live brain, and Phase-0 validation. |

## Repository source mappings

### Neuron-model implementation tracked by the SNN session

The final SNN session snapshot tracks 63 `neuron-model/` paths. The current
repository is the primary location for them. They fall into the following
groups.

| Source group and representative retained history | Current repository destination | Interpretation |
| --- | --- | --- |
| `neuron/neuron.py`, `37b72b59ef705ec4@v4` | `../neuron-model/neuron/neuron.py` | Shared PAULA neuron model. The session introduced the gated `reward_hebb` path; preserve default-path compatibility when comparing history. |
| Standard vision/SNN builders, mostly `@v2` on 2026-07-04 | `../neuron-model/snn_classification_realtime/{build_activity_dataset.py,network_builder.py,network_builder_direct.py,realtime_classification_fusion.py}` and `core/`, `activity_dataset_builder/`, `activity_preparer/`, `realtime_classifier/` | Initial CIFAR audit, dataset/input fixes, and calibration path. |
| Foveation core, 2026-07-05–07 history | `../neuron-model/snn_classification_realtime/foveation/{fovea.py,retina.py,retinal_conv.py,saliency.py,signals.py,perception.py,viewer.py}` | Retina/fovea, glow, visualisation, and basic experiments. |
| Mini-brain, notably `core.py` `5c062a649ac282e1@v7`, `reservoir.py` `784f81f423ab229d@v6`, `heads.py` `47dd2d…@v4` | `../neuron-model/snn_classification_realtime/foveation/minibrain/` | Encoder/readout/plasticity/gaze variants and reproducible CIFAR experiment drivers. |
| Research log, `50c2bd0eb5c2c8f4@v15` | `../neuron-model/snn_classification_realtime/foveation/FOVEATION_RESEARCH.md` | Canonical maintained experiment narrative and retractions. |

The session history also tracks later direct work in `cifar_ceiling/`,
`foveation_results/basin_generalization/`,
`snn_classification_realtime/paula_vision/`, and the three top-level
basin/completion/kappa runners. These are direct repository material, not
artifact-corpus imports.

### Scratchpad tools from the SNN session

The original private temporary scratchpad is no longer present. Its 31
recoverable non-generated tools were promoted, without overwriting maintained
implementations, to:

[`../../neuron-model/snn_classification_realtime/foveation/recovered_snn_debug_tools/`](../../neuron-model/snn_classification_realtime/foveation/recovered_snn_debug_tools/)

| Source class | Count | Destination |
| --- | ---: | --- |
| Diagnostic/smoke probes | 15 | `diagnostics/` |
| Launch and resume scripts | 10 | `launchers/` |
| Analysis/artifact helpers | 5 | `scripts/` |
| Mini-brain HTML report | 1 | `reports/minibrain_report.html` |

The historical launchers are supported by the maintained
`exp_attractor_class.py` compatibility aliases. They remain opt-in because
full CIFAR arms are long-running and create material output.

### Active-inference implementation tracked by the continuation

The final continuation snapshot tracks 54 direct active-inference paths and
37 direct neuron-model paths. The active-inference paths are now in their
intended source locations, principally:

| Component | Current repository destination | Retained history example |
| --- | --- | --- |
| Agent core and 3D body/world | `simulations/active_inference/{aif_agent3d.py,world3d.py,visual_cortex.py}` | `aif_agent3d.py` `a6f73d64199b1269@v30`; `world3d.py` `3a2bd650f47fed70@v14` |
| Compass/path integration | `simulations/active_inference/{central_complex.py,cx_navigator.py,pi_accum.py,pi_agent.py,pi_stone.py,ladder_integrator.py}` | `central_complex.py` `ee7678495641eedf@v18`; `cx_navigator.py` `b187143cceb07c47@v11` |
| Mushroom body, foraging, arbitration | `simulations/active_inference/{mushroom_body.py,mushroom_agent.py,mushroom_learn.py,forager.py,neural_nav.py,aif_arbiter.py}` | `mushroom_agent.py` `84328295e82e43be@v6` |
| Live brain and visualisation | `simulations/active_inference/{live_brain.py,topo_live.py,build_brain_page.py,export_brain.py}` | `live_brain.py` `2bb6450960f6ff7d@v11` |
| Architecture/status documentation | `simulations/active_inference/{ARCHITECTURE.md,LAB_RULES.md,RESEARCH_LOG.md,ROADMAP.md}` | `ARCHITECTURE.md` `dd2b5d72d689b327@v7`; `RESEARCH_LOG.md` `31da94d754ba8b61@v6` |
| Earlier organism scaffolding | `simulations/organism/` | All eight tracked files were restored at their original paths. |

`decode.py` was the one direct source file restored because it was absent from
the working tree; it is now at
`simulations/active_inference/decode.py`. Its final session record was version
87 at 2026-07-30T05:09:06.824Z, without a separately named history blob.

### Session job corpus

The continuation's non-conversational job directory contained exactly 2,624
files and occupied 3.1 GB when audited.  It was a temporary recovery corpus and
was removed on 2026-08-03 after its durable conclusions, source modules, and
accepted fixtures were migrated.  The retained embodied probes are listed in
the [maintained experiments directory](../simulations/active_inference/experiments/).

| Original material | Repository organisation | Audit state |
| --- | --- | --- |
| Python and shell experiment code | `scripts/` | Syntax checked during recovery. |
| HTML/JavaScript interactive or static artifacts | `web/` | Retained as experiment evidence/visualisation, not promoted product UI. |
| JSON/YAML configurations and captured records | `config/` | Ordinary JSON parsed; large JSON remains retained without eager parse. |
| Logs, arrays, media, and multi-file runs | `results/{logs,arrays,media,runs}/` | 207 NumPy artifacts were readable during recovery. |
| Text and ancillary material | `notes/`, `misc/` | Preserved for provenance. |

The old `run_recovered.py` and `audit_artifacts.py` launchers belonged to that
temporary corpus and are no longer supported.  The current harnesses under
`simulations/active_inference/experiments/` and the retained root-level probes
are the supported rerun surface.

### Divergent historical source versions

Files that differed from a newer working-tree implementation were preserved
instead of overwritten.

| Repository | Preservation destination | Contents |
| --- | --- | --- |
| neuron-model | `../neuron-model/recovered_versions/snn_debug_2026_07/` | Historical `neuron/neuron.py`, mini-brain `core.py`, and `exp_attractor_class.py`. |
| active-inference | `../simulations/active_inference/recovered_versions/snn_debug_2026_07/` | Historical `RESEARCH_LOG.md`. |

## Final task-state translation

The continuation's persisted task files were session infrastructure and were
not copied. The final task meanings have been made durable in the agent
roadmap. Task 24 was closed negative after the structured shift-bank tests;
task 27 was closed as a root-cause diagnosis. Tasks 22–23 are parked vision
work, 25 is parked behind the compass, 26 is the unmeasured existing `trise`
circuit, 28–30 are Phase-1 measurements, 31 is the in-progress Phase-0 matrix,
and 32 is the gait-sustained-yaw bottleneck investigation. Exact acceptance
criteria and the experimental paths are in
[`simulations/active_inference/ROADMAP.md`](../simulations/active_inference/ROADMAP.md).

## Excluded source material

The raw session JSONLs, 118 continuation task-output files, Claude task queues,
assistant memory, and file-history backing store are intentionally excluded
from the repositories. They are neither source code nor independently
reproducible result artifacts. The missing original SNN scratchpad is the only
source location that could no longer be re-read directly; its final recoverable
tools were already promoted as documented above.
