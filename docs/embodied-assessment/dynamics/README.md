# Recorded dynamics

`populations.html` now opens the real audiovisual preparation: 1,152 PAULA
neurons receiving a movie and its original soundtrack. It includes paired
exposure, silent movie, a still image, audio-only reference and a shifted-sound
control. The source movie, transduced sensory inputs and 3D population activity
share a tick cursor. Sound is opt-in and is not played during audio-withheld
conditions. No neural-generated bark is claimed. The earlier binary-pattern
population run and pathway cuts remain selectable.

The viewer is a research sketch, not a production lab. Node positions describe
wiring groups, not anatomical locations or an inferred attractor manifold.
All neural activity is recorded PAULA data; the page never runs a neural model.
The real media and transformations are credited in `media/README.md`.

`regimes.html` is the lightweight 3D research sketch requested after the owner
clarified that hierarchical regimes, rather than polished single-circuit UI,
are the object of investigation. It offers time ribbons and state-space
projections of recorded sensory activity, predictions, modulation and weights.
Vertical position groups measured variables. It does not assert a discovered
hierarchy, attractor, multimodal learner, or action–perception loop.

`index.html` retains the earlier paired, tick-level inspection instrument.
The three comparisons concern temporal credit, feedback-path degradation, and
timing-dependent recall. It provides a spike raster, selected-cell traces,
synapse inputs, complete queues and a raw tick download. Both pages start paused.

These are read-only static pages. The server at `127.0.0.1:8810` serves files
only; it starts no PAULA or embodied simulation. From `active-inference/`:

```sh
.venv/bin/python -m http.server 8810 --bind 127.0.0.1 --directory docs/embodied-assessment
```

The data loader uses local script envelopes and browser gzip decompression, so
the pages also support direct local-file use in compatible browsers. HTTP use
was checked in the in-app browser; direct-file use was not browser-verified.
The older association-view cache retains at most six full-trial chunks plus
five small metadata records; its data occupies about 32 MB. The population view
loads one complete binary float32 presentation at a time, with float64 research
records stored separately. There is no external JavaScript, font request,
telemetry, trained decoder or controller in the viewer.

## Data and verification

`association_visual_export.py` first re-runs the independent equation audit,
then exports every completed tick from five existing recordings. The 92,480
raw records preserve the original state, delivered inputs and event queues.
`derived` adds reconstructed pre-reset membrane, current threshold, spike age,
credit eligibility and exact per-tick weight deltas. Original post-reset
membrane remains in `state`. All times are simulator ticks, not milliseconds.
The inspector plots hold each sample until the next tick. The 3D state-space
view connects consecutive samples geometrically, not as a claim about an
unobserved continuous path between ticks. Plot amplitude scales are labeled.

```sh
.venv/bin/python -m simulations.active_inference.experiments.association_visual_export
uv run --offline --no-sync --with pytest python -m pytest tests/test_association_visual_export.py -q
```

Nine tests passed, including exact source equality for every exported raw tick,
the temporal-credit sign reversal, and the silent-transfer/successful-return
classification. Raw-data tests skip if the local research recordings are absent.
The paired view was checked for stepping, play/pause, selected-cell changes,
both recall outcomes and the feedback-breakdown event. The 3D sketch received
only a load/render check, consistent with the owner's request to stop treating
this as a production webpage project.

## Research boundary

The next experiment must instantiate the missing coupled hierarchy: a relation
between distinct sensory populations becomes an associative state, influences
action, changes subsequent sensory input, and is regulated through neural
feedback. Match individual sensory activity statistics while changing their
relationship; test ascending and descending cuts, matched replay, retained
distinctions, and recovery without stopping adaptation. Current eight-cell
recordings cannot answer those questions. A 3D stack does not fill that gap.

## Population experiment and real media

The population baseline is a first explicit wiring of sensory cores, reciprocal
connections, an LLGC-like interface, an upper recurrent population and two
regulatory populations. It is not yet a successful realization of the notebook
architecture. Source and findings:

- `simulations/active_inference/experiments/population_hierarchy.py`: shared
  sparse PAULA graph, matched pathway cuts, recording, weight-only transplant
  probes and binary visual export.
- `audiovisual_population.py`: real-video and audio transduction. Its network
  uses the same graph. The stable channel-2 group IDs retain their earlier
  `touch`/`tactile_core` identifiers, but the metadata, actual inputs and display
  labels explicitly identify them as auditory in the movie experiment.
- `population_recording_audit.py`: independent data reader, with no PAULA
  import or simulator startup. Reports initial-wiring false positives,
  withdrawal transients, input imbalance and count/recording discrepancies.
- `POPULATION_FINDINGS_2026-09-08.md`: bounded findings and remaining tests.

From `active-inference/`, using a new output directory for each experiment:

```sh
.venv/bin/python -m simulations.active_inference.experiments.audiovisual_population --source docs/embodied-assessment/dynamics/media/dog-command-source.ogv --output .live/research/NEW_ALIGNED_RUN --condition aligned
.venv/bin/python -m simulations.active_inference.experiments.audiovisual_population --source docs/embodied-assessment/dynamics/media/dog-command-source.ogv --output .live/research/NEW_SHIFTED_RUN --condition shifted
.venv/bin/python -m simulations.active_inference.experiments.population_recording_audit .live/research/NEW_ALIGNED_RUN --control .live/research/NEW_SHIFTED_RUN
.venv/bin/python -m simulations.active_inference.experiments.population_hierarchy export --input .live/research/NEW_ALIGNED_RUN --output docs/embodied-assessment/dynamics/data/population-media-aligned.js
uv run --offline --no-sync --with pytest python -m pytest tests/test_population_hierarchy.py -q
```

Every completed tick has cellular observables and outgoing information
amplitudes. The main aligned run also records every incoming information weight
using lossless float64 XOR coding; `read_information_weights` reconstructs it.
It does not record every queue or all plasticity fields every tick. Initial
configuration, source hashes, actual sensory features, protocol and final full
state are retained. These distinctions matter when diagnosing an unrecorded
intracellular mechanism; rerun from the declared source rather than invent it.

## Paper asset

`assets/paper.webp` was generated with the built-in image tool, then converted
to WebP. The exact prompt and provenance are in `assets/paper.webp.json`.
It is decorative material, not experimental imagery.
