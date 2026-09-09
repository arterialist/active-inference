# PAULA neural microscope and component laboratory

Start the local web lab with:

```bash
uv run aif-live lab --port 8850
```

Then open `http://127.0.0.1:8850/lab`. The lab is an operator console rather
than a shell runner: the left rail selects a live V1--V4 session and controls
run/step/pause/reset, the center keeps the actual 3D topology visible, and the
evidence rail seeks the bounded post-tick trace. Seeking replays body pose,
physical afferents, firing cells, and intracellular state rather than
interpolating a summary.

Double-click a cell for current or historical membrane state, parameters,
metadata, and synapses; the protocol also exposes historical neuron and
per-synapse traces.

The **Batch harnesses** dock stays below the microscope so causal experiments
do not discard the live context. It filters the whitelist of maintained
embodied harnesses (motor, food collection, mushroom-body valence, arbiter,
metabolic SLEEP, compass, and obstacle detour), accepts bounded parameters,
shows asynchronous logs and recent run records, and writes full-trace evidence
under the ignored `.live/lab-runs/` directory. The lab never accepts arbitrary
commands or Python source.

Version-specific acceptance is fail-closed. Strict harnesses forward the
selected V1--V4 profile to the builder, record the component manifest and
neuron count, require a complete neural tick trace, and run causal controls.
Food, diagonal-toxin, and head-on-toxin harnesses each expose ten deterministic
worlds ordered from simple to difficult; V4 likewise runs ten physical wall,
corner, chicane, and maze geometries.  The release gate is the embodied
matrix protocol: every strict suite runs all ten worlds with the fixed seeds
`11, 23, 44, 77, 101`, and derives its common horizon from the hardest
world's declared completion time.  A one-world or one-seed lab launch remains
useful for exploration, but is explicitly non-protocol and cannot be reported
as version acceptance. The compass remains labelled
`legacy_experimental` because it intentionally uses the historical full-brain
probe and is not evidence for any strict agent version. Run
`version_integrity_audit.py` before accepting a new evidence bundle.
For the complete cross-version run, use
`version_acceptance_matrix.py --output <evidence-directory> --workers 10`; a non-zero child
or missing/invalid acceptance record makes the matrix fail.  Each child is
then independently checked with
`version_evidence_validator.py <child-directory>`: it recomputes the causal
predicates from raw traces (not `summary.json` or `acceptance.json`) and the
matrix records SHA-256 hashes for every child file as provenance only.  A red behavioural result
with structurally valid evidence is intentionally still red; it is not
silently reclassified as a diagnostic pass.  If the control never encounters
the imposed constraint, the result is `inconclusive`, not a claimed success.
Every strict record also stores the effective PAULA build parameters, and the
validator requires ablation conditions to share the same fixture, initial
state, and horizon.
Every embodied record also emits synchronized `pov.mp4`, `third_person.mp4`,
and `top_down.mp4` artifacts under its evidence `artifacts/` directory.  The
artifact manifest records the sampling interval and fails closed if any view
cannot be encoded, so visual review cannot silently disappear from a release
run.

The live introspection endpoints are `/api/introspection` (static graph
manifest and trace range), `/api/trace` (seek timeline),
`/api/tick/{tick}?detail=neurons|synapses|all` (exact post-tick state),
`/api/neuron/{id}/history`, and `/api/synapse?...&tick=...`.

The lab server also exposes `/api/runs` for recent harness records. Keyboard
shortcuts in the console are `R` run, `S` step, `P` pause, and `L` jump to the
live edge.

The maintained live-brain/replay implementation remains at its established
entrypoints during the compatibility pass: `../live_brain.py`,
`../compass_replay.html`, and `../experiments/compass_replay_bundle.py`.
