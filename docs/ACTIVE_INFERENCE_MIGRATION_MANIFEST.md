# Active-Inference Organism Migration Manifest

## Purpose

This migration preserves proven PAULA-based organism work while making it
composable, testable, and legible.  It adopts the constraint-driven
co-evolution rule: a version adds only the smallest body and neural delta
needed to recover from a named environmental failure.

This is a retention manifest, not a claim that older experiments were wrong.
Accepted components, their causal harnesses, and the research record are kept
even where an attempted integration did not yet meet embodied acceptance.

## In scope

Only `simulations/active_inference/` and its directly supporting custom-agent
tests, documentation, and experiments are being reorganized.  Shared
simulation infrastructure stays in place.  C. elegans and zebrafish are out
of scope and must not be changed.

## Retained work

- The PAULA motor, sensory, food-seeking, toxin-avoidance, CPG, Mushroom Body,
  arbitration, and navigation source modules.
- All causal/acceptance harness source in
  `simulations/active_inference/experiments/` and the current standalone
  compass probes in `experiments/`.
- The live brain server, topology/replay generator, and browser UI source.
- The research charter, provenance record, and self-contained handoff.
- Versioned agent packages and component specifications added by this
  migration.  They wrap/inherit existing working components before source is
  relocated, so compatibility is continuously testable.

## Deliberately retained status distinctions

- **accepted:** causal behavior passes its designated harness;
- **experimental:** useful isolated or partial embodied evidence, not yet an
  agent requirement;
- **quarantined:** retained for reproducibility and further research, but not
  enabled by a default agent.

In particular, the compass variants are retained as experimental/quarantined
navigation components.  They are not discarded merely because the present
body/loop does not yet support their full embodied objective.  The existing
motor/CPG work is retained as an accepted reusable component, not replaced to
make an artificial V1/V2 narrative.

## Removal queue (safe, reproducible artifacts only)

1. `simulations/active_inference/experiments/claude_recovery_2026_07/` — a
   3.1 GB recovered-session corpus of copies, generated traces, and temporary
   material.  Current maintained sources and the research handoff preserve the
   useful conclusions.
2. Python `__pycache__/` directories — derived bytecode.
3. Superseded local backup files after their maintained counterpart is
   confirmed.
4. Generated compass JSON traces in `experiments/results/` after their source
   generators/harnesses are retained.  These are reproducible and are not
   golden test fixtures.

Each destructive step is executed against explicit paths and reported with
reclaimed space.  No source module, acceptance harness, or other-organism
file is removed under this manifest.

## Executed cleanup (2026-08-03)

- Removed the explicit recovered-session directory
  `simulations/active_inference/experiments/claude_recovery_2026_07/`
  (3.1 GB).
- Removed Python bytecode caches and reproducible JSON/plot/animation outputs
  from the custom-agent result directories (including the 104 MB compass
  result bundle directory).  Their producing harnesses remain.
- Removed only three superseded backup copies: `live_brain.bak.py`,
  `export_brain.bak.py`, and `brain_live.bak.html`.
- Retained the non-disposable former `ARCHITECTURE.prev.md` as
  `docs/ACTIVE_INFERENCE_ARCHITECTURE_ARCHIVE_2026-07-29.md` because it
  contains measured failure modes and readout rules.
- Added per-directory ignore rules so future generated result traces do not
  repopulate the repository accidentally.

## Target organization

The custom organism remains under `simulations/active_inference/`, preserving
the existing repository boundary:

```text
core/          component contracts, composition, configuration
body/          World3D and explicit transducers
components/    sensory, motor, learning, navigation, arbitration (extensible)
agents/        version packages: V1 reactive, V2 memory, V3 metabolic interoception
challenges/    named environmental failures and acceptance contracts
lab/           live brain, replay, and viewer assets
experiments/   acceptance and diagnostic harnesses
```

The categories are navigation aids rather than a closed taxonomy.  Every
component also carries explicit `requires`, `provides`, dependencies,
conflicts, status, and acceptance harness metadata, so a future component can
be registered without being forced into an unsuitable category.

## Versioning rule

An agent version is a package, never a self-contained script.  Its README and
composition declare inherited components, the one new environmental challenge,
the minimum body capability, the neural delta, and an ablation/acceptance
test.  The examples in the charter are examples only; future versions are
chosen by observed environment-induced failure, not by a precommitted feature
roadmap.

## Delivered composition (2026-08-03)

The first complete three-stage composition is now available as importable
packages under `simulations/active_inference/agents/`:

| Version | Inherited PAULA foundation | Constraint-driven delta | Package |
| --- | --- | --- | --- |
| V1 reactive | olfactory food/toxin route + the established CPG → relay → graded-muscle body | none; this is the stable reactive baseline | `agents/reactive_v1/` |
| V2 memory | V1 unchanged | accepted Mushroom Body learned-valence route | `agents/memory_v2/` |
| V3 interoceptive | V2 unchanged | delayed gut/energy body transducers + a PAULA FORAGE/HOME/EXPLORE/SLEEP WTA | `agents/interoceptive_v3/` |

There is no empty rhythmic V2: the former rhythmic placeholder was merged into
the inherited motor substrate.  V1 therefore retains the proven rhythmic body
instead of deleting working locomotion to manufacture a version boundary.

V3 is a single PAULA brain.  The body updates `gut_load`, `energy_store`, and
realized activity cost from contact, digestion, and MuJoCo movement.  It sends
only bounded afferent currents to ordinary PAULA cells.  The SLEEP population
is built and connected in `components/arbitration/paula.py` and
`components/arbitration/metabolic_parts.py`; Python does not choose a mode or
issue a motor command.  SLEEP inhibition suppresses the search/motor route,
while the body continues to digest and pay basal cost.

The complete embodied causal harness is
`experiments/embodied_metabolic_rest_causal.py`.  It records one row per
neural/physics tick and compares unchanged V2 with intact V3, SLEEP-output
ablation, and metabolic-afferent ablation.  Seeds 11 and 23 both pass with the
same measured separation:

- intact V3: final energy `0.58834`, post-meal SLEEP spikes `576`, actuator
  absolute sum `171.48`;
- unchanged V2: final energy `0.36616`, no SLEEP, actuator sum `1383.91`;
- SLEEP-output ablation: final energy `0.41568`, actuator sum `1224.62`;
- metabolic-afferent ablation: final energy `0.42216`, no SLEEP, actuator sum
  `1266.58`.

The retained full-trace fixture is the complete two-seed
`embodied_metabolic_rest_causal_20260803T101248Z` bundle, plus the full-horizon
food-route causal bundle under
`experiments/results/embodied_food_collection_causal_20260803T101252Z`.

## Refactor boundary

The refactor is mechanical where moving it can preserve neuron IDs, synapse
ordering, and compatibility, and explicit where the legacy builder is still
the safest source of accepted topology.  The real body implementation now
lives at `components/body/world.py`, the arbiter at
`components/arbitration/paula.py`, and V3's afferent fragment at
`components/arbitration/metabolic_parts.py`.  `core/brain_composer.py` is the
single build seam: it validates a named component selection and delegates the
one whole-brain build.  The old `world3d.py` and `aif_arbiter.py` paths remain
small compatibility shims, so existing scripts do not silently fork the
implementation.

The central-complex, motor, visual, and Mushroom Body blocks remain in the
maintained `aif_agent3d.py` builder for now; they are not duplicated as fake
independent controllers.  This is an honest partial extraction rather than a
claim that every historical block has already been rewritten.  Future
extractions must preserve the same one-network contract and pass the existing
causal harnesses before their legacy block is removed.

The shared neuron substrate follows the same boundary: graded and conjunctive
extensions are first-class imports under
`neuron-model/neuron/extensions/`, with default-preserving compatibility shims
at the old module paths.  The phase-locked sample/reset/hold extension remains
under `extensions/experimental/` and is not used by the accepted V1--V3 line.
