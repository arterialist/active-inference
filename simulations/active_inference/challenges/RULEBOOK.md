# Constraint-Driven Co-Evolution Rulebook

## The authorization chain

Do not add a brain feature because it seems biologically interesting.  A new
component is authorized only by this chain:

```text
one environmental change → measured failure of the frozen prior agent
→ minimum body transducer/effector → minimum PAULA component
→ recovery + ablation evidence
```

If removing the proposed body capability or circuit does not restore the
induced failure, it is not yet a justified part of that agent version.

## Challenge contract

Every challenge module documents:

1. the unchanged prior agent and its baseline success;
2. exactly one altered environmental rule;
3. quantitative failure criterion and seeds/duration;
4. the minimum new body capability needed to observe or act on that rule;
5. the neural component and the signals it consumes/provides;
6. recovery and causal ablation harnesses;
7. which earlier versions must remain unchanged.

Examples such as uneven traction, delayed hazard sensing, or a memory-demanding
odour reversal are prompts, not an obligatory roadmap.

The first concrete V4 challenge is `obstacle_detour`: freeze V3, remove food
from the fixture entirely, and run a deterministic geometry suite (full-width
wall, L-corner, alternating chicane, compact maze). The minimum body delta is
two forward-biased physical whisker/range channels; the minimum neural delta
is bilateral obstacle onset plus a crossed PAULA turn/stop/wall-follow route.
Acceptance requires unchanged V3 collision/stall, intact V4 contact-free
lateral deflection, and failure under independent body-afferent and
reflex-output ablations. The chicane and maze are diagnostic route-depth
probes, so their limitations remain visible rather than being converted into a
false binary success. The harness records every neural tick, not only the
final pose.

## Scaling rules learned from this work

- Preserve accepted circuits as reusable components.  Do not delete a working
  CPG/motor route to create a cosmetically simpler version history.
- Use full tick traces for neural/body causal questions.  Windowed summaries
  are useful outputs, but cannot diagnose phase reversals, body-neural lag, or
  missed events.
- Match the measurement to the state: spike-window fills for latches,
  unwrapped cumulative displacement for ring motion, and in-tick hooks for
  within-tick populations.
- Make every build knob observable and reject unknown configuration keys.
  Silent keyword filtering, fan-in-dependent thresholds, and neuron-ID
  collisions previously created false conclusions.
- Validate a component at three levels: isolated circuit, whole brain in a
  virtual stimulus harness, and embodied closed loop.  Passing one level is
  evidence, not transfer proof.
- A body movement that alternates faster than a circuit can integrate is a
  body–brain interface constraint, not automatically a compass defect.  Test
  a neural buffer and a changed movement regime separately, then retain the
  trace that explains transfer failure.
- Keep experimental extensions explicit.  Graded tonic release is an opt-in
  neuron extension; phase sample/hold is quarantined experimental machinery,
  not a biological claim or default controller.
- Scale a circuit only after the environmental constraint demands it. Add one
  transducer at a time, retain the prior version unchanged, and make the new
  path observable from body signal → PAULA population → descending relay →
  actuator. More cells are justified when they provide a measurable temporal,
  bilateral, or precision code—not as an untested complexity multiplier.

## Embodied Matrix acceptance protocol

The program treats the simulator as a controllable family of embodied
realities: worlds are designed to expose a constraint, not sampled until a
trajectory happens to look favourable.  The minimum release gate for every
strict V1--V4 agent is therefore a **10 × 5 embodied matrix**:

1. Each strict harness declares exactly ten different physical worlds in a
   fixed catalog order, from the simplest expression of its challenge to the
   hardest.  The challenge must be present in the fixture before the run; a
   random source or an unreachable control is not evidence.
2. Every world is run with the same five preregistered seeds:
   `11, 23, 44, 77, 101`.  Fewer seeds or a subset of worlds is an exploratory
   lab run and must be labelled non-protocol; it cannot support a version
   acceptance claim.
3. When the horizon is omitted, the harness derives it as
   `max(world_catalog[w].completion_steps)` over the selected worlds.  The
   completion value is in that harness's declared units (body steps for
   movement, training trials for the memory phase).  An explicit shorter
   horizon is rejected.  All causal conditions for a world/seed share the
   same fixture, initial pose, and horizon.
4. Acceptance requires every record, every raw tick trace, the strict PAULA
   topology, the synchronized POV/third-person/top-down video artifacts, and
   the causal ablations to pass.  A structurally complete but
   behaviourally failing or inconclusive world remains red; summaries cannot
   override raw-trace recomputation.
5. `version_acceptance_matrix.py` schedules independent strict suites in
   parallel child processes.  Parallel execution is an orchestration detail:
   each child owns its MuJoCo/PAULA state, writes an immutable evidence
   directory, and is independently validated before the matrix is reduced.

The matrix is a minimum reproducibility floor, not a claim that ten worlds
fully characterize an organism.  New world families, perturbation axes,
longer horizons, and cross-world causal analyses should be added when the
scientific question needs them.  A passing row means the declared component
survived that embodied reality under the stated causal test; it does not
license extrapolation to untested environments.

## Default composition policy

Accepted components may be inherited by normal agent packages.  Experimental
and quarantined components require explicit prototype selection and a named
harness.  Their source and evidence remain available; non-default status is a
guardrail against accidental claims, not disposal.

## Continuing adaptation and regulatory pathways

The isolated experiments recorded in
[`ASSOCIATION_FINDINGS_2026-09-08.md`](../experiments/ASSOCIATION_FINDINGS_2026-09-08.md)
add these checks. They do not change any embodied version's acceptance status.

- A positive learning-rate parameter does not prove effective adaptation.
  Record eligible updates and represented weight changes. In one tested PAULA
  variant, `eta_retro=1e-9` produced zero basal terminal changes through float32
  rounding, although modulation still produced changes.
- Test the rate and the temporal credit rule together. Inhibiting an obsolete
  prediction left its earlier spikes inside `t_ref`; amplifying learning then
  strengthened the old association. A local modulatory window change reversed
  the update direction in the matched preparation.
- Audit the pathways that regulate adaptation. Fast retrograde updates drove
  prediction export gains below zero and interrupted comparator inhibition.
  Slower, numerically resolved adaptation recovered the tested cases. This is
  finite-horizon evidence for separate adaptation timescales, not a license to
  freeze regulatory pathways or a proof of indefinite stability.
- A quiet mismatch population is not an acceptance criterion. Test false
  predictions and contradictory or omitted outcomes. A one-sided positive-error
  detector can be silent when the network predicts every outcome.
- Treat stimulus timing as part of the component's tested operating range.
  Preserve pulse counts while varying spacing, order, delay, and downstream
  consumers. Label transfer failures even when acquisition and retention pass.
- Use independent raw-trace recomputation and executable replay. A rendering
  of recorded states is not causal replay; replay must run the saved network
  from its inputs and compare state without overwriting it.
