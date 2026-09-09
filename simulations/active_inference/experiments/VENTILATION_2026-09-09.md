# A bodily consequence for the retained sensorimotor preparation

This is a completed physical-necessity screen, not a neural regulation result.
It follows the [return-to-resistance experiment](ACTIVE_SWEEP_2026-09-09.md).
The retained predictor helped movement without solving the observer-defined
sweep task. Before adding regulation, this screen asks whether that movement
can maintain a resource and whether excess activation has a distinct cost.

## Declared body and evidence boundary

`components/body/ventilation.py` adds a chamber linked to the existing MuJoCo
hinge: volume is `6 + 5*tanh(20*q)` mL. Only increasing volume admits fresh gas.
Oxygen fraction is .21, reservoir capacity .3 mL, initial reserve .15 mL and
continuous demand .3 mL/s. Thus an unmoving chamber with no exchange exhausts
its initial reserve after .5 s. Inspiration, uptake, consumption, overflow and
unmet demand are retained separately at every tick.

The existing energy organ is reused with capacity .3 J, initial energy .225 J,
gut capacity 2 J, initial gut contents 1.2 J, digestion .12 W, basal cost .02 W,
mechanical efficiency .25 and activation cost .1 times summed squared muscle
release, integrated over time. Both antagonists contribute to activation cost.
Positive work is charged per muscle, rather than inferred from their net
torque. Cancellation cannot make activation free. No host motor brake, task
label, gate reward or organ-based action selector was added.

These are illustrative model units and equations, fixed before the screen.
The chamber has ideal extraction and negligible added mechanical load. There
is no dead space, pressure-volume work, perfusion, CO2 or coupling of oxygen to
digestion. Oxygen and energy are separate constraints, not a biochemical model.
Unmet demand is a measured deficit, not a validated biological death threshold.
Its duration and amount remain available, even if the reserve later recovers.
Zero deficit is the stringent bounded criterion used here, not a claim that
real organisms tolerate no transient physiological error.

[Yao et al., 2023](https://doi.org/10.1016/j.cub.2023.01.019) report a mouse
carotid-body-to-brainstem pathway whose manipulation affects hypoxia-induced
sighing. The accessible abstract and summary were checked for this screen.
That supports considering organ feedback to respiratory circuitry; it does
not justify these chamber parameters or establish a faithful neural model.
No respiratory neural circuit was implemented in this screen.

## What was actually executed

For each seed 11, 23, 44 and 77, six complete recorded 1024-tick muscle histories
were replayed through MuJoCo: released and loaded transfer branches with intact
or reset predictive weights, followed by current and reset return branches.
Five additional controls use the return/current history: no activation,
actuator transmission cut, gas exchange cut, doubled activation and quadrupled
activation. These last two controls multiply recorded muscle release in the
experiment driver. They are explicitly open-loop physical capacity probes,
not PAULA learning or a permitted organism controller.

There are 44 courses, each 4.096 seconds, totaling 45,056 independently checked
physical/resource ticks. Unmodified muscle replay reproduces every saved
MuJoCo integration state exactly. The separate auditor re-integrates the
original hinge and reconstructs both resource recurrences without calling the
organ update method. It also checks muscle-to-action mapping, work, activity,
exchange, gate records and every state sample. This is not a new audit of all
neural equations; the original full neural traces remain the source evidence.

New organs start with identical stated contents at the beginning of each
course. They did not exist during the prior acquisition. They receive no
information about which weights were retained or reset. The brain has not yet
received their afferents. This cannot establish learned interoception.

## Full trajectories and consequential observations

All four released-world intact and reset histories meet the model's oxygen
and energy demands over the bounded course. Loaded histories periodically
exhaust oxygen while energy remains available. In the return/current histories,
the first meaningful oxygen deficit occurs at local ticks 448, 440, 442 and
438 for the four seeds. Reset histories reach it at 429, 427, 420 and 424.
These indices are zero-based; state samples are after the corresponding step.
The reporting tolerance is 1e-12 model units, only to avoid counting numerical
round-off as an onset. All values remain in the raw arrays.

The current histories therefore delay the first oxygen shortfall compared
with reset in this transducer, but do not prevent it. This is a retrospective
physical consequence of a previously measured weight intervention. It does
not mean those weights learned an oxygen-maintenance objective.

Doubling the return muscle histories avoids both deficits in every seed during
the course. Quadrupling avoids oxygen deficit but first incurs energy deficit
at ticks 295, 296, 295 and 296. Oxygen reserve curves for 2x and 4x nearly
overlap after their rise, while their energy trajectories separate sharply.
The extra activation largely spills oxygen beyond capacity and incurs cost.
This establishes a nonempty operating range in the tested body, not an optimum.

Silence and actuator disconnection both run short of oxygen at ticks 180 to
182. They do not fail at tick 125 because the continuing body is initially
displaced and moving: passive relaxation admits some gas. The exchange cut
fails at tick 125 while retaining the measured chamber motion. The distinction
would be lost if controls silently reset the body to rest.

Final reserves alone give the wrong answer. Seed 11's natural return finishes
with .064312 mL oxygen despite .123116 mL cumulative unmet demand. Its 4x
control finishes with .023222 J energy despite .471135 J cumulative unmet
demand. Later replenishment does not erase earlier shortfalls. The figure
plots every reserve and deficit sample for every seed on shared scales.

## What this permits next, and what it does not

The environment makes movement necessary and excessive activation costly.
It does not yet make closed-loop regulation necessary: the fixed 2x replay
control meets both demands in this particular course. A future neural
regulator must be compared with that alternative rather than credited solely
for beating the original amplitude. Different physical demand or resource
conditions should test whether feedback contributes beyond a constant gain.
Finite gut contents also prevent extrapolating a four-second course to
indefinite self-maintenance.

The next composition can retain the actual acquired brain, its delayed sensory
history and its imperfect predictor, while adding organ afferents and neural
regulation of the existing motor rhythm. Test intact feedback, matched pathway
cuts and constant-drive controls without supplying decoded oxygen error or an
action choice from the host. Keep plasticity available, track changes in the
old predictor, and account for the added neural and afferent latencies. The
bounded question is coexistence and useful regulation, not respiratory anatomy
or proof of ALERM's whole hierarchy. No V1–V4 or PAULA core changes were made.

## Reproduction and tests

From the `active-inference` repository:

```bash
uv run --offline --no-sync --with cloudpickle==3.1.2 python -m simulations.active_inference.experiments.ventilation_screen .live/research OUTPUT
uv run --offline --no-sync --with cloudpickle==3.1.2 python -m simulations.active_inference.experiments.ventilation_figures OUTPUT FIGURE_OUTPUT
uv run --offline --no-sync --with cloudpickle==3.1.2 --with pytest python -m pytest tests/test_ventilation.py tests/test_metabolic_population.py -q
```

The retained screen is `.live/research/20260909_ventilation_screen_final` and
the figure is `20260909_ventilation_figures/ventilation-resources.png`. The
manifest records parameters, columns, source-record identities and hashes;
the output stores each complete physical/resource trajectory, not just event
times. Source recordings are required for reproduction and remain local.

Twenty-four focused tests passed in 15.91 seconds, including independent-audit
corruption checks, conservation through overflow and deficit, reloading organ
state, unchanged physical integration, co-contraction costs and existing
metabolic-population and active-sweep tests. Empty or shortened evidence and
undeclared initial reserves are also rejected. An initial corruption test mistakenly added .01
to an integer gate array, making no change; it was corrected to add one and
then confirmed that the auditor rejects it. The body now rejects a net-command
`step()` call because that interface would omit antagonist activation costs.
Use `step_muscles()` with both actual releases.

Two intermediate copies were compared field-for-field with all 44 final
records and then removed, reclaiming about 15 MiB. Their trajectories were
identical; the reruns followed interface and audit validation additions. The
final data use about 7.5 MiB. The removed copies are not needed for reproduction.
