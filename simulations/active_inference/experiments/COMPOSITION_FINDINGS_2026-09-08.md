# PAULA composition and plasticity: experimental findings

This is a completed isolated mechanism study, not embodied acceptance or evidence
of consciousness. The biological agents and base `neuron.py` were not changed.
The freeze conditions below are causal controls, not candidate agents. The user's
requirement is continuing adaptation with weak basal plasticity and transient
neuromodulatory increases. The final experiment implements that requirement.

The subsequent [closed-loop association investigation](ASSOCIATION_FINDINGS_2026-09-08.md)
replaces the scheduled modulatory stimulus with a neural comparator and tests
acquisition, retention, reversal, and timing transfer with continuing adaptation.

## Question and preparation

Can a sustained relation between two PAULA populations be used by another PAULA
cell while adaptation continues, and what does connecting that reader change?

Two three-cell delayed synfire rings store one of two relative phases, initiated
by a single cue at tick 3. Two ordinary coincidence neurons distinguish the
phases. Wiring and reader thresholds are designed, not learned. Phase is a
relational code, but this is not proof that a new cognitive entity has emerged.
Five seeds select recurrent weights in [3.8,4.2]. Each seed is tested with both
phase cues. These are five gain initializations and two conditions, not ten
independent biological replications. Seed 11 was also used for calibration.

The exact original preparation, questions and limitations were recorded before
the first run in `COMPOSITION_PROBE.md`. All initial experiments use the base
Neuron class. The later rate-gate experiment uses an explicitly experimental
subclass, described below. There are no fitted software decoders.

## Results before the rate-gate extension

Each row has five initializations, both phase cues, 1,260 ticks. Persistence
means the complete expected spike sequence, not merely positive average activity.
Disconnected readers are not scored as successful readers.

| Adaptation | Source wiring | Exact recurrence | Exact neural readout | First missing ring spike |
|---|---|---:|---:|---|
| Frozen diagnostic | Shared reader terminal | 10/10 | 10/10 | None within horizon |
| Postsynaptic only, eta=.01 | Shared reader terminal | 10/10 | 10/10 | None within horizon |
| Retrograde only, eta=.01 | No reader attachment | 0/10 | Not applicable | 298 |
| Retrograde only, eta=.01 | Shared reader terminal | 0/10 | 0/10 | 242–263 |
| Both, eta=.01 | No reader attachment | 0/10 | Not applicable | 319–340 |
| Both, eta=.01 | Shared reader terminal | 0/10 | 0/10 | 284–326 |
| Both, eta=.01 | Separate reader terminal | 0/10 | 0/10 | 319–340 |

The full factorial also includes the other wiring combinations; none were
discarded. Two source dynamics problems are separable.

First, the source circuit itself drifts. After its startup transient, an incoming
recurrent pulse arrives 15 ticks after the receiver's previous spike, inside its
approximately 17.075-tick learning window. Its postsynaptic weight increases.
The same contact sends retrograde error proportional to incoming amplitude minus
weight. Because weight exceeds amplitude here, this reduces presynaptic release.
Effective transmitted current eventually falls below firing threshold.

For seed 11, phase 0, no reader attachment, neuron 1 misses its tick-340 spike.
Its queued local potential is 2.3506772518. Dendritic attenuation and membrane
integration predict 2.3506772518 × .95^6 / 3 = .5759879284, below threshold .6.
The recorded membrane value is .5759879351. The two downstream failures at ticks
347 and 354 have no arriving pulse, so they are consequences, not the initiating
failure. With shared readers, neuron 5 first fails at tick 284 with S=.59816933.

Second, the reader changes its source through the shared terminal's retrograde
updates. All twenty paired retro-enabled comparisons first diverge in source
state at tick 6. Separating export terminal 901 from recurrent terminal 900,
or selectively cutting only reader-generated retrograde events, makes the source
cell trajectories identical to the disconnected source for the whole run.
The comparison includes source membrane, queues, weights and recurrent terminal;
it deliberately excludes the independently adapting export terminal 901.
This removes attachment interference, not the source circuit's own drift.

Freezing t_ref does not rescue these preparations. Reducing both learning rates
to .001 postpones the first missing spike to 2,923–2,965 ticks without readers,
or 2,923–2,951 with shared readers, across five phase-0 runs per condition.
All ten are silent by the end of the 8,400-tick recording. A short run at this
rate would therefore have produced a misleading stability claim.

## A stable-looking rhythm can conceal loss of viability

The 21-tick rhythm is unchanged until failure. At the same point in successive
cycles, neuron 1's t_ref approaches 17.07525029354, while its predicted pulse
margin above threshold falls from +.25638 at input tick 145 to +.02012 at tick
313, then -.02401 at tick 334. The activity-derived regulator has essentially
the same observation during that decline. This particular timing regime does
not let the firing-rate homeostat correct the drifting transmission gain.

This matters for the proposed hierarchy: a supervisor that sees only the
apparently healthy rhythm can miss loss of the conditions sustaining it.
Faster observation alone does not solve an uninformative observation. A useful
next hypothesis is regulation of transmission margin or susceptibility to a
small perturbation, using locally available signals rather than an external
observer's access to all state variables.

The follow-up interventions distinguish parameters from ongoing activity. Across
all five seeds and both phases, freezing adaptation at tick 160 retains exact
recurrence and readout through tick 1,259. Freezing at 400 does not restart them.
Restoring initial synaptic and release gains at 400 also does not restart them.
Restoring gains AND repeating the original cue causes a second transient bout,
85–94 source spikes and 10–11 reader spikes after tick 400, before another loss.
That is externally re-creating a state, not autonomous memory retrieval.
Residual intracellular history remains, so we do not claim complete erasure of
all information from every state variable.

## Continuing plasticity with a local rate modulator

The opt-in extension is
`neuron-model/neuron/extensions/experimental/plasticity_rate.py`.
It inherits the unchanged base spike, dendritic and adaptation rules. Its local
rate multiplier is

    g(M) = 1 + boost × max(M,0) / (half_saturation + max(M,0))
    eta_post_effective = eta_post_basal × g(M)
    eta_retro_effective = eta_retro_basal × g(M)

The source neuron's local M gates its retrograde adaptation; the receiver's M
gates its postsynaptic adaptation. Both use the previous completed tick's M,
respecting the existing scheduler. Basal rates are strictly positive. Disabled
metadata follows the exact base path. Enabled cells own their parameter object,
and temporary rate substitution is restored even on an exception.

This is a phenomenological receptor mechanism, not a demonstrated cellular
model. It has no task labels, global loss, error decoder, hold counter or
behavioural policy. With basal rates 1e-5, boost 999 and half-saturation .1,
a ninth PAULA neuron releases the modulator through ordinary connections.
Its informational release is zero, which is checked every tick. The experiment
stimulates that neuron for an 80-tick burst, or continuously in the stress arm.
Thus neural delivery is implemented, but an autonomous decision to release the
modulator is not. The unused channel 0 carries it in this isolated preparation.
Adding a third channel to existing agents needs separate work because several
current network buffers assume exactly two modulatory dimensions.

Forty confirmation runs use five gain initializations, both phase cues, four
conditions and a 4,200-tick horizon, following eight calibration runs at 840 ticks.
No weights are frozen in any of these runs.

| Condition | Exact recurrence and readout | Change in recurrent weights |
|---|---:|---:|
| Nonzero basal plasticity | 10/10 | +.01299 to +.01541 |
| Brief neuron-delivered modulator | 10/10 | +.26192 to +.34364 |
| Same burst, rate receptor disabled | 10/10 | Identical to basal condition |
| Continuous modulator | 0/10 | First ring spike missed at tick 536 |

The brief burst peaks at a rate multiplier of 840.47, then decays back toward
1. The effective baseline remains 1e-5. Recurrent weights change throughout the
study; the burst changes them roughly twenty times more than basal adaptation.
This demonstrates temporary acceleration of adaptation while preserving an
existing relational state. It does NOT yet demonstrate acquisition of new
content, indefinitely balanced adaptation, autonomous gating or embodiment.

## Verification and records

282 exploratory runs recorded 529,200 network ticks. Simulation/recording time
reported by the runs totals 147 seconds, excluding analysis, setup and tests.
One simulation worker was used. Recorded artifacts occupy approximately 107 MB;
there are no generated videos or restarted live servers.

Records are under `active-inference/.live/research/20260908_composition_*`.
Every run has its configuration, resolved classes/parameters, source hashes,
seed/RNG provenance, complete tick records and descriptive summary. Queues,
delivered input vectors and terminal gains are included. These are observational
records, not an arbitrary-checkpoint restoration API.

`composition_analysis.py` streams the traces, checks their recorded-state digest,
reconstructs spike lists independently of the summaries and identifies the first
missing expected event. In the gated confirmation alone, it checks 83,528
postsynaptic and 62,648 retrograde updates against the implemented equations.
Maximum residuals are 9.71e-9 and 8.18e-8. It independently reconstructs rate
gain from each cell's preceding local modulator state.

Paired observed/unobserved runs have identical state digests for frozen and
adaptive conditions. Ten intervention-driver controls reproduce the original
baseline state sequence exactly. Tests explicitly reject a falsified summary,
do not treat silent readers as success, and distinguish re-cueing from retention.
Ten experiment/instrument tests and six cell-extension tests pass.

To rerun, from active-inference, choose a new output path:

    .venv/bin/python -m simulations.active_inference.experiments.composition_plasticity_gate --output .live/research/NEW_RUN
    .venv/bin/python -m simulations.active_inference.experiments.composition_analysis .live/research/NEW_RUN --output .live/research/NEW_ANALYSIS.json
    uv run --no-sync --with pytest python -m pytest tests/test_composition_probe.py -q

From neuron-model:

    .venv/bin/python -m unittest discover -s tests -p test_plasticity_rate.py -v

## Research direction, not a universal failure claim

This tests a deliberately sparse, autonomously recurring, unmodulated motif,
then one controlled rate-gate extension. It does not characterize all PAULA
architectures or require organisms to maintain every rhythm indefinitely.
An appropriate transition to silence can be useful; it fails this specific
persistent-state preparation. Silent synaptic memory would need a different
recall experiment.

The next experiment should make modulator release conditional on a neurally
available mismatch or novelty signal, test acquisition of a second state without
destroying the first, and compare closed-loop release against rate-dose-matched
open-loop pulses. A positive scalar learning-rate multiplier does not by itself
provide a restoring force for persistent drift in an unchanged timing regime.
The question is whether the coupled neural regulation changes that regime while
retaining useful adaptation, not how small we can make eta to pass a short test.

This direction connects to biological work on multiple interacting plasticity
processes. Zenke, Gerstner and Ganguli discuss neuromodulatory gating and the
timing requirements of compensatory feedback in their
[2017 review](https://doi.org/10.1016/j.conb.2017.03.015).
Zenke, Agnes and Gerstner's
[2015 model](https://www.nature.com/articles/ncomms7922) combines Hebbian,
heterosynaptic, homeostatic and consolidation mechanisms to form and recall
assemblies despite ongoing plasticity. Neither paper validates our extension.
They motivate testing interacting regulators instead of treating persistent
activity, plasticity, and consolidation as interchangeable achievements.
