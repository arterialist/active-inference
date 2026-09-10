# Connectome-grounded fly circuits

The active research objective is to reconstruct identified fruit-fly circuits
with PAULA, reproduce specified experimental findings, then reunite circuits
through their measured projections and test learning and adaptive coordination.
Anatomical preservation, assumed dynamics and demonstrated function are separate
claims. Earlier custom organisms and C. elegans remain intact.

## First preparation

The initial preparation is the left adult mushroom-body KC/APL circuit with its
actual antennal-lobe projection-neuron providers. Selection uses FlyWire 783 and
annotation release 2.1.0. It includes all directed pairs among selected cells,
including KC recurrence and reciprocal PN connections. It does not replace the
graph with a textbook circuit diagram.

| Retained anatomy | Neurons or directed pairs | Counted synapses |
| --- | ---: | ---: |
| Kenyon cells | 2,580 neurons | |
| APL | 1 neuron | |
| ALPN providers | 174 neurons | |
| Internal connections | 179,282 pairs | 514,524 |
| Incoming cut boundary | 72,253 pairs | 317,156 |
| Outgoing cut boundary | 135,892 pairs | 620,579 |

The boundary touches 9,211 additional neurons. Their identities, annotations
and incident connections are retained, but those neurons do not run in this
preparation. Incoming boundary ports remain present and undriven; outgoing
boundary terminals remain present and disconnected. No boundary drive is
imputed. Missing activity cannot be mistaken
for evidence that these pathways are physiologically dispensable.

The internal counts include 203,202 KC→KC, 179,488 ALPN→KC, 60,561 KC→APL,
53,486 APL→KC, 13,157 ALPN→ALPN, 1,559 ALPN→APL, 1,613 APL→ALPN and
1,458 KC→ALPN synapses. These are neuron-pair aggregates, not spatially located
PAULA contacts. The left APL identity is `720575940624547622`.

## Evidence and limits

The [PN/LN paired transmission assay](PN_LN_ELECTRICAL_2026-09-10.md) separates
chemical release from an explicitly hypothetical electrical contact. Forty-eight
recorded conditions show subthreshold voltage transfer, finite-window direction
differences and feedback through a nonspiking partner. These are model findings,
not evidence that the selected FlyWire pair has the assumed gap junction.

The [DL5 antennal-lobe reunion](ANTENNAL_REUNION_2026-09-10.md) now adds the
actual ORN and local-interneuron neighborhood to the original PN/KC/APL graph.
Six full-tick recordings separate short-term terminal depression, recurrent
recruitment, a single-cell release lesion and transmitter-polarity uncertainty.
One identified LN's release is necessary for the first train's broad recurrent
regime in this model. Blocking it is a diagnostic, not a reconstructed solution;
the intact circuit still lacks physiological and behavioral acceptance.

[Lin et al., 2014](https://doi.org/10.1038/nn.3660) provides the first experimental
targets. Its Figure 3 tests KC activation and KC-output blockade against APL
responses. Figures 4 and 5 test APL activation/blockade, KC activity, sparseness
and odor-response correlations. The behavioral experiments distinguish learned
discrimination of similar mixtures from dissimilar odors. These observations
were obtained with calcium/release imaging and behavior, not PAULA spike counts.
The current execution probe has not reproduced those assays.

[Amin et al., 2020](https://doi.org/10.7554/eLife.56954) shows spatially restricted
activity and inhibition within non-spiking APL. This prevents interpreting a
single global APL integrator as a complete cellular reconstruction. The current
pair table has no synapse locations. A separate spatial snapshot now locates
all 128,435 linked APL contacts and exactly matches all 6,315 incident pair
counts. It also retains 42,572 connectors with no linked partner. See
[the spatial findings and reproduction commands](SPATIAL_FINDINGS_2026-09-10.md).
An opt-in passive-cable extension now uses those contacts within one APL neuron.
The global graded model remains the default reference. Full-tick comparisons
and a numerical refinement expose early local release saturation rather than
establishing a physiological fit. See
[the branch-local APL findings](LOCAL_APL_FINDINGS_2026-09-10.md).

The [Amin spatial benchmark](AMIN_SPATIAL_BENCHMARK_2026-09-10.md) now reproduces
all nine published straight-backbone predictions and compares the measured
stimulus/calcium profiles with a schematic conservative cable. A uniform-input
diagnostic exposes different spatial operators despite the same nominal decay
length. This is not yet a physiological fit of the FlyWire APL or an independent
rerun of the authors' anatomical predictions.

The [registered APL follow-up](REGISTERED_APL_FINDINGS_2026-09-10.md) recovers the
authors' saved 181,741-node tree and reproduces every node-to-region assignment.
Its full cable traces expose underpredicted vertical propagation. An independent
anatomical reference calculation also still differs from the published curves.
Registration recovery and numerical verification are established; a physiological
fit is not. The code keeps this saved specimen separate from FlyWire.

The [input-physics follow-up](APL_INPUT_PHYSICS_2026-09-10.md) tests shared
electrical parameters and current versus conductance drive. Full-tick controls
show that a better regional fit can largely survive removal of axial coupling.
Distributed stimulus transduction must therefore be separated from propagation
before interpreting the fit as a cellular repair.

The [regional connected-network test](APL_REGIONAL_FINDINGS_2026-09-10.md)
uses public neuropil meshes as readouts without altering the graph. KC-output
blockade nearly removes vertical-lobe local release while calyx release persists,
at two electrical-coupling settings. Full ticks expose saturation and negative
PN input that prevent calling this a calcium-physiology replication. A whole-cell
average cannot adjudicate the regional feedback claim.

The [PN input controls](PN_INPUT_CONTROLS_2026-09-10.md) now trace inhibitory PNs
recruited without direct experimental current, through their actual input ports
and onward to APL. The regional distinction survives this and a VP-label input
control. A fixed-recorded-input cable decomposition attributes the blocked
vertical lobe's negative voltage to PN inhibition, while leaving a large
regional feedback effect. It is not a connected-network lesion prediction.

Current distinctions:

| Item | Evidence status |
| --- | --- |
| Neuron identity and directed pair counts | Imported from the pinned FlyWire-derived tables. Anatomical reconstructions have their own uncertainty. Weak one-count pairs are retained. |
| Cell labels | Pinned annotation release. `cell_class` identifies all KCs; a nonempty `cell_type` is not required. |
| APL spatial anatomy | Public CATMAID m783 snapshot, 360,309 tree nodes. Linked contact counts match the pair table exactly. Used by the opt-in local cable; coordinates, source rows and open connectors remain separately auditable. |
| Neurotransmitters | Predicted transmitter/confidence and known-transmitter/source fields remain separate. |
| Forward sign | Uses the Shiu table's `Excitatory` model column, not a measured receptor effect. |
| Initial strength | Assumed linear conversion, `sign × count × 0.02`, on postsynaptic information only. No per-edge fitting or clipping. |
| Timing | One network tick plus two dendritic ticks. No physical duration assigned to a tick; no inferred axonal length. |
| PAULA dynamics | Ordinary PAULA, with global graded APL by default or an opt-in local passive cable. Shared integration, thresholds and cable electrical properties are assumptions, not fly measurements. |
| Adaptation | Native postsynaptic and retrograde rules remain enabled at weak positive rates. They are not a validated fly learning rule. Graded APL inherits spike-timing machinery without a spiking clock. |
| Experimental input | A separate zero-delay input per selected neuron. It is explicitly not an anatomical synapse. |
| Composition | Global biological/PAULA identities survive different cuts. Cut-edge records support future reunion. State-preserving reunion has not been implemented or tested here. |

Biological root IDs are decimal strings in JSON. PAULA uses the source table's
global index, so a cut does not renumber neurons. Terminal/port numbers are
assigned by original source-row order across all incident connections,
including the boundary. A different cut from the same pinned source therefore
keeps port identities and the input-count-dependent plasticity bounds. PAULA
has 4,096 addresses in each port namespace.
The adapter fails before construction if its graph plus experimental inputs
does not fit. It never truncates connections to make a run possible.

## What actually ran on 10 September 2026

The importer verified all three pinned file hashes and checked all 15,091,983
connectivity rows against the global root/index map, positive counts and source
sign arithmetic. Those rows contain 54,492,922 counted synapses. It checked
duplicate directed pairs across the entire retained graph and boundary, not
across discarded rows. The extracted artifact occupies about 14 MiB locally.

Two 32-tick execution probes used the same graph and current schedule. At tick
4, an artificial pulse stimulated 32 dispersed KCs; at tick 20 another pulse
stimulated APL. This is not an odor presentation, thermal activation model or
learning task. There is no biological seed replication claim for this one
specimen and deterministic input.

The actual trace shows APL quiet through tick 6 and releasing at tick 7 with
`S=0.6498000622`, `O=0.0064980006`. A recorded receiving port then has positive
input `0.0064980006` and negative local potential `−0.0023392802` at tick 8.
This verifies a functioning signed route through the imported graph, not
restored odor coding or a physiological fit.

The first run used both learning rates at `1e-8`. It changed 4,765 postsynaptic
coefficients, including 33 artificial drive inputs, but no terminal coefficients.
A direct native-rule regression
shows that small float32 return updates can round away repeatedly at unit
terminal strength. The second run changes only `eta_retro` to `1e-6`; 4,732
terminal coefficients then change, while remaining positive during this probe.
The first run is retained, not replaced. This rate change establishes effective
numerical adaptation in this short preparation, not useful learning or long-run
sign stability. The shared neuron equations were not edited.

A third run retains incoming boundary slots and outgoing boundary terminals in
the instantiated cells. Without them, cutting this circuit changes PAULA's
input-count-dependent plasticity-window bounds in every selected neuron. For
APL, the initial upper bound changes from 8,163 to 9,633 ticks. The final
preparation has 254,290 input slots including experimental drive, and 315,174
terminals. Its largest cell has 3,210 anatomical input partners and 3,105 output
partners, below the address limits. The early KC→APL response and the counts
of changed coefficients remain the same in this short test. Future reunion
still changes incoming activity and return signals; preserved slots do not
guarantee preserved function.

Each probe records all selected cells and receiving ports every tick: actual
arriving input vectors, native local potentials, soma fields, neuromodulator
states, postsynaptic weights, terminal information coefficients and queue
counts. Explicit ID arrays define every column. The trace omits complete event
queues and retrograde error vectors; it is not a restart checkpoint. An exact
small-network comparison verifies that recording leaves native trajectories
and final weights unchanged. A separate uninstrumented run of the actual
second preparation also matched all 617,120 recorded soma values and every
final postsynaptic and terminal coefficient exactly.

The second probe took about 7 seconds including loading and trace compression,
with peak resident memory about 764 MiB. Its compressed trace is 6.4 MiB. This
rose to 9.2 seconds, about 1,016 MiB resident memory and a 10.8 MiB trace when
boundary ports were retained. These are short-run observations, not long-run
scaling benchmarks. All processes
finished. No live agent servers or large simulation suites were started.

## PN-drive interventions

The first matched intervention course is complete. Five recordings cover the
default strength, a stronger shared count conversion, KC-output blockade,
APL-output blockade and APL activation. They retain the same measured graph,
boundary slots and positive native adaptation. See
[the findings and tick-level causal witness](INTERVENTION_FINDINGS_2026-09-10.md).

The stronger conversion recruits KCs and produces inhibitory feedback. This is
an operating-range result, not a reproduction of odor discrimination. In one
weak-drive window, KC blockade delays a PN spike and a downstream KC spike,
reducing the window's count despite increasing KC activity later. Averages
alone would miss that preserved KC→PN→KC route.

`intervention_probe` records all inputs, native local potentials, soma fields
and information coefficients in 16-tick chunks. `intervention_analysis` checks
the input schedule, actual experimental-port arrivals, silent cut boundary,
blockade targets, continuous state and independently recomputed summaries.
Comparisons require matching parameters, anatomy, port identities and initial
recorded state. These are diagnostic checks, never biological acceptance.

Both intact 224-tick courses also replayed exactly without tick instrumentation,
including every recorded postsynaptic and terminal coefficient, not just their
endpoints. The intervention recordings omit complete event queues and cannot
be used as executable checkpoints. Release blockade removes forward events;
the blocked cell's `O` can remain positive. It must not be displayed as delivered
release.

## Next physiological test

The Prisco et al. primary tables have now been acquired and audited. They
provide an animal-level APL-blockade target, but only nine glomerulus labels,
no physiological time series, and three unresolved pooled-KC count mismatches.
Twenty of the 174 selected PNs are exact-label candidates, not individually
measured inputs. See [the primary physiology audit](PRISCO_PHYSIOLOGY_2026-09-10.md)
and `prisco.py` for the read-only importer, identity mapping and reanalysis.

The branch-local implementation and matched release-block controls now execute,
but the first response already caps some local releases. Finer time steps do
not remove that effect. Current density, local release, spatial stimulation and
observation models need physiological constraints before more inhibition can
be interpreted as a better fit.

Obtain an empirically grounded odor-to-PN input mapping before claiming odor
decorrelation. Test multiple odors and similar mixtures, with liveness and
response-strength controls so silence cannot count as sparse discrimination.
Use shared cell-class parameters across interventions. Any failure must be
examined against cut-boundary input, physiological uncertainty, missing local
APL processing and PAULA equations before changing anatomy or fitting edges.
The regional comparison now supplies separate calyx and lobe measurements;
the input-subset controls retain that distinction. The next input experiment
must address temporal structure and empirical odor selectivity, not treat the
broad artificial ALPN course as an olfactory stimulus. Keep both regions and
actual recurrent projections.

The [DL5 primary-current audit](PN_CURRENT_PHYSIOLOGY_2026-09-10.md) now supplies
12 time-resolved unitary ORN→PN current traces and binned intrinsic ramp
responses from Gugel et al. The native receptor-to-hillock witness retains all
ports of the identified DL5 PN and separates one-tick arriving current from
lingering somatic voltage. Receptor-current kinetics and membrane integration
must be constrained separately. These data do not yet provide an odor drive
for the full selected PN population or calibrate a physical PAULA clock.

The [current-kernel and reunion experiment](PN_CURRENT_REUNION_2026-09-10.md)
adds an opt-in, per-port PAULA current state and tests its fitted tail with
peak-matched and charge-matched controls. The connected preparation preserves
every pair. Exact target-PN replay exposes APL-driven spike-time changes that
the unchanged total spike count conceals. Effective current shape is partly
constrained; physiological gain, conductance, depression and odor coding are
not yet reproduced. `PNCurrentKernel` must be explicitly selected for one
identified PN and presynaptic type; all default preparations remain unchanged.

The [joint intrinsic-current calibration](PN_INTRINSIC_CALIBRATION_2026-09-10.md)
now tests the published DL5 current ramp through actual PAULA ticks. Electrical
injection bypasses plastic receptor inputs. A close control-average fit still
has substantial cell-held-out errors, and changing current gain and integration
together changes the same diagnostic train from 69 to 13 spikes. This is an
effective candidate, not measured membrane biophysics. Full source-bin and
tick records remain available.

The [closed-loop current-step study](PN_CURRENT_STEPS_2026-09-10.md) now applies
that candidate to the identified PN only, inside the unchanged 2,755-cell graph.
Isolated, intact and APL-blocked courses share an identical target spike raster
across four selected one-second steps. APL changes target voltage and one input
coefficient; its blockade recovers the isolated voltage exactly. Native terminal
adaptation continues in both connected conditions. Feedback supplies only about
0.2% of injected charge, and no other PN or KC spikes, so this is not a
strong-recruitment, odor or learning acceptance. All 27,000 target ticks pass
exact current/state/postsynaptic-weight replay. Synaptically driven reunion
with depression and reconstructed ORN/LN input remains the next test.

Learned discrimination remains a later benchmark requiring actual learning and
neural consumers, not a decoder trained to call the response successful.

## Reproduce

From `active-inference/`, the optional `flywire` extra supplies PyArrow. Existing
dependencies and the C. elegans connectome dependency keep their locked versions.

```sh
uv sync --extra flywire --extra dev
uv run python -m simulations.drosophila sources
```

Download the three files printed by `sources` to a local directory. Downloads
are never implicit. Then use a new output directory:

```sh
uv run python -m simulations.drosophila prepare \
  .live/research/flywire783 .live/research/flywire783/kc-apl-left-v1
uv run python -m simulations.drosophila.execution_probe \
  .live/research/flywire783/kc-apl-left-v1 \
  .live/research/flywire783/kc-apl-left-boundary-preserved
uv run python -m pytest tests/test_drosophila_connectome.py -q
```

The full intervention course and its independent replay check:

```sh
uv run python -m simulations.drosophila.intervention_probe \
  .live/research/flywire783/kc-apl-left-v1 \
  .live/research/flywire783/pn-course-new-intact \
  --condition intact --weight-per-count 0.075
uv run python -m simulations.drosophila.intervention_analysis \
  .live/research/flywire783/pn-course-new-intact \
  .live/research/flywire783/pn-course-new-intact/analysis.json
uv run python -m simulations.drosophila.intervention_analysis \
  .live/research/flywire783/pn-course-new-intact \
  .live/research/flywire783/pn-course-new-intact/unobserved-verification.json \
  --source .live/research/flywire783/kc-apl-left-v1
uv run python -m pytest tests/test_drosophila_connectome.py tests/test_drosophila_interventions.py -q
```

Repeat the recording with a distinct output directory and `--condition`
`kc_release_block`, `apl_release_block` or `apl_activation`, keeping
`--weight-per-count 0.075`. Analyze each record on its own, then compare using
`--reference PATH_TO_INTACT_RECORD`. The `--source` replay option runs the
network again; inspection and comparison only read existing files. None starts
a live server. The new 13 tests cover release lesions, chunk boundaries,
current attribution, misleading metadata, source-file sampling and exact
uninstrumented state agreement, bringing the fly package to 39 tests.

Existing outputs are never overwritten. To repeat the rate comparison with the
current boundary-preserving builder, call
`run_probe(graph, new_output, Dynamics(eta_retro=1e-8))`. This does not recreate
the earlier omitted-slot construction; its original recordings remain local.
There are 26 tests covering identities, complete cut extraction, source/schema
failures, duplicate pairs, bounds, inhibition, latency, active adaptation,
round-trip integrity and observational equivalence. Fixture tests are software
checks, not physiological acceptance tests.

## Source lineage

`connectome.SOURCES` pins the exact URLs and SHA-256 digests. The connectivity
and global index come from the [Shiu research repository](https://github.com/philshiu/Drosophila_brain_model/tree/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960).
Its original [2024 paper](https://doi.org/10.1038/s41586-024-07763-9) used
materialization 630; this preparation uses the repository's separately supplied
783 table and does not claim an exact replication of their original simulation.
Their Brian2 model is not imported or run.

Annotations come from [FlyWire annotation release 2.1.0](https://github.com/flyconnectome/flywire_annotations/tree/ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0),
associated with [Schlegel et al., 2024](https://doi.org/10.1038/s41586-024-07686-5)
and [Dorkenwald et al., 2024](https://doi.org/10.1038/s41586-024-07558-y).
These sources concern an adult female brain, not all adult fly tissues and not
the individual animals used in the physiological experiments. No other specimen
or release is silently merged.

The Shiu repository includes an MIT license. The annotation repository's
release-level data licensing still needs explicit confirmation before
redistributing derived tables. Raw inputs and derived anatomical artifacts
remain local under ignored `.live/`; the published source contains the importer,
tests, provenance references and findings only.
