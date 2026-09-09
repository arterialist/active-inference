# A current tail changes gain, spike timing and recruited feedback

An opt-in PAULA current extension now carries a fitted DL5 current-tail
hypothesis into the existing 2,755-cell PN/KC/APL preparation. It preserves
all 179,282 internal directed pairs and every incident boundary port. This is
an input-mechanism experiment, not a physiological or behavioral acceptance.
No default agent or other organism selects this extension.

## Physiological constraint and its limits

The [Gugel et al. source audit](PN_CURRENT_PHYSIOLOGY_2026-09-10.md) supplies
12 peak-aligned, trial-averaged DL5 uEPSC traces. We fit signed,
baseline-subtracted postpeak current over 0–100 ms. Each cell is normalized by
its measured peak and has equal weight. Peak amplitude, rise, latency and
individual-trial variability are not predicted. This analysis is exploratory,
not preregistered, and does not treat time points as biological replicates.

| Effective tail | Shared fit, all 12 cells | Mean leave-one-cell-out RMSE | Solvent-trained, exposed-cell test RMSE |
| --- | --- | ---: | ---: |
| One exponential | 14.363 ms | 0.061854 | 0.065578 |
| Two exponentials | 10.293 and 48.249 ms; peak fractions 0.81894 and 0.18106 | 0.052388 | 0.058808 |

RMSE is in measured-peak-normalized current units. Two components reduce
mean held-out error by 15.3%, improving 8 of 12 cells and worsening 4. Neither
time constant hits the declared 0.1–1000 ms bounds. Baselines ending at 30 or
40 ms retain the ordering. Cross-condition transfer is not evidence that the
exposure conditions are equivalent. This modest shape improvement does not
establish two molecular receptor populations.

[Nagel, Hong and Wilson (2015)](https://pubmed.ncbi.nlm.nih.gov/25485755/)
provide independent motivation for testing multiple time scales. Their Figure 2
separates fast and slow ORN→PN current components pharmacologically and shows
different depression during trains. These are different glomeruli and assays;
we have not reproduced their pharmacology or depression. Their fitted slow
conductance also depends on the model and fitting protocol.

Somatic current is not a direct readout of a local receptor. [Gouwens and
Wilson (2009)](https://doi.org/10.1523/JNEUROSCI.0764-09.2009) show strong
compartment-dependent attenuation and imperfect somatic voltage control.
[Kazama and Wilson (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2429849/)
also distinguish direct fast input from a slower lateral contribution in their
nerve-stimulation preparation. Consequently, the fitted kernel is an effective
hillock-current hypothesis. It is not a molecular or recording-electrode model.

## Separate timing from added charge

The extension lives in `neuron.extensions.experimental.input_current`. Each
declared port retains its own current components. One queued release adds
current once; its tail causes no synthetic receptor events or extra plasticity
updates. Spiking, reset, thresholds, native delays and positive adaptation
remain inherited. The experimental current port bypasses filtering.

For the fitted fractions and a provisional 1 ms/tick conversion, preserving
the impulse's peak multiplies its integrated current by 17.67230. The
area-matched version scales the same waveform down to conserve charge.
This provisional clock converts the tail only. It does not calibrate the
20-tick membrane constant, refractory period, pA scale or firing rate in Hz.

The identified receiving cell is DL5_adPN `720575940617207185`, PAULA index
30002. Source row 66370 binds ORN_DL5 `720575940604352689` to port 0 with 44
counted contacts. We impose one release every five ticks for ticks 0–199.
The ORN itself and an odor stimulus are not simulated. The isolated preparation
retains all 244 incoming and 729 outgoing pairs. Each condition is recorded
for 800 ticks at three provisional clocks, 0.5, 1 and 2 ms/tick.

| Condition at provisional 1 ms/tick | Integrated current during recording, model units | Isolated PN spikes |
| --- | ---: | ---: |
| Native impulses | 119.130 | 0 |
| Same-shape tail, area matched | 119.130 | 0 |
| Tail, peak matched | 2105.300 | 69 |
| Native impulses scaled to match tail charge | 2105.304 | 40 |

Small charge differences include ongoing learning and the explicitly reported
unrecorded tail. Changing the provisional clock moves peak-matched spike
counts to 96 or 39. Current-injection controls produce the same 33 spikes in
every variant. There is no intrinsic-current calibration hidden in the filter.

The charge-matched comparison exposes a second effect beyond gain. Native
impulses produce mean pre-reset voltage 2.63163; the filtered train produces
1.37562. Spike reset discards that voltage in PAULA. The impulse condition
therefore loses more input to each reset and produces fewer spikes despite
similar integrated current. This is a property of the current model's reset,
not evidence that the filtered condition matches a real PN's firing.

## What changes when the circuit is reunited

The connected experiment uses the full existing graph and its spatial APL
extension. All other current kernels remain native. The same 200-tick input
course runs for 320 ticks. Every cell's S, O and average activity are recorded;
DL5 input buffers, per-port arriving impulses, effective currents, component
states, input weights and terminal weights are retained tick by tick. APL
terminal releases are recorded too. Full APL compartment arrays and other
cells' per-tick synaptic weights are omitted, explicitly, to keep this diagnostic
small. It is not a checkpoint.

| Condition | DL5 spikes | Other PN spikes | KC cells / spikes | Maximum local APL terminal release |
| --- | ---: | ---: | ---: | ---: |
| Native | 0 | 0 | 0 / 0 | 0 |
| Area-matched tail | 0 | 0 | 0 / 0 | 0 |
| Peak-matched tail | 69 | 8 | 1 / 2 | 0.10760 |
| Charge-matched native impulses | 40 | 0 | 0 / 0 | 0.08674 |

The 69-spike total conceals a timing change. Isolated spikes at 229, 239 and
253 move to 230, 240 and 255 in the connected preparation. APL input reaches
DL5 port 122 at receptor tick 13 and as current at tick 15, the first difference
from isolation. A recruited M_vPNml50, `720575940631630021`, reaches port 175
at receptor tick 72 and as current at tick 74. Both inputs retain the source
model's inhibitory sign. Their summed currents are -11.51858 and -0.54150.

Replaying every recorded input into a fresh isolated DL5 reproduces its
recorded S/O trajectory and final postsynaptic coefficients exactly in all
four conditions. In open-loop replay, deleting APL input recovers the isolated
S/O trace; deleting only the other PN input does not shift the spike raster.
The latter input can change pre-reset current without changing the recorded
post-reset S/O. This is why the record keeps current separately from spikes.
These are input-attribution interventions. They do not establish what a full
network with APL release blocked would do.

## What is and is not established

Established here: an explicit current-state mechanism, a cell-held-out shape
comparison, a gain control, a reset-loss explanation, and reproducible changes
in real connected neural consumers. Native adaptation stays active. In the
connected peak condition, three DL5 input coefficients and 178 terminal
coefficients change. A fit has not been converted into a claim of learned
discrimination, odor coding or complete PN physiology.

The next physiological constraint is joint current/voltage and train behavior.
We need to distinguish local synaptic conductance from somatic filtering,
constrain effective input charge and membrane response together, and include
the measured short-term depression and antennal-lobe inhibition. Adding a tail
while silently retaining the old peak/count gain is not an acceptable shortcut.
The geometry-and-physiology approach in [Tobin et al. (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5440167/)
is relevant to that next comparison. Missing ORNs and LNs must be reconstructed
from their full incident source records before treating them as reunited cells.

## Reproduction

From `active-inference`, use the verified Gugel extraction and graph already
described in the source audit. Each command refuses an existing output.

```sh
.venv/bin/python -m simulations.drosophila.pn_current fit CELLS_JSON NEW_FIT
.venv/bin/python -m simulations.drosophila.pn_current probe GRAPH NEW_FIT/analysis.json NEW_ISOLATED
.venv/bin/python -m simulations.drosophila.pn_current_reunion run \
  GRAPH NEW_FIT/analysis.json NEW_REUNION --spatial APL_SPATIAL_DIRECTORY
.venv/bin/python -m simulations.drosophila.pn_current_reunion analyze \
  GRAPH NEW_REUNION NEW_ISOLATED NEW_REUNION/replay-attribution.json
```

Retained local directories under `.live/research/flywire783/` are
`dl5-current-fit-controls-20260910`, `dl5-current-probe-controls-20260910`, and
`dl5-current-reunion-verified-20260910`. The earlier connected run is retained
for independent array-level repeatability comparison. Each connected record
is about 22 MiB. The first four-condition run took 105 seconds, with 1.00 GiB
maximum resident memory and no swaps. No live simulations were started.
The second four-condition run also took 105 seconds. Every recorded array
matches the first run exactly, and all four target-PN replays pass.

Verification: 165 fly tests and 67 neuron-extension tests pass. One unrelated
checkpoint test remains skipped for missing `cloudpickle`. Tests include
independent convolution, charge conservation, genuine held-out-cell exclusion,
unchanged anatomical bindings, exact replay and a false-voltage record whose
updated hash still cannot pass neural replay. These are mechanism and
instrumentation checks, not biological acceptance thresholds.
