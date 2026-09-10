# PN/LN chemical and electrical transmission

The paired assay exposes a missing transmission channel in the chemical-only
adapter and a limit of the new passive electrical extension. Subthreshold
voltage does not leave a silent ordinary PAULA cell. An explicit electrical
contact transfers that voltage, but constant passive conductance cannot by
itself produce polarity-dependent transmission. None of the results establishes
an electrical contact for the selected FlyWire pair or repairs the full circuit.

## Anatomical and physiological evidence

The assay selects DL5_adPN `720575940617207185` and lLN2T_e
`720575940633483807`. This LN was the first positive-sign interneuron recruited
in the earlier full-network trace. Blocking its release prevented the first
train's broad recruitment. Selection therefore follows the observed failure,
not a search for an isolated pair that behaves well.

The cut preserves both directed chemical pairs. PN→LN has 94 counted contacts,
source row 3267788; LN→PN has 36, row 12261797. It also retains 967 incoming
boundary pairs with 8,916 contacts and 2,266 outgoing boundary pairs with 14,669
contacts. The 2,355 boundary cells are absent and undriven. Input counts, native
plasticity bounds, cell IDs, port order and chemical signs are unchanged.

[Yaksi and Wilson, 2010](https://doi.org/10.1016/j.neuron.2010.08.041) found
electrical eLN→PN transmission of depolarization and hyperpolarization. Blocking
chemical transmission did not remove that response. PN→eLN transmission had
electrical and chemical contributions. Their injections lasted 500 ms, recurred
every 5 s for 40–50 trials, and were adjusted per cell for approximately ±40 mV
deflections. The unstimulated response was low-pass filtered at 50 Hz before
computing the ratio of voltage changes. Somatic transfer could underestimate
local junction coupling. Depolarization and hyperpolarization were not equally
effective. Their gap-junction mutation also affected some chemical connections,
so a developmental mutation is not equivalent to an acute electrical lesion.

The paper's driver-defined eLN population is not identified with this exact
FlyWire root. The LN's predicted transmitter is low-confidence and its curated
transmitter field is empty. The assay therefore tests a declared possible
mechanism on an identified chemical pair, not a cell-matched replication of
those recordings. No measured coupling coefficient is fitted here.

## Explicit dynamical assumptions

The opt-in extension is documented in the sibling neuron-model repository at
`neuron/extensions/experimental/ELECTRICAL.md`. Each cell receives its partner's
pre-tick somatic voltage and computes passive junction current locally. The
network wrapper only transports synchronous endpoint snapshots. Chemical and
electrical pathways coexist; no chemical edge is converted into a gap junction.
No labels, phase estimates or stabilizing controller enter the cells.

DL5 retains its fitted current tails, ORN gains and effective lambda of
60.4891 nominal ticks. The LN retains the unfitted lambda of 20 ticks and the
0.075/contact chemical gain. Relative leaks are both one in abstract units.
Conductances 0, 0.003, 0.03 and 0.3 are declared sensitivity values, not estimates
from anatomy or physiology. Native receiving and terminal learning remain on.

Each condition has two 500-tick current steps separated by 4,500 recovery ticks,
after 500 baseline ticks. Each source is tested separately at −0.5, +0.5 and +2
model-current units, with and without chemical release. There are 48 courses,
504,000 network ticks and 1,008,000 recorded cell ticks. Two deterministic repeats
are not biological replicates and do not reproduce the paper's trial count.
The clock is nominal. LN current and voltage scales are not calibrated.

The model release block filters outgoing chemical events only. It retains cell
activity and native return events. It is neither cadmium pharmacology nor a
gap-junction mutation. Every condition starts from the same constructed state;
weights continue adapting across both trials within that condition.

## What the paired results establish

Neither cell drives a subthreshold response in its partner at zero electrical
conductance. Nonzero conductance transmits both polarities. Positive and negative
subthreshold responses are exactly sign-symmetric for a given source.

| Assumed conductance | PN→LN 500-tick transfer ratio | LN→PN 500-tick transfer ratio |
| --- | ---: | ---: |
| 0 | 0 | 0 |
| 0.003 | 0.002856 | 0.002616 |
| 0.03 | 0.027848 | 0.025575 |
| 0.3 | 0.222897 | 0.208499 |

These are ratios of mean model-voltage changes during the first subthreshold
step. Different membrane time constants produce different finite-window ratios
despite reciprocal conductance and equal assumed leaks. This is filtering,
not junction rectification. The independent steady-state check gives
`S_receiver/S_source = g/(1+g)` for equal leaks.

A spike-silent PN can still participate in feedback. At g=0.3, injecting +2
into the LN produces 33 LN spikes with chemical transmission and 31 with
chemical release blocked. The PN emits zero spikes in both cases. Chemical
LN→PN input raises PN voltage and reduces the LN's net electrical current loss,
from 67.652 to 49.363 integrated model-current units during the first step.
The source spike trains first differ at tick 562. Both repeats have the same
spike counts, although native learning changes the second intact voltage trace.
An analysis that regarded a nonspiking PN as functionally absent would miss this
loop. This is a demonstrated model effect, not a novel biological discovery.

No case sustains firing into the 4,500-tick recovery. This does not establish
whether the pair would sustain the earlier intact network's spike history.
Both the stimulus and boundary conditions differ. A matched connected comparison
is needed before attributing that difference to circuit isolation.

PAULA resets somatic voltage on each spike. That trace is not the paper's
recorded membrane waveform. The analysis therefore refuses to label spiking
epochs with a physiological coupling coefficient. The extension still lacks
dendritic junction position, action-potential waveform, active conductances and
electrical rectification. Subthreshold transfer is necessary evidence for this
mechanism, but insufficient for physiological reproduction.

## Verification and retained evidence

The calibration was rerun after extending the builder. All 2,047,194 recorded
values match the preceding calibration exactly. The ordinary-cell and zero-gap
paired courses match in 6,262,248 recorded values. A second run of every paired
condition matches 25,048,992 values, including all recorded chemical inputs,
internal-pair weights, event counts and final all-port weights.

Independent checks cover current conservation, dissipation, membrane integration,
spike/reset decisions and declared release blocks. A regression test modifies a
receiving weight and updates the artifact hash; replay still rejects it. Hashes
identify records but do not establish that their contents follow the equations.

The raw paired evidence is 19 MiB. It retains both somata and currents at every
tick, all active chemical input channels, internal-pair receiving and terminal
coefficients, and initial/final all-port weights. It is not a complete queue or
intracellular checkpoint. Source data and derived anatomical tables remain local
under `.live/research/flywire783/`; no unresolved-license anatomy is republished.

## Reunited circuit result

Two further experiments restore all 3,005 selected cells and all 213,855 measured
internal pairs. They use the original first ORN train, with no depression, and
compare the same hypothetical PN/LN contact at g=0 and g=0.03. Both run for 1,600
ticks, including the 1,000-tick first train and 400 recovery ticks. They do not
test the later 50-Hz train or establish long-run stability.

The zero-gap run exactly reproduces 20,923,200 values from the corresponding
prefix of the original recording. The nonzero gap changes only the two endpoint
somatic traces. The first voltage difference occurs at tick 204. Maximum voltage
changes are 0.0024474 in the PN and 0.0097662 in the LN. No spike changes anywhere
in the network. PN input histories, recorded PN receiving weights and all LN
event counts remain identical.

The added current reaches an absolute peak of 0.0353564 model units. The PN and
LN chemical-current peaks are 80.5966 and 214.8384. These peaks alone would not
prove functional irrelevance, but the complete spike comparison does show that
the perturbation never crosses a firing boundary in this course. Other cells
receive unchanged chemical outputs. The 0.03 conductance tested here is one
assumption, not a bound on all possible electrical effects.

| Window | DL5 spikes | ALLN spikes | ALPN spikes | KC spikes | Maximum APL release |
| --- | ---: | ---: | ---: | ---: | ---: |
| First train, both conditions | 290 | 34,037 | 21,485 | 13,621 | 1.0, at cap |
| First 400 recovery ticks, both | 73 | 17,089 | 10,734 | 6,638 | 1.0, at cap |

The extension therefore passes the isolated passive-transfer test but does not
resolve the connected recruitment failure at the tested conductance. This is
not a physics/host handoff failure: both PN histories replay exactly from the
recorded receptor inputs and partner voltage. Both endpoint membrane equations
pass the independent float32-bounded audit without clipping or ambiguous spike
decisions. LN chemical current is measured here, not replayed from all its
receptors. There are 3,200 exact PN replay ticks and 6,400 endpoint audit ticks.

The next constraint belongs on chemical PN/LN transfer and cell-specific
excitability in the recurrent neighborhood. Increasing an unmeasured junction
until the network settles would not establish the missing physiology. Electrical
coupling remains available as a separately testable pathway, while the
driver-to-FlyWire identity mapping and quantitative biological fit remain open.

Calibration: `dl5-intrinsic-electrical-source-20260910/`.
Paired courses and exact replay: `dl5-ln-paired-electrical-20260910/`.
Connected onset comparison: `dl5-electrical-reunion-20260910/`.

## Reproduction

From `active-inference/`, with the existing pinned inputs and a new output path:

```sh
uv run python -m simulations.drosophila.paired_recording \
  .live/research/flywire783/dl5-antennal-expansion-20260910/graph \
  .live/research/flywire783/dl5-intrinsic-electrical-source-20260910/analysis.json \
  .live/research/flywire783/dl5-current-fit-controls-20260910/analysis.json \
  NEW_PAIRED_OUTPUT
uv run python -m simulations.drosophila.electrical_analysis pairs \
  NEW_PAIRED_OUTPUT \
  .live/research/flywire783/dl5-intrinsic-electrical-source-20260910/analysis.json \
  .live/research/flywire783/dl5-current-fit-controls-20260910/analysis.json \
  NEW_PAIRED_OUTPUT/replay.json
```

Recorded source hashes bind each run to its implementation. Historical records
remain tied to their original source revision. Changing code requires a new
recording or an explicit comparison, not rewriting those manifests.

For the connected comparison, use the existing `orn_train` command with
`no_depression --stop-tick 1600 --junction 720575940617207185
720575940633483807 G`, once with G=0 and once with G=0.03, in distinct output
directories. The extension-aware `electrical_analysis reunion` command takes
the graph, intrinsic fit, tail fit, comparison directory containing `zero_gap/`
and `gap_003/`, original complete reference directory, and a new output JSON.
The older chemical-only analyzer rejects electrical records explicitly.

Verification after this change: 188 fly-package tests and 80 neuron-model tests.
These test software and recorded equations. They do not certify fly physiology,
odor discrimination, learning behavior or integrated adaptive coordination.
