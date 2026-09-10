# DL5 current shape and firing need a joint calibration

The fitted current tail can now be combined with an experimentally constrained
firing response in the same PAULA neuron. A single effective calibration fits
the control-average ramp curve, but does not explain the variation across
individual cells. It is a candidate model, not physiological acceptance or a
measured membrane parameter set. This initial study is isolated; the subsequent
[current-step reunion](PN_CURRENT_STEPS_2026-09-10.md) tests the candidate in
the connected preparation. Default agents remain unchanged.

## Recover the actual observation protocol

[Gugel, Maurais and Hong (2023), Figure 7C](https://doi.org/10.7554/eLife.85443)
specify a triangular current ramp at 4.5 pA/s, with firing measured in 50 ms
windows overlapping by 25 ms. The source workbook contains four DL5 PNs per
exposure condition. Its increasing-current rows run from 0 to 100 pA. The
0.1125-pA increments after the first nonzero sample agree with the stated
window stride, but the first increment is 0.100 pA. We preserve that axis.

The caption does not establish bin-center versus bin-edge alignment, the
number of averaged trials, or their spike times. Simulations therefore expose
all three alignment conventions. The decreasing branch is not present in the
acquired table. Its final 20 all-zero rows are not interpreted as a second
trial or as a command to silence the neuron.

For parameter proposals, we aggregate both data and predictions into 5-pA
intervals between 5 and 95 pA. This is an explicit analysis choice, not the
published estimator. It avoids the uncertain endpoints and reduces sensitivity
to an unknown phase within a 50-ms window. Every original measured bin remains
in the retained records. Each experimental cell keeps its own curve.

## An electrode is not a plastic synapse

`electrophysiology.CurrentElectrode` adds a predetermined current at PAULA's
membrane-current boundary. It creates no receptor inputs or retrograde events.
Native synaptic input and adaptation remain available. Under pure electrical
injection, the weights stay unchanged because no chemical inputs occur, not
because learning is disabled. Tests also deliver genuine chemical input during
injection and verify its delayed current, learning and return event.

The instrument is process-scoped and must not be attached to a running live
server. It restores the original neuron method on completion or failure.
The isolated PN retains every anatomical port, including undriven inputs.
This is not a literal recreation of biological antennal-nerve transection:
the real preparation can retain other neurons and background input absent
from the isolated model.

## Fit proposals, then run actual PAULA ticks

A continuous scalar integrate-and-fire approximation proposes an effective
current scale and integration coefficient. It never supplies the simulated
brain's output. Actual PAULA ticks generate every reported firing curve, with
the native threshold, cooldown/reset and positive learning rates. Each run
includes the approximately 22.22-s ascending ramp and one second of descent
for observation-window support. Only the measured ascending branch is scored.

| Proposal | Effective rheobase/current scale, pA | Integration coefficient, ms | PAULA error against the control-average curve, Hz RMSE |
| --- | ---: | ---: | ---: |
| Gain-only fit with the integration coefficient fixed | 50.713 | 20.000 | 10.433 |
| Joint scalar fit | 25.413 | 60.489 | 1.736 |
| Same joint fit at half the tick duration | 25.413 | 60.489 | 1.800 |

The gain-only fit is an analytic proposal, not a claim of globally optimal
parameters for the discrete neuron. A reference using the earlier count gain
and a newly declared current-unit mapping gives 145.265 Hz RMSE. That large
error is conditional on the mapping; the earlier work never claimed a physical
pA calibration. The gain-only comparison is the more informative control.

Leading and trailing observation windows give 1.753 and 1.785 Hz RMSE for the
joint fit. The half-tick comparison concerns this current-injection response,
not convergence of the entire network's time-dependent plasticity and delays.

The small average error must not hide the individual-cell result. When each
control cell is withheld and the other three determine the proposal, actual
PAULA errors are:

| Withheld control source column | Held-out curve RMSE, Hz |
| --- | ---: |
| G | 28.124 |
| H | 10.598 |
| I | 11.017 |
| J | 9.445 |

Column G is retained, not discarded as an outlier. The exposed cells are
additional transfer comparisons, not evidence that chronic exposure has been
modeled. Column letters in different source tables do not identify paired
cells across the current and firing experiments.

The 60.489-ms coefficient is an effective fit. It could absorb population
mixing, omitted active currents, compartments or background input. It is not
an independently measured membrane time constant. The compartment dependence
demonstrated by [Gouwens and Wilson (2009)](https://doi.org/10.1523/JNEUROSCI.0764-09.2009)
is a reason to test voltage and recording-location constraints before assigning
this parameter a specific cellular interpretation.

## Put the current and firing mechanisms together

The identified DL5 PN has 41 incoming ORN_DL5 partners, with 20–72 counted
contacts per partner and mean 40.951. Assuming uniform efficacy and equal
sampling of those partners, matching the control mean uEPSC peak of 35.023 pA
under the fitted current scale requires a per-count coefficient of 0.0372903,
instead of the earlier exploratory 0.075. The optogenetic recruitment
probabilities are not known, and physiological cells are not the connectome
specimen. Matching the mean peak sets the gain; it is not a prediction.

Every partner is tested separately through its real input port. Predicted
unitary peaks span 17.105–61.578 pA. None evokes a spike in the otherwise
undriven PN. These are model predictions, not a reproduced biological
distribution. The voltage state S has no established mV conversion.

The earlier train diagnostic is also rerun at source row 66370, the 44-contact
ORN input. Forty releases occur five ticks apart, followed by silence through
tick 799. The fitted tail is identical in all four factorial conditions:

| Integration coefficient | Per-count gain 0.075 | Per-count gain 0.0372903 |
| --- | ---: | ---: |
| 20 ms | 69 spikes | 39 spikes |
| 60.489 ms | 29 spikes | 13 spikes |

The gain change removes 30 spikes with the shorter coefficient but 16 with
the longer one. Their effects depend on each other. This is a controlled
composition result inside the neuron, not a biological train-response match.
Short-term depression is still absent. The predicted 13-spike course must not
replace the earlier connected evidence as though it had already been tested
with APL, other PNs and KCs reunited.

## Verification and next constraints

The independent ramp auditor checks the commanded current, every membrane
transition and every threshold/reset decision. All tested transitions have
zero voltage-equation residual. The audit rejects a falsified current even on
a spike tick where reset would hide its effect in the recorded voltage alone.
It establishes execution of the model, not agreement with biological spike
times, which are unavailable in this source.

The subsequent current-step study adds a connected preparation with closed-loop
APL-release controls. Voltage/current constraints, short-term synaptic
depression and synaptically driven reunion remain incomplete. Cell-to-cell
variation must remain visible. A fit to one averaged curve cannot authorize assigning the same
parameters to all PNs, KCs or APL. The construction helper explicitly refuses
to apply this isolated calibration to a multi-cell graph.

The final records are in `.live/research/flywire783/dl5-intrinsic-final-20260910`.
They include all source bins, every simulated tick, observation conventions,
cell-held-out comparisons, all 41 unitary probes and the four train controls.
The eight ramp executions and unitary/train probes take about 6 seconds and
approximately 218 MiB maximum resident memory. No live simulation was started.
Intermediate duplicate recordings were checked array by array before removal;
their numerical contents remain available in retained records.

```sh
.venv/bin/python -m simulations.drosophila.pn_intrinsic \
  GRAPH_DIRECTORY CELLS_JSON ARTICLE_V2_JSON CURRENT_TAIL_FIT_JSON NEW_OUTPUT
.venv/bin/python -m pytest tests -k drosophila -q
```

The source and graph paths are the same verified inputs used in the
[current-tail study](PN_CURRENT_REUNION_2026-09-10.md). The new instrument and
assay add seven tests; all 172 fly tests pass. One unrelated checkpoint test
remains skipped for missing `cloudpickle`.
