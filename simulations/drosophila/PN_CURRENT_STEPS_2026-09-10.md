# Reuniting the calibrated DL5 neuron with spatial APL feedback

The candidate DL5 calibration has now been exercised in the full selected
PN/KC/APL graph. Across all 9,000 ticks, the intact circuit and an
APL-release blockade both produced the same target spike counts as isolation.
The intact circuit nevertheless changed the target voltage. This is evidence
about a weakly recruited feedback loop, not general robustness or physiological
acceptance.

## What was held constant

The connected conditions retain 2,755 identified cells, 179,282 internal
directed pairs and all incident boundary ports. There are 9,211 boundary cells
whose neurons are not simulated. APL uses the existing passive spatial cable
and local graded release. No default agent, C. elegans code or PAULA neuron
implementation was changed.

Only DL5_adPN `720575940617207185`, global index 30002, receives the
[previous calibration](PN_INTRINSIC_CALIBRATION_2026-09-10.md). Its effective
integration coefficient is 60.4890558 ticks, with a nominal 1 ms/tick mapping.
The electrical-current scale is 25.4125365 pA per model-current unit. Its 41
ORN_DL5 input coefficients use 0.0372903 per signed counted contact, with the
same fitted two-component current tail. Other neurons keep their existing
parameters. These are explicit pre-run assignments, not online controllers.

The ORN ports are undriven in this electrical assay. Thus the fitted synaptic
tail and gain are present but not functionally tested here. The experiment
asks whether the PN's intrinsic response survives reunion; it does not replace
the still-needed connected ORN-train and odor-response tests.

## Protocol and intervention

[Gugel, Maurais and Hong, Figure 7B](https://elifesciences.org/articles/85443/figures)
describe one-second current pulses separated by one-second intervals, increasing
in 5-pA steps. This experiment samples 25, 50, 75 and 100 pA in that order.
It adds a one-second baseline and retains every one-second recovery interval,
including the final interval. Each complete course has 9,000 ticks.

The acquired source workbook supplies ramp responses but lacks the numerical
step series despite its broad source-data label. Consequently these step
responses are model predictions, not fitted or held-out biological step
measurements. Neither isolated nor connected simulation literally recreates
antennal-nerve transection, which can leave other biological inputs present.

Three fresh preparations receive the identical predetermined electrical
command through `CurrentElectrode`: isolated target, intact connected graph,
and connected graph with APL forward release blocked. The electrode bypasses
chemical receptors. The blockade filters APL's completed outgoing event list;
it does not silence APL's state, remove edges, freeze weights or suppress its
native return events. It is a closed-loop intervention, unlike the earlier
open-loop removal of recorded target input.

## Step responses and the first causal difference

| Injected current | Isolated spikes in 1 s | Intact spikes in 1 s | APL-blocked spikes in 1 s |
| ---: | ---: | ---: | ---: |
| 25 pA | 0 | 0 | 0 |
| 50 pA | 23 | 23 | 23 |
| 75 pA | 40 | 40 | 40 |
| 100 pA | 55 | 55 | 55 |

At the 50-pA step, the first target spike occurs at tick 3042. APL first has
positive output at tick 3045. Its release reaches target input port 122 at
tick 3046 and contributes current at tick 3048. That is the first changed
target-voltage tick relative to isolation. The blockade eliminates this input.
All three complete target spike rasters match, not merely their step totals.
Blocking APL recovers the isolated target S/O trajectory exactly. With APL
intact, maximum absolute S difference is 0.00238465. No other PN or KC spikes
in these courses. Their subthreshold responses remain recorded, but this is
not a demonstration of KC population coding or a failure of biological KCs.

The anatomical loop has five counted contacts from this PN to APL at source
row 3268238, and 19 from APL back to the PN at row 7243136. The receiving
coefficient at PN port 122 starts at -1.425 under the existing count/sign
conversion. These are measured contact counts combined with assumed signs,
efficacies and delays, not measured physiological loop gain.

During the 50, 75 and 100-pA steps, signed feedback-current sums are -3.76535,
-6.47792 and -9.09771 model-current tick units. Their magnitudes are only about
0.19%, 0.22% and 0.23% of the respective injected-current integrals. Equal
firing counts under such weak feedback do not establish strong-coupling
stability. A nominal one-tick observation also cannot exclude smaller physical
timing differences.

The blocked run changes 178 target terminal coefficients despite an unchanged
somatic response and no returning target input. Native return signals from
the target's neural consumers account for this distinction. It is ongoing
PAULA adaptation, not a demonstration of learned discrimination or useful
long-term memory.
The intact condition also changes one target postsynaptic coefficient, on the
APL feedback port. Maximum APL compartment voltage is 10.9171 model units and
maximum local release is 0.0717849 in both connected conditions, below the
release cap of 1. Neither quantity is a calibrated biological measurement.

## Inspection and falsification

`pn_current_steps.py` records every cell's S, O and average activity, plus all
target input buffers, per-port currents, postsynaptic coefficients and terminal
coefficients. Target thresholds and plasticity-window state are retained too.
APL records include maximum compartment voltage, maximum local release and
forward/return event counts. Recording uses 1,000-tick chunks.

`pn_step_analysis.py` replays each complete target input history through a fresh
PAULA neuron under the identical electrode course. It checks every current,
S/O/F_avg/t_ref/r/b state and postsynaptic weight. A separate equation audit
checks command, integration, threshold crossing and reset. Mutation tests reject
fabricated voltage, current or postsynaptic learning even when record hashes
are updated. Intact and blocked anatomical bindings must match exactly.

Terminal return-event histories are not replayed; terminal weights are recorded
but not certified by the target replay. Full APL compartment arrays, other
cells' synaptic histories and complete event queues are omitted. These records
are not restart checkpoints. The raw field `delivered_forward_events` counts
events admitted to the network router after the intervention, including
boundary terminals with no simulated receiver. It is not a receptor-delivery
count and APL's positive graded outputs must not be called spikes.

All three complete target replays pass, checking 13,446,000 scalar values
across 27,000 target ticks. The bindings match between intact and blocked
conditions. All 176 fly tests and 67 neuron tests pass; one unrelated checkpoint
test remains skipped because `cloudpickle` is absent. The two connected runs
took 668 and 420 seconds, respectively, while running concurrently. Sampled
resident memory stayed around 0.8-0.9 GiB per worker. The retained records use
about 88 MiB, and all simulation workers exited. No live servers were started.

## Reproduction and remaining work

Retained records are under
`.live/research/flywire783/dl5-current-steps-20260910/`. The analysis command
requires all three conditions to complete and rejects missing or overlapping
tick chunks. From `active-inference`:

```sh
.venv/bin/python -m simulations.drosophila.pn_current_steps \
  GRAPH INTRINSIC_ANALYSIS CURRENT_TAIL_ANALYSIS NEW_CONDITION_OUTPUT \
  isolated --spatial APL_SPATIAL_DIRECTORY
```

Repeat with `intact` and `apl_release_block`, each in its own new output
directory, then run:

```sh
.venv/bin/python -m simulations.drosophila.pn_step_analysis \
  GRAPH INTRINSIC_ANALYSIS CURRENT_TAIL_ANALYSIS CONDITIONS_DIRECTORY NEW_ANALYSIS_JSON
.venv/bin/python -m pytest tests -k drosophila -q
```

The next step is synaptically driven reunion with measured temporal input
constraints, including short-term depression and antennal-lobe inhibition.
Electrical injection into one PN cannot establish odor representation or
sufficient KC recruitment. ORNs and local interneurons must be reconstructed
with their full incident edges before they can supply those signals as neurons.
The control-average fit still has large cell-held-out errors, and neither the
rest of the graph's physical clock nor its synaptic gains are calibrated.
