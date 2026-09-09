# DL5 input current is not the same observable as somatic voltage

The original Figure 7 workbook from
[Gugel, Maurais and Hong (2023)](https://doi.org/10.7554/eLife.85443)
provides time-resolved unitary ORN→PN currents. These are a constraint on the
input mechanism, not an empirical odor drive for all 174 PNs. The current
PAULA implementation retains voltage after an input has arrived, but its
receptor-to-hillock event does not itself have a sustained current waveform.
This distinction is verified below without changing the neuron equations.

## Primary source and preparation

The [publisher's original workbook](https://cdn.elifesciences.org/articles/85443/elife-85443-fig7-data1-v2.xlsx)
is 402,813 bytes, SHA-256
`a8ae6fcd3bf0d8effab7a0ecbfa88fccf192f134144072282758ca8125bd8c78`.
The hash was computed from the download, not supplied by the publisher.
The paper and its source data are attributed to the authors under CC BY 4.0.
The full publisher article JSON and the original workbook remain local.

Figure 7A–C measures DL5 PN firing under somatic current injection after
bilateral antennal-nerve transection. Figure 7D–G records DL5 PNs in voltage
clamp while optogenetic minimal stimulation recruits a presynaptic ORN axon.
A unitary connection includes multiple anatomical release sites. It is not a
measurement of one connector in the FlyWire table. The study uses different
flies from the connectome specimen.

The authors align individual uEPSCs by their peaks before averaging.
Consequently, the source traces cannot identify synaptic conduction delay,
light-to-spike latency or trial-to-trial temporal variability. In particular,
the approximately 23 ms light-evoked latency described in the paper is not a
value to assign to a PAULA cleft. The reported 10 kHz acquisition rate is also
not an established simulation clock.

The new `gugel.py` reader preserves every workbook cell and all 888 formulas.
It independently resolves the formulas from constants using only the source's
reference-plus/minus-literal grammar. All cached values agree exactly with that
calculation. It does not execute Excel, recalculate or rewrite the workbook.

| Source section | What is present | What is not established |
| --- | --- | --- |
| Rows 5–914, DL5 firing | Four experimental cells per condition. 890 increasing-current rows ending at 100 pA, followed by 20 all-zero rows. | No raw spike times or separate 5-pA current-step table. The binned ramp is not a sequence of independent cells or a step-response assay. |
| Rows 921–2921, DL5 uEPSCs | 2,001 time points in each of seven solvent and five E2-hexenal exposure columns. Currents in pA, time in ms. | Not individual unaligned trials, not membrane voltage, not calcium. |
| Rows 2928–2932, VA6 lateral responses | Odor-evoked depolarization after deafferentation, with unequal sample counts across stimuli. | Not direct ORN drive to VA6, and not proof of the missing lateral circuit's dispensability. |

Earlier browsing had incorrectly nominated Figure 5 as the intrinsic assay.
The publisher captions resolve that: Figure 5 concerns mixtures and its
supplement concerns LN innervation. Figure 7 is the relevant experiment.
The download's contents, rather than its broad “B–C” label, determine what can
be reproduced from it. The 1-s, 5-pA step protocol cannot yet be independently
reconstructed from the acquired workbook.

## Reanalysis of the actual current traces

Each source column remains one experimental cell. We subtract its mean current
before source time 20 ms, define inward current as baseline minus recorded
current, and align the analysis to its minimum current. No absolute value or
rectification is applied. All 12 minima occur at source time 49.900499 ms,
consistent with the authors' peak alignment.

Integration uses the recorded time coordinates, with linear interpolation only
at exact window boundaries. The following estimates are our reanalysis, not a
claim to reproduce the authors' fitting procedure. The 1/e crossing is a
descriptive crossing time, not an exponential fit or a membrane time constant.

| Quantity | Solvent exposure, 7 cells | E2-hexenal exposure, 5 cells |
| --- | ---: | ---: |
| Mean inward peak, pA | 35.023 | 38.949 |
| Mean first postpeak 1/e crossing, ms | 13.684 | 12.618 |
| Individual crossing range, ms | 9.602–19.078 | 8.367–21.945 |
| Fraction of 0–50 ms postpeak charge delivered after 5 ms, individual range | 0.663–0.752 | 0.600–0.767 |
| Same fraction using a 0–100 ms window | 0.697–0.802 | 0.617–0.788 |

Repeating the calculation with baseline windows ending at 30 and 40 ms leaves
the combined 0–50 ms tail fraction between 0.600 and 0.767. Thus the sustained
current is not an artifact of one selected baseline interval. The workbook
does not supply unaveraged trials with which to estimate trial-level uncertainty.
Time points are not treated as biological replicates, and these descriptive
comparisons do not establish equivalence between exposure conditions.

## Native PAULA witness on the identified input

The selected FlyWire DL5_adPN is `720575940617207185`, PAULA global index
`30002`. Isolating it from the existing preparation retains all 244 incoming
pairs, 729 outgoing pairs, their original addresses, and the extra experimental
input. No incoming coefficient is renumbered or discarded.

The earliest source-row ORN_DL5 provider is `720575940604352689`, source row
`66370`, with 44 counted contacts onto PN port 0. We impose one positive unit
release at that receiving receptor at tick 0. The ORN is not simulated. This
is an input-kernel assay, not a reenactment of the biological deafferentation
or optogenetic stimulation protocol.

The same native assumptions used in the connected operating-range probe remain:
count conversion 0.075, dendritic delay two ticks, attenuation 0.95 per tick,
integration constant 20 ticks, and weak positive adaptation. The cleft is not
included because the assay begins at the receiving receptor.

All 160 ticks are recorded. Current arrives only at tick 2, with magnitude
`2.9782497882843018` in model units. It is zero on every later tick. Somatic
voltage remains positive and follows the native passive decay. Independent
queue inspection and current inferred from the somatic update agree to within
`1.1920928955078125e-7`, with an explicit float32 rounding allowance.
The stimulated coefficient changes from `3.3` to `3.2999999234400015`.
Adaptation was not frozen.

This is not yet a biological voltage-clamp simulation. There is no assigned
physical seconds-per-tick or pA-per-model-current conversion. Its result is
structural: propagation delays move the event in time; membrane integration
retains voltage; neither supplies a receptor-current tail. Treating the soma's
decaying voltage as a successful uEPSC fit would compare different observables.
Conversely, this assay does not disprove PAULA's extensible framework.

## Consequence for reconstruction

The next mechanism test should separate receptor-current relaxation from
membrane integration inside PAULA. Preserve measured partners and contacts,
then test one shared ORN→DL5 kinetic hypothesis against these waveforms and PN
current-to-firing responses. Keep extracellular release, receptor state,
somatic voltage and observation models separately inspectable.

Do not deliver a prerecorded EPSC as many independent presynaptic spikes and
call the receptor mechanism solved. That would change both input timing and
the number of native plasticity updates. Do not infer each contact's strength
from a unitary axon-to-PN peak, assign these kinetics to KC/APL without evidence,
or retune every anatomical edge. Realistic odor-selective input, the missing
antennal-lobe population, connected KC/APL physiology and learned behavioral
discrimination remain unfinished requirements.

## Reproduce

Download the linked workbook without re-exporting it. From `active-inference/`,
use the bundled document Python for standalone read-only extraction:

```sh
/Users/arterialist/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  simulations/drosophila/gugel.py extract \
  PATH_TO/elife-85443-fig7-data1-v2.xlsx NEW_EXTRACTION_DIRECTORY

.venv/bin/python -m simulations.drosophila.gugel native-input \
  .live/research/flywire783/kc-apl-left-v1 NEW_PROBE_DIRECTORY
.venv/bin/python -m pytest tests/test_drosophila_gugel.py -q
```

The standalone extractor avoids importing MuJoCo into the bundled document
runtime. Source workbooks are never modified and outputs refuse overwrite.
The retained local results are `gugel2023-extracted-baseline-controls-20260910`
and `dl5-native-input-bound-20260910`, under `.live/research/flywire783/`.
No large connected-network recordings or live simulations were started.

Verification: 160 fly tests passed. The broader collection skipped one unrelated
checkpoint test because `cloudpickle` is absent. The 12 new tests cover formula
dependencies and failures, analytic waveform integrals, signed currents,
missing support and native current-versus-voltage behavior. A separate reader
of the workbook's original XML matched all 34,269 numeric cells to extraction.
These checks establish the analysis and probe, not physiological acceptance.
