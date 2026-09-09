# Primary physiological constraints for the PN–APL–KC preparation

The published source tables are now available locally, independently of their
plots. Twenty workbooks and the README total 321,239 bytes. All match Dryad's
published SHA-256 digests. They are CC0 data from dataset version 160926,
[Tavosanis and Prisco](https://doi.org/10.5061/dryad.bk3j9kdd1), accompanying
[Prisco et al., eLife 74172](https://doi.org/10.7554/eLife.74172).

Earlier direct downloads failed. On 10 September the public metadata API worked,
its download endpoint returned 401, and normal individual-file downloads in the
browser succeeded without login. The 8 GB anatomy-render collection was not
downloaded. No source workbook was edited or recalculated.

## What the experiments actually measure

The imaging protocol uses two 5-second odor puffs, separated by 20 seconds of
clean air. Calcium imaging runs at about 9 Hz in the calyx and 5 Hz for the
antennal-lobe preparation. The first 30 frames establish the fluorescence
baseline. These physical units do not establish a PAULA tick duration.

The measurements differ by preparation:

| Preparation | Measured quantity | What cannot be substituted |
| --- | --- | --- |
| GH146-GCaMP6m PN dendrites | Odor-evoked calcium peaks in selected AL glomeruli | Input current, spike rate, or a response for every PN |
| NP225-Syp::GCaMP3 PN boutons | Mean peak among responding optical ROIs in each animal | Whole-PN soma activity or a population-wide mean including inactive boutons |
| MB247-homer::GCaMP3 KC claws | Mean peak among responding microglomerular ROIs in each animal | Total KC spikes or a measurement of each individual claw |
| APL-GCaMP6m | Calcium signal from a manually selected calyx region | Whole-neuron mean voltage or whole-neuron release |

The authors identify active regions separately for each odor using image
variation, Otsu thresholding and an expected 5 µm ROI diameter. The two odors
can therefore select different ROIs. This is explicit in their methods and
response to reviewer 6. Normalization of this conditional distribution is not
equality of total activity, identical responding neurons, or enforced silence.

## A usable intervention target, with actual experimental spread

The APL-output-blockade table contains ten animals per condition and two odors
per animal. The odor gap is Oct minus Mch, in percentage points of peak ΔF/F0.

| Condition | Mean within-animal odor gap | Individual gaps |
| --- | ---: | --- |
| APL ON | 7.192 | 17.90, 0.32, -0.67, 29.31, 27.72, -20.73, 4.46, 0.91, -8.28, 20.98 |
| APL OFF | 61.288 | 75.18, 106.25, 90.70, 39.78, 35.86, 42.36, 83.99, 7.24, 74.13, 57.39 |

Our independent Welch comparison of these animal-level differences gives an
OFF-minus-ON effect of 54.096, with a 95% interval of 30.847 to 77.345. The
calculation pairs odors within each animal, not animals across ON/OFF groups.
It is not the paper's two-way ANOVA with Tukey comparisons. This uncertainty
interval describes this dataset; it is not a proposed pass tolerance for a
neural model. The exact differences remain in the machine-readable analysis.

The useful mechanistic target is a change in the odor-response distribution
under APL blockade while maintaining responsive neural activity. It does not
require every intact trial to have exactly equal amplitudes. This comparison
still needs a defensible calcium/optical observation model before it can score
PAULA. The source workbooks contain peaks, not full physiological trajectories,
so they cannot independently validate latency, oscillation or sustained dynamics.

## How much of the reconstructed input is covered

The two glomerulus tables contain 360 values: nine labels, ten rows and two odors
in each of two cohorts. Eight unambiguous labels yield 20 candidate PNs in our
174-PN FlyWire preparation: DM6 3, DC2 2, VA6 1, DA1 9, DL1 2, DL5 1, DM3 1,
DC1 1. The remaining 154 PNs do not receive imputed measurements or zero drive.

The ninth label is VA1. It does not resolve the eight VA1d/VA1v candidates.
Even the 20 exact-label candidates are not measured individual cells. They
include a GABAergic DA1_vPN alongside cholinergic DA1_lPNs; the available table
does not identify their individual GH146 expression. The importer preserves
that distinction, exact root IDs and the unresolved cells. It never turns a
glomerulus peak into a per-cell current. The paper's hemibrain connectivity
tables also remain separate from our FlyWire identities.

## Source inconsistencies and interpretation limits

Three pooled KC columns do not match the summed ROI counts in their associated
per-animal tables:

| KC measurement | Sum of reported active ROIs | Pooled peak values |
| --- | ---: | ---: |
| Mch/Oct cohort, Oct | 183 | 184 |
| δ-DL/Oct cohort, δ-DL | 109 | 93 |
| δ-DL/Oct cohort, Oct | 165 | 107 |

The PN-bouton columns and all four APL ON/OFF columns reconcile in count, and
their count-weighted mean discrepancies are below 0.002 ΔF/F0 percentage points.
Pooled columns lack animal and ROI identifiers; matching counts do not justify
inventing either identity. Some SD columns are on a different numerical scale
from the means. They are preserved but not used as uncertainty weights.

Figure 3B's paired t-test reproduces to its reported precision, p=0.0002. The
nine Figure 3F pairs give p=0.23981 with a two-sided paired t-test, whereas the
caption reports 0.1648. The separately deposited delta column reproduces every
one of those nine differences, so that discrepancy is not explained by an odor
subtraction error. Its cause remains unresolved. Neither nonsignificant value
establishes equivalence. No data were altered to reproduce the reported p-value.

Three delta-column headers say Mch–Oct, but their numbers equal Oct–Mch. The
importer keeps both the original text and the checked numeric direction.
An independent ZIP/XML reading confirmed all 1,645 numeric cells in the six KC
workbooks involved in the count checks, separately from openpyxl extraction.
These issues limit particular quantitative fits; they do not erase the observed
APL-blockade effect or show that the paper's general conclusion is false.

## Reproduction and next experiment

`prisco2021_sources.json` pins each filename, Dryad ID, size and hash.
`prisco.py extract SOURCE_DIRECTORY NEW_DIRECTORY` reads the workbooks with
openpyxl in read-only mode, retains every cell including blanks, and copies the
verified originals. Run this standalone command with the bundled document
Python runtime; importing the simulation package there requires unrelated
simulation dependencies. Analysis runs in the normal simulation environment:

```sh
uv run --locked --extra dev --extra flywire python -m simulations.drosophila.prisco analyze \
  .live/research/flywire783/prisco2021-primary-20260910 NEW_ANALYSIS.json \
  --graph .live/research/flywire783/kc-apl-left-v1
uv run --locked --extra dev --extra flywire python -m pytest tests/test_drosophila_physiology.py -q
```

The local extraction is `.live/research/flywire783/prisco2021-primary-20260910/`.
`cells.json` preserves the source grids. `analysis-final.json` retains every
glomerular value, animal-level odor difference, candidate cell and unmatched
cell. It also records source and graph hashes. `article-v2.json` preserves the
publisher's full text and methods. Existing outputs are never overwritten.

No brain or neuron equations changed in this work, and no new connectome course
or live simulation was run. All 91 fly-package tests pass, including 12 new
measurement and identity checks. These tests do not certify a physiological fit.
The next physiological experiment must resolve stimulation and observation
together: an identified PN pattern with explicitly unknown channels, calyx-local
APL readout, and KC input-region measurements distinct from KC spike output.
Fit a shared transduction hypothesis on one condition and test its intervention
predictions without retuning it. Obtain time-resolved experimental signals for
the dynamical comparison. Do not treat partial input coverage as full odor
reconstruction or replace the connected network with a feedforward surrogate.
