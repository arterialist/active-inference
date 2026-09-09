# APL spatial benchmark, 10 September 2026

The published straight-backbone calculation is reproduced. The current
FlyWire/PAULA circuit is **not yet physiologically calibrated**. This experiment
separates a reproduced reference calculation, the authors' published anatomical
predictions, and a new schematic test of PAULA's conservative cable operator.
It does not change any neuronal equations or agent defaults.

## Data and observation model

[Amin et al. 2020](https://doi.org/10.7554/eLife.56954) locally activated APL with
ATP and measured GCaMP6f and co-ejected red dye. Dye spread matters because the
stimulus itself spreads through the tissue. Figure 8 contains normalized spatial
measurements and predictions for horizontal-lobe, vertical-lobe and calyx
stimulation. These are spatial means, not raw imaging time series. The roughly
5 Hz imaging does not determine a PAULA tick duration.

Original Figure 7 and Figure 8 workbooks, including their original sheet names,
blank cells and 74 saved formula expressions, are preserved without export.
Formula caches remain explicitly saved values. The analysis uses the numeric
observation/prediction columns and independent MAT dye arrays. An independent
ZIP/XML read agrees with all 324 numeric calcium/dye cells in the three Figure 8
stimulus sheets. The workbook and MAT representations agree on all dye-fit
values and all branch-index mappings.

The branch plots share 19 source segments. Counting both complete curves would
overweight the shared stem. Each of the 35 source regions is counted once here.
Two distinct regions have identical schematic coordinates at the junction.
They remain distinct observations and distinct input sites in the reproduced
reference calculation.

## What ran

The authors' `convolveExpDecay.m` specifies a sum of stimulus-site contributions,
each attenuated exponentially with city-block distance on their 2D backbone.
Independent evaluation reproduced all nine published straight-backbone curves
at 25, 50 and 75 micrometres. Maximum absolute discrepancy was
`7.77e-16`. This verifies that calculation and the input/observation mapping.
The anatomical connectome predictions below are read from the workbook, not
independently regenerated from the hemibrain reconstruction.

The new comparison uses the existing PAULA `PassiveCable` operator on the
schematic backbone. This is a geometry ablation, not a 34-compartment APL
reconstruction. The two coincident sites become one physical junction with
their drive densities averaged. The circuit's 360,309-node FlyWire APL is not
replaced. Assumptions are uniform 0.2 micrometre radius, sealed ends, uniform
membrane properties and current density proportional to the fitted dye.
The relation `lambda^2 = radius * (Rm/Ra) / 2` defines the nominal length scale.
The input is held for 400 model ticks at `dt/lambda_param = 0.05`, not assigned
the duration of the ATP puff. Every node voltage at every tick is retained.

For each run, the full trajectory satisfies the discrete cable equation with
maximum residual below `3.2e-17`. The final state differs from an independently
solved stationary equation by less than `3.7e-10`. No plastic network runs in
this operator test, so it neither demonstrates adaptation nor freezes a running
agent's learning.

## Spatial comparison

Descriptive RMSE against normalized observed calcium, with equal weight per
source region. These scores have no animal-level confidence interval and are
not acceptance thresholds. Calcium is not assumed to be calibrated voltage.

| Candidate at nominal 50 micrometres | Horizontal stimulation | Vertical stimulation | Calyx stimulation | All sites |
| --- | ---: | ---: | ---: | ---: |
| Published anatomical predictions | 0.13492 | 0.13166 | 0.17021 | 0.14664 |
| Reproduced straight-backbone sum | 0.11845 | 0.09896 | 0.19842 | 0.14514 |
| Schematic conservative cable | 0.20250 | 0.06878 | 0.17739 | 0.16042 |

For context, a normalized spatially uniform prediction scores `0.77060` and
the fitted dye alone scores `0.26278`. Their errors support testing spatial
propagation, but do not identify a unique cellular mechanism.

The conservative cable improves the vertical-stimulation comparison but worsens
the horizontal one. At the vertical tip under horizontal stimulation, it predicts
`0.76371` against observed `0.26671`. The actual anatomical 50-micrometre curve
also has substantial local errors. One vertical-stimulation region reads
`0.17146` predicted against `0.70125` observed. These residuals must not disappear
behind the all-sites score.

With this explicitly chosen scoring rule, published anatomical curves score
`0.20083`, `0.14664`, `0.14362` at 25, 50 and 75 micrometres respectively.
The small aggregate preference for 75 over 50 is driven by improved calyx fit;
horizontal and vertical stimulation each favor 50 among those three candidates.
This is not evidence for changing PAULA's parameter to 75 or disputing the
paper's qualitative choice of 50. The scoring rule and shared-region weighting
are ours, not an inferred published optimization criterion.

## Why the same length scale gives different responses

A separate uniform-input diagnostic tests an operator property that a single
localized decay curve cannot establish. With unit input at every schematic
site, the finite exponential sum gives normalized minima of `0.38750`, `0.42418`
and `0.48514` at 25, 50 and 75 micrometres. Its maximum is one. Locations have
different totals of distance-weighted contributors.

Under uniform membrane-density input, the conservative sealed cable stays
spatially uniform at **every** one of the 401 recorded states. Maximum spatial
range is below `3e-15`. Its uniform solution follows `v(t) = 1 - 0.95^t` from
rest. Axial exchange cannot create a gradient in that condition, and all local
membrane drive/leak balances are identical. These three additional full traces
are retained alongside the nine localized-drive traces.

This difference is a consequence of the declared input measures and boundary
conditions. It is not a coding error in the reproduced equation, and it does not
establish which preparation best represents the real cell. The paper explicitly
does not model branching effects. An exponential distance scale is therefore
not enough to specify a conservative input-to-output operator. In particular,
the earlier numerical refinement of the full FlyWire cable cannot be promoted
to biological validation by borrowing that scale.

## Next physiological test and remaining obstacle

The next comparison should use a real neurite tree, its registered measurement
regions and measured stimulus footprint together. It must keep anatomical
geometry, stimulus density, electrical propagation and calcium observation as
separate assumptions. It should predict all three stimulation sites with one
declared parameter set and retain the per-region residuals.

The authors' saved `apl200607.mat` was acquired from their pinned repository.
SciPy sees a MATLAB opaque object plus an internal workspace, not accessible
node-to-region fields. MATLAB/Octave is not installed. Its presumed region
mapping has not been decoded or silently replaced with a guessed coordinate
alignment. Recovering that mapping, or making an independently justified and
documented registration, remains necessary for the same-specimen anatomical
comparison. Their hemibrain specimen must remain separate from FlyWire 783.

Neither normalized spatial calcium means nor the dye fit supplies absolute
current, release gain, channel kinetics or a time conversion for the existing
PAULA preparation. Those remain unresolved. This benchmark establishes a
reproducible observation comparison and a specific model-assumption difference,
not odor learning, embodied behavior or an accepted APL cellular model.

## Reproduce and inspect

Source code: [amin_spatial.py](amin_spatial.py).
Primary data URLs are under the article's Figure 7 and Figure 8 source-data
links. The three small MAT files are in `data/` at the authors' repository
commit [`d16f97f26e605ec0591043db636b7ff8e9801e0f`](https://github.com/aclinlab/amin-et-al-2020/tree/d16f97f26e605ec0591043db636b7ff8e9801e0f).
Author code has GPL-3.0 licensing. The mathematical reference equation is
independently evaluated here; no author MATLAB code is executed.

From `active-inference/`, with new output paths:

```sh
# Use the bundled document Python, which has openpyxl, for the read-only import.
DOCUMENT_PYTHON simulations/drosophila/amin_spatial.py extract SOURCE_DIR NEW_CELLS_JSON
.venv/bin/python -m simulations.drosophila.amin_spatial analyze SOURCE_DIR NEW_CELLS_JSON NEW_OUTPUT_DIR
.venv/bin/python -m pytest -q tests/test_drosophila*.py
```

Retained local sources: `.live/research/flywire783/amin2020-primary-20260910/`.
Final results and all 12 traces:
`.live/research/flywire783/amin-spatial-observation-20260910-v2/`.
`analysis.json` includes every observed and predicted profile, source hashes,
per-site scores and trace filenames. Each NPZ includes coordinates, source-to-node
mapping, current density, current, membrane areas normalized as capacities,
electrical parameters, tick indices and the complete voltage trajectory.
Final results occupy 959,758 bytes; acquired source material occupies about
25 MiB. The older candidate output is retained, not overwritten.

All 104 fly tests pass, including 13 new observation/operator tests. These are
software and numerical checks, not 104 physiological acceptances.
