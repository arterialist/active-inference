# Registered APL test, 10 September 2026

The missing anatomical registration is recovered. The unchanged PAULA cable
operator now runs on the authors' saved 181,741-node APL tree with their
stimulation and observation regions. It reaches a numerically verified steady
state but underpredicts propagation in part of the vertical lobe. This is a
conditional intracellular test, not physiological acceptance of the connected
FlyWire circuit, and not a learning experiment.

## Recovered primary anatomy

The saved `apl200607.mat` from
[Amin et al. 2020](https://doi.org/10.7554/eLife.56954) contains an opaque MATLAB
class instance. [mat-io 1.0.0](https://github.com/foreverallama/matio) reads its
stored properties without executing MATLAB methods. The new optional
`physiology` dependency pins this reader. Existing dependencies keep their
locked versions.

The original `aplmanualSkel200303.mat` contains the 35-region raster mask.
Independently applying the authors' coordinate normalization to every saved
node reproduces all 181,741 region assignments exactly. The 1,834 unassigned
nodes remain in the tree. They can conduct axial current but receive no
experimental drive. Exact agreement verifies the saved registration procedure,
not its biological accuracy or identity with an imaged animal.

The saved numerical sampling set has 9,969 points. The authors' ROI and spatial
selection retains 9,579. These are integration/readout locations, not extra
neurons or biological synapses. All lie inside their declared edges, with
maximum projection error `4.07e-12` source pixels. All 35 regions have samples.

The separately published `APLskelv1.1.csv` has 182,631 nodes and is a different
revision. Exact coordinates match 181,223 saved nodes, but 27 comparable parent
relationships and three matched radii differ. No nearest-neighbor match changes
the experimental geometry. The saved healed tree, the newer CSV, and the
360,309-node FlyWire APL remain separate preparations. Source hashes and the
cross-revision comparison are in `extraction.json`.

## Static reference equation

The anatomical exponential-distance sum is independently evaluated with two
tree passes. Storage and work grow linearly with node count instead of storing
every pairwise distance. Adding numerical sample locations subdivides existing
edges without adding paths. This computation is an offline reference, never a
controller or replacement PAULA neuron.

Independent Dijkstra evaluation at seven sample destinations includes all
9,579 sources. It agrees with the factorized sums to within `6.26e-12` across
all three published decay constants. A separate read-only check against the
authors' saved sample graph compared 69,783 distances. Its maximum difference
was `1.02e-10` in square-root-pixel electrotonic units. These checked paths
support the distance implementation, not equality of every possible path.

The resulting normalized regional profiles still differ from the published
anatomical predictions. Maximum regional differences range from `0.03912` to
`0.13420` across the nine comparisons. The exact cause remains unresolved.
Source revision and numerical sample selection are candidates, not findings.
The earlier straight-backbone calculation remains exactly reproduced; this
anatomical rerun must not inherit that success label.

## Dynamic cable test on the registered tree

All original 181,741 nodes run through the existing `PassiveCable` operator.
Coordinates and radii use the documented conversion of 8 nm per source pixel.
Each stimulation site starts from zero and receives held current density
proportional to the fitted dye profile. Current per node is this density times
normalized membrane area. Shared parameters are `Rm/Ra = 25000 micrometres`
and `dt/lambda_param = 0.05`. No per-region or per-site parameters are fitted.

Voltage is interpolated at the saved sample locations, averaged within each
region and peak-normalized. Comparing that profile with normalized calcium is
an explicit spatial observation assumption. Calcium is not calibrated voltage.
RMSE below is descriptive, equally weighted across the 35 unique regions; it
has no animal-level confidence interval or acceptance threshold.

| Stimulation | Spatial RMSE | Maximum final voltage difference from independently solved steady state |
| --- | ---: | ---: |
| Horizontal lobe | 0.21334 | 5.84e-10 |
| Vertical lobe | 0.18234 | 7.55e-10 |
| Calyx | 0.19669 | 7.14e-10 |

Under vertical stimulation, region 17 has observed normalized calcium
`0.70125`, against predicted normalized voltage `0.12744`. Its recorded
trajectory rules out insufficient settling as the explanation.

| Tick | Region 17 voltage divided by that tick's regional peak |
| --- | ---: |
| 1 | 0.09056 |
| 20 | 0.11681 |
| 100 | 0.12723 |
| 200 | 0.12744 |
| 400 | 0.12744 |

Every node is recorded at every tick, including initial state, for all three
sites. A separate streaming audit checked all 218,634,423 node-state values,
file hashes and consecutive tick coverage. Reconstructing regional readouts
from the full states gives exact agreement. Independently rebuilding the
cable equation gives maximum residual below `3.45e-16`. Total area-weighted
voltage follows the closed-form mass balance to within `2.51e-14`.
These checks establish a correctly recorded numerical solution, not a correct
biophysical model.

## What this narrows down

The schematic cable's stronger vertical propagation did not carry over to the
registered morphology under this input assumption. Anatomical path lengths,
caliber and branch loading now matter explicitly. The source/observation
assignment and insufficient run length are no longer the same unresolved
obstacles they were before this test.

The next matched comparison should hold registration and readout fixed while
testing shared electrical parameters and input physics across all three sites.
The paper activates P2X2 channels; the current test instead holds an imposed
current density. A conductance input of the form `g(ATP) * (E_rev - V)` is a
distinct hypothesis worth testing. It has not been implemented or shown to
improve these results. The reference kernel also samples sources along length,
while this cable applies current over membrane area. Those input measures must
remain explicit. Absolute current, channel kinetics, the calcium observation
map and physical tick duration are still uncalibrated.

## Reproduce and inspect

Implementation: [amin_anatomy.py](amin_anatomy.py). Primary anatomy and author
code are pinned at
[`d16f97f26e605ec0591043db636b7ff8e9801e0f`](https://github.com/aclinlab/amin-et-al-2020/tree/d16f97f26e605ec0591043db636b7ff8e9801e0f).
The authors' code is GPL-3.0. No MATLAB code is executed or copied into this
independent implementation. The extraction verifies hashes before decoding.

From `active-inference/`, use new output paths and the primary cells JSON from
the [preceding benchmark](AMIN_SPATIAL_BENCHMARK_2026-09-10.md):

```sh
uv sync --locked --extra dev --extra flywire --extra physiology
.venv/bin/python -m simulations.drosophila.amin_anatomy extract SOURCE_DIR NEW_REGISTERED_DIR
.venv/bin/python -m simulations.drosophila.amin_anatomy benchmark NEW_REGISTERED_DIR SOURCE_DIR CELLS_JSON NEW_KERNEL_DIR
.venv/bin/python -m simulations.drosophila.amin_anatomy cable NEW_REGISTERED_DIR SOURCE_DIR CELLS_JSON NEW_CABLE_DIR
.venv/bin/python -m pytest -q tests/test_drosophila*.py
```

Final local artifacts, all under `.live/research/flywire783/`:

- Primary source material: `amin2020-primary-20260910/`.
- Registered geometry and extraction audit: `amin-registered-apl-20260910-v2/`.
- Final static reference results: `amin-anatomical-kernel-20260910-v2/`.
- Full cable states, inputs and observations: `amin-registered-cable-20260910/`.

The cable recording occupies about 1.4 GiB. It remains local, outside Git.
Its recorded analysis-source hash is
`b126554d3a8989a787c8c616a96040bd209d3ddc24153797ea485250e77de06e`.
Earlier candidate extractions and static outputs are retained, not overwritten.
All 116 fly tests pass, including 12 new geometry/kernel tests. No live servers,
embodied suites or new agent versions were started by this experiment.
