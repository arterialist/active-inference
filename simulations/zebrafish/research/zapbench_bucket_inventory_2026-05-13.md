# ZAPBench Public Asset Inventory - 2026-05-13

This inventory was collected from the ZAPBench landing page, dataset README,
Google Cloud Storage bucket listing API, the public `google-research/zapbench`
repository, and the released processing notebooks.

## Primary Sources

- Landing page: https://zapbench-release.storage.googleapis.com/landing.html
- Dataset README: https://zapbench-release.storage.googleapis.com/volumes/README.html
- GCS bucket browser root: https://storage.googleapis.com/zapbench-release/browse.html
- GitHub repository: https://github.com/google-research/zapbench
- ICLR 2025 paper: https://openreview.net/forum?id=oCHsDpyawq
- Google Research blog: https://research.google/blog/improving-brain-models-with-zapbench/
- Raw stimulus/ephys notebook object: https://storage.googleapis.com/zapbench-release/notebooks/20241018/stimuli_and_ephys.ipynb

## Dataset Scope

- Whole-brain larval zebrafish neural activity, acquired with 4D light-sheet calcium imaging.
- Extracted trace matrix: 7,879 imaging frames by 71,721 segmented neurons.
- Stimulus conditions: `gain`, `dots`, `flash`, `taxis`, `turning`, `position`, `open loop`, `rotation`, `dark`.
- Public connectome status: the same fish is undergoing synaptic-level anatomical mapping, but the connectome is not released in the public ZAPBench bucket as of this inventory.

## High-Value GCS Prefixes

- `volumes/20240930/raw/`: raw functional activity volume.
- `volumes/20240930/anatomy/`: functional anatomy volume.
- `volumes/20240930/aligned/`: motion-stabilized aligned activity.
- `volumes/20240930/df_over_f/`: aligned and normalized activity.
- `volumes/20240930/segmentation/`: segmentation volume and dataframes.
- `volumes/20240930/traces/`: Zarr3 extracted activity traces used for time-series forecasting.
- `volumes/20240930/traces.zip`: downloadable trace archive, 1,978,749,092 bytes.
- `volumes/20240930/position_embedding/`: 71,721 x 192 neuron position embedding Zarr.
- `volumes/20240930/stimuli_features/`: 7,879 x 26 stimulus covariate matrix.
- `volumes/20240930/stimulus_evoked_response/`: stimulus-evoked response Zarr3.
- `volumes/20240930/stimuli_raw/stimuli_and_ephys.10chFlt`: raw 10-channel stimulus/ephys stream, 1,728,269,600 bytes.
- `volumes/20240930/annotations/`: training and evaluation subvolumes for segmentation.
- `ffn_checkpoints/20240930/ckpt-332.flax`: one-shot FFN segmentation checkpoint.
- `dataframes/20250131/combined.json`: benchmark result dataframe, 1,307,618 bytes.
- `figures/20250131/`: benchmark HTML/JSON/PDF/SVG result charts.
- `figures/20250131/fluroglancer/`: per-condition prediction chart JSON/HTML files for interactive visual comparison.
- `inference/20250131/`: published model prediction outputs used by result visualizers.
- `neuroglancer/20250131/segmentation.json`: Neuroglancer state for segmentation.
- `neuroglancer/20250131/traces_rastermap.json`: Neuroglancer state for rastermap-sorted traces.
- `neuroglancer/20250131/volumetric_activity.json`: Neuroglancer state for volumetric activity.
- `fluroglancer/open.html`: WebGL viewer entrypoint for calcium fluorescence and predictions.

## Released Notebook / Code Facts Used by the Decoder

- `zapbench/constants.py` defines condition offsets `(0, 649, 2422, 3078, 3735, 5047, 5638, 6623, 7279, 7879)` and the nine condition names.
- The trace TensorStore spec is `gs://zapbench-release/volumes/20240930/traces/` with transform shape `[7879, 71721]`.
- The stimulus feature TensorStore spec is `gs://zapbench-release/volumes/20240930/stimuli_features/` with shape `[7879, 26]`.
- The position embedding TensorStore spec is `gs://zapbench-release/volumes/20240930/position_embedding/` with shape `[71721, 192]`.
- The raw stimulus/ephys notebook identifies the 10-channel raw stream and uses channel 2 for TTL, channel 3 for `stimParam4`, channel 4 for condition index, channel 6 for `stimParam3`, and channel 8 for visual velocity.

## Local Decoder Artifacts

- Compact direct-ephys linear decoder: `active-inference/simulations/zebrafish/data/zapbench_ephys_tail_action_decoder.npz`.
- Same-volume strict no-lag baseline: `active-inference/simulations/zebrafish/data/zapbench_ephys_tail_action_decoder_lag0.npz`.
- Stimulus-implied fallback decoder: `active-inference/simulations/zebrafish/data/zapbench_tail_action_decoder.npz`.
- Direct raw labels and reports: `zebrafish_live_demo_v2/analysis/out/zapbench_ephys_action_decoder/`.
- Experimental nonlinear report/artifact: `zebrafish_live_demo_v2/analysis/out/zapbench_ephys_action_decoder_sklearn/`.

