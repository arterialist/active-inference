# ZAPBench Source Notes

Sources extracted on 2026-05-13:

- Landing page: https://zapbench-release.storage.googleapis.com/landing.html
- Dataset README: https://zapbench-release.storage.googleapis.com/volumes/README.html
- ICLR/OpenReview paper: https://openreview.net/forum?id=oCHsDpyawq
- Official code: https://github.com/google-research/zapbench
- Google Research blog: https://research.google/blog/improving-brain-models-with-zapbench/
- Public bucket browser: https://storage.googleapis.com/zapbench-release/browse.html#zapbench-release/volumes/

## Dataset Facts

- ZAPBench is the Zebrafish Activity Prediction Benchmark for cellular-resolution whole-brain activity prediction in larval zebrafish.
- The public benchmark traces are stored at `gs://zapbench-release/volumes/20240930/traces/`.
- Trace matrix shape is `7879 x 71721`, zarr3, float32, with time dimension first.
- Aligned stimulus covariates are stored at `gs://zapbench-release/volumes/20240930/stimuli_features/`.
- Stimulus covariate shape is `7879 x 26`, zarr v2.
- The dataset README lists additional volumes for raw functional activity, anatomy, aligned activity, aligned normalized activity, annotations, segmentation, and traces.
- Dataset license is CC-BY 4.0. Official code is Apache 2.0.

## Experimental Facts Relevant to Action Decoding

- The fish was a larval zebrafish recorded in a virtual-reality fictive-behavior setup.
- The fish was paralyzed and embedded; agarose was removed around the head and dorsal tail region.
- Left and right dorsal tail motor nerve electrical activity was recorded using suction pipettes.
- The paper defines swim signal from the 10 ms rolling standard deviation of the motor nerve electrical signal.
- In closed-loop conditions, swim signals above 2.5 standard deviations of baseline drove backward stimulus motion proportional to swim power.
- The public aligned 26-column stimulus covariate matrix is not direct left/right swim-power ground truth. Direct motor labels require processing the raw `stimuli_and_ephys.10chFlt` file.

## Condition Offsets and Names

Offsets in imaging timesteps:

```python
CONDITION_OFFSETS = (0, 649, 2422, 3078, 3735, 5047, 5638, 6623, 7279, 7879)
CONDITION_NAMES = (
    "gain",
    "dots",
    "flash",
    "taxis",
    "turning",
    "position",
    "open loop",
    "rotation",
    "dark",
)
```

The official benchmark holds out `taxis` for the standard forecasting split, while training conditions are gain, dots, flash, turning, position, open loop, rotation, and dark.

## Stimulus Feature Schema

The paper describes dimensions using 1-indexed labels. Converted to Python 0-indexed columns:

- `0`: gain, low `-1` or high `1`; `1`: gain condition indicator.
- `2`: dots orientation/coherence setting `-1` or `1`; `3`: dots condition indicator.
- `4`: flash, dark `-1` or bright `1`; `5`: flash condition indicator.
- `6`: taxis left hemifield, dark `-1` or bright `1`; `7`: taxis right hemifield; `8`: taxis condition indicator.
- `9`: turning velocity; `10` and `11`: sine/cosine direction encoding; `12`: turning condition indicator.
- `13` to `15`: position grating type one-hot; `16`: delay in `[0, 0.9]`; `17`: position condition indicator.
- `18`: open-loop condition indicator.
- `19`: rotation direction, rightward `-1` and leftward `1`; `20`: rotation condition indicator.
- `21`: dark condition indicator.
- `22` to `25`: specimen identity variables; not meaningful for this single-specimen benchmark context.

## Implementation Hook

The action decoder code lives in:

- `zebrafish_live_demo_v2/analysis/zapbench_action_decoder.py`
- `zebrafish_live_demo_v2/analysis/zapbench_ephys_action_decoder.py`
- `active-inference/simulations/zebrafish/action_decoder.py`

The first decoder trains a ridge model from random-projected ZAPBench trace windows to stimulus-implied tail action:

- `kick`: nonzero expected fictive swim/startle/turn drive.
- `side`: none/left/right.
- `force`: normalized action magnitude.

The direct decoder now also processes the public raw 10-channel ephys file:

- Raw file: `gs://zapbench-release/volumes/20240930/stimuli_raw/stimuli_and_ephys.10chFlt`.
- Local cache path used for analysis: `zebrafish_live_demo_v2/analysis/cache/zapbench/stimuli_and_ephys.10chFlt`.
- SHA-256 of the downloaded file: `589b0045ffc4cf9e4a29183579756c23419f45af0a1553dfefca578142ea46d4`.
- Raw channel inference identifies channels `0` and `1` as the two continuous motor-ephys candidates. Channels `2`, `3`, `4`, `6`, and `8` match TTL, stimulus parameter, condition-index, and visual-velocity roles from the official notebook.
- TTL alignment found `7872` high TTL peaks, `566769` plane-level low TTL peaks, `7871` aligned imaging intervals, a low/high ratio of about `71.998`, and median `5484` raw samples per volume.
- The inferred raw sample rate is about `5994 Hz`, so the 10 ms motor-envelope window is `60` samples.
- Direct labels are stored in `zebrafish_live_demo_v2/analysis/out/zapbench_ephys_action_decoder/direct_ephys_labels.npz`.
- Best calcium-delay decoder artifact: `active-inference/simulations/zebrafish/data/zapbench_ephys_tail_action_decoder.npz`, trained with `label_lag=-8`.
- Strict same-volume artifact: `active-inference/simulations/zebrafish/data/zapbench_ephys_tail_action_decoder_lag0.npz`, trained with `label_lag=0`.

The negative-lag artifact is a classifier of motor action from calcium activity after accounting for slow nuclear GCaMP dynamics. The lag-0 artifact is the stricter same-volume baseline for causal/controller experiments.
