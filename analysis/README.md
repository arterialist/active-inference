# C. elegans simulation — offline analysis

Uses dependencies already declared on **`active-inference`** (`numpy`, `scipy`, `matplotlib`).

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `WORM_NPZ` | per-script (e.g. `worm_capture.npz`) | Input `.npz` when no CLI path is given |
| `WORM_OUTDIR` | `.` or `worm_plots` | Output directory for figures |

## Layout

| Path | Role |
|------|------|
| `analysis/plots/worm_kymograph.py` | Kymographs + motor/neural panels from a capture |
| `analysis/plots/worm_long_run.py` | Run–turn–run style heading / ω statistics |
| `analysis/plots/worm_continuous.py` | Welch spectra, D–V muscle stats, long-run summaries |
| `analysis/plots/worm_decay.py` | Compare several captures (yaw amplitude vs time) |
| `analysis/plots/worm_instability.py` | Diagnostics for large `angle_max` captures |
| `analysis/dev/extract_param_spec_map.py` | Print `sim.*` → attribute map |
| `analysis/dev/list_dead_spec_fields.py` | Static list of spec fields marked unread |
| `analysis/dev/grep_spec_attribute_reads.py` | `grep` selected attributes under `simulations/` |
| `analysis/neural/a_mn_oscillator_smoke.py` | A-MN + AVA gate smoke test (no MuJoCo) |

## Run (examples)

From `active-inference/`:

```bash
uv run python -m analysis.plots.worm_kymograph ./worm_capture.npz ./worm_plots
uv run python -m analysis.plots.worm_long_run
WORM_NPZ=my.npz WORM_OUTDIR=out uv run python -m analysis.plots.worm_continuous
uv run python -m analysis.neural.a_mn_oscillator_smoke
uv run python -m analysis.dev.grep_spec_attribute_reads
```

Lab-side capture scripts (WebSocket) live in **`celegans-live-demo/analysis/`**.
