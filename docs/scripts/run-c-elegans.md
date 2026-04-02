# run_c_elegans.py

← [Index](../INDEX.md) | **File:** `scripts/run_c_elegans.py`

Main simulation runner.

## Usage

```bash
# Basic simulation
uv run python scripts/run_c_elegans.py --steps 1000

# With plot and animation
uv run python scripts/run_c_elegans.py --steps 5000 --save-plot --save-animation

# With evolved config
uv run python scripts/run_c_elegans.py --steps 50000 --evol-config evolved_food_seeking_config.json

# Interactive 2-D viewer
uv run python scripts/run_c_elegans.py --interactive --evol-config evolved_food_seeking_config.json

# 3-D MuJoCo viewer (requires mjpython)
uv run mjpython scripts/run_c_elegans.py --viewer --steps 500

# Multiple food sources
uv run python scripts/run_c_elegans.py --food-positions "0.5,0.5 -0.5,-0.5" --food-scale 0.01

# Ablation: disable neuromodulators
uv run python scripts/run_c_elegans.py --no-M0 --no-M1
```

## Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | 500 | Number of physics steps |
| `--food-x` | 0.0005 | Food x-position (metres) |
| `--food-z` | 0.0 | Food z-position (metres) |
| `--food-positions` | — | Multiple foods as "x1,z1 x2,z2 ..." |
| `--food-scale` | 1.0 | Scale factor for food coordinates |
| `--save-plot` | false | Save trajectory + motor wave plot |
| `--plot-output` | c_elegans_run.png | Plot file path |
| `--save-animation` | false | Save animated GIF |
| `--animation-output` | c_elegans_run.gif | Animation file path |
| `--animation-fps` | 15 | Animation frame rate |
| `--animation-frames` | 300 | Max animation frames |
| `--save-log` | false | Save full run log to disk |
| `--log-dir` | auto-timestamped | Log directory path |
| `--viewer` | false | 3-D MuJoCo viewer |
| `--interactive` | false | 2-D matplotlib viewer |
| `--no-cache` | false | Force fresh connectome download |
| `--no-M0` | false | Disable stress modulator |
| `--no-M1` | false | Disable reward modulator |
| `--verbose` | false | PAULA INFO logging |
| `--evol-config` | — | Path to evolved config JSON |

## StreamingCollector

Pre-allocates numpy arrays and fills them step-by-step to avoid holding full `SimulationStep` dicts in memory.

Arrays collected: `positions`, `head_positions`, `elapsed_ms`, `ticks`, `prediction_error`, `motor_entropy`, `joint_angles`, `motor_outputs`, `chemicals`, `neural_S`, `neural_fired`.

## Plot Output (3 panels)

1. Head trajectory on floor plane (mm scale) with food markers.
2. Motor wave heatmap: dorsal − ventral activation per body segment over time.
3. Prediction error and motor entropy traces.

## See Also

- [Interactive Viewer](../c-elegans/interactive-viewer.md) — `--interactive` mode details
- [Data Logging](../data-logging/run-logs.md) — `--save-log` output format
- [Evolved Config](../evolution/output.md) — `--evol-config` file format
