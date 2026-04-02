# Run Logs

← [Index](../INDEX.md) | **File:** `simulations/run_log.py`

Each logged run produces a directory under `logs/` with a timestamped name.

## Directory Contents

| File | Format | Contents |
|------|--------|----------|
| `config.json` | JSON | RunConfig, RunSummary, FreeEnergyTrace, data_keys |
| `data.npz` | Numpy compressed | All time-series arrays |
| `positions.csv` | CSV | [tick, x, y, z] centre-of-mass |
| `head_position.csv` | CSV | [tick, x, y, z] head position |
| `joint_angle.csv` | CSV | [tick, joint_0, joint_1, ...] |
| `motor_output.csv` | CSV | [tick, muscle_0, muscle_1, ...] |
| `chemical.csv` | CSV | [tick, NaCl, butanone, ...] |
| `neural_S.csv` | CSV | [tick, neuron_0_S, neuron_1_S, ...] |
| `neural_fired.csv` | CSV | [tick, neuron_0_fired, ...] |
| `elapsed_ms.csv` | CSV | [tick, ms] |

## Loading Logs

```python
from simulations.run_log import load_run_log
config, data = load_run_log(Path("logs/my_run"))
# config: dict with RunConfig, RunSummary, data_keys
# data: dict[str, ndarray] with tick, position, head_position, etc.
```

## See Also

- [run_c_elegans.py](../scripts/run-c-elegans.md) — `--save-log` flag
- [analyze_neuromod_food_seeking.py](../scripts/analysis-tools.md) — consumes log dirs
- [SensorimotorLoop](../engine/sensorimotor-loop.md) — source of FreeEnergyTrace data
