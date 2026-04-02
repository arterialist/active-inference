# Evolution — Output Files

← [Index](../INDEX.md) | [Algorithm](algorithm.md)

## Best Genome: `evolved_food_seeking_config.json`

```json
{
    "K_STRESS_SYN": 2655.68,
    "K_REWARD_SYN": 2678.00,
    "K_VOL_STRESS": 3689.72,
    "K_VOL_REWARD": 1205.26,
    "STRESS_DEADZONE": 0.00228,
    "CHEM_EMA_ALPHA": 0.0255,
    "TONIC_FWD_CMD": 0.262,
    "TONIC_FWD_MOTOR": 0.098,
    "neuron_params": {
        "motor":   { "w_tref": [29.82, -16.58] },
        "sensory": { "w_tref": [14.97, -8.83] }
    }
}
```

## Checkpoint Format

Adds metadata to the above:

| Field | Description |
|-------|-------------|
| `best_x` | Raw normalised parameter vector |
| `best_dist_mm` | Best minimum distance achieved (mm) |
| `generation` | Last completed generation |
| `n_evals` | Total fitness evaluations |
| `elapsed_min` | Wall-clock minutes elapsed |
| `evol_args` | CLI arguments used |

## All Evolved Config Files

| File | Description |
|------|-------------|
| `evolved_food_seeking_config.json` | Best parameters from primary run |
| `evolved_food_seeking_config_2.json` | Alternative run |
| `evolved_food_seeking_checkpoint.json` | Latest checkpoint with metadata |
| `*.bak` | Backup copies |

## See Also

- [apply_evolved_config.py](../scripts/analysis-tools.md) — writes evolved params back to source code
- [run_c_elegans.py](../scripts/run-c-elegans.md) — `--evol-config` flag to use evolved params
- [Configuration Reference](../config/nervous-system-constants.md) — what evol_config keys mean
