# Evolution — Parameter Space

← [Index](../INDEX.md) | [Algorithm](algorithm.md)

13-dimensional optimisation vector. All values normalised to `[0, 1]` internally, mapped to biological ranges via `x_to_config()`.

## Search Space

| Index | Parameter | Range | Scale |
|-------|-----------|-------|-------|
| 0 | K_STRESS_SYN | [1000, 8000] | Linear |
| 1 | K_REWARD_SYN | [1000, 8000] | Linear |
| 2 | K_VOL_STRESS | [500, 4000] | Linear |
| 3 | K_VOL_REWARD | [500, 4000] | Linear |
| 4 | STRESS_DEADZONE | [1e-6, 0.01] | Log |
| 5 | CHEM_EMA_ALPHA_SLOW | [0.01, 0.10] | Linear |
| 6 | CHEM_EMA_ALPHA_FAST | [0.05, 0.40] | Linear |
| 7 | TONIC_FWD_CMD | [0.1, 0.5] | Linear |
| 8 | TONIC_FWD_MOTOR | [0.05, 0.2] | Linear |
| 9 | Motor w_tref M0 | [15, 45] | Linear |
| 10 | Motor w_tref M1 | [-25, -5] | Linear |
| 11 | Sensory w_tref M0 | [8, 25] | Linear |
| 12 | Sensory w_tref M1 | [-12, -3] | Linear |

## Evolved Values (Best Run)

| Parameter | Default | Evolved |
|-----------|---------|---------|
| K_STRESS_SYN | 4000.0 | 2655.7 |
| K_REWARD_SYN | 4000.0 | 2678.0 |
| K_VOL_STRESS | 2000.0 | 3689.7 |
| K_VOL_REWARD | 2000.0 | 1205.3 |
| STRESS_DEADZONE | 0.00005 | 0.00228 |
| CHEM_EMA_ALPHA_SLOW | 0.01 | 0.0255 |
| TONIC_FWD_CMD | 0.25 | 0.262 |
| TONIC_FWD_MOTOR | 0.0 | 0.098 |
| Motor w_tref M0 | 30 | 29.82 |
| Motor w_tref M1 | −15 | −16.58 |
| Sensory w_tref M0 | 12 | 14.97 |
| Sensory w_tref M1 | −6 | −8.83 |

## See Also

- [Neuromodulation Gain Constants](../neuromodulation/overview.md#gain-constants) — indices 0–4
- [Dual-EMA Filter](../neuromodulation/dual-ema-filter.md) — indices 5–6
- [Tonic Drives](../neuromodulation/tonic-drives.md) — indices 7–8
- [NeuronParameters w_tref](../paula/neuron-parameters.md) — indices 9–12
