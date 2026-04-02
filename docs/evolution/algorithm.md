# Evolutionary Optimisation — Algorithm

← [Index](../INDEX.md) | **File:** `scripts/evolve_food_seeking.py`

## Algorithm

Uses `scipy.optimize.differential_evolution` with strategy `"best1bin"` (best-of-generation, binomial crossover).

All parameters are normalised to `[0, 1]` and mapped to biological ranges via `x_to_config()`.

## Contents

| Topic | Page |
|-------|------|
| Parameter space (13-D) | [parameter-space](parameter-space.md) |
| Fitness function & robustness | [fitness-function](fitness-function.md) |
| Output files | [output](output.md) |

## Usage

```bash
# Default (30 gen × 20 pop × 50k ticks)
uv run python scripts/evolve_food_seeking.py

# Large-scale robust evolution
uv run python scripts/evolve_food_seeking.py --generations 100 --population 40 --robust

# Low-memory mode (Raspberry Pi / 8 GB machines)
uv run python scripts/evolve_food_seeking.py --low-memory --measure-memory

# Resume from checkpoint
uv run python scripts/evolve_food_seeking.py --checkpoint my_checkpoint.json
```

## Memory Management

| Flag | Effect |
|------|--------|
| `--low-memory` | Forces GC after each eval, sets `max_history=1`, suppresses connectome summary |
| `--measure-memory` | Prints peak RSS (MB) each generation |

## See Also

- [Parameter Space](parameter-space.md)
- [Fitness Function](fitness-function.md)
- [apply_evolved_config.py](../scripts/analysis-tools.md) — writes evolved params to source
