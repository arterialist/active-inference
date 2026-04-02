# Evolution — Fitness Function & Robustness

← [Index](../INDEX.md) | [Algorithm](algorithm.md)

## Fitness Function

Each genome is evaluated against **3 distinct food positions** to prevent directional overfitting:

```python
TEST_ENVIRONMENTS = [
    (-0.002,  0.002, 0.0),   # Top-left
    ( 0.002, -0.002, 0.0),   # Bottom-right
    ...
]
```

**Fitness** = mean (or worst with `--robust`) minimum distance to food across environments. **Lower is better** (DE minimises).

**Early exit:** if the worm reaches within `EAT_RADIUS_M = 0.0001` (0.1 mm), that environment scores 0.

## Robustness Protocol

| Mechanism | Description |
|-----------|-------------|
| Multi-environment testing | Prevents "lucky torpedo" genomes that only work in one direction |
| `--robust` flag | Scores on the *worst* environment instead of the average |
| Atomic JSON checkpointing | Temp file + rename on every generation — crash-safe |
| SIGINT handling | Saves checkpoint before exiting |

## See Also

- [Output Files](output.md)
- [Algorithm](algorithm.md)
