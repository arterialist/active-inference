# Ablation Flags

← [Index](../INDEX.md) | [NeuronParameters](neuron-parameters.md)

The `ablation` parameter on `Neuron.__init__` accepts a comma-separated string of flags to selectively disable PAULA mechanisms.

## Flags

| Flag | Effect |
|------|--------|
| `thresholds_frozen` | r and b do not update with M_vector |
| `tref_frozen` | t_ref stays at initial value |
| `directional_error_disabled` | Direction always +1 (no LTD) |
| `weight_update_disabled` | Postsynaptic weights frozen |
| `retrograde_disabled` | No retrograde signal generation |

## Usage

```python
neuron = Neuron(params, ablation="weight_update_disabled,retrograde_disabled")
```

## See Also

- [Scripts — run_c_elegans.py](../scripts/run-c-elegans.md) — `--no-M0` / `--no-M1` flags for neuromodulator ablation
- [tick() Method](tick-method.md) — the phases these flags disable
