# Neuromuscular Junction

← [Index](../INDEX.md) | [C. elegans Overview](overview.md) | **File:** `simulations/c_elegans/muscles.py`

`NeuromuscularJunction` is a stateless translator with three static methods.

## Methods

| Method | Description |
|--------|-------------|
| `to_ctrl(activations)` | Converts 0-indexed internal muscle names (`seg{N}_{QUAD}`) to 1-indexed MuJoCo actuator names (`muscle_seg{N+1}_{QUAD}`) |
| `dorsal_minus_ventral(activations)` | Returns `(N_BODY_SEGMENTS,)` array of (dorsal − ventral) per segment. Positive = dorsal contraction |
| `mean_activation(activations)` | Average activation across all muscles |

## See Also

- [MuJoCo Body](mujoco-body.md) — receives the MuJoCo-named actuator commands
- [Motor Decoding](../motor-decoding/overview.md) — upstream stage that produces muscle activations
