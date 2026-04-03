# Agar Plate Environment

← [Index](../INDEX.md) | [C. elegans Overview](overview.md) | **File:** `simulations/c_elegans/environment.py`

`AgarPlateEnvironment` models a circular agar plate with chemical point sources.

## ChemSource Dataclass

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `molecule` | `str` | — | Chemical name |
| `position` | `ndarray` | — | 3-D position in metres |
| `max_concentration` | `float` | 1.0 | Peak concentration |
| `decay_constant` | `float` | 1800.0 | Exponential decay rate |
| `valence` | `str` | "attractive" | "attractive" or "aversive" |

## Concentration Formula

```
C(p) = max_concentration × exp(−decay_constant × ‖p − source‖)
```

## Chemical Sources

**Food items** generate two gradients:
- **NaCl** — sensed by ASE neurons
- **butanone** — sensed by AWC neurons

**Additional sources:**
- 2-nonanone aversive odour (sensed by AWB neurons), at fixed offset
- Nociceptive zone — circular region where `is_nociceptive()` returns true (sensed by ASH neurons)

## Food Dynamics

| Method | Description |
|--------|-------------|
| `step()` | Checks if head is within `FOOD_CONSUMPTION_RADIUS_M` → removes food |
| `add_food(position)` | Creates new food source at runtime (used by interactive viewer) |
| `remove_food_near(position, radius)` | Removes nearest food within radius |

## Observation Construction (`_build_observation`)

- **Chemicals:** max (strongest) concentration per molecule at the head position, keyed by molecule name
- **Touch:** generates `"nociceptive"` and `"wall"` contact-force entries based on worm position relative to nociceptive zone and plate boundary
- **Proprioception:** empty dict (joint-angle data is available via `BodyState`, not the environment observation)

## See Also

- [Sensor Encoder](sensor-encoder.md) — maps chemicals to neuron names
- [Biological Constants](biological-constants.md) — FOOD_GRADIENT_DECAY, FOOD_CONSUMPTION_RADIUS_M
- [Abstract Interface](../engine/interfaces.md) — BaseEnvironment
