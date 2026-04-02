# Interactive Viewer

← [Index](../INDEX.md) | [C. elegans Overview](overview.md) | **File:** `simulations/c_elegans/interactive_viewer.py`

`CElegansInteractiveViewer` provides a real-time 2-D matplotlib display.

## Panels

| Panel | Content |
|-------|---------|
| **Left** | Worm trajectory on agar plate with food markers (magenta stars). Left-click adds food; right-click removes nearest food. |
| **Right** | Rolling metrics — prediction error, motor entropy, M0 (red), M1 (green). |

## Usage

```bash
uv run python scripts/run_c_elegans.py --interactive --evol-config evolved_food_seeking_config.json
```

The main loop calls `engine.step()` repeatedly and updates matplotlib artists each frame.

## See Also

- [run_c_elegans.py](../scripts/run-c-elegans.md) — `--interactive` flag
- [SensorimotorLoop](../engine/sensorimotor-loop.md) — source of prediction error / motor entropy metrics
