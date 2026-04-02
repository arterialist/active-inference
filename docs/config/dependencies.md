# Dependencies

← [Index](../INDEX.md)

From `pyproject.toml`. Python ≥ 3.11. Build system: Hatchling.

## Runtime

| Package | Purpose |
|---------|---------|
| numpy | Array computation |
| mujoco | Physics simulation |
| loguru | Structured logging |
| networkx | Graph utilities |
| pandas | Data analysis |
| scipy | Optimisation (differential_evolution) |
| matplotlib | Plotting and animation |
| tqdm | Progress bars |
| cect | OpenWorm ConnectomeToolbox (Cook 2019 data) |

## Dev

| Package | Purpose |
|---------|---------|
| pytest | Test runner |
| pytest-asyncio | Async test support |
| ipython | Interactive exploration |

## External: neuron-model

The PAULA neuron and network implementations live in a separate repo:
<https://github.com/arterialist/neuron-model>

Loaded at runtime via `simulations/paula_loader.py`, which ensures the repo is on `sys.path`.
