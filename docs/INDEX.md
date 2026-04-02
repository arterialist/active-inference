# Active Inference Sensorimotor Simulations — Reference Index

A framework for simulating artificial life through active inference. Organisms are assembled from [[paula/neuron-parameters|PAULA]] spiking neurons wired by biological connectomes and embodied in MuJoCo physics environments.

**Upstream references**

| Resource | URL |
|----------|-----|
| PAULA paper | <https://al.arteriali.st/blog/paula-paper> |
| ALERM framework | <https://al.arteriali.st/blog/alerm-framework> |
| Neuron model source | <https://github.com/arterialist/neuron-model> |

---

## Sections

| # | Topic | Files |
|---|-------|-------|
| 1 | [Architecture Overview](architecture/overview.md) | Overview, module layout |
| 2 | [Theoretical Foundations](theory/active-inference.md) | Active inference, PAULA, ALERM |
| 3 | [Core Simulation Engine](engine/simulation-engine.md) | SimulationEngine, SensorimotorLoop, interfaces |
| 4 | [PAULA Neuron Model](paula/neuron-parameters.md) | Parameters, synapses, state, tick(), network, ablation |
| 5 | [Connectome Pipeline](connectome/connectome-data.md) | ConnectomeData, build_paula_network, C. elegans loader |
| 6 | [C. elegans Implementation](c-elegans/overview.md) | Body, environment, sensors, muscles, engine, viewer |
| 7 | [Neuromodulation System](neuromodulation/overview.md) | ALERM M0/M1, dual-EMA, volume transmission, tonic drives |
| 8 | [Motor Decoding](motor-decoding/overview.md) | Segment map, graded output, E/I, reciprocal inhibition |
| 9 | [Evolutionary Optimisation](evolution/algorithm.md) | DE algorithm, parameter space, fitness, output |
| 10 | [Scripts and Entry Points](scripts/run-c-elegans.md) | run_c_elegans, evolve, analyze, apply, download |
| 11 | [Data, Logging, and Analysis](data-logging/run-logs.md) | Run logs, evolved configs |
| 12 | [Configuration Reference](config/nervous-system-constants.md) | Constants, neuron presets, motor constants, dependencies |
