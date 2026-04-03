# Module Layout

← [Index](../INDEX.md) | [Architecture Overview](overview.md)

```
active-inference/
├── simulations/                    Core engine (organism-agnostic)
│   ├── engine.py                   SimulationEngine
│   ├── sensorimotor_loop.py        SensorimotorLoop + FreeEnergyTrace
│   ├── base_body.py                BaseBody, BodyState
│   ├── base_environment.py         BaseEnvironment, EnvironmentObservation
│   ├── base_nervous_system.py      BaseNervousSystem
│   ├── types.py                    Type aliases and protocols
│   ├── connectome_loader.py        ConnectomeData → PAULA NeuronNetwork
│   ├── paula_loader.py             Ensures neuron-model on sys.path
│   ├── run_log.py                  Run logging and post-hoc loading
│   ├── interactive/                Base interactive viewer
│   │   └── base.py                 BaseInteractiveViewer (ABC)
│   │
│   └── c_elegans/                  C. elegans organism
│       ├── simulation.py           CElegansEngine factory
│       ├── connectome.py           Cook 2019 connectome loader + cache
│       ├── neuron_mapping.py       CElegansNervousSystem (302 neurons)
│       ├── body.py                 CElegansBody (MuJoCo wrapper)
│       ├── body_model.xml          MJCF model (13 segments, 12 actuated joints, 48 actuators)
│       ├── environment.py          AgarPlateEnvironment
│       ├── sensors.py              SensorEncoder
│       ├── muscles.py              NeuromuscularJunction
│       ├── config.py               Biological constants
│       └── interactive_viewer.py   2-D real-time viewer
│
├── scripts/
│   ├── run_c_elegans.py            Main simulation runner
│   ├── evolve_food_seeking.py      Differential evolution of neuromod gains
│   ├── analyze_neuromod_food_seeking.py  Post-hoc analysis
│   ├── apply_evolved_config.py     Write evolved params back to source
│   └── download_connectome.py      One-shot connectome cache
│
├── data/c_elegans/                 Cached connectome JSON
├── logs/                           Per-run output directories
└── evolved_food_seeking_config.json  Best evolved parameters
```

## Key Packages

| Path | Docs |
|------|------|
| `simulations/engine.py` | [SimulationEngine](../engine/simulation-engine.md) |
| `simulations/sensorimotor_loop.py` | [SensorimotorLoop](../engine/sensorimotor-loop.md) |
| `simulations/connectome_loader.py` | [Connectome Pipeline](../connectome/connectome-data.md) |
| `simulations/c_elegans/` | [C. elegans](../c-elegans/overview.md) |
| `scripts/` | [Scripts](../scripts/run-c-elegans.md) |
