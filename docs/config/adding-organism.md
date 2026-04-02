# Adding a New Organism

← [Index](../INDEX.md) | [Module Layout](../architecture/module-layout.md)

Steps to extend the framework with a new organism.

## Checklist

1. **Create `simulations/<organism>/`** — new module directory.

2. **Define or download connectome** as a `ConnectomeData`.  
   See [ConnectomeData](../connectome/connectome-data.md).

3. **Call `build_paula_network(connectome, ...)`** to get a `NeuronNetwork`.  
   See [build_paula_network()](../connectome/build-paula-network.md).

4. **Subclass `BaseNervousSystem`** — implement `reset()`, `tick()`, `get_neuron_states()`.  
   See [Interfaces](../engine/interfaces.md).

5. **Subclass `BaseBody`** — implement MuJoCo wrapper with `reset()`, `step()`, `get_state()`.  
   See [Interfaces](../engine/interfaces.md), [MuJoCo Body](../c-elegans/mujoco-body.md) for reference.

6. **Subclass `BaseEnvironment`** — implement stimulus generation and food dynamics.  
   See [Interfaces](../engine/interfaces.md), [Agar Plate Environment](../c-elegans/agar-plate-environment.md) for reference.

7. **Create a factory function** (like `build_c_elegans_simulation`) that assembles all components.  
   See [Engine Factory](../c-elegans/engine-factory.md) for the assembly pattern.

8. **`SensorimotorLoop` wraps automatically** — active inference framing (FreeEnergyTrace) is inherited for free.

## Notes

- Motor decoding is organism-specific — implement your own `_decode_motor_outputs` or use raw motor neuron spikes directly.
- Neuromodulation is optional — `enable_m0`/`enable_m1` flags in the factory are the standard pattern.
- The [Biological Constants](../c-elegans/biological-constants.md) and [Neuron Presets](neuron-presets.md) files are *C. elegans*-specific; define your own per organism.
