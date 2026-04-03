# Active Inference

← [Index](../INDEX.md)

The simulation frames each organism as an active inference agent. At every tick, two processes run simultaneously:

**Perception (plasticity):** Each PAULA synapse computes a prediction error vector:

```
E_dir = [ info_received − u_i.info,
          plast_received − u_i.plast,
          mod_0, mod_1, ... ]
```

The postsynaptic learning rule `Δu_i = η × direction × ‖E_dir‖ × u_i.info` adjusts synaptic weights to minimise future surprise. Over time, the network's weights encode a **generative model** of the sensory statistics the organism encounters.

**Action (motor output):** Motor neuron spikes drive muscle activations, physically moving the body. Movement changes the sensory stream, closing the perception–action loop. The organism's behaviour implicitly minimises **prediction error (free energy)** by seeking out predictable sensory states.

## See Also

- [PAULA Neuron Model](paula.md) — implements the learning and spiking mechanics
- [ALERM Framework](alerm.md) — neuromodulatory overlay on active inference
- [SensorimotorLoop](../engine/sensorimotor-loop.md) — FreeEnergyTrace instrumentation
- [Prediction Error §4.4](../paula/tick-method.md) — Phase E plasticity details
