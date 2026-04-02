# build_paula_network()

← [Index](../INDEX.md) | [ConnectomeData](connectome-data.md)

Converts a `ConnectomeData` into a PAULA `NeuronNetwork` + a `name_to_paula_id` mapping.

## Signature

```python
def build_paula_network(
    connectome:         ConnectomeData,
    base_params:        NeuronParameters | None = None,
    sensory_params:     NeuronParameters | None = None,
    motor_params:       NeuronParameters | None = None,
    interneuron_params: NeuronParameters | None = None,
    weight_max:         float = 5.0,
    log_level:          str = "WARNING",
    param_overrides:    dict[str, NeuronParameters] | None = None,
) -> tuple[NeuronNetwork, dict[str, int]]
```

## Steps

1. **Assign degrees** — compute in/out degrees and stable PAULA IDs.
2. **Create neurons** — select parameter preset by type (sensory / motor / interneuron / base), with optional per-name overrides. `num_inputs` is set to the neuron's actual in-degree.
3. **Create connections** — for each edge, assign a terminal ID on the presynaptic neuron and a synapse ID on the postsynaptic neuron. Chemical synapses get `distance = 2`, gap junctions get `distance = 1`.
4. **Assemble network** — build `_EmptyTopology` (duck-type for `NetworkTopology`) and construct `NeuronNetwork`.

## Synapse Distance Constants

| Name | Value | Description |
|------|-------|-------------|
| `PAULA_SYNAPSE_LIMIT` | 4095 | 12-bit terminal/synapse ID ceiling |
| `CHEMICAL_SYNAPSE_DISTANCE` | 2 | Cable distance for chemical synapses |
| `GAP_JUNCTION_DISTANCE` | 1 | Cable distance for gap junctions |

## See Also

- [NeuronNetwork](../paula/neuron-network.md) — the returned network type
- [NeuronParameters](../paula/neuron-parameters.md) — parameter types
- [Neuron Presets](../config/neuron-presets.md) — preset functions for each neuron class
