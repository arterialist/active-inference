# ConnectomeData

← [Index](../INDEX.md) | **File:** `simulations/connectome_loader.py`

## Dataclass

```python
@dataclass
class ConnectomeData:
    neurons:             list[NeuronInfo]
    chemical_edges:      list[SynapticEdge]
    gap_junction_edges:  list[SynapticEdge]
    name_to_info:        dict[str, NeuronInfo]   # built in __post_init__
```

## NeuronInfo

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | e.g. "ASEL", "DB3" |
| `neuron_type` | `str` | "sensory", "interneuron", "motor", "unknown" |
| `paula_id` | `int` | Stable integer ID for PAULA |
| `in_degree_chem` | `int` | Incoming chemical synapses |
| `in_degree_gap` | `int` | Incoming gap junctions |
| `out_degree_chem` | `int` | Outgoing chemical synapses |
| `out_degree_gap` | `int` | Outgoing gap junctions |

## SynapticEdge

| Field | Type | Description |
|-------|------|-------------|
| `pre_name` | `str` | Presynaptic neuron |
| `post_name` | `str` | Postsynaptic neuron |
| `synapse_type` | `str` | "chemical" or "gap_junction" |
| `weight` | `float` | Raw synapse count from the connectome |

## See Also

- [build_paula_network()](build-paula-network.md) — converts ConnectomeData to NeuronNetwork
- [C. elegans Loader](celegans-loader.md) — loads Cook 2019 data into ConnectomeData
