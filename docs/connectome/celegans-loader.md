# C. elegans Connectome Loader

← [Index](../INDEX.md) | [ConnectomeData](connectome-data.md) | **File:** `simulations/c_elegans/connectome.py`

```python
def load_connectome(use_cache: bool = True) -> ConnectomeData
```

## Behaviour

**First call:** Parses Cook 2019 hermaphrodite data via the `cect` library (OpenWorm ConnectomeToolbox). Classifies each of the 302 neurons as sensory, motor, interneuron, or unknown. Extracts chemical synapses and gap junctions. Saves to `data/c_elegans/connectome_cache.json`.

**Subsequent calls:** Loads from JSON cache (fast). An in-memory cache (`_connectome_memory_cache`) persists across calls within a process — important during evolution.

## Result

| Metric | Value |
|--------|-------|
| Neurons | 302 |
| Chemical synapses | ~3,709 |
| Gap junctions | ~1,092 |

## See Also

- [ConnectomeData](connectome-data.md) — the returned type
- [Scripts — download_connectome.py](../scripts/run-c-elegans.md) — one-shot download
- [Evolutionary Optimisation](../evolution/algorithm.md) — uses in-memory cache heavily
