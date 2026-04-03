# NeuronNetwork

← [Index](../INDEX.md) | [PAULA Theory](../theory/paula.md) | **File:** `neuron-model/neuron/network.py`

Manages a collection of `Neuron` objects with inter-neuron signal propagation.

## Event Scheduling

Uses a pair of **calendar-queue wheels** (circular buffers of size `max_delay + 1`). Events are placed in slots by `arrival_tick % wheel_size`, giving O(1) insertion and retrieval.

## run_tick() Sequence

1. Extract events from current wheel slot.
2. Write external inputs directly to neuron `input_buffer` arrays.
3. Deliver presynaptic events via `fast_connection_cache` — a dict mapping `(source_id, terminal_id)` to a list of `(target_buffer_ref, synapse_index)` pairs. Avoids dictionary lookups in the hot loop.
4. Deliver retrograde events to target neurons.
5. Call `tick()` on every neuron, collecting output events.
6. Schedule new events into future wheel slots.
7. Record history (membrane potentials, firing flags, activity).

## Performance Optimisations

| Technique | Benefit |
|-----------|---------|
| Pre-allocated `input_buffer` numpy arrays (shape `(num_inputs, 4)`, hardcoded) | Avoids per-tick allocation |
| `fast_connection_cache` stores direct numpy array references | Signal delivery is a single indexed addition |
| Tuple events `(source_id, terminal_id, info_value)` | Avoids object allocation |
| Calendar-queue wheels | O(1) insertion/retrieval vs priority queues |

## See Also

- [tick() Method](tick-method.md) — the per-neuron tick called by run_tick()
- [build_paula_network()](../connectome/build-paula-network.md) — assembles a NeuronNetwork from connectome
