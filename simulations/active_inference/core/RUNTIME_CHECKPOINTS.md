# Exact neural runtime checkpoints

`runtime_checkpoint.py` saves the actual PAULA object graph, including queued
signals, extension state, numeric types and shared input-buffer references.
Use this to branch an experiment without replaying its acquisition history.
JSON state records remain useful for inspection but are not this executable state.

Run from `active-inference` with the cached dependency:

```sh
uv run --offline --no-sync --with cloudpickle==3.1.2 python your_experiment.py
```

```python
from simulations.active_inference.core.runtime_checkpoint import (
    save_checkpoint, load_checkpoint,
)
from neuron.neuron import setup_neuron_logger

save_checkpoint(net, checkpoint_path, sources=[config_path, sensory_path])
setup_neuron_logger('CRITICAL')
branch = load_checkpoint(checkpoint_path, trusted=True)
# Deliver physical inputs to branch.network through the normal transducer.
branch.step()
save_checkpoint(branch, next_checkpoint_path, sources=[config_path, sensory_path])
```

Load only checkpoints you created and kept in a trusted local workspace. The
format executes Python during loading. Its hashes detect changes, not malicious
authors. Never load uploaded or downloaded checkpoints through a research server.

Each branch retains its own Python and legacy NumPy random-generator states.
Use `branch.step()` to advance them. It temporarily changes process-global
generators, so parallel branches need separate processes, not threads. Explicit
NumPy Generator objects inside the network are serialized with the object graph.

Python, NumPy, cloudpickle and recorded source versions must match. Neuron class
sources and their inheritance chains are recorded automatically. Supply other
experiment dependencies, configuration and input files through `sources`.
Files cannot be overwritten. Logging sinks are rebound to the local logger.
Configure that logger before loading in a fresh worker. Loading does not call
neuron constructors or restore the previous process's logging level. In the
9 September cached-factor experiment, omitting this step printed every spike
despite quiet settings in the saved cells. The workers were stopped, logs
compressed losslessly, and complete probe files reused by a verified resume.
Body state, external input cursors and environmental state are not included.
An embodied checkpoint must save those separately at the same simulation boundary.

The full-graph test saved 1,152 neurons with 1,344 signals in flight after 64
real-media ticks. All recorded fields and final recorded state matched an
uninterrupted 128-tick continuation exactly. The checkpoint was 804,944 bytes;
one measured load took 0.128 seconds. This verifies that preparation, not every
possible extension or stochastic body. See `experiments/runtime_checkpoint_probe.py`
and `.live/research/20260909_runtime_checkpoint_full_graph_seed11/summary.json`.
