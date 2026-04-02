# PAULA (Neuron Model)

← [Index](../INDEX.md) | [Theory](active-inference.md)

PAULA (**Predictive Adaptive Unsupervised Learning Agent**) is a spiking neuron model — the concrete implementation of the [ALERM framework](alerm.md) — that unifies:

- **Leaky integration** with configurable time constant.
- **Spike-timing-dependent plasticity (STDP)** via a directional learning window.
- **Retrograde signaling** from post- to presynaptic terminals.
- **Neuromodulation** of threshold, post-cooldown threshold, and learning window duration.
- **Dendritic cable propagation** with exponential decay and configurable delay.

Source: `neuron-model/neuron/neuron.py`, `neuron-model/neuron/network.py`

See the upstream paper: <https://al.arteriali.st/blog/paula-paper>

## Detailed Reference

| Topic | Page |
|-------|------|
| NeuronParameters | [neuron-parameters](../paula/neuron-parameters.md) |
| Synaptic structures (u_i, u_o) | [synaptic-structures](../paula/synaptic-structures.md) |
| Neuron state variables | [neuron-state](../paula/neuron-state.md) |
| tick() method (full state transition) | [tick-method](../paula/tick-method.md) |
| NeuronNetwork | [neuron-network](../paula/neuron-network.md) |
| Ablation flags | [ablation](../paula/ablation.md) |
