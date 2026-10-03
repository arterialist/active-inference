# Deferred research possibilities

These are retained hypotheses, not active tasks or prerequisites for the current
experiments. Recording a possibility does not establish its necessity or
authorize a change to PAULA's defining dynamics.

## Spatial waves and extracellular field coupling

Recorded 9 September 2026 from the research side conversation. Status: deferred
at the user's request. No implementation or new experiment was started.

### Question to retain

Could physically organized interactions between PAULA populations support
integration across sensory, memory, regulatory and motor processes that the
current nonspatial wiring does not provide? In particular, could an endogenous
electric field add a useful coupling mechanism beyond synapses?

The user's broader hypothesis concerns global experience integration through
the physics of neurons. Collective dynamics may coordinate populations across
distances and durations unavailable to an individual cell. A functional result
would still need a separate argument connecting it to conscious experience.

### Distinctions that matter

| Mechanism | What would need to be represented |
| --- | --- |
| Oscillations and coordination through synapses | Cellular dynamics, excitation/inhibition, recurrence and transmission delays. Explicit 3D geometry is not inherently required. |
| Spatial traveling activity waves | Spatially organized connectivity and propagation. A 2D sheet or coupled sheets can support waves; a 3D volume is one option, not a prerequisite. |
| Extracellular electric fields affecting neurons, or ephaptic coupling | Current sources, spatial geometry, a conducting medium and a specified response of the membrane to the field. This is an additional physical interaction. |

Neuronal ionic currents generate the extracellular electric and magnetic signals
measured by EEG and MEG. A traveling pattern of neuronal activation is not the
same thing as radiated electromagnetic energy. Also, "neural field" models can
describe spatial population activity without simulating an electromagnetic
field. A field visualization or forward model alone does not make the field
feed back onto the neurons.

### PAULA's inspected boundary

In the code inspected on this date, `neuron/neuron.py` in the sibling
`neuron-model` repository schedules an input using
`current_tick + self.distances[synapse_id]` and attenuates it using
`delta_decay ** distance`. These are effective delay/attenuation parameters,
not physical coordinates. The current audiovisual preparation samples graph
connections without a tissue geometry. No extracellular-field solver or
ephaptic feedback was found in the inspected neural runtime.

This does not establish an inability to produce collective waves. A spatial
construction could generate connections and delays while retaining PAULA's
equations. Positions used only by a viewer would have no dynamical effect.
Physical-field claims would require additional source and membrane assumptions;
PAULA's current activity values are not automatically calibrated electric
currents or EEG measurements.

### What could justify revisiting it

A useful occasion would be a concrete question about cross-population timing,
spatial propagation or shared regulation. A failure of multimodal learning by
itself would not identify missing field coupling as its cause.

One possible contrast would test whether joint sensory information becomes
usable through collective coordination while individual sensory representations
remain available. Disrupting and restoring cross-population timing could help
separate that contribution from changes in local activity or memory. Keep
adaptation enabled and account for input, rate and gain confounds. This is a
candidate experiment, not a new mandatory test campaign.

If direct field coupling becomes the question, compare a declared endogenous
field model with a matched synaptic-only system. A reduced, physically justified
coupling model may suffice; a full 3D electromagnetic wave engine should not be
assumed necessary. Quasi-static volume-conductor models are established for
neural field measurements, but closing the field-to-neuron feedback loop is a
separate modeling step. Any new PAULA extension needs a specific justification.

The target would be useful, selective coordination. Maximum synchrony, an
attractive wave animation or a good EEG-like spectrum would not establish
integration, biological fidelity or consciousness.

### Sources and evidence status

- [Cardin et al., 2009, *Driving fast-spiking cells induces gamma rhythm and controls sensory responses*](https://www.nature.com/articles/nature08002).
  Causal in-vivo evidence for inhibitory-cell involvement in gamma generation
  and phase-dependent sensory responses. Not evidence for field-mediated consciousness.
- [Davis et al., 2021, *Spontaneous traveling waves naturally emerge from horizontal fiber time delays and travel through locally asynchronous-irregular states*](https://www.nature.com/articles/s41467-021-26175-1).
  Spatial spiking-network modeling and comparison with recordings support
  synaptic-delay waves without cellular lockstep. Not a PAULA result.
- [Anastassiou et al., 2011, *Ephaptic coupling of cortical neurons*](https://pubmed.ncbi.nlm.nih.gov/21240273/).
  Rat cortical-slice experiments show extracellular fields affecting membrane
  potentials and spike timing. This does not establish brain-wide experience integration.
- [Hagen et al., 2018, *Multimodal modeling of neural network activity: computing LFP, ECoG, EEG, and MEG signals with LFPy 2.0*](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2018.00092/full).
  Biophysical forward models relate current sources and geometry to measurable
  fields. Computing a field is distinct from demonstrating its causal feedback.
- [Das et al., 2026, *Planar, spiral, and concentric traveling waves distinguish behavioral states in human memory*](https://www.nature.com/articles/s41467-026-71386-z).
  Human recordings associate wave patterns with memory processes and items.
  Observational evidence, not proof of a consciousness mechanism.
- [Banaie Boroujeni and Kastner, July 2026 preprint, *Ephaptic coupling improves the neural population code*](https://www.biorxiv.org/content/10.64898/2026.07.28.741366v1).
  Follow-up lead only: the indexed abstract and text excerpts report different
  effects of endogenous versus externally imposed fields, including improved
  coding with decorrelation in their model. Full methods were not reviewed
  because access failed. Recheck publication status and methods before relying
  on it. Its relevance is that more synchrony need not mean better integration.
