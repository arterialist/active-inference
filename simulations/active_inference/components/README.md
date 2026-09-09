# Component fragments

The custom organism is one PAULA topology.  Components contribute populations,
synapses, sensory ports, and declared capabilities to that topology; they are
not independent controllers.

Current extracted fragments:

- `body/world.py`: MuJoCo body and physiological state/transducers;
- `body/metabolism.py`: normalized metabolic afferent contract;
- `arbitration/paula.py`: accepted FORAGE/HOME/EXPLORE arbiter plus optional
  SLEEP population;
- `arbitration/metabolic_parts.py`: V3 body afferents and neural WTA wiring.
- `body/obstacles.py`: V4 physical barrier/transducer contract;
- `sensory/obstacle.py`: bilateral range and delayed-onset PAULA populations;
- `motor/obstacle_reflex.py`: crossed obstacle turn, brake, and wall-follow
  outputs converging on the established motor core.

Experimental population building blocks, separate from V1-V4:

- `learning/regional_regulation.py`: build-time local activity-regulator wiring
  and a degree-preserving shuffled control. Short audiovisual integration is
  measured across four graph seeds; long-run memory and embodiment are unproved.
- `learning/projection_terminals.py`: candidate separation of native output
  terminal adaptation by projection family. Includes a capacity/fan-out-matched
  shuffled control. Initial forward wiring and local retrograde isolation are
  tested. Paired/swapped audiovisual experiments at seeds 11 and 23 do not
  establish a replicated recall repair; some seed-23 readouts change in the
  predicted directions without consistent replication/control superiority.
  No demo uses it.
- The population learning experiments can select the sibling neuron library's
  `neuron.extensions.experimental.bounded_plasticity.BoundedPlasticityNeuron`.
  This local rule addresses inhibitory-weight sign reversal. It is not enabled
  in the existing demo agents, and numerical stability is not memory acceptance.

The central-complex, visual, navigation, Mushroom Body, and motor blocks remain
in their existing maintained modules because their neuron-ID/synapse ordering
is still part of the accepted compatibility surface; they are not duplicated
as parallel controllers.  `core/brain_composer.py` is the single build seam:
it validates named components before invoking that compatibility builder, and
it has no runtime behaviour after the network is built.  This is a real partial
extraction with explicit contracts, not a claim that every historical block
has already been mechanically split into files.
