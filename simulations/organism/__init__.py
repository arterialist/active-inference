"""An insect-brain-inspired embodied organism built from PAULA spiking neurons.

Three functional brain regions, composed via the PAULA construction kit (`paula_agent.ckit`),
embodied in a MuJoCo body and closed-loop:

  - mushroom_body   : olfactory associative learning (sparse Kenyon cells + dopamine-gated MBONs)
  - central_complex : path integration -> home vector (analog leaky integrators)
  - action_selection: winner-take-all arbitration over drives
  - body            : the MuJoCo planar body (antennae, nest, predator)
  - organism        : the full closed-loop creature (Organism)
  - record          : run a simulation and render a self-contained HTML visualization

See README.md for how to run.
"""
