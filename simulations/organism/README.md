# The Organism — an insect brain in a body

An insect-brain-inspired agent built from **PAULA spiking neurons**, embodied in a **MuJoCo** body and
run closed-loop. Three functional brain *regions* (not reflexes) produce genuinely context-dependent,
learned behaviour: it forages, learns which smells are food, flees a predator, and path-integrates home
like a desert ant.

Everything is composed with the PAULA construction kit (`paula_agent.ckit`) from the sibling
`neuron-model` repo — **no edits to the neuron model**. PAULA is put on `sys.path` via this repo's
`simulations/paula_loader.py`, so no manual `PYTHONPATH` is needed.

## The three brain regions (~150 neurons)

| module | region | what it does |
|---|---|---|
| `mushroom_body.py` | mushroom body | sparse Kenyon-cell code + APL feedback inhibition + dopamine-gated `reward_hebb` MBONs → **learns odour → approach/avoid** |
| `central_complex.py` | central complex | four analog leaky integrators accumulate a **path-integration home vector** (<2° error) |
| `action_selection.py` | action selection | winner-take-all over drives (`Sel4`, 4-way: forage/flee/home/explore) |
| `body.py` | MuJoCo body | planar bug with two odour antennae, a nest disc, and a mocap predator |
| `organism.py` | the creature | `Organism` — wires the regions into a closed sensorimotor loop |
| `record.py` | visualization | run a sim → self-contained `organism_demo.html` (2D canvas replay + real MuJoCo frames) |

## Requirements

Uses this repo's own venv (`.venv`), which already has `mujoco`, `numpy`, and `pillow`. The PAULA neuron
model must be present at the sibling path `../neuron-model` (that's what `paula_loader` points at).

## Run it

From the repo root (`active-inference/`), using its venv:

```bash
# Full organism — 6 seeds x 2200 steps, prints the ethogram (~5-7 min):
.venv/bin/python -m simulations.organism.organism

# Each brain region standalone (fast self-tests):
.venv/bin/python -m simulations.organism.central_complex     # path-integration accuracy
.venv/bin/python -m simulations.organism.mushroom_body       # odour learning (approach A / avoid B)
.venv/bin/python -m simulations.organism.action_selection    # WTA arbitration
.venv/bin/python -m simulations.organism.body                # physics + offscreen render check

# Build the visualization (renders seed 2, 2200 steps -> organism_demo.html; open it in a browser):
.venv/bin/python -m simulations.organism.record 2 2200
#   ...or auto-pick the richest of 8 seeds:
.venv/bin/python -m simulations.organism.record -
```

Each file also runs directly (`.venv/bin/python simulations/organism/organism.py`) thanks to a small
`sys.path` bootstrap at the top.

## Use it in your own code

```python
from simulations.organism.organism import Organism

o = Organism(seed=2)
for _ in range(2200):
    mode = o.step()          # 'forage' | 'flee' | 'home' | 'explore'
# behaviour tallies: o.modes, o.good, o.tox, o.trips, o.avoid_events, o.hurt
# full per-step trace:  o.log   (x, y, mode, energy, smell, valence, dnest, tasted, ...)
```

## What "complex" does and doesn't mean here

The behaviour is genuinely context-dependent and shaped by experience — which action wins is decided each
tick by the spiking WTA reading four drives; approach-vs-avoid comes from a mushroom body that *learned*
the odour→value association by being poisoned; the way home is a path integral.

But this is action-**selection**, not planning: there is no world-model and no goals beyond the drives, so
it cannot form multi-step plans, and over long excursions the path-integrated home vector accumulates
drift — if the predator harasses it far enough off course it will **abandon the trip** rather than find
the nest. It is real ethology, and more than reflexes, but it is not open-ended cognition. The behaviour
balance is also sensitive to tuning (explore-flood ↔ flee-flood ↔ starvation), so `record.py` selects the
seed with the fullest ethogram.

### A representative run (seed 2, 2200 steps)

`forage 51% · flee 24% · home 13% · explore 13%` — ate 6 nutritious + 1 toxic, and after being poisoned
once **learned to steer away from the toxic smell** (299 avoidance steps), made 3 nest returns, never got
caught.
