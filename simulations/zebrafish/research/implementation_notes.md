# Implementation Notes

## Target organism

Start with larval zebrafish, not adult zebrafish. The strongest public sources
are concentrated around 5 to 7 dpf larvae: whole-brain optical imaging,
light/electron microscopy registration, larval locomotion, and compact body
kinematics.

## Suggested module split

Future files should mirror `simulations/c_elegans/`:

- `connectome.py`: load Fish1/CAVE, Svara 2022, or reduced connectome products
  into `ConnectomeData`.
- `neuron_mapping.py`: define neuron classes, atlas regions, neurotransmitter
  labels, and PAULA IDs.
- `body.py`: MuJoCo larval body with head/trunk/tail segments and water drag.
- `muscles.py`: axial myotome actuator groups and left/right tail control.
- `sensors.py`: visual flow, lateral-line water flow, vestibular, olfactory,
  thermosensory, mechanosensory, and proprioceptive encoders.
- `environment.py`: shallow aquatic arena with visual flow, prey/food,
  thermal gradients, odor gradients, light/dark, water-flow, and acoustic/startle
  stimuli.
- `simulation.py`: factory for larval zebrafish simulation.
- `config.py`: dpf/stage, body length, time step, fluid-drag constants, neural
  scaling, and source IDs used for each parameter.

## Minimal viable simulation path

1. Body first: model a 5 to 7 dpf larva as a head plus segmented trunk/tail.
   Use Mueller and van Leeuwen 2004 plus Budick and O'Malley 2000 for tail
   wave timing, bend categories, and burst/slow swim distinctions.
2. Behavior validation: reproduce bout-level kinematic categories from Marques
   et al. 2018 before attempting full brain behavior.
3. Sensory loop: start with optomotor/optic-flow signals because whole-brain
   imaging and connectomic resources are strongest there.
4. Neural scope: begin with a reduced region-level network (pretectum,
   hindbrain, reticulospinal/spinal projection, optic tectum, ARTR-like turning
   regions). Add cell-level connectome only after a reduced CAVE/Fish1 export is
   available.
5. Data bridge: keep atlas coordinates in a source coordinate frame and record
   transforms explicitly. FishExplorer/mapzebrain and Z-Brain are the obvious
   initial standard spaces.

## Important caveats

- There is no single canonical C. elegans-style complete, hand-curated
  zebrafish connectome. Public resources are large, evolving, and partially
  proofread.
- Fish1 is the strongest current whole-brain connectomics anchor, but practical
  access uses CAVE and may require credentials or a Google account.
- Svara et al. 2022 and Boulanger-Weill et al. 2025 provide smaller, more
  interpretable circuit slices that may be better first implementation targets.
- Larval zebrafish swim in water. A MuJoCo implementation must include fluid
  drag/added resistance approximations; crawling-style agar assumptions from
  `c_elegans` should not be reused.
- Genomic and single-cell atlases are useful for cell typing and marker
  selection, but they should not drive neural dynamics directly until mapped to
  atlas regions/cell classes.

