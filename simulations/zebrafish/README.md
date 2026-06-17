# Zebrafish Simulation And Research Corpus

This directory contains the larval zebrafish (`Danio rerio`) implementation
under `active-inference` plus the source corpus used to keep the model
traceable.

Simulation modules:

- `body.py`: MuJoCo 5-7 dpf larval body with a free root, segmented tail,
  yaw/pitch hinges, and per-quadrant tail actuators.
- `environment.py`: 3D aquatic arena with odor, thermal, optic-flow,
  lateral-line, wall, and startle channels.
- `connectome.py` and `neuron_mapping.py`: PAULA-backed reduced
  tectum/pretectum/hindbrain/reticulospinal/spinal CPG circuit. This is a
  source-annotated stand-in until a denser public synapse-level zebrafish
  export is integrated.
- `simulation.py`: organism factory and MuJoCo/PAULA loop wiring.

Research corpus:

- `research/source_catalog.md`: human-readable source inventory grouped by
  connectome, imaging, behavior, anatomy, and tooling.
- `research/source_manifest.json`: machine-readable registry of every source
  found in this pass.
- `research/sources.bib`: seed BibTeX bibliography for paper-writing or future
  citation export.
- `research/search_log.md`: reproducible search/API queries and raw result
  summaries.
- `research/candidate_literature.md`: untriaged recent PubMed hits preserved
  from the live search.
- `research/sourced_articles.md`: local fetch status for public article/PDF
  sourcing.
- `research/implementation_notes.md`: how the sources map onto future organism
  modules.
- `research/data_policy.md`: what can be stored in git versus downloaded
  locally on demand.

Large public datasets are not vendored here. The corpus records source URLs,
DOIs, dataset IDs, access requirements, and expected use in the simulation so a
simulation run can pull only the needed subset.
