# Search Log

Checked: 2026-05-13.

This file records the search/API paths used to build the source corpus. It is
meant to make the inventory refreshable.

## Web Queries

Core searches:

```text
zebrafish connectome dataset larval zebrafish brain electron microscopy public data
zebrafish whole brain calcium imaging dataset public light sheet microscopy
Z-Brain zebrafish brain atlas Randlett 2015 data
Mapzebrain cellular resolution atlas larval zebrafish brain data
DANDI zebrafish calcium imaging dataset NWB
larval zebrafish locomotor repertoire dataset behavior open
zebrafish larva body mechanics swimming kinematics dataset
zebrafish anatomical atlas body structure public database ZFIN anatomy ontology
Fish1 connectomic resource larval zebrafish brain companion paper DOI
FishExplorer multimodal cellular atlas platform larval zebrafish brain
Ahrens whole brain functional imaging at cellular resolution zebrafish 2013 Nature Methods data
Brain-wide neuronal dynamics during motor adaptation zebrafish Ahrens 2012 Nature data
Whole-brain activity maps reveal stereotyped distributed networks for visuomotor behavior zebrafish data
Dunn brain-wide mapping neural activity controlling zebrafish exploratory locomotion dataset
Budick O'Malley locomotor repertoire zebrafish larvae 2000
Marques zebrafish locomotor repertoire unsupervised behavioral clustering dataset 2018 current biology
zebrafish larval swimming kinematics body curvature dataset tail beat bouts
zebrafish larva neuromechanical model swimming body muscle simulation
Z-Brain atlas download zebrafish source data
mapzebrain.org download data API zebrafish brain atlas
Danio rerio genome assembly GRCz11 Ensembl zebrafish
NCBI Danio rerio genome assembly GRCz11
```

Pages inspected directly:

```text
https://fish1-release.storage.googleapis.com/index.html
https://fish1-release.storage.googleapis.com/tutorials.html
https://fish1-release.storage.googleapis.com/atlas.html
https://zebrafishatlas.zib.de
https://mapzebrain.org
https://www.nature.com/articles/s41592-022-01621-0
https://zenodo.org/records/19231045
https://www.janelia.org/open-science/whole-brain-functional-recordings
https://www.janelia.org/publication/fishexplorer-a-multimodal-cellular-atlas-platform-for-neuronal-circuit-dissection-in
https://zfin.org
https://zfin.org/ZFA%3A0100000
https://zfin.org/downloads
https://bioportal.bioontology.org/ontologies/ZFA
https://www.zfap.org/download.php
https://data.mendeley.com/datasets/r9vn7x287r/1
https://research.google/pubs/zapbench-a-benchmark-for-whole-brain-activity-prediction-in-zebrafish/
```

## DANDI API Search

Commands:

```bash
curl -fsSL 'https://api.dandiarchive.org/api/dandisets/?search=zebrafish&page_size=25'
curl -fsSL 'https://api.dandiarchive.org/api/dandisets/?search=Danio%20rerio&page_size=25'
```

Relevant DANDI IDs found:

```text
DANDI:000235 Thermoregulatory Responses Forebrain
DANDI:000236 Thermoregulatory Responses Midbrain
DANDI:000237 Thermoregulatory Responses Hindbrain
DANDI:000238 Thermoregulatory Responses Reticulospinal system
DANDI:000350 Glia Accumulate Evidence that Actions Are Futile and Suppress Unsuccessful Behavior
DANDI:000485 Thermal Plaid Experiments
DANDI:000486 Thermal Plaid Replay Experiments
DANDI:000697 Constant temperature behavior 16C
DANDI:000698 Constant temperature behavior 18C
DANDI:000699 Constant temperature behavior 20C
DANDI:000700 Constant temperature behavior 22C
DANDI:000701 Constant temperature behavior 24C
DANDI:000702 Constant temperature behavior 26C
DANDI:000703 Constant temperature behavior 28C
DANDI:000704 Constant temperature behavior 30C
DANDI:000705 Constant temperature behavior 32C
DANDI:000706 Constant temperature behavior 34C
DANDI:000707 Gradient behavior 18C to 26C
DANDI:000708 Gradient behavior 24C to 32C
DANDI:000888 Thermal Plaid Experiments - steep gradients
DANDI:000889 Thermal Plaid Replay Experiments - steep gradients
DANDI:001076 OMR Robot CaImaging
DANDI:001291 Imaging the voltage of neurons distributed across entire brains of larval zebrafish
DANDI:001337 Fixed hot/cold stim medulla
DANDI:001339 Fixed hot/cold stim trigeminal ganglion
DANDI:001453 Neural circuits underlying divergent visuomotor strategies of zebrafish and Danionella cerebrum
DANDI:001553 Raw light-sheet voltage imaging datasets for larval zebrafish brains
DANDI:001569 improv platform calcium fluorescence images
DANDI:001840 Cardiac dynamics of larval zebrafish in response to UV stimulus
```

## PubMed Queries

Connectome query:

```bash
curl -fsSL 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=zebrafish%20connectome&retmode=json&retmax=20'
```

Top result IDs found:

```text
41526386 41087271 40661440 40644546 40161766 39712646 39702668
39578573 39386457 39229236 38994821 38452770 37546897 37252868
37236180 37066422 36280716 35410342 35311003 35115916
```

Selected result titles:

```text
2025 Jun 15 - A connectomic resource for neural cataloguing and circuit dissection of the larval zebrafish brain.
2025 Jul 11 - Structural and genetic determinants of zebrafish functional brain networks.
2025 Jun 1 - Correlative light and electron microscopy reveals the fine circuit structure underlying evidence accumulation in larval zebrafish.
2024 Dec - Predicting modular functions and neural coding of behavior from a synaptic wiring diagram.
2023 Jun 5 - Cyclic structure with cellular precision in a vertebrate sensorimotor neural circuit.
2022 Nov - Mapping of the zebrafish brain takes shape.
2022 - RealNeuralNetworks.jl.
2021 - Towards a Comprehensive Optical Connectome at Single Synapse Resolution via Expansion Microscopy.
```

Whole-brain calcium imaging query:

```bash
curl -fsSL 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=zebrafish%20whole-brain%20calcium%20imaging&retmode=json&retmax=20'
```

Top result IDs found:

```text
42058463 42015506 41993382 41905435 41676547 41648186 41509353
41467186 41087271 40961930 40894042 40644546 40631170 40549559
40161766 40139932 39975109 39925410 39719697 39694033
```

Selected result titles:

```text
2026 - CaMPARI2 enables stimulus-locked whole-brain activity mapping at cellular resolution in unrestrained larval zebrafish.
2026 Apr 7 - Whole-brain cellular-resolution functional network properties of seizure susceptibility.
2026 Feb 5 - Astrocyte-induced internal state transitions reshape brainwide sensory, integrative, and motor computations.
2025 Dec 19 - Optogenetic interrogation of the zebrafish lateral line reveals brain-wide neural circuits involved in pattern separation.
2025 Jul 11 - Structural and genetic determinants of zebrafish functional brain networks.
2025 Jun 23 - The geometry and dimensionality of brain-wide activity.
2025 Jun 1 - Correlative light and electron microscopy reveals the fine circuit structure underlying evidence accumulation in larval zebrafish.
2024 Sep 23 - Spontaneous Brain Activity Emerges from Pairwise Interactions in the Larval Zebrafish Brain.
```

## Refresh Notes

- Re-run the DANDI search before implementation because draft dandisets and
  published versions are changing quickly.
- Re-run PubMed queries for years 2025+ before committing to a connectome or
  calcium-imaging source as "latest".
- Bulk data should be downloaded by future explicit downloader scripts, not by
  this source inventory pass.

