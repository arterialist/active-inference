# Zebrafish Source Catalog

Checked: 2026-05-13.

This catalog records the public zebrafish sources found during this pass. It is
not a claim that the public literature is exhausted; it is a reproducible seed
corpus for implementing the organism.

## Highest-Priority Sources

1. Fish1 connectomics release
   - URL: https://fish1-release.storage.googleapis.com/index.html
   - Type: 7 dpf larval zebrafish whole-brain connectomics resource.
   - Key facts: correlated EM and confocal LM from the same specimen; EM
     resolution 4 nm x 4 nm x 30 nm; 187,053 cell bodies across brain, spinal
     cord, and ganglia; 26,915 vglut2a-positive and 14,510 gad1b-positive cells;
     CAVE access and community proofreading.
   - Simulation use: primary candidate for a reduced cell/synapse connectome.

2. FishExplorer
   - URLs: https://zebrafishatlas.zib.de and
     https://fish1-release.storage.googleapis.com/atlas.html
   - Type: multimodal atlas and query/visualization platform.
   - Key facts: shared coordinate space linking Fish1, mapzebrain, and
     Boulanger-Weill et al.; direct neuron downloads across registered spaces
     are described by the Fish1 atlas page.
   - Simulation use: coordinate system and neuron-region lookup bridge.

3. Svara et al. 2022, "Automated synapse-level reconstruction of neural
   circuits in the larval zebrafish brain"
   - DOI: https://doi.org/10.1038/s41592-022-01621-0
   - Type: 5 dpf larval zebrafish SBEM resource and visual-motion circuit
     reconstruction.
   - Key facts: 14 x 14 x 25 nm voxel SBEM; 208 functionally characterized
     neurons involved in visual motion processing; queryable address-book style
     resource; registered to mapzebrain.
   - Simulation use: tractable first synapse-level circuit for optic-flow
     behavior.

4. Hildebrand et al. 2017, "Whole-brain serial-section electron microscopy in
   larval zebrafish"
   - DOI: https://doi.org/10.1038/nature22356
   - Type: complete 5.5 dpf larval zebrafish ssEM/open-access image resource.
   - Key facts: complete brain ssEM data, projectome reconstruction, functional
     atlas co-registration and same-specimen two-photon data.
   - Simulation use: structural morphology and long-range axon constraints.

5. Z-Brain / Randlett et al. 2015
   - DOI: https://doi.org/10.1038/nmeth.3581
   - ZFIN page: https://zfin.org/ZDB-PUB-160119-2
   - Type: 3D larval zebrafish brain atlas and whole-brain activity maps.
   - Key facts: open-source atlas with molecular labels and anatomical region
     definitions; MAP-Mapping behavior/stimulus activity maps.
   - Simulation use: initial brain-region ontology and validation maps.

6. mapzebrain / Kunst et al. 2019
   - URL: https://mapzebrain.org
   - Publication: https://zfin.org/ZDB-PUB-190601-15
   - Type: cellular-resolution larval zebrafish brain atlas.
   - Key facts: thousands of neurons registered to a standard brain, region
     masks, marker patterns, and morphology data.
   - Simulation use: cell morphology, brain regions, and registration target.

7. Janelia whole-brain functional recordings / Ahrens et al. 2013
   - DOI: https://doi.org/10.1038/nmeth.2434
   - Tool page: https://www.janelia.org/open-science/whole-brain-functional-recordings
   - Type: light-sheet whole-brain GCaMP5G recording and analysis software.
   - Key facts: larval whole-brain recordings at 0.8 Hz, capturing more than
     80 percent of neurons at single-cell resolution.
   - Simulation use: neural timescale and large-scale activity validation.

8. Ahrens et al. 2012
   - DOI: https://doi.org/10.1038/nature11057
   - Type: two-photon brain-wide calcium imaging during fictive motor
     adaptation.
   - Simulation use: virtual-reality optic-flow/motor learning loop.

9. Dunn et al. 2016
   - DOI: https://doi.org/10.7554/eLife.12741
   - Harvard open copy: https://dash.harvard.edu/entities/publication/73120379-2d45-6bd4-e053-0100007fdf3b
   - Type: whole-brain light-sheet imaging during exploratory locomotion.
   - Simulation use: ARTR-like turn-sequence dynamics and spontaneous behavior.

10. Marques et al. 2018 locomotor repertoire
    - DOI: https://doi.org/10.1016/j.cub.2017.12.002
    - ZFIN: https://zfin.org/ZDB-PUB-180109-2
    - Data: https://data.mendeley.com/datasets/r9vn7x287r/1
    - Type: larval behavior clustering and tracking software/data.
    - Key facts: millions of swim bouts; 13 basic swimming patterns.
    - Simulation use: primary behavior validation target.

11. ZFIN, ZFA, and ZFIN downloads
    - ZFIN: https://zfin.org
    - ZFA root: https://zfin.org/ZFA%3A0100000
    - Downloads: https://zfin.org/downloads
    - BioPortal ZFA: https://bioportal.bioontology.org/ontologies/ZFA
    - Type: curated organism database, anatomy ontology, expression and
      phenotype downloads.
    - Simulation use: anatomical names, developmental stages, gene markers,
      transgenic lines, and expression data.

12. ZFAP / FishNet 3D body atlas
    - ZFAP download page: https://www.zfap.org/download.php
    - FishNet paper: https://doi.org/10.1186/1741-7007-5-34
    - Type: 3D OPT body models from 24 hpf through adult stages.
    - Simulation use: body proportions and developmental morphology.

## Connectome and Structural Data

- Fish1 release: whole-brain 7 dpf EM/LM, CAVE, programmatic notebooks, and
  FishExplorer integration.
- Petkova et al. 2025 preprint, "A connectomic resource for neural cataloguing
  and circuit dissection of the larval zebrafish brain"
  - DOI: https://doi.org/10.1101/2025.06.10.658982
  - Key facts found: more than 180,000 segmented soma, more than 40,000
    molecularly annotated neurons, about 30 million synapses.
- Boulanger-Weill et al. 2025 preprint, "Correlative light and electron
  microscopy reveals the fine circuit structure underlying evidence accumulation
  in larval zebrafish"
  - DOI: https://doi.org/10.1101/2025.03.14.643363
  - Dataset: https://zenodo.org/records/19231045
  - Key facts found: functional calcium imaging plus ultrastructural
    reconstruction in anterior hindbrain; dataset includes 589.2 MB functional
    H5, 1.8 GB ANTs field, and 5.6 GB traced neurons/axons zip.
- Petkova 2020 dissertation, "Correlative Light and Electron Microscopy in an
  Intact Larval Zebrafish"
  - URL: https://dash.harvard.edu/handle/1/37365549
- FishExplorer 2025 preprint
  - DOI: https://doi.org/10.1101/2025.07.14.664689
  - Janelia page: https://www.janelia.org/publication/fishexplorer-a-multimodal-cellular-atlas-platform-for-neuronal-circuit-dissection-in
- High-precision registration between zebrafish brain atlases
  - DOI: https://doi.org/10.1093/gigascience/gix056
- Zebrafish brain atlases review
  - Article: "Zebrafish brain atlases: a collective effort for a tiny
    vertebrate brain" (2023).

## Whole-Brain Activity and Calcium/Voltage Imaging

- Ahrens et al. 2012, brain-wide two-photon calcium imaging during motor
  adaptation.
- Ahrens et al. 2013, light-sheet whole-brain functional imaging at cellular
  resolution.
- Portugues et al. 2014, whole-brain activity maps for visuomotor behavior.
- Dunn et al. 2016, whole-brain exploratory locomotion and ARTR.
- Janelia whole-brain functional recordings software/videos.
- ZAPBench 2025, whole-brain activity prediction benchmark.
  - Google Research: https://research.google/pubs/zapbench-a-benchmark-for-whole-brain-activity-prediction-in-zebrafish/
  - Key facts found: 4D light-sheet recordings of more than 70,000 neurons plus
    motion-stabilized voxel-level segmentations.
- PyZebrascope 2022.
  - DOI: https://doi.org/10.3389/fcell.2022.875044
  - Code: https://github.com/KawashimaLab/PyZebraScope_public
- Sy et al. 2023 FishChip chemosensory behavior and brainwide representation.
  - Paper DOI: https://doi.org/10.1038/s41467-023-35836-2
  - Data DOI: https://doi.org/10.5281/zenodo.7306139
- Boulanger-Weill et al. 2026 Zenodo dataset for functional/structural
  hindbrain data.
- DANDI zebrafish datasets found in the API search are listed in the manifest.

## Behavior, Locomotion, and Body Mechanics

- Budick and O'Malley 2000, "Locomotor repertoire of the larval zebrafish:
  swimming, turning and prey capture"
  - Type: high-speed kinematic descriptions of 6 to 9 dpf larvae.
  - Key facts found: slow vs burst swim distinctions and brainstem descending
    neuron context.
- Mueller and van Leeuwen 2004, "Swimming of larval zebrafish: ontogeny of body
  waves and implications for locomotory development"
  - DOI: https://doi.org/10.1242/jeb.00821
  - Key facts found: 2 to 21 dpf body-wave kinematics; tail beat frequencies up
    to 100 Hz; muscle strain/strain-rate constraints.
- Marques et al. 2018 plus Mendeley tracking data.
- Body dynamics and hydrodynamics of swimming fish larvae
  - DOI: https://doi.org/10.1098/rsif.2012.0516
- Mechanical characteristics of ultrafast zebrafish larval swimming muscles
  - DOI: https://doi.org/10.1016/j.bpj.2020.06.031
- Orger et al. 2008, visually guided behavior and spinal projection neurons.
- Severi et al. 2014, neural control/modulation of swimming speed.
- Kubo et al. 2014 and related reticulospinal/startle/optic-flow literature are
  relevant for motor output mapping.

## Anatomy, Genome, Ontologies, and Atlases

- ZFIN: central curated zebrafish organism database.
- ZFA/ZFS: anatomy and stage ontologies; BioPortal showed the 2026-03-31 ZFA
  release and OBO/CSV/RDF downloads.
- ZFIN downloads: expression data, stage/anatomy mappings, marker records,
  phenotype/expression datasets.
- ZFAP download area: 24 h, 48 h, 72 h, 96 h, 120 h and 5 mm to 17 mm TIF
  anatomy models.
- FishNet paper: 3D lifespan atlas with more than 36,000 images and more than
  1,500 annotated images.
- ZeBrain adult zebrafish brain atlas.
  - URL: https://zebrain.nig.ac.jp/zebrain/page.do?p=welcome
- ZMAP single-cell meta-atlas.
  - URL: https://wagnerlabucsf.github.io/zmap/
- ZSCAPE perturbed embryo atlas.
  - URL: https://cole-trapnell-lab.github.io/zscape/downloads/
- FaceBase zebrafish craniofacial atlas.
  - URL: https://www.facebase.org/resources/zebrafish/anatomical-nav/
- Ensembl zebrafish GRCz11.
  - URL: https://www.ensembl.org/Danio_rerio/Info/Annotation
- NCBI GRCz11 assembly.
  - URL: https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000002035.6/
- Complete de novo assembly and re-annotation of the zebrafish genome (2025)
  describes GRCz12tu/GRCz12ab and pangenome resources.

## DANDI Sources Found

The DANDI API search for "zebrafish" and "Danio rerio" returned these relevant
dataset IDs:

- DANDI:000235, Thermoregulatory Responses Forebrain.
- DANDI:000236, Thermoregulatory Responses Midbrain.
- DANDI:000237, Thermoregulatory Responses Hindbrain.
- DANDI:000238, Thermoregulatory Responses Reticulospinal system.
- DANDI:000350, Glia Accumulate Evidence that Actions Are Futile and Suppress
  Unsuccessful Behavior.
- DANDI:000485, Thermal Plaid Experiments.
- DANDI:000486, Thermal Plaid Replay Experiments.
- DANDI:000697 to DANDI:000706, constant-temperature behavior from 16 C to
  34 C.
- DANDI:000707 to DANDI:000708, thermal-gradient behavior.
- DANDI:000888 to DANDI:000889, steep-gradient thermal plaid/replay.
- DANDI:001076, OMR Robot CaImaging.
- DANDI:001291, light-sheet voltage imaging across entire larval brains.
- DANDI:001337, fixed hot/cold stimulus medulla.
- DANDI:001339, fixed hot/cold stimulus trigeminal ganglion.
- DANDI:001453, visuomotor strategies of zebrafish and Danionella cerebrum.
- DANDI:001553, raw light-sheet voltage imaging datasets for larval zebrafish
  brains.
- DANDI:001569, improv platform calcium fluorescence images.
- DANDI:001840, larval zebrafish cardiac dynamics under UV stimulus.

## Practical First Pulls

For the first actual implementation pass, pull in this order:

1. Mendeley locomotor repertoire data for bout validation.
2. ZFAP 120 h or 5 mm body model for body proportions.
3. Z-Brain/mapzebrain region masks for a reduced atlas.
4. Janelia or DANDI whole-brain activity data for neural time constants.
5. A reduced CAVE/Fish1 or Svara 2022 export for a first circuit-level
   connectome.

