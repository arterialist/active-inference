# Zebrafish Behavior And Movement Sources

Date: 2026-05-13

Scope: recent public sources used as behavioral targets for the larval zebrafish MuJoCo/PAULA simulation. The current v2 simulation is a single free-swimming larva with no food/prey in the arena, so prey-capture and social sources define future scenario targets rather than baseline no-food acceptance criteria.

## Source Set

1. Slangewal et al. (2026), "Visuomotor decision-making through multifeature convergence in the larval zebrafish hindbrain", Nature Communications. Freely swimming larvae make discrete decision bouts at stochastic interbout intervals; bout detection is based on body-orientation variance and left/right heading changes. Neural imaging maps visual feature integration to hindbrain/pretectal/tectal populations. https://www.nature.com/articles/s41467-026-69633-4

2. Chen et al. (2026), "High-throughput multi-camera array microscope platform for automated 3D behavioral analysis of swimming zebrafish larvae", Communications Biology. Provides current 3D skeletal tracking and swim-bladder/kinematic measurement expectations for unconstrained larvae. https://www.lifescience.net/publications/1859178/high-throughput-multi-camera-array-microscope-plat/

3. Ravan, Chemla, and Gruebele (2025), "Larval zebrafish swim bouts in three dimensions reveal both new and redundant behaviours", Journal of the Royal Society Interface. Identifies 3D behavioral clusters including short-latency C-turns, O-turns, free swims, voluntary turns, and dorso-ventral components in escape behavior. https://experts.illinois.edu/en/publications/larval-zebrafish-swim-bouts-in-three-dimensions-reveal-both-new-a-2/

4. Rubinstein et al. (2025), "A detailed quantification of larval zebrafish behavioral repertoire uncovers principles of hunting behavior", iScience. Builds a low-dimensional continuous model of hunting movements; fish rotate around a body-centered point and couple heading change with translation. https://www.sciencedirect.com/science/article/pii/S2589004225004742

5. Lau, Fitzgerald, and Bianco (2025), "Supraspinal commands have a modular organization that is behavioral context specific", Current Biology. Calcium imaging and high-resolution behavior tracking relate reticulospinal activity to many swim types, with eight functional archetypes and context-specific hunting modules. https://www.sciencedirect.com/science/article/pii/S0960982225010036

6. Davis, Zhu, and Schoppik (2025), "Larval zebrafish maintain elevation with multisensory control of posture and locomotion", Journal of Experimental Biology / bioRxiv version. Long unrestrained recordings show pitch-axis posture/elevation control, swim bouts, passive interbout intervals, climb/dive/flat categories, and light-dependent compensation strategies. https://pubmed.ncbi.nlm.nih.gov/38328242/

7. Mearns et al. (2025), "Diverse prey capture strategies in teleost larvae", eLife. Zebrafish prey capture uses eye convergence, binocular centering, discrete prey-capture bouts, and capture strikes; useful for distinguishing zebrafish-specific hunting syntax from other larval fish. https://elifesciences.org/articles/98347

8. Davies and De Marco (2024), "Saccades and pivoting during positive mechanotaxis in larval zebrafish", MicroPublication Biology. Larvae approach minute water motion using exploratory movements and pivoting maneuvers with low displacement. https://pubmed.ncbi.nlm.nih.gov/39450186/

9. Agha, Kishore, and McLean (2024), "Cell-type-specific origins of locomotor rhythmicity at different speeds in larval zebrafish", eLife. Links speed-dependent swimming rhythms to spinal circuit cell types and ventral-root activity. https://pubmed.ncbi.nlm.nih.gov/39287613/

10. Harpaz et al. (2024), "Experience-dependent modulation of collective behavior in larval zebrafish", bioRxiv/PMC. Uses virtual neighbors and group-density experience to quantify adaptive collective movement changes. https://pubmed.ncbi.nlm.nih.gov/39149341/

11. Lamire et al. (2023), "Functional and pharmacological analyses of visual habituation learning in larval zebrafish", eLife. Repeated dark flashes evoke O-bends and distributed calcium responses; motor-correlated hindbrain populations are separated from stimulus-tuned populations. https://elifesciences.org/articles/84926

12. "A brainstem circuit for gravity-guided vertical navigation" (2024), Nature / PMC. Uses SAMPL to quantify pitch, posture, and depth navigation; larvae translate in short bouts separated by inactive intervals while controlling vertical behavior. https://pmc.ncbi.nlm.nih.gov/articles/PMC10980031/

13. Carbo-Tano et al. (2023), "The mesencephalic locomotor region recruits V2a reticulospinal neurons to drive forward locomotion in larval zebrafish", Nature Neuroscience. Classifies forward and turn swim bouts from tail kinematics and links V2a reticulospinal neurons to locomotor components. https://www.nature.com/articles/s41593-023-01418-0

14. Naumann lab / PLOS Computational Biology (2023), "A behavioral and modeling study of control algorithms underlying the translational optomotor response in larval zebrafish with implications for neural circuit function". Defines closed-loop visual-motion control targets for OMR-style behavior. https://pmc.ncbi.nlm.nih.gov/articles/PMC9998047/

15. Zhang et al. (2023), "An optofluidic platform for interrogating chemosensory behavior and brainwide neural representation in larval zebrafish", Nature Communications. Combines controlled chemical delivery, behavior readouts, and whole-brain cellular-resolution activity. https://www.nature.com/articles/s41467-023-35836-2

16. Zhu and Goodhill (2023), "From perception to behavior: The neural circuits underlying prey hunting in larval zebrafish", Frontiers in Neural Circuits. Summarizes prey hunting syntax: brief 100-180 ms bouts, 1-2 s inactive periods, 1-10 bouts per hunting event, J-turns, approach/slow swims, and capture strikes. https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2023.1087993/full

17. Nakayama et al. (2022), "Long descending commissural V0v neurons ensure coordinated swimming movements along the body axis in larval zebrafish", Scientific Reports. Slow swimming depends on coordinated S-shaped body forms and head-yaw stabilization rather than one-sided C-bends. https://www.nature.com/articles/s41598-022-08283-0

18. Koning, Ahemaiti, and Boije (2022), "A deep-dive into fictive locomotion - a strategy to probe cellular activity during speed transitions in fictively swimming zebrafish larvae", Biology Open. Describes fictive swim episodes and speed-transition physiology targets. https://pubmed.ncbi.nlm.nih.gov/35188534/

19. Barbara et al. (2022), "PyZebrascope: An Open-Source Platform for Brain-Wide Neural Activity Imaging in Zebrafish", Frontiers in Cell and Developmental Biology. Open-source neural-imaging acquisition/analysis context for whole-brain activity. https://pubmed.ncbi.nlm.nih.gov/35663407/

20. Ghosh and Rihel (2020), "Hierarchical Compression Reveals Sub-Second to Day-Long Structure in Larval Zebrafish Behavior", eNeuro. Large-scale long-duration data extract millions of movements and pauses into modules and motifs across circadian timescales. https://pubmed.ncbi.nlm.nih.gov/32241874/

21. Mearns et al. (2020), "Deconstructing Hunting Behavior Reveals a Tightly Coupled Stimulus-Response Loop", Current Biology. Maps a continuum of swim bouts and shows hunting sequences respond immediately to visual prey-state changes. https://pubmed.ncbi.nlm.nih.gov/31866365/

## Behavioral Targets For The Simulation

- Baseline spontaneous locomotion should be bout-and-coast, not continuous propulsion: short active bouts, inactive/coast periods, stochastic interbout intervals, and repeated same-side turning chains.
- Body mechanics should produce coordinated S-shaped axial waves during slow/forward swimming, with stable head yaw and no persistent corkscrew or vertical collapse.
- The body must be capable of 3D movement: pitch, climb/dive/flat trajectories, dorso-ventral displacements, and water-column stabilization.
- Startle/dark-flash behavior should support high-amplitude C/O-like bends and quick heading reorientation.
- Mechanosensory/lateral-line behavior should support pivoting or turning with small translational displacement.
- Visual/optic-flow behavior should support OMR/phototaxis-style heading decisions and speed modulation.
- Prey/hunting behavior, when prey is enabled in a future scenario, should support eye/target alignment proxies, J-turn/approach/slow-swim/capture-like syntax, and visual-feedback-dependent sequence interruption.
- Social/collective behavior requires multi-agent or virtual-neighbor stimuli and is not a baseline single-fish/no-food requirement.
- Neural activity should be compared as calcium-like continuous state trajectories and behavior-correlated population motifs. The current PAULA scaffold is not a replay of published calcium datasets and should not be treated as source-calibrated calcium imaging until those datasets are explicitly fitted.
