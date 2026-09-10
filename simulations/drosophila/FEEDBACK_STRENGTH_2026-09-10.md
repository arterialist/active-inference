# Preserving feedback connections while changing their assumed strength

The [pathway lesions](GAIN_REUNION_2026-09-10.md) identified two routes that
could sustain unwanted activity. This follow-up leaves those routes connected
and tests their initial receiving strength. It supplies a non-lesion operating
range under a controlled inhibitory input, not a physiological fit or an
autonomous olfactory circuit.

## The change is in initial PAULA receptors

Only the initial `info` coefficients on the 16,957 internal positive-model
LN/PN-to-LN receiving ports are multiplied by a declared scale. All coefficients
remain nonzero. Boundary weights, source terminal coefficients, synaptic counts,
neuron identities and native events remain unchanged at initialization. Native
postsynaptic and retrograde learning continue with their existing weak positive
rates. No observer adjusts gain or activity during the run. No cell is reset.

The scales 0.125, 0.25 and 0.5 are a coarse sensitivity test, not values inferred
from physiology. The existing 0.075-per-contact conversion for these pathways
was itself an assumption. The experiment asks whether the previously observed
failure requires a new cellular mechanism or can already depend on that assumed
conversion. It does not validate the source-model transmitter signs.

The sensory preparation, presynaptic-inhibition gain 1, decay 100 ticks, seed 11,
and 2,200-tick baseline/stimulus/recovery course are unchanged. The first tests
retain the independent 80-pulse LN command from the isolated gain assay.
That command is experimental assistance, not autonomous public sensory input.

## A conditional operating range

At sensory command 50, all three reduced strengths reproduce the isolated
target-PN soma trace exactly in the 424-cell antennal cut: 129 stimulus spikes
and 3 recovery spikes. But the only spiking nonsensory cells are the target PN
and the commanded inhibitory LN. This is preservation of a local function with
largely inactive neighbors, not evidence that the neighbors cooperate.

The least-reduced setting, 0.5, was therefore tested with stronger sensory drive
and with all 3,005 PN/LN/KC/APL partners present.

| Preparation | Sensory command | PN stimulus spikes | PN recovery spikes | KC spikes over course |
| --- | ---: | ---: | ---: | ---: |
| Isolated reference | 50 | 129 | 3 | absent |
| Antennal, feedback scale 0.5 | 50 | 129 | 3 | absent |
| Full, feedback scale 0.5 | 50 | 129 | 2 | 0 |
| Isolated reference | 100 | 251 | 6 | absent |
| Antennal, feedback scale 0.5 | 100 | 250 | 5 | absent |
| Full, feedback scale 0.5 | 100 | 251 | 6 | 1 |

At the stronger drive, 14 LNs and 5 PNs respond in the first 200 stimulus ticks.
The antennal circuit has no LN recovery spikes; the full circuit has one.
The full preparation's maximum local APL release is about 0.108, below its
imposed cap of 1. This does not establish that its spatial response is calibrated.

Full target-PN spike timing differs from isolation despite matching stimulus
counts. At sensory 50 the first spike difference is tick 359; at sensory 100
it is tick 647. The gate remains driven by the commanded LN, whose spike trace
does not differ from isolation in either case. Conditional PN receiving replays
without ORN inputs produce no spikes, unlike the earlier high-recurrence state.
These are conditional replay results, not predictions of a closed-loop ORN lesion.

The stronger full course has one spike from one of 2,580 KCs, and no KC recovery
activity. This cannot establish odor discrimination, associative memory or useful
mushroom-body cooperation. Only one sensory channel is provided here.

## Removing experimental assistance brings the failure back

The additional antennal course removes the LN command while retaining sensory
100, feedback scale 0.5 and every connection. It fails to recover despite
recruiting the regulator itself. The same parameter setting therefore cannot
be reported as autonomous gain control.

| Antennal, sensory 100, feedback 0.5 | LN command 80 | No LN command |
| --- | ---: | ---: |
| Regulator's first spike | 206 | 224 |
| Regulator spikes before tick 500 | 24 | 71 |
| LNs recruited before tick 500 | 14 | 135 |
| Mean sensory release fraction, ticks 300–399 | 0.2120 | 0.1292 |
| Target PN stimulus spikes | 250 | 277 |
| Target PN recovery spikes | 5 | 169 |
| Regulator recovery spikes | 0 | 333 |
| All LN recovery spikes | 0 | 28,160 |
| ORN recovery spikes | 0 | 0 |

The unassisted regulator starts later but then fires more, and eventually
suppresses sensory release more strongly. Its mean recovery release fraction
is about 0.0492, versus 0.8132 after the assisted circuit's regulator stops.
Nevertheless, 129 LNs and 151 PNs remain active during unassisted recovery,
without ORN spikes. Conditional target receiving replay without ORN input still
produces 167 recovery spikes. Removing positive-model LN input in the replay
produces zero. Ongoing recurrent input, not a long sensory train or absent
inhibitory activation, supplies the target's remaining drive.

This comparison changes the whole LN command course. It does not isolate the
18-tick onset difference from subsequent differences in firing. An appropriate
next test separates a brief early regulatory input from continued assistance,
with a charge-matched delayed input as a timing control. Only then could timing
be identified as the decisive factor, rather than inferred from two trajectories.
Any successful timing mechanism would need a neural source in the final brain;
an experimenter-scheduled pulse is not that mechanism.

The half-strength preparation is thus a conditional functional reference and
a useful failure case. It is not adopted as a repaired autonomous agent.

The distinction matters for learning as well as stability. [Sudhakaran et al., 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC6622292/)
found that PN output could drive odor-selective behavioral habituation and
implicated plasticity of recurrent inhibition. Their result motivates testing
output-dependent neural regulation, not permanently suppressing feedback.
The present recordings demonstrate neither that habituation nor its molecular
mechanism. [Hallem and Carlson, 2006](https://pubmed.ncbi.nlm.nih.gov/16615896/)
measured odor responses across a receptor repertoire; one electrically driven
channel cannot substitute for such a multi-channel sensory pattern when testing
odor identity or public-input normalization.

## Retained evidence

Records under `.live/research/flywire783/`:

- `dl5-gain-antennal-feedback0125-20260910/`
- `dl5-gain-antennal-feedback025-20260910/`
- `dl5-gain-antennal-feedback05-20260910/`
- `dl5-gain-antennal-feedback05-high-20260910/`
- `dl5-gain-full-feedback05-20260910/`
- `dl5-gain-full-feedback05-high-20260910/`
- `dl5-gain-antennal-feedback05-unassisted-20260910/`

The runner adds `--feedback-scale`; default 1 preserves the original initial
weights. Initial and final selected receptor weights are retained separately.
Analysis checks their bindings, initial values against counted contacts, declared
scaling and actual final changes. At scale 0.5, 92 selected receiving weights
change at sensory 50, and 736 at sensory 100, in both cuts. The changes are small;
their existence is not evidence of useful learning. In the unassisted condition,
15,763 of the selected receiving weights change, with maximum absolute change
about 0.000255. Ongoing adaptation is present but has not restored the function
within this course.

The target's complete receiving trace independently reproduces its recorded soma,
intrinsic state, current and receiving weights. Local gate dynamics and target
routing are checked each tick. These checks verify the claimed execution and
declared change, not every intracellular state or biological correctness.
Runs use new output directories and preserve historical source hashes.
