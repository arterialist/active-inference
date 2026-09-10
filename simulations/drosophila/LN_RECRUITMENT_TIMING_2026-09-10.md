# Recruitment is not maintenance

Blocking lLN2T_e `720575940633483807` before the first sensory train prevents
broad recruitment in this PAULA preparation. Blocking its chemical output only
from tick 600 does not stop the established activity. The earlier result
therefore identified a route into the regime, not a cell whose continued output
is necessary to maintain it. This is an informative failure of the current
assumed dynamics, not successful memory or sensory coding.

The full connected test keeps 3,005 cells and 213,855 measured internal pairs.
It uses the same staggered ORN current pulses as the earlier experiment, with
no terminal depression or added electrical junction. It is not an odor assay.
The train lasts from tick 200 through 1199. Recording stops at 1600, after only
400 recovery ticks. All weak positive native learning rates remain enabled.

## Receiving-path experiment

Four identified LNs replay their original full receiving histories with native
PAULA dynamics. All incident ports remain present, including silent boundary
ports. Nine conditions per cell either retain all inputs, select/remove DL5 or
ALLN inputs, vary the initial DL5 coefficient, or remove every input. Outputs
do not feed back into the recorded source trains. Incoming return histories to
outgoing terminals are not reconstructed.

| Cell type and root suffix | First spike, intact replay | First spike, DL5 only | First spike, without DL5 |
| --- | ---: | ---: | ---: |
| lLN2P_c, 628343634 | 258 | 258 | 406 |
| lLN2T_e, 633483807 | 334 | 334 | 403 |
| lLN2T_c, 618757666 | 340 | 340 | 403 |
| il3LN6, 623636701 | 400 | none | 400 |

DL5 alone reproduces onset in three cells. But without DL5, recorded inputs from
other cells later drive all four near their refractory-limited rate, including
133 spikes each during the 400-tick recovery. ALLN-only inputs drive three of
the four at that rate; lLN2T_c fires 66 times. Different receiving paths can
therefore support substantial activity in the conditional replay.

This cannot establish that those paths would generate the same input after a
lesion. The forced source trains were recorded from the already-recruited
intact network. Quartering or halving DL5 gain in this experiment changes onset
without preventing later recruitment, but is not a closed-loop gain repair
test. Every cell stays silent with all received inputs removed.

The all-input replay reproduces 6,776,000 recorded soma, intrinsic and receiving
weight values exactly. Signed current attribution uses pre-arrival receiving
weights and native dendritic delay/attenuation. The result locates input paths;
exact execution does not validate their assumed physiological strengths.

## Closed-loop intervention

The new run filters this LN's outgoing chemical events beginning at tick 600.
It does not silence its membrane, erase its weights, alter receiving inputs,
remove its anatomical connections, or block native return events.

| Recovery, ticks 1200–1599 | Intact | Block from tick 0 | Block from tick 600 |
| --- | ---: | ---: | ---: |
| ALLN spikes | 17,089 | 2 | 17,041 |
| ALLNs that fire | 169 | 2 | 169 |
| ALPN spikes | 10,734 | 3 | 10,459 |
| KC spikes | 6,638 | 0 | 6,324 |
| KCs that fire | 550 | 0 | 538 |
| DL5 PN spikes | 73 | 3 | 71 |
| Target LN spikes | 133 | 1 | 133 |
| Maximum APL release | at cap | 0.0791 | at cap |

The late lesion changes neural state and spikes first at tick 604. Eventually
650 cells differ in spike timing from intact. The perturbation does propagate,
so near-preserved population activity is not evidence that the intervention
failed to reach its targets. It is also not evidence that every downstream
function is preserved: substantial spike-timing changes can matter to a neural
consumer even when counts are similar.

The late run matches all 7,846,200 compared pre-intervention values and retains
the same anatomical bindings. Only the specified LN's forward events are
withheld: 512,487 events from onset to recording end. Its 47,629 return events
remain admitted. All other LNs' attempted forward events remain admitted.
An independent receiving-history replay reproduces the DL5 PN's soma, current,
intrinsic fields and receiving weights for all 1,600 ticks.

The experiment demonstrates a dependence on intervention history. It does not
separate ongoing synaptic adaptation from fast recurrent state, establish
bistability, identify a minimal sustaining subnetwork, or prove indefinite
activity without sensory drive. Three ORNs still spike during recovery in the
intact and late-lesion conditions. The modest observation window cannot
establish long-run stability.

## Consequence for the next functional experiment

The useful question is whether antennal-lobe processing regulates sensory gain
while preserving distinguishable signals. Merely suppressing this excessive
activity could destroy the function we need for subsequent KC learning.

[Olsen and Wilson, 2008](https://doi.org/10.1038/nature06864) found lateral
inhibition that scales with total antennal-lobe input and acts substantially at
ORN terminals. The current adapter instead routes LN-to-ORN signed events to
the soma. That is a specific missing regulatory action, not proof that a
larger graph or stronger global inhibition will solve the problem.

[Olsen, Bhandawat and Wilson, 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2866644/)
separated a target channel's private input from public input to other channels.
Lateral input shifted the direct input needed to reach PN saturation and made
responses more transient. This supplies an input-to-output functional test.
Their fitted population equation is an observational comparison, not a Python
controller to insert into the PAULA circuit.

The next bounded assay should measure target-PN transfer over independently
varied direct and lateral activity. Test direct input alone, lateral input alone,
both together, and a selective regulatory-pathway lesion. The required result
is retained sensitivity and direct-input ordering alongside context-dependent
gain regulation, with recovery and trial-history effects visible. Silence,
ceiling firing and a post-hoc rescaled plot do not pass. Any added terminal
regulation must operate locally through declared neural inputs, preserve native
ongoing plasticity, and leave reference defaults unchanged.

The existing graph has only DL5 ORN providers. Its silent boundary cannot stand
in for independently controlled public sensory input. A functional isolated
preparation must explicitly supply that boundary through documented neural
input histories or include the necessary sensory providers. Those are different
experiments and must be labeled accordingly. No new graph expansion is justified
until that boundary and the neural readout are specified.

After establishing this transformation, test it with the reciprocal PN/KC/APL
partners active and ask whether KCs retain distinguishable input-dependent
responses. Only then does this branch support an associative learning claim.
This sequence makes calibration subordinate to demonstrated circuit function.

## Retained evidence and reproduction

All paths below are within ignored `.live/research/flywire783/`. Raw recordings
remain local. No anatomical table is redistributed.

- `dl5-ln-conditional-inputs-20260910/`: 36 conditional receiving replays.
- `dl5-ln-late-release-block-20260910/`: whole-network late-lesion ticks and
  `comparison.json` against original intact and early-lesion records.
- `dl5-orn-onset-20260910/`: original four-LN and early-lesion records.
- `dl5-orn-trains-20260910/no_depression/`: historical intact reference.

Run `python -m simulations.drosophila.ln_input_replay --help` for replay inputs.
The late intervention uses the existing `orn_train` command with
`no_depression --stop-tick 1600 --release-block 720575940633483807 600`.
Run `python -m simulations.drosophila.ln_release_timing --help` for the comparison.
All commands require a new output path. Historical source hashes are retained;
they are not rewritten to claim that old records used the current source tree.
