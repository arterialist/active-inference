# Does established sensory regulation survive a lower input?

## Protocol fixed before the recordings

The 424-cell antennal preparation responds and recovers at sensory command 100
but develops persistent activity at command 50. Both observations survive the
curated-GABA receiving-sign sensitivity control. The next test changes sensory
history without changing a neuron, connection, initial weight or learning rate.

Compare constant command 50 during ticks 200 through 1599 against command 100
during ticks 200 through 599 followed by the identical command-50 train during
ticks 600 through 1599. Both then receive no input for 1,000 ticks. The future
commands must match per source and tick, without resetting source phases at
the switch. This leaves a full 1,000-tick low-input period after preconditioning,
not a short observation that could miss the previously delayed failure.

Run both histories in the antennal cut under original signs and under the
existing curated-GABA control. Run both histories in the 44-cell isolated cut
as well. No curated-GABA candidate is present in that cut, so a duplicate
isolated sign-control course is unnecessary. All six courses retain seed 11,
feedback scale 0.5, sensory afferent scale 2, gain 1, no regulator electrode,
and ongoing native plasticity. No parameter will be selected from these runs.

The preconditioned course must reproduce the existing high-input neural history
before the first changed command. New constant-low courses must reproduce
the existing low-input history before the extended stimulus differs. These
checks make the boundary change explicit, rather than assuming the extended
recorder preserves previous behavior.

## Interpretation

If preconditioning preserves response and recovery where constant low input
fails, that supports history dependence of the complete adaptive state. It
does not alone distinguish persistent fast dynamics from learned weights.
If both histories fail, the established high-input regime does not provide
durable protection against this lower input under the tested boundary.
Different failure times remain evidence; they are not a repaired function.

Inspect all LN and PN activity during the common future and after input removal,
not only a target count. Verify all 2,600 target-neuron ticks, source current
integration, sensory gate delivery and initial/final selected weights. Report
the isolated comparison separately so a response that needs the restored
network is not assigned to the isolated gain circuit.

## Completed comparison

All six courses completed. Future commands and all 42 realized sensory-cell
spike trains match exactly from tick 600 onward within each history comparison.
Original stationary-course prefixes also match the extended recordings exactly.
The circuit, initial weights and learning rates are unchanged between histories.

| Preparation | History | Target PN spikes, ticks 600–1599 | LN spikes, ticks 600–1599 | Target PN recovery spikes, ticks 1600–2599 | LN recovery spikes |
| --- | --- | ---: | ---: | ---: | ---: |
| Isolated, 44 cells | Constant low | 201 | 33 | 6 | 0 |
| Isolated, 44 cells | High then low | 189 | 33 | 6 | 0 |
| Antennal, original signs | Constant low | 236 | 28,724 | 167 | 28,314 |
| Antennal, original signs | High then low | 190 | 33 | 6 | 0 |
| Antennal, GABA control | Constant low | 248 | 18,541 | 167 | 25,120 |
| Antennal, GABA control | High then low | 191 | 33 | 6 | 0 |

Both preconditioned antennal courses retain a low-input response and recover
after input removal. The final LN spike occurs at tick 1570 under original
signs and 1583 under the GABA control. The final target PN spike occurs at 1666
in both. Constant-low antennal courses still have LN and PN spikes on the last
recorded tick, 2599. The same final target PN recovery count in those two failed
courses does not imply identical surrounding dynamics or recruitment histories.

After tick 600, only the identified regulator fires among the 208 LNs in either
preconditioned antennal course. This is preservation of the tested sensory
response and recovery, not evidence that a richly active antennal population
has become self-balancing. The isolated circuit recovers under either history;
the pathological persistence appears after restoring the surrounding network.

Every target PN tick was replayed across all 2,600 ticks in each course. Source
integration and gate-delivery audits report no clipping or ambiguous threshold
decisions. Native plasticity remains active. All 35 selected afferent weights
change in each course. In the preconditioned antennal courses, 736 selected
feedback weights change under original signs and 565 under the GABA control.
These updates do not identify the carrier of the history effect.

The supported result is finite-time history dependence of this complete
adaptive preparation under two sign assumptions. One timing seed, assumed
interneuron gains and dynamics, and no fast-state/weight separation do not
establish bistability, associative memory, a robust autonomous repair, or a
general property of PAULA compositions. Preconditioning was prescribed by the
experimenter. It is not a learned action of the network.

## Evidence and closeout

The six directories under `.live/research/flywire783/` are named
`dl5-regulator-history-{isolated,original,gaba}-{low,preconditioned}-20260910`.
Each has a full-course `comparison-final/analysis.json`. The final paired
analyses are `dl5-regulator-history-original-comparison-final-20260910.json`,
`dl5-regulator-history-isolated-comparison-final-20260910.json` and
`dl5-regulator-history-gaba-comparison-20260910.json`.

`analysis/plots/fly_regulation_closeout.py` renders the completed history and
regulatory-branch comparisons through the existing checked-recording reader.
It retains exact plotted spike events, commands, gate fractions and provenance.
The local figures are under
`simulations/drosophila/figures/regulation-closeout-20260910/`.
The focused suite passes all 15 tests; the full fly suite passed all 212 tests
after the experimental and analysis changes. Figure generation runs no neurons.

This investigation stops here. No additional learning or memory experiment is
started as part of this closeout. The broader embodied-agent objective remains
unresolved.

## Publication assessment

This is useful evidence about the present preparation, but it does not meet
the requested major-breakthrough standard. History-dependent population
responses were already demonstrated by
[Wilson and Cowan, 1972](https://pmc.ncbi.nlm.nih.gov/articles/PMC1484078/).
Inhibitory control of recurrent amplification and transient trajectories has
also been modeled explicitly by
[Hennequin, Vogels and Gerstner, 2014](https://pubmed.ncbi.nlm.nih.gov/24945778/).
Neither work is the same PAULA model. Their existence means that observing
history dependence or inhibitory stabilization here is not itself the new
scientific contribution. The present experiments do not yet derive a
PAULA-specific law, delimit its domain, or discriminate the responsible state
variables well enough to establish such a contribution.

The other completed candidates do not change that decision. The registered APL
control demonstrates a transduction-versus-propagation confound in this model,
but its conductance and calcium-to-voltage mapping are unconstrained.
[Amin et al., 2020](https://elifesciences.org/articles/56954) already discuss
stimulus localization and the distinction between calcium suppression and
depolarization. The passive PN/LN extension demonstrates subthreshold feedback
and finite-window filtering, not new electrical-junction physiology; the
biological transmission mechanisms were established by
[Yaksi and Wilson, 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2954501/).
The PN current-tail extension addresses a model-observable mismatch. Its
implementation and fit do not independently constitute a scientific discovery.

No article or website deployment was made from this closeout. The figures are
local evidence for independent review, not a publication or a claim about
consciousness, organism-level learning, or actual fly physiology.
