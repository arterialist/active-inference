# Brief assistance does not preserve the connected sensory function

The earlier half-strength feedback circuit worked under sustained independent
stimulation of its inhibitory regulator. Removing that command brought back
persistent activity. The matched timing test now rules out one proposed shortcut:
eight early regulatory pulses do not make continued assistance unnecessary.

## Equal injected dose, different neural effects

The early and delayed conditions use the same eight one-tick current pulses of
40 model units. They arrive at ticks 206–293 or 606–693, an exact 400-tick shift.
All 42 sensory command traces are identical. Each condition has a separately
recorded isolated reference with the same initial feedback scale of 0.5.
The stimulus lasts from tick 200 through 1199; recording continues through 2199.
These are model ticks, not a calibrated physical duration for the whole circuit.

| Regulatory input | Isolated PN stimulus / recovery spikes | Connected PN stimulus / recovery spikes | Connected LN recovery spikes |
| --- | ---: | ---: | ---: |
| Eight early pulses | 331 / 12 | 282 / 241 | 28,432 |
| Eight delayed pulses | 318 / 12 | 277 / 169 | 28,160 |
| Sustained reference, 80 pulses | 251 / 6 | 250 / 5 | 0 |
| Unassisted reference | 331 / 12 | 277 / 169 | 28,160 |

The two reference rows retain the previously recorded isolated assay. The new
early/delayed rows use the new matched isolated recordings, including their
half-strength PN-to-LN receiving weight. Reference counts do not establish
state equivalence across these preparations.

Early assistance and the sustained reference are identical before the first
omitted pulse at tick 306. Their regulator spikes first differ at that tick,
target input at 307, and sensory gate state at 309. In ticks 300–399, the early
condition recruits 101 LNs while the sustained reference has 12 active LNs.
By recovery, the early condition has no ORN spikes but continued activity in
131 LNs and 151 PNs. Its final 200 ticks still contain 5,665 LN spikes and 48
target PN spikes. This is not merely a brief offset transient.

## Why the delayed command has no outgoing effect

Against the unassisted course, the delayed condition has an identical complete
spike raster for every recorded cell. Its gate, target inputs and regulator
release-event counts are also identical. Its membrane state is not identical:
the current pulses do reach the regulator and change its voltage.

Six pulses arrive less than three ticks after its previous spike. PAULA's
declared cooldown prevents another spike then. The other two arrive on ticks
631 and 643, when the unassisted neuron already spikes. Thus the delayed
electrode is an effective current intervention but not an effective outgoing
spike intervention in this state. Equal injected charge cannot be interpreted
as equal neural influence, or as a successful test of what an additional
inhibitory spike would have done.

The independent source membrane/reset audit covers each tick using recorded
current and intrinsic thresholds. It finds no clipping or threshold-ambiguous
spikes in any of the four connected courses. Its largest residual is below
2.2e-7. Exact target receiving replay and gate/routing checks also pass. These
checks support this explanation of execution, not the biological adequacy of
the three-tick cooldown or the assumed receptor strengths.

## Consequence for the next experiment

Presynaptic control of sensory release cannot, by itself, turn off a recurrent
source of target drive that bypasses those sensory terminals. In the early
course, conditional target replay without ORNs still produces 241 recovery
spikes; excluding positive-model LN input produces none. This replay holds
source histories fixed and is not a closed-loop lesion prediction.

Earlier regulator recruitment remains relevant, but the brief-onset hypothesis
failed. The [next test](REGULATOR_RECRUITMENT_2026-09-10.md) asks whether the
regulator's measured sensory afferents can provide sustained recruitment
without an electrode. It changes an explicitly uncalibrated initial strength,
not the measured connection graph. Physiological work also shows that LN
responses depend on intrinsic dynamics and differently timed excitation and
inhibition, so adjusting strength alone is not a complete cellular explanation.
[Nagel and Wilson, 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4829653/).

## Retained evidence

The four new recordings are `.live/research/flywire783/dl5-gain-regulator-`
followed by `early-isolated-20260910`, `late-isolated-20260910`,
`early-antennal-20260910`, or `late-antennal-20260910`. Each has its own
`comparison/analysis.json` and conditional target replay.
`dl5-gain-regulator-timing-comparison-20260910.json` retains interval-aligned
population counts, individual regulator pulse annotations, exact shared-past
comparisons, source audits and manifest hashes. `schedule_followup()` in
`gain_reunion_analysis.py` regenerates that comparison from the four connected
courses and the graph. No new bulk trace format or live server was introduced.
