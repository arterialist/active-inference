# Does local-neuron stabilization depend on uncertain current polarity?

## Question fixed before the recordings

The previous branch intervention implicated the identified regulator's outputs
to other local neurons in preventing persistent antennal activity. One of the
first two local neurons with changed spikes was il3LN6, root
720575940623636701. Its curated annotation says GABA, while the imported source
model assigns positive output signs. This motivates testing the branch result
under a previously defined polarity control before attributing its failure to
missing receptor kinetics or adding another regulatory mechanism.

The control reuses `orn_onset.negative_gaba_control`. It reverses the initial
receiving-current coefficients of every internal projection from selected
ALLNs whose curated annotation includes GABA and whose imported outgoing signs
are all positive. Selection is anatomical and annotation-based, not based on
whether a cell's sign change improves a run. The measured graph is unchanged.
Magnitudes are preserved after the previously declared feedback initialization.
Every native learning rate remains positive.

This is not a claim that all these targets have the same GABA receptor, that
GABA must always hyperpolarize, or that cotransmitters can be ignored. It is a
test of dependence on an uncertain signed-current assumption. The biological
reference retains its original imported signs.

## Three matched courses

All courses retain the 424-cell antennal cut, feedback scale 0.5, sensory
afferent scale 2, seed 11, no regulator electrode, and 2,200 recorded ticks.
Sensory stimulation occupies ticks 200 through 1199. Recovery is observed
through tick 2199. Only the following experimental factors differ.

| Course | Sensory command | Additional branch intervention |
| --- | ---: | --- |
| High input | 100 | None |
| Low input | 50 | None |
| High input, regulator-to-LN block | 100 | Withhold forward events on the same 93 internal regulator-to-ALLN pairs from tick zero |

The matched original-sign recordings already exist. High and low intact
courses test whether the operating point and low-input persistence survive the
polarity assumption. The high-input branch block tests whether the regulator's
local-neuron branch remains necessary under the alternative signs. No outcome
will be used to retune a gain during this comparison.

The isolated 44-cell reference contains none of the control's candidate
sources. It can therefore be reused without changing its dynamics. Analysis
checks that exclusion explicitly. Connected causal comparisons must have
identical polarity settings; this exception does not apply to them.

## Evidence required

Compare every population during stimulation and recovery, not only the target
PN. Locate the first changed somatic state and spike against each original-sign
course. For the branch comparison, establish the first withheld forward event
and exact shared recorded history before it. Replay every target PN tick with
the alternative receiving signs, verify sensory terminal gating and delivery,
and audit source current integration. Current groups and conditional receiving
tests use the initial effective signs, explicitly distinguished from imported
model signs. Conditional receiving replays are not closed-loop predictions.

The changed receiving coefficients must match the graph identities, contact
weights and prior scaling. The experimental recorder preserves their actual
initial values and records changes from ongoing learning. These checks can
detect implementation mistakes; they cannot establish physiological validity.

## Completed results

The control changed 1,595 internal receiving coefficients representing 16,255
contacts from ten candidate sources. All three courses completed. The table
reports activity during the entire 1,000-tick recovery window, including brief
offset responses. Persistent cases still have LN and PN spikes at tick 2199.

| Course | Original-sign target PN recovery spikes | Control target PN recovery spikes | Original-sign ALLN recovery spikes | Control ALLN recovery spikes |
| --- | ---: | ---: | ---: | ---: |
| High input, intact | 5 | 5 | 0 | 1 |
| Low input, intact | 167 | 167 | 28,446 | 25,154 |
| High input, regulator-to-LN block | 244 | 168 | 29,300 | 25,769 |

The high-input operating point survives this control. Its last LN spike is at
1201 and last PN spike at 1255. During stimulation, the identified regulator's
count decreases from 98 to 85 while target PN count increases from 219 to 226.
Mean sensory-terminal release fraction increases from 0.181 to 0.200. Added
negative receiving signs reduce overall local activity but also weaken the
regulator's sensory gate. Output response is therefore not a monotonic function
of how many connections have inhibitory signs.

The low-input failure is delayed, not repaired. Under original signs, 133 LNs
fire during ticks 300–399. Under the control, seven LNs fire in that interval;
only seven to eleven participate in each 100-tick interval through tick 799.
Recruitment then expands to 96 in ticks 800–899 and 121 in ticks 900–999. The
regulator spikes at 228, 686 and 850 before entering frequent firing from 867.
Its later saturation is not evidence that it prevented the transition. The
target PN's unchanged recovery count conceals both this different onset and
the change in surrounding population activity.

The branch result also survives. Blocking the same 93 regulator-to-ALLN pairs
under the control produces persistent activity, while its matched intact
reference recovers. The first withheld forward event is at 213, first changed
somatic state at 216 and first changed spike at 224. All recorded variables
share an exact past before the first withheld event. There are 59,613 withheld
forward events and 77,561 preserved return events from the selected source.
The regulator-to-ORN gate remains engaged, with mean recovery release fraction
0.0492. Sensory gain suppression alone does not stop this recurrent activity.

Every target PN tick replays exactly with the declared receiving signs. Source
integration audits report no clipping or ambiguous threshold decisions. Native
learning changes all 35 selected sensory afferent weights in each course and
565, 14,730 and 14,685 selected feedback weights in the high, low and blocked
courses respectively. This verifies ongoing updates, not useful learning.

These results rule out the ten curated-GABA polarity conflicts as the sole
explanation for either failure. They do not resolve the remaining transmitter,
receptor, strength or cellular-response assumptions. Nor do they establish
that every blocked connection is independently necessary.

## Next discriminating experiment

Keep the circuit fixed and compare low input from rest with the identical low
input after a high-input period has established regulated activity. Follow both
through stimulus removal. This tests whether the failure depends on how the
complete adaptive state is reached, or whether lowering input also destroys
an already established regime. Such history dependence would not by itself
separate fast neural state from weight-mediated memory. No new gain or neuron
extension is justified by the present comparison alone.

## Records

Under `.live/research/flywire783/`, the three new courses are
`dl5-regulator-gaba-high-20260910`, `dl5-regulator-gaba-low-20260910` and
`dl5-regulator-gaba-high-lnblock-20260910`. Each contains an audited PN replay
analysis and `polarity-comparison.json` against its matched original-sign
course. `dl5-regulator-gaba-low-recruitment-20260910.json` retains the interval
and regulator-spike details. The recordings occupy about 32 MiB together.
The focused and full fruit-fly test suites pass, with 210 tests in the latter.
