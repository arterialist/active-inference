# A conditional sensory gain mechanism

The isolated DL5 ORN/PN/LN preparation now demonstrates context-dependent gain
control under explicitly imposed sensory and LN current commands. A local
PAULA terminal-regulation extension changes how much a sensory spike transmits,
while preserving native somatic integration and ongoing plasticity. The result
does not establish public-odor normalization, an appropriate biological
operating range, adaptive self-regulation, learned discrimination or survival
of the function in the reunited brain.

## The selected functional boundary

This cut contains all 42 selected ORN_DL5 cells, DL5_adPN
`720575940617207185`, and left lLN2F_b `720575940623000858`. The LN was chosen
before observing functional outcomes: it is the strongest left source of
model-negative input to the selected ORNs. It has 444 counted contacts onto
all 42 ORNs and 15 onto the target PN. Its GABA prediction confidence is 0.574;
its curated transmitter field is empty. No claim is made that this exact cell
matches the driver-defined populations used in the cited physiology.

The cut retains 673 internal directed pairs with 3,108 contacts. These include
ORN recurrence, ORN→LN, PN→LN and PN→ORN feedback. It also retains all incident
boundary ports: 3,109 incoming pairs and 4,979 outgoing pairs. The 2,991 boundary
neurons are absent and undriven. Neither an omitted partner nor its activity
is silently replaced by a generic neuron.

Direct drive consists of staggered one-tick current pulses to all ORNs. The
independently varied lateral drive is current injected into the selected LN.
These electrodes ask what the circuit does under controlled activity. The LN
command is not a simulated public odor and does not establish that the absent
upstream circuit will generate appropriate lateral activity autonomously.
Strong pulses also make this assay insensitive to modest changes in ORN somatic
excitability; it cannot adjudicate odor transduction at the sensory periphery.

The commanded direct rates are 0, 2, 5, 10, 20, 50 and 100 pulses per nominal
second. LN commands are 0, 20 and 80. Each course has 200 baseline ticks and
two identical 1,000-tick stimulation periods, each followed by 1,000 recovery
ticks. We record actual spikes rather than assume a pulse generated a spike.
Seeds 11, 23 and 44 permute sensory-neuron phase assignments. They are timing
realizations, not biological replicates. Repeated trials share ongoing plasticity
and intracellular state; no resetting or retraining occurs between them.

## The native adapter fails to regulate transmission sufficiently here

With the existing somatic inhibitory action, LN stimulation barely changes PN
output. Sensory drive at 50 and 100 produces approximately 330 and 331 PN spikes
per nominal second regardless of the tested LN command. Different high sensory
inputs therefore approach the same refractory ceiling.

At direct rate 10 and lateral rate 80, blocking LN→ORN release leaves the PN
response unchanged. Blocking LN→PN release instead restores the small rate
reduction seen in two of the three timing seeds. The current adapter's modest
effect comes through direct PN inhibition in these conditions. Its LN→ORN
somatic route does not alter the PN spike output. This is not evidence that
the anatomical sensory-terminal route is biologically unimportant.

[Olsen and Wilson, 2008](https://doi.org/10.1038/nature06864) found substantial
lateral inhibition at ORN terminals. The existing mapping lacks that regulatory
action. [Olsen, Bhandawat and Wilson, 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2866644/)
separated direct and lateral sensory input and found a context-dependent shift
in the PN transfer curve, along with more transient responses. Those findings
motivate the functional test. They do not calibrate the currents or cells here.

## Added cellular hypothesis

The opt-in `PresynapticInhibitionNeuron` reads the designated anatomical
inhibitory port through its native current coefficient, delay and attenuation.
That input drives a decaying intracellular quantity. The cell multiplies its
native ALPN-directed terminal release by `1/(1+gain*state)`. Other outgoing
terminals are unchanged. Ordinary somatic inhibition and all native receiving
and retrograde learning remain active. No population firing-rate calculation,
stimulus label or target output enters the neuron.

This is a supplied receptor-to-release hypothesis. The existence of suppression
is therefore not an emergent discovery. The measured neural transfer curve and
its pathway dependence test the consequences of composing that hypothesis with
the selected circuit. The law is an effective approximation, not fitted GABA
receptor kinetics. A shared 100-tick decay and gains 0, 0.1 and 1 are explicitly
declared sensitivity settings. The reference builder and core PAULA source
remain unchanged. A temporary constructor factory selects the subclass only
during experimental assembly and is restored before neural execution.

## Functional result and tradeoff

The table shows first-trial PN spikes per nominal second for seed 11 with LN
drive 80. The input numbers are also the observed mean sensory firing rates in
these courses. Nominal rates are not full-circuit physiological calibration.

| Sensory rate | Native somatic action | Terminal gain 0.1 | Terminal gain 1 |
| --- | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 |
| 2 | 21 | 9 | 0 |
| 5 | 68 | 42 | 1 |
| 10 | 140 | 92 | 19 |
| 20 | 245 | 186 | 49 |
| 50 | 330 | 330 | 129 |
| 100 | 331 | 331 | 251 |

Gain 1 makes previously near-ceiling inputs distinguishable by PN spike count.
At sensory rate 50, its PN output is 129 across all three phase seeds. At rate
100 it is 251, 251 and 253. This is evidence that a downstream neuron could
receive a useful rate difference; an actual trained neural consumer has not
yet been demonstrated.

The same setting loses the weakest signal under strong lateral drive. Rate 2
is completely suppressed, and rate 5 becomes an almost purely onset response.
Gain 0.1 preserves weak responses but does not resolve high-drive saturation.
Neither setting is declared the correct operating point. Context-dependent
masking may be useful or harmful depending on the sensory competition and the
consumer's task. We cannot select that operating range by maximizing quietness.

The stronger gate also changes response time course. At direct rate 50, the PN
emits 37 spikes in the first 200 ticks and 22 in the last 200, compared with
64 and 66 in the native condition. It emits three recovery spikes rather than
18. This is an imposed inhibitory-state buildup shaping the circuit's transient
response, not a fitted replication of the physiological waveform.

## Pathway interventions

At gain 1 and lateral rate 80, the matched pathway tests give:

| First-trial PN rate, seed 11 | Direct 10 | Direct 50 |
| --- | ---: | ---: |
| All pathways active | 19 | 129 |
| LN→ORN chemical release blocked | 140 | 330 |
| LN→PN chemical release blocked | 20 | 130 |
| All LN chemical release blocked | 142 | 330 |

The gate's major effect requires the measured LN→ORN route. It largely survives
removal of direct LN→PN inhibition. The same conclusion holds in seeds 23 and
44. Release interventions leave neuronal state, receiving learning and return
events active. They do not edit the anatomical pair table.

Across every matched recorded condition, ORN and LN spike times are identical
to the corresponding native intact course. The PN change is therefore a change
in transmission and postsynaptic integration, not an unnoticed change in the
source spike trains. At direct rate 100 with no LN electrode, seed 11 has 58 PN
spike-tick differences for gain 0.1 and 68 for gain 1, despite unchanged trial
spike totals. Counting spikes alone misses this alteration of the neural signal.

Silent-source and zero-gain comparisons test compatibility separately. Full per-tick
PN receiving histories, receiving weights and current contributions allow an
independent native PN replay. Native and effective terminal amplitudes are
recorded as per-ORN sums across configured terminals, not as complete individual
terminal histories. Their sum comparison requires a float32 accumulation bound;
gate state evolution and the release fraction are checked separately.
All 183 primary courses pass PN receiving replay, totaling 768,600 ticks.
Actual-LN-silent conditions match their native records in 190,771,848 values.
Three nonzero-lateral-drive courses with coupling gain set to zero separately
match 23,846,481 native values. These checks verify the interventions and
readouts; they are not physiological or behavioral acceptance criteria.

## What must happen next

The demonstrated component is a conditional regulator, not a validated
autonomous normalization network. A zero LN electrode command does not imply
a silent LN. Direct sensory rate 100 recruits approximately 33 LN spikes per
nominal second without injected lateral drive. The gate changes neural timing
in that condition despite unchanged total PN counts. At direct rates up to 50,
the LN remains silent without its electrode. Thus the cut already contains a
sensory-driven regulatory loop, but it engages too late or too weakly to prevent
the PN ceiling under the current assumptions.

Its missing functional input is lateral drive that responds appropriately to
sensory context over the relevant range. Test that boundary using actual
upstream partners or clearly specified neural input histories before claiming
public-odor normalization. Do not infer it from the existence of anatomical LN
inputs alone: anatomical feedback exists here without adequate gain regulation.

The next composition test must also restore identified PN/KC/APL partners
without changing this local mechanism to suit the new result. Measure whether
the gain effect persists, whether the selected PN remains controlled by sensory
input rather than recurrent recruitment, and whether KCs retain distinguishable
responses. Repeat the receptor-coupling null and the specific release-pathway
lesion in that connected preparation. A calm network with no useful KC response
is a failure, as is a large recurrent response that erases sensory differences.

This is the bridge to subsequent learning: provide distinguishable, regulated
neural input to a plastic consumer, then show that learning and reciprocal
coordination survive composition. An isolated transfer curve is not a claim
that the brain-building objective has been achieved.

## Evidence locations

The raw evidence is local under `.live/research/flywire783/`:

- `dl5-ln-gain-intact-20260910/`: 63 native courses.
- `dl5-ln-gain-lesions-20260910/`: 18 native pathway-lesion courses.
- `dl5-ln-gain-terminal-01-20260910/`: 21 weak-gate courses, seed 11 only.
- `dl5-ln-gain-terminal-1-20260910/`: 63 strong-gate courses and comparison.
- `dl5-ln-gain-terminal-lesions-20260910/`: 18 strong-gate lesion courses.

Use `python -m simulations.drosophila.ln_gain --help` for explicit input files,
rates, seeds, lesions and optional gate settings. Omit `--inhibition-gain` for
the native reference. Use `ln_gain_analysis` for full-record summary checks,
PN replay and matched parity comparisons. No live server is started. Existing
outputs are never overwritten, and old manifests retain their original source
hashes when the observation or analysis code changes.
