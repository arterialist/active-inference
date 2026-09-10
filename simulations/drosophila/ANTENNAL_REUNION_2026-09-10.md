# DL5 upstream reunion and the onset of broad recurrent activity

The identified DL5 PN now receives native spikes from reconstructed ORNs,
alongside its measured local-interneuron partners, in the connected PN/KC/APL
preparation. Six completed recordings expose a specific model failure. Simple
ORN terminal depression postpones broad recruitment; it does not prevent it at
the stronger drive. Blocking release from one identified interneuron prevents
the first train's broad recruitment. A separate transmitter-polarity control
reduces activity but does not resolve it. None is an odor-coding or behavioral
replication, and neither lesion nor sign control changes the default circuit.

## Anatomy preserved across reunion

`antennal_lobe.py` retains all 2,755 previously selected cells and adds all 42
cells annotated ORN_DL5 plus 208 ALLNs with at least one counted connection in
either direction with those ORNs or the identified DL5 PN. Both hemispheres
are included. The selection does not filter by predicted transmitter, sign,
connection strength above one count, or presence of a detailed cell-type label.
This is a declared neighborhood, not a closed or complete antennal lobe.

The resulting 3,005-cell graph contains 213,855 internal pairs and 909,634 counted
contacts. Incoming boundary pairs retain 514,692 contacts; outgoing boundary
pairs retain 831,622. All 387,427 incident rows of the old preparation, all
11,966 previously retained node records, and the old cells' port ordering survive
exactly. No PAULA address overflow occurs.

New cells were re-extracted from the original pinned connectivity table.
Promoting boundary stubs from the old cut would omit connections among those
new cells and other boundary cells. For example, that shortcut would omit
84,213 incoming and 82,913 outgoing incident pairs across the added ALLNs.
These per-cell incident counts are not unique-edge totals.

The recorded graph includes 213,037 ALLN→ALLN, 87,891 ALLN→ALPN, 82,144
ALPN→ALLN, 4,137 ALLN→ORN and 5,339 ORN→ALLN contacts. Earlier KC/APL motifs
remain present. Incoming boundary activity is absent and explicitly recorded as
a boundary condition, not inferred dispensability.

## Terminal dynamics and the stimulus

The new native subclass is documented in the sibling neuron-model repository
at `neuron/extensions/experimental/RELEASE_DEPRESSION.md`. Each selected
terminal emits its native forward coefficient times a recoverable resource.
It consumes resource only on its own release. Native soma dynamics, weak
positive receiving plasticity and retrograde updates continue. No odor labels,
stimulus schedules or decoded network error enter the extension. Zero depletion
is bit-exact with ordinary PAULA in its regression test.

[Nagel, Hong and Wilson, 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4289142/)
Figure 1 supplies the simple control's per-spike retained fraction 0.78 and
recovery time 893 ms. Their simple model fits depression during a 10-Hz train
but cannot explain sustained odor responses. The paper distinguishes fast and
slow components and presynaptic inhibitory regulation. We do not claim to have
implemented that complete mechanism. These control values were transferred
from DM6/VM2 physiology to DL5 and all its ALPN-directed terminals, an explicit
cross-glomerulus assumption.

The 4,200-tick course has 200 baseline ticks, a 1,000-tick 10-Hz-equivalent
train, 1,000 recovery ticks, a 1,000-tick 50-Hz-equivalent train, then 1,000
recovery ticks. Each source receives a one-tick pulse of 40 model-current units,
staggered by source-order rank. The input is electrical, not odor transduction.
The source soma decides whether to spike; recurrent input can veto or supplement
the pulse. The 1-ms/tick interpretation is nominal. The full network and ORN
intrinsic dynamics are not physiologically clock-calibrated.

DL5 retains the previous target-only current-tail, gain and membrane calibration.
Other cells retain the uncalibrated 0.075/count gain and ordinary soma settings;
APL retains its full local passive cable. ALLN→ORN effects are still somatic
signed currents, not a reconstruction of presynaptic GABA regulation.

## Three full courses

Counts below are spikes during the specified 1,000-tick window, not calcium or
inferred firing rates. ALPN counts include the target DL5 PN.

| Condition and window | DL5 | All ALLNs | All ALPNs | All KCs | Maximum local APL release |
| --- | ---: | ---: | ---: | ---: | ---: |
| No depression, first train | 290 | 34,037 | 21,485 | 13,621 | 1.0, at cap |
| Depression, first train | 78 | 7 | 78 | 0 | 0.07884 |
| Depression + all-ALLN release block, first train | 78 | 7 | 78 | 0 | 0.07871 |
| No depression, stronger train | 333 | 43,061 | 26,996 | 16,762 | 1.0, at cap |
| Depression, stronger train | 267 | 41,887 | 26,188 | 16,091 | 1.0, at cap |
| Depression + all-ALLN release block, stronger train | 101 | 59 | 101 | 0 | 0.10384 |
| No depression, final recovery | 184 | 42,617 | 26,789 | 16,466 | 1.0, at cap |
| Depression, final recovery | 198 | 43,373 | 27,168 | 16,343 | 1.0, at cap |
| Depression + all-ALLN release block, final recovery | 1 | 0 | 1 | 0 | 0.07186 |

The blocked ALLNs can still spike and learn. Their forward events are filtered
before routing, while return events remain. Low KC activity under blockade is
therefore not an olfactory success: it removes the very interactions being
reconstructed. Similarly, bounded high activity is not numerical divergence,
but it prevents treating this preparation as established stimulus-selective
olfactory coding.

The depression and no-depression DL5 voltages first differ at tick 303, their
spikes at 317, and their ORN spikes at 481. During the no-depression first train,
there are 36 off-pulse ORN spikes and two pulses without same-tick spikes.
The intact depressing run realizes all requested spikes during both trains,
then generates 54 ORN spikes without commanded pulses in the final recovery.
This closes the modeled cause-effect loop back to the sensory population.

All 12,600 DL5 ticks replay exactly from recorded receiving buffers, including
per-port current, soma fields and receiving weights. Across the three runs,
529,200 ORN-cell ticks pass independent membrane checks and 3,087,000 terminal
resource values pass recovery/depletion checks. No source membrane clipping or
precision-ambiguous spikes occurred. Source equations and release amplitudes
have explicit float32 rounding bounds; they are not full ORN-history replays.
Receiving and terminal coefficients changed in every condition. The recordings
omit full queues, all other synaptic histories and the full evolving APL tree;
they are not restart checkpoints.

## A single-cell causal intervention

The follow-up retains all 3,005 cells and the same no-depression pulse course,
stopping at tick 1,600, after 400 ticks of the first recovery. It records the
complete receiving histories of four identified ALLNs and replays each as it
runs. There are three conditions: intact, forward release blocked from
`lLN2T_e` root `720575940633483807`, and the separate polarity control described
below. No higher-drive or long-run stability claim follows from this shorter test.

The new intact run matches all 14,424,000 original soma values over those 1,600
ticks exactly. Across the three conditions, 19,200 watched-cell ticks replay
exactly in soma, receiving weights and intrinsic fields. First-spike input
attribution reconstructs per-port current within a disclosed float32 bound;
the largest full-course residual is 7.99e-5 model-current units.

The first negative-sign LN, `lLN2P_c` root `720575940628343634`, fires at 258.
Its DL5 input supplies approximately 1.02768 of the 1.04820 modeled voltage
at threshold crossing. The first positive-sign LN, `lLN2T_e`, fires at 334.
The DL5→LN pair has 94 contacts, receiving port 168 and source row 3267788.
It supplies approximately +1.03030 voltage; the early inhibitory LN contributes
−0.01918, and direct ORN inputs supply the remaining small positive contribution.
The summed voltage is 1.03003 against a threshold of one.

Removing this LN's release first changes soma state at tick 337 in 251 cells.
The first spike difference is at 340 in the already recruited inhibitory
`lLN2P_c`, not simply the next excitatory cell in a serial chain. In particular,
`lLN2T_c` root `720575940618757666` first fires at 340 with or without the
block. Its first spike does not require `lLN2T_e`, although that input contributes
0.11607 voltage in the intact case. Later recruitment differs radically.

| First train, ticks 200–1199 | ALLN spikes / cells | ALPN spikes / cells | KC spikes / cells |
| --- | ---: | ---: | ---: |
| Intact | 34,037 / 169 | 21,485 / 151 | 13,621 / 712 |
| One LN's forward release blocked | 61 / 3 | 138 / 1 | 0 / 0 |
| Curated-GABA negative-current control | 28,795 / 140 | 16,624 / 135 | 9,578 / 519 |

The blocked cell still fires 16 times across the shorter course. Its 24,624
attempted forward events are withheld; 348 return events pass through. During
the final 400 ticks, the block condition has two ALLN spikes, three DL5 spikes,
no ORN spikes and no KC spikes. Intact has 17,089 ALLN, 10,734 ALPN and 6,638 KC
spikes in the same window. This establishes a necessary contribution of that
cell's forward release to the original broad regime in this protocol. It does
not identify a uniquely responsible outgoing edge or show that the biological
cell should be absent.

## What the transmitter audit does and does not establish

All 208 ALLNs have consistent source-table outgoing signs: 120 positive and
88 negative. Only 26 have nonempty curated transmitter fields. Fourteen are
acetylcholine-labelled, eight GABA/MIP, two GABA, and two glutamate; all 26 have
positive source-model signs. Seventeen predictions fall outside the curated
transmitter set. GABA predicted for a GABA/MIP annotation is not counted as a
conflict. Prediction scores, curated labels and their literature sources remain
separate in `antennal_identity.py`; none is relabelled automatically.

The sensitivity control sets the receiving current sign negative for internal
targets of the ten curated-GABA/positive-model sources, preserving magnitudes,
all anatomical rows, port addresses, dynamics and adaptation. It changes neither
the pinned sign table nor absent boundary receptors. This control starts to
affect neural state at 403, after the early 334-tick recruitment. It reduces
activity but leaves APL at its release cap and 4,682 KC spikes during recovery.
These ten polarities are therefore not a sufficient explanation of the failure.

The biological interpretation must remain target-specific.
[Shang et al., 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2866183/) identified
cholinergic local neurons and evidence for lateral excitation.
[Yaksi and Wilson, 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2954501/) found
electrical eLN→PN transmission and opposing direct excitation and indirect
inhibitory effects. [Huang et al., 2010](https://doi.org/10.1016/j.neuron.2010.08.025)
reported mixed cholinergic/electrical reciprocal connections. These findings
cannot be collapsed into one chemical sign or used to add invented gap-junction
edges to this chemical pair table. Nor do their driver-labelled populations
establish an exact physiological match to our particular FlyWire root.

[Sizemore et al., 2023](https://www.nature.com/articles/s41467-023-41012-3) found
GAD1 overlap in AL MIP neurons, but also different MIP effects across glomeruli
depending on receptor expression. Their hemibrain MIP assignments are candidate
matches. A negative-current test does not implement that peptide system, identify
all postsynaptic receptors, or validate the cross-specimen annotation mapping.

## Next experiment justified by these results

Restore the full unblocked pathway. Constrain the recurrent PN↔LN interaction
and the LN's recruited targets with identified physiological responses, then
test the same trains and recovery windows again. Separate fast chemical
transmission, electrical coupling where independently supported, and terminal
inhibition. Do not solve the failure by leaving the causal hub blocked, changing
its weak serotonin prediction into a biological assertion, or counting silence
as sparse coding. The original intrinsic PN fit remains useful, but it cannot
calibrate the newly active recurrent loop by itself.

## Reproduction and retained evidence

The raw anatomical graph and recordings stay in ignored `.live/research/flywire783/`.
Source, tests and these findings are versioned. The PAULA extension is in
neuron-model commit `ffe9a5d`. No original records were overwritten.

| Artifact directory | Contents |
| --- | --- |
| `dl5-antennal-expansion-20260910` | Full graph, selection witnesses, original-cut comparison, transmitter audit |
| `dl5-intrinsic-release-source-20260910` | Intrinsic rerun with current builder-source provenance |
| `dl5-orn-trains-20260910` | Three full courses and independent analysis |
| `dl5-orn-onset-20260910` | Three targeted courses, full watched receiving histories and analysis |

The intrinsic rerun matched 2,047,194 prior saved array values exactly. Its
new manifest records the builder with the opt-in depression support; older
manifests are not rewritten to pretend the source was unchanged.

From the active-inference package, with the pinned sources and prior calibration
artifacts available, the entry points are:

```sh
uv run python -m simulations.drosophila.antennal_lobe SOURCE PARENT_GRAPH NEW_EXPANSION
uv run python -m simulations.drosophila.antennal_identity GRAPH NEW_AUDIT_JSON
uv run python -m simulations.drosophila.orn_train GRAPH INTRINSIC_JSON TAIL_JSON SPATIAL_APL NEW_OUTPUT no_depression
uv run python -m simulations.drosophila.orn_train GRAPH INTRINSIC_JSON TAIL_JSON SPATIAL_APL NEW_OUTPUT depressing
uv run python -m simulations.drosophila.orn_train GRAPH INTRINSIC_JSON TAIL_JSON SPATIAL_APL NEW_OUTPUT depressing_ln_block
uv run python -m simulations.drosophila.orn_train_analysis GRAPH INTRINSIC_JSON TAIL_JSON COURSE_PARENT NEW_ANALYSIS_JSON
uv run python -m simulations.drosophila.orn_onset GRAPH INTRINSIC_JSON TAIL_JSON SPATIAL_APL ORIGINAL_NO_DEPRESSION NEW_OUTPUT intact
uv run python -m simulations.drosophila.orn_onset GRAPH INTRINSIC_JSON TAIL_JSON SPATIAL_APL ORIGINAL_NO_DEPRESSION NEW_OUTPUT first_positive_ln_block
uv run python -m simulations.drosophila.orn_onset GRAPH INTRINSIC_JSON TAIL_JSON SPATIAL_APL ORIGINAL_NO_DEPRESSION NEW_OUTPUT curated_gaba_negative_control
uv run python -m simulations.drosophila.orn_onset_analysis GRAPH ONSET_PARENT NEW_ANALYSIS_JSON
uv run python -m pytest tests/test_drosophila*.py -q
```

Use a different output directory for each condition. Parent directories must
contain subdirectories named for their conditions when invoking the comparison
analyzers. Strict source checks require the generating revision and matching
calibration provenance. Commands are bounded offline recordings; none starts a
live agent or server. All six workers completed. The fly tests total 183 passing
tests, and the neuron-model suite totals 72. These counts measure software
verification coverage, not physiological acceptance.
