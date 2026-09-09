# APL feedback depends on where the response is measured

The connected PAULA preparation now shows a strong distinction between APL's
calyx and vertical-lobe responses to KC-output blockade. In the strongest
artificial-input window, vertical-lobe local release falls by 99.82% at the
original electrical coupling and 97.97% at eight times that coupling. Calyx
release persists, falling by 12.93% and 11.63%, respectively. These are measured
model effects, not a replication of calcium physiology.

This changes the interpretation of the earlier whole-cell result. Persistent
APL activity after KC blockade does not by itself refute a regional KC-feedback
mechanism. Both the intervention and its observation location belong in the
experimental specification. Anatomical composition can support different local
roles within one neuron.

## The physiological distinction

[Lin et al., 2014, Figure 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC4000970/)
measured the APL vertical-lobe response during KC activation and KC-output
blockade. [Prisco et al., 2021](https://elifesciences.org/articles/74172) examined
APL in the calyx and its reciprocal relationships with PN boutons and KC claws.
Their discussion explicitly distinguishes calyx input processing from broad
feedback inhibition. [Amin et al., 2020](https://elifesciences.org/articles/56954)
provides experimental evidence for localized APL activity. The papers do not
establish that our voltage-to-release function predicts any of their indicators.

The present test asks whether the already connected model supports a comparable
regional distinction. It does not equate atlas regions with the authors' optical
ROIs or claim that their flies and the reconstructed FlyWire specimen are the
same animal.

## Anatomical readouts, without rewiring

The public FlyWire CATMAID project provides closed triangle meshes for four
left mushroom-body regions. `neuropil.py` preserves the source responses and
classifies all 360,309 APL tree nodes in those mesh volumes. No coordinate fit,
guessed bounding-box ROI, branch pruning or neural input is involved. Exact
duplicate vertices are merged without moving coordinates or repairing meshes.
Three ray directions agree for every node in this snapshot. This is not an
independent anatomical validation; all three use the same intersection library.

| Region | APL tree nodes | Selected PN→APL contacts | KC→APL contacts | APL→KC contacts |
| --- | ---: | ---: | ---: | ---: |
| Calyx, volume 49 | 70,461 | 1,463 | 9,825 | 12,874 |
| Peduncle, volume 53 | 48,694 | 21 | 12,458 | 9,592 |
| Vertical lobe, volume 55 | 33,747 | 0 | 5,851 | 5,018 |
| Medial lobe, volume 51 | 123,634 | 0 | 20,978 | 16,631 |
| Outside all four | 83,773 | 75 | 11,449 | 9,371 |

There are no overlapping memberships in this snapshot. Outside nodes remain
in the neuron. Another 6,212 incoming boundary contacts are retained and undriven.
Readouts classify the skeleton attachments where the cable actually receives
current, not the nearby connector centroids. Atlas membership remains an
observation choice, particularly at anatomical boundaries.

## Matched full-network recordings

All four courses contain the same 2,755 cells and original directed pairs,
including KC recurrence and PN reciprocity. Each has 224 ticks. Ticks 32 through
191 supply successive 40-tick PN current levels of 1.25, 2.5, 5 and 10. The
preceding and following 32 ticks have no experimental input. Every one of the
174 selected ALPNs receives this artificial course, including non-olfactory
cells. It is not an odor pattern or four independent trials.

At each of `Rm/Ra=25,000 µm` and `200,000 µm`, an intact course is compared with
KC-output blockade. The latter coupling came from the preceding input-physics
sensitivity work; it was not selected to match this regional result. All other
parameters, anatomy, initial states and external PN inputs match within each
pair. The ordinary builder defaults are unchanged. Native postsynaptic and
retrograde adaptation remain positive and their recorded coefficients change.
This does not validate inherited spike-timing plasticity in nonspiking APL.

The following are means over ticks 152 through 191, not endpoint measurements.
Voltage is membrane-area-weighted within each region. Local release is the mean
of `clip(0.01*v, 0, 1)` at all recorded APL output attachments in that region,
including unlinked attachments. It precedes pair averaging, learned terminal
gain and any blockade. It is not the delivered synaptic event or a calcium trace.

| Coupling, µm | Region | Voltage, intact → KC blocked | Local release, intact → KC blocked |
| --- | --- | ---: | ---: |
| 25,000 | Calyx | 146.7768 → 107.0058 | 0.919023 → 0.800152 |
| 25,000 | Vertical lobe | 24.9040 → -2.1236 | 0.253595 → 0.000461 |
| 200,000 | Calyx | 133.3504 → 95.3680 | 0.935742 → 0.826869 |
| 200,000 | Vertical lobe | 23.8861 → -2.0294 | 0.254930 → 0.005165 |

The preceding level, PN drive 5, also separates the regions. Vertical release
falls by 99.37% and 93.67% at the two couplings, versus 7.88% and 9.96% in the
calyx. The two lower windows have identical intact and KC-blocked regional
traces. No KC output has yet arrived at APL in those windows.

The release ceiling compresses the calyx effect. At coupling 25,000, the final
window has 73.97% of calyx output attachments at the cap in the intact run and
51.07% under KC blockade, averaged over ticks. Calyx voltage falls by 27.10%,
which is appreciably more than its 12.93% release reduction. Reporting only the
release percentage would conceal this nonlinear observation effect.

## What the ticks establish, and what they do not

PN current first reaches APL at tick 66. At coupling 25,000, the first KC current
enters the calyx at tick 119 and the vertical lobe at tick 123. Vertical voltage
already differs slightly between intact and blocked runs at tick 119, before
local KC input arrives. The passive tree transmits the effect from elsewhere.
That first difference is only 0.00001226 voltage units and is not a biological
latency estimate. Implicit branch exchange has no hard propagation front.

Under KC-output blockade, recorded direct input to the vertical and medial
lobes is zero at every tick. Their remaining voltage comes through axial
exchange from other parts of APL. The observer reconstructs each region's input
from actual delayed port currents and contact counts, and checks it against
the independently recorded node currents. Regional charge balance supplies
the net axial inflow. It does not identify a unique origin for every later
voltage component.

The negative vertical voltage matters. In the original-coupling blocked course,
PNs deliver 2,455.38 positive and -157.84 negative current units over the whole
course. Negative input enters the calyx, peduncle and outside region; none enters
the vertical lobe directly. The model's negative source signs and artificial
activation of the broad ALPN population therefore need investigation before
interpreting near-zero rectified vertical release as faithful physiology.
The current experiment has not isolated the contribution of those negative
inputs. Rectification can hide hyperpolarization, just as saturation can hide
extra excitation.

Whole-KC output blockade also removes KC→KC and KC→PN transmission. It is not
a selective lesion of KC→APL. PN spikes subsequently differ despite identical
external current. Both KC feedback and other recurrent effects contribute to
the later trajectories. The preserved connections are part of the explanation,
not irrelevant edges to discard for a cleaner diagram.

## Verification and retained evidence

`regional_activity.py` keeps every tick of regional voltage, local release,
saturation, signed source current and net axial inflow. Empty regions have a
zero denominator and a missing value, never fabricated silence. It first runs
the full-record auditor over the original intracellular and synaptic traces.
Those traces, rather than the regional summaries, remain the primary model
evidence.

All four records pass node-equation, current-placement and release checks.
Maximum per-node equation residual across them is `1.262e-12`; maximum regional
input-reconstruction discrepancy is `2.274e-13`. The new intact 200,000 course
also replays without tick instrumentation, exactly matching 81,069,525 recorded
branch-voltage values, 80,709,216 node-current values, every recorded synaptic
coefficient and the other recorded states. The original-coupling intact replay
was already verified in the preceding work. Numerical agreement does not
establish physiological accuracy.

The new three raw neural courses occupy about 1.55 GiB. The four regional
readouts and their full-record audit reports together occupy about 468 KiB.
Raw data remains under `.live/research/flywire783/`, not in Git. Existing intact
25,000 data was reused, not regenerated. No live agent server was started.

Local paths under that directory:

- Public source: `mb-neuropil-meshes-20260910`.
- Anatomical memberships: `apl-neuropil-bound-20260910`.
- Existing intact course: `pn-course-local-cable-intact`.
- New courses: `pn-course-local-cable-kc-block-r25000`,
  `pn-course-local-cable-intact-r200000`, `pn-course-local-cable-kc-block-r200000`.
- Readouts: `apl-regions-intact-r25000`, `apl-regions-kc-block-r25000`,
  `apl-regions-intact-r200000`, `apl-regions-kc-block-r200000`.

Each readout contains `per_tick.npz`, `full_record_audit.json` and `analysis.json`
with hashes of its source recording and anatomical binding. The optional mesh
parser was tightened after binding; its final version reproduces every stored
vertex and triangle exactly. The per-node memberships did not change.

## Reproduction and next experiment

From `active-inference`, use fresh output paths:

```sh
uv sync --locked --extra dev --extra flywire --extra physiology --extra spatial
uv run python -m simulations.drosophila.neuropil acquire NEW_MESH_SNAPSHOT
uv run python -m simulations.drosophila.neuropil bind \
  NEW_MESH_SNAPSHOT .live/research/flywire783/apl-left-spatial-bound-20260910 NEW_BINDING
uv run python -m simulations.drosophila.intervention_probe \
  .live/research/flywire783/kc-apl-left-v1 NEW_COURSE \
  --condition kc_release_block --weight-per-count 0.075 \
  --apl-representation local_cable --apl-cable-rm-over-ra-um 200000 \
  --spatial .live/research/flywire783/apl-left-spatial-bound-20260910
uv run python -m simulations.drosophila.regional_activity \
  NEW_COURSE NEW_BINDING NEW_REGIONAL_READOUT
```

Use `intact` for the matched reference, then compare the two raw courses with
`intervention_analysis --reference`. The public server can change; exact
reproduction requires the retained source responses, not a fresh download
assumed identical. `uv run pytest tests -q -k drosophila` currently passes
135 tests. Mesh geometry and spatial-current checks are not replaced by a
successful file-hash comparison.

The next discriminating experiment should address the input course, especially
the inhibitory and non-olfactory PNs, while retaining both observation regions
and the actual graph. A declared input-subset control can diagnose the present
artifact. A physiological claim still requires an empirically grounded odor
mapping and an observation model. Another gain sweep or adding downstream
learning before resolving that distinction would not answer this question.
