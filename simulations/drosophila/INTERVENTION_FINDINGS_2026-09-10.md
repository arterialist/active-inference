# KC/APL intervention findings, 10 September 2026

The imported 2,755-cell FlyWire preparation supports KC recruitment and
KC/APL inhibitory feedback under a declared artificial drive. It does not yet
reproduce odor coding, calcium imaging or learned behavior. The important
result is a traceable intervention in the actual interconnected graph,
including a small response with the opposite sign from the later response.

## Experiment

Every course starts from the same explicitly constructed state. There is no
random wiring or biological replication claim for this one specimen. All 174
actual ALPN providers receive current 1.25, 2.5, 5 and 10, respectively, during
ticks 32–71, 72–111, 112–151 and 152–191. Ticks 0–31 and 192–223 have zero
experimental input. These are consecutive levels in one adapting trajectory,
not independent dose-response trials. A tick has no assigned physical duration.
The ALPN selection includes non-olfactory cells, so this course is not an odor.

At the adapter default of 0.02 per signed synapse count, no KC spikes anywhere
in the 224-tick course. The maximum KC potential is 0.9597557 against threshold
1.0. This demonstrates inadequate recruitment in this course, not permanent
silence under every possible input.

A proposed conversion of 0.08 failed before simulation. The retained
1,276-count APL→DPM boundary edge would have weight −102.08, outside PAULA's
bound. The tested conversion is 0.075 for every counted connection. No edge was
clipped, omitted or individually fitted. The default remains 0.02. All four
intervention arms use 0.075 with otherwise unchanged parameters.

KC or APL blockade removes all forward terminal events from those cells,
including events on disconnected boundary terminals. Soma dynamics, native
postsynaptic plasticity and retrograde events remain active. This is a defined
computational lesion, not a model of temperature-sensitive vesicle depletion.
APL activation adds current 50 at its experimental input during ticks 32–191.

## The response trajectories

With intact transmission, PNs first spike at tick 63 and global APL first
releases at tick 66, before any KC spike. At tick 66, APL's attributed hillock
current is 96.860812 from PNs and zero from KCs. The first three KCs spike at
tick 101. Their output reaches APL's input ports at tick 102 and its integrator
at tick 104. The KC-attributed current at tick 104 is 4.128937 intact and zero
under KC blockade, while the PN-attributed current is identical at that tick.
APL output becomes 0.0820023 intact versus 0.0799379 under blockade.

APL's remaining response under KC blockade is not unexplained activity. The
recorded direct PN inputs drive it; at tick 168 they supply 45.62155 units while
KC current remains zero. Incoming boundary slots are silent throughout.
At tick 223, after inputs cease, APL still has output 0.2037293 intact and
0.0677442 under KC blockade. The short washout does not establish return to rest.

The following counts summarize the recorded intervals, not acceptance scores.

| PN current / ticks | Intact KC spikes | KC release blocked | APL release blocked | APL activated |
| --- | ---: | ---: | ---: | ---: |
| 1.25 / 32–71 | 0 | 0 | 0 | 0 |
| 2.5 / 72–111 | 6 | 5 | 56 | 0 |
| 5 / 112–151 | 701 | 1,243 | 3,292 | 15 |
| 10 / 152–191 | 1,780 | 3,759 | 6,623 | 720 |

At current 10, respectively 1,120, 1,651, 1,953 and 483 distinct KCs fire in
these four arms. Broad recruitment under uniform drive is not sparse odor
coding. APL activation suppresses but does not abolish the strongest response.
Its assumed graded-release cap is reached at ticks 172–185, 188 and 191.
Intact APL reaches that cap at ticks 168–172. Raising the cap to improve the
endpoint would be another dynamical assumption, not physiological validation.

Under APL blockade, KC and PN membrane trajectories first differ at tick 69;
PN output differs at tick 72 and APL state at tick 75. KC output first differs
at tick 91. Thus the intervention changes upstream PN activity before the
observed KC spike difference. The APL `O` trace remains positive because it is
the attempted output; all actual forward release is removed in this arm.

## Why the weak-drive count decreases under KC blockade

The change from six spikes to five is not a broken recorder or an absence of
disinhibition. It follows a retained excitatory return route through a PN.

| Tick | Intact route | KC-blocked route |
| --- | --- | --- |
| 101 | KC `720575940625175845` emits | Its soma still spikes, but release is removed |
| 102 | Its input creates potential 0.375 on port 166 of VA2_adPN `720575940611079236` | That input is absent |
| 104 | The PN reaches S=0.6761057 | S=0.6591838 |
| 108 | The PN spikes | S=0.9873763, no spike |
| 109 | Its release creates potential 1.6500039 on port 6 of KC `720575940616654166` | That input is absent at this tick; the PN spikes one tick later |
| 111 | The receiving KC spikes | S=0.9275967, no spike |
| 116 | The intact receiving KC has not spiked again | The blocked-condition KC now spikes |

APL inhibition onto that receiving KC is already weaker under blockade. At
tick 109 its APL input potential is −0.1150545 instead of −0.1189286. Nevertheless,
the missing excitatory arrival changes threshold crossing. This witness
identifies a specific path and timing change; it does not partition all later
network effects into independent causes. KC blockade also removes recurrence,
KC→APL and other KC→PN projections. No one-path lesion was run here.

## What the papers require beyond this result

[Lin et al., 2014](https://doi.org/10.1038/nn.3660) measures KC/APL interventions
with calcium and release imaging, including APL responses in mushroom-body
lobes. Global PAULA `O` is neither a lobe signal nor calcium. The model's
residual PN-driven APL response therefore cannot be called a contradiction of
the paper's KC-blockade observation. Nor do the favorable signs in the table
constitute a reproduction of its odor-panel and learned-discrimination results.

[Amin et al., 2020](https://doi.org/10.7554/eLife.56954) demonstrates localized
APL activity and inhibition. [Prisco et al., 2021](https://doi.org/10.7554/eLife.74172)
examines reciprocal PN/APL/KC interactions and normalization in the calyx.
Together they require distinguishing local APL input/output relations from a
single whole-cell integrator. Pair counts alone do not specify those relations.
The next representation needs spatial anatomy and an explicit intracellular
model, while keeping the same biological APL identity.

Prisco's [data catalog](https://doi.org/10.5061/dryad.bk3j9kdd1) lists small
primary tables of PN glomerular, APL and KC responses to named odors. The
catalog was inspected, but the attempted file downloads returned HTTP 403.
Those tables have not been read or used as input. The entire 8.02 GB deposit
was not downloaded. An accessible public copy, plus a declared mapping between
its measurements and this specimen's annotated PNs, is a useful next step.

## Evidence and recording limits

The five raw courses occupy about 438 MiB under ignored
`.live/research/flywire783/`. Each retains all 254,290 receiving slots and
315,174 terminal information coefficients in 16-tick chunks. Chunk boundaries
share exactly equal states. The analyzer checks the actual experimental input
ports, silent boundaries, blockade targets and every summary against its raw
trajectory. Source-tagged currents use the recorded local potentials and delays;
their float64 sums are diagnostic, not an exact copy of native heap summation.

Both intact courses also matched separate uninstrumented execution for all 224
ticks. Per course, this compares 4,339,125 soma values, 1,239,750 neuromodulator
values, 57,215,250 postsynaptic coefficients and 70,914,150 terminal coefficients,
including initial state. This is exact numeric equality, not a tolerance check.

All courses changed postsynaptic and terminal coefficients; terminal
information stayed positive during these runs. KC blockade removed 485,171
attempted forward terminal events while its cells still generated 783,482
native return events. APL blockade removed 490,590 forward events while APL
generated 14,345 native returns. These are event counts, not spike counts.
Boundary terminals are included in attempted forward counts.

The recordings omit full pending queues, individual return-error vectors and
terminal modulation coefficients. They are not restart checkpoints. The first
recordings hashed the probe source at finish only; a recording guard was added
while the baseline process was running. Its original manifest is retained
unchanged. Exact replay now confirms agreement with the current implementation,
but cannot retroactively attest which file bytes were loaded. Future runs
sample source files at import, start and finish, including shared neuron,
graded-neuron and network code.

There are 39 passing software tests. Five recorded experiments and both replay
checks terminated. No live servers or whole-agent suites were started. These
results justify the next physiological preparation, not an assertion that the
full research objective has been achieved.

### Local record identities

Manifest SHA-256 digests identify the unchanged raw courses:

| Directory suffix under `.live/research/flywire783/` | Manifest SHA-256 |
| --- | --- |
| `pn-course-baseline` | `7318e4fe40596f1cbf64d2620b957fe365d9a848529925c8c4bc21868c82bbc1` |
| `pn-course-gain0075-intact` | `1706041c38403746f1e46d9edc909e8de52d0399987cf705adab33785c792196` |
| `pn-course-gain0075-kc-block` | `8289b4f2849b97c5d804ccd47b7f107ab3f201dba81d5073fb9fbe4d8b661451` |
| `pn-course-gain0075-apl-block` | `b55df4f8db524e18f9586df3a92f51c44b2c26cedd4556db4b7b282b8633180d` |
| `pn-course-gain0075-apl-activation` | `044de080d3dcfbb7d0128e4a943ac12c082e7c782f4cf22fdae2d7a6c6587d2a` |
