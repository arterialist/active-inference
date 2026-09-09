# Undriven inhibitory PNs remain part of the connected response

Turning off the experimental current to inhibitory PNs does not silence them.
The retained PN network recruits them, and their output still reaches APL.
The calyx-versus-vertical-lobe distinction under KC-output blockade survives
this control and a separate control that removes current from VP-labelled PNs.
These are results from the connected PAULA model, not replicated fly physiology.

A further fixed-input cable decomposition separates two effects. Negative PN
current explains the blocked vertical lobe's below-zero voltage. It does not
account for most of the regional feedback dependence. Without that negative
electrical contribution, while keeping every other recorded current fixed,
vertical-lobe release still falls by 98.22% under KC blockade. This decomposition
is not a connected-network lesion prediction.

## What changed, and what did not

The preparation retains all 2,755 selected cells, all original directed pairs,
boundary ports, and native postsynaptic and retrograde adaptation at positive
rates. The controls change only which dedicated experimental ports receive
current. An undriven neuron can still receive synaptic input and release output.
KC-output blockade remains the previously declared whole-KC forward-event
filter, including KC→PN and KC→KC output as well as KC→APL.

The common course has 224 ticks, with zero external input for ticks 0–31 and
192–223. Ticks 32–71, 72–111, 112–151 and 152–191 deliver current amplitudes
1.25, 2.5, 5 and 10, respectively, to each chosen PN. Dose per driven cell stays
fixed; total input is not renormalized. These are successive windows in one
adaptive trajectory, not independent trials or four odors. No physical duration
has been assigned to a tick.

All comparisons below use `Rm/Ra=25000 µm`, `weight_per_count=0.075`,
`lambda_ticks=20`, one cleft tick plus two dendritic ticks, `signal_decay=0.95`,
`eta_post=1e-8`, `eta_retro=1e-6`, APL release gain 0.01 and cap 1.
These remain assumed dynamics. The builder's ordinary defaults are unchanged.

| Input panel | Driven PNs | Undriven PNs | Selection rule |
| --- | ---: | ---: | --- |
| Existing all-PN reference | 174 | 0 | All selected ALPNs |
| Without inhibitory drive | 158 | 16 | Exclude source-table model sign -1 |
| Without VP drive | 154 | 20 | Exclude explicit VP1–VP5 hemibrain labels, including mixed labels |

The sixteen negative-sign PNs include thirteen with a known-GABA annotation,
one further predicted-GABA cell, and two predicted-glutamate cells. Model sign
is not measured receptor pharmacology. The recorded panel keeps transmitter
predictions, known-transmitter annotations and source-model signs separate.

The VP-label control is motivated by the thermo- and hygrosensory pathways
described by [Marin et al., 2020](https://pubmed.ncbi.nlm.nih.gov/32619476/).
Their multiglomerular PNs can combine VP and other inputs. A missing VP label
does not establish exclusive olfactory input. This control must not be called
an "olfactory-only" preparation. Four cells overlap the two exclusion sets.
A combined exclusion is implemented in `input_panels.py` but has not been run
on the full graph.

## The complete recruitment path

In both no-inhibitory-drive records, the sixteen undriven cells produce 204
spikes in total. The first is `M_vPNml50`, FlyWire root
`720575940608124338`, PAULA ID 5541, at tick 66. Its identity is also listed by
[Virtual Fly Brain](https://www.virtualflybrain.org/term/m_vpnml50-vfb_fw035663/)
as a GABAergic multiglomerular AL projection neuron. That annotation does not
validate the modeled current magnitude or timing.

| Tick | Recorded event |
| ---: | --- |
| 63 | 116 distinct incoming PNs each have actual somatic output 1. |
| 64 | Their signals arrive at the corresponding M_vPNml50 receptor ports. |
| 66 | The delayed potentials reach integration. M_vPNml50 emits output 1 despite zero experimental current and previous membrane voltage 0. |
| 67 | Its terminal 282 delivers information amplitude 1 to APL port 154. |
| 69 | APL receives actual delayed current -1.2183749676 from that port. |

The incoming potentials sum to 101.73431255 using float64 diagnostic arithmetic.
The corresponding unclipped pre-reset membrane estimate is 5.08671563 against
threshold 1. The recorded post-spike membrane is zero. This is consistent with
the observed spike, not a bit-exact reconstruction of native float32 heap
accumulation. All 116 sources and receptor contributions remain in the audit.

The M_vPNml50→APL source pair is row 579880, with eighteen anatomical contacts.
The assumed count conversion gives postsynaptic information weight -1.35.
Its actual native local potential at tick 67 is -1.3500000238. The delayed
recorded cable current differs slightly from a float64 multiplication by
`0.95**2`; the audit reports both, rather than treating that rounding difference
as a new mechanism. No negative release amplitude is sent. Inhibition enters
through the receiving weight, consistent with PAULA's positive-input mask.

## Regional effects survive the input controls

The table uses means over ticks 152–191. Voltage is membrane-area-weighted.
Release is the mean rectified, capped local output at anatomical attachments,
before terminal gain or blockade. It is not a calcium measurement or delivered
synaptic output. The all-PN pair is reused from the
[preceding regional experiment](APL_REGIONAL_FINDINGS_2026-09-10.md).

| Input panel | Region | Voltage, intact → KC blocked | Local release, intact → KC blocked |
| --- | --- | ---: | ---: |
| All PNs | Calyx | 146.7768 → 107.0058 | 0.919023 → 0.800152 |
| All PNs | Vertical lobe | 24.9040 → -2.1236 | 0.253595 → 0.000461 |
| No inhibitory drive | Calyx | 147.9564 → 107.8356 | 0.920516 → 0.802865 |
| No inhibitory drive | Vertical lobe | 24.9614 → -2.0963 | 0.253404 → 0.000471 |
| No VP drive | Calyx | 141.5268 → 105.4288 | 0.912006 → 0.792121 |
| No VP drive | Vertical lobe | 23.9662 → -2.1274 | 0.245814 → 0.000447 |

Removing inhibitory experimental drive reduces whole-course negative PN current
into APL from -157.840729 to -115.469962 in the KC-blocked comparison. It does
not eliminate it. The no-VP value is -153.305751. Each new panel's intact and
blocked courses have equal total negative PN current over all 224 ticks;
that equality does not imply identical per-port or per-tick currents.

The twenty undriven VP-labelled cells produce three spikes in each no-VP
record. Their first spike is in `VP2_adPN`, root `720575940619895125`, at tick
165. The matched intact and KC-blocked trajectories first diverge in APL at
tick 119 without inhibitory drive and tick 121 without VP drive. PN outputs
later diverge at ticks 128 and 133. Identical external current does not imply
identical PN activity once their actual recurrent connections operate.

Saturation remains substantial. In the last no-inhibitory-drive window, an
average 74.27% of calyx output attachments are capped in the intact condition,
versus 51.76% after KC blockade. Persistence of capped calyx release alone
cannot establish an accurate calyx transfer function.

## Separate electrical contributions without a new neural run

`regional_activity --decompose-negative-pn` uses the actual delayed input
currents from the full recording. For each tick it extracts the negative part
of every PN→APL port current, distributes it over that port's measured contacts,
and solves its contribution on the complete 360,309-node passive tree. This
component begins at zero. Subtracting it from the recorded voltage leaves the
response to all other recorded currents and the original initial state.

The passive voltage operator is linear for these prescribed currents. The
observer checks both component and remainder equations at every node and tick.
Their largest residuals are `9.99e-15` and `6.70e-14`. Release is nonlinear, so
the observer rectifies and caps the remainder's node voltages before averaging;
it never subtracts mean release values as if release were linear.

| No-inhibitory-drive condition | Region | Negative PN voltage contribution | Remainder voltage | Remainder local release |
| --- | --- | ---: | ---: | ---: |
| Intact | Calyx | -0.03736 | 147.99376 | 0.920554 |
| KC blocked | Calyx | -0.03736 | 107.87300 | 0.802935 |
| Intact | Vertical lobe | -2.51769 | 27.47908 | 0.258652 |
| KC blocked | Vertical lobe | -2.51769 | 0.42142 | 0.004613 |

Thus negative PN input reverses the blocked vertical-lobe mean voltage's sign,
but the fixed-input remainders still differ greatly. Their local release falls
by 98.22% in the vertical lobe and 12.78% in the calyx. That narrows the artifact
explanation. It does not predict a fly or even the full PAULA network with a
negative-PN pathway lesioned. Such a lesion would change PN and KC activity,
APL output, return signals and subsequent weights. Those downstream changes
are deliberately held fixed in this electrical decomposition.

## Evidence, verification and next experiment

The four full neural courses occupy about 2.1 GiB and remain local under
`.live/research/flywire783/`. They were not regenerated for this analysis. The
two new decompositions and their full-record audits together add about 412 KiB.
No live agent server, embodied suite or video renderer was started.

Raw courses are `pn-panel-no-inhibitory-intact-r25000`,
`pn-panel-no-inhibitory-kc-block-r25000`, `pn-panel-no-vp-intact-r25000` and
`pn-panel-no-vp-kc-block-r25000`. Their corresponding regional directories start
with `apl-regions-no-inhibitory-` or `apl-regions-no-vp-`. Decomposition directories
are `apl-negative-pn-split-no-inhibitory-intact-r25000` and
`apl-negative-pn-split-no-inhibitory-kc-block-r25000`.

Every regional result has a full-record audit, every-tick readouts and source
hashes. The decomposition audits also retain the completed recruitment path.
Both new intact input panels have exact uninstrumented replays, covering
all recorded soma states, synaptic coefficients, branch voltages and currents.
Each replay matches 81,069,525 branch-voltage values and 80,709,216 node-current
values, as well as all other recorded state fields.
These records omit complete event queues and are not restart checkpoints.

The Drosophila selection passes 148 tests. These include a four-cell recruited-PN
test, an independent dense solve of the fixed-input remainder, a zero-negative-
current control, missing-region handling, matched input/graph checks and
full-tick recorder checks. An unrelated checkpoint module is skipped because
`cloudpickle` is unavailable. Passing these tests establishes neither fly
learning nor behavioral replication.

To reproduce the small analysis from retained raw data, from `active-inference`:

```sh
OPENBLAS_NUM_THREADS=1 uv run python -m simulations.drosophila.regional_activity \
  .live/research/flywire783/pn-panel-no-inhibitory-intact-r25000 \
  .live/research/flywire783/apl-neuropil-bound-20260910 \
  NEW_READOUT --decompose-negative-pn
uv run pytest tests -q -k drosophila
```

To generate a new matched input-panel course, use the existing
`intervention_probe` command with `--pn-drive-panel without_inhibitory_drive`
or `--pn-drive-panel without_vp_drive`. Keep the parameters above and select
`intact` or `kc_release_block`. Each new full recording takes roughly half a
GiB; do not launch another batch merely to recreate these readouts.

The next physiological experiment needs temporally varied, odor-selective PN
input with explicit empirical coverage. The present identical-current schedule
creates a highly synchronous initial volley. We have not established whether
that synchrony occurs under biological input. A controlled timing experiment
can test sensitivity, but must not be presented as measured odor stimulation.
Retain both regional observations and the actual recurrent graph. A selective
PN→APL release intervention would test the full feedback consequences that the
electrical decomposition cannot. Odor discrimination, useful learning and
coordination with downstream neural consumers remain unestablished.
