# Neural reference for plasticity, separate from somatic inhibition

## Question and model change

The completed local-competition courses reduced shared prediction bias while
also weakening the joint audiovisual component. They did not establish retained
association. This candidate asks whether a population can influence which
synapses are rewritten without directly inhibiting the representation being
expressed. It is not another inhibition-strength sweep or an accepted memory
repair.

`neuron/extensions/experimental/contrast_eligibility.py` adds an explicit
subclass of the current magnitude-feedback predictor. The inherited local
context trace is `x_i <- d*x_i + (1-d)*arrival_i`. A separate neural reference
trace is `z_r <- d*z_r + (1-d)*reference_arrival_r`. For the selected prediction
ports only, the update becomes:

```
q_i <- clip(q_i + eta(e_previous) * e_previous * (x_i - g*z_r), 0, cap)
```

The error, positive basal rate, bounded error-dependent amplification and
context timescale remain as before. The change allows negative eligibility:
activity below the neural reference can reverse a synapse's update even when
the neuron-wide error has the same sign. It does not require a negative release
amplitude, a host-computed error or a negative information weight.

The references arrive on declared zero-throughput receptor ports. They do not
directly contribute to membrane integration. Forward potentials and returns
use pre-update weights in the inherited order; only selected information
weights are replaced after that tick. With no reference mapping, the inherited
path is exact. With a mapping but strength zero, the same added neural circuit
remains and the old learning rule is used. Existing classes and `neuron.py`
are unchanged.

This is a phenomenological eligibility extension, not a faithful OLM cell or a
claim that the original PAULA equations already implemented it. The primary
abstract of [Sejnowski, 1977](https://pubmed.ncbi.nlm.nih.gov/925522/) motivates
distinguishing covariance-dependent storage from raw activation. Its nonlinear
model and proposed plasticity are not reproduced here. The current spatial
reference is not an estimate of statistical covariance across experiences.

## Neural construction and limits

`components/learning/eligibility_reference.py` adds one graded reference cell
for each 192-cell mixed bank, bringing this preparation from 590 to 592 cells.
Each reference receives all members of its bank. Birth input weights total
`1/.99`, compensating the declared one-tick dendritic attenuation of `.99`.
Its membrane time constant is two ticks. It projects to both predictive cells
through zero-throughput reference receptors; each selected input uses the
reference belonging to its actual source bank. Source membership is anatomy,
not a stimulus label. No external input is added.

This is a delayed reference, not instantaneous normalization. The extra
synaptic path and integration remain in the simulation. Unequal feature
statistics, reference lag, lower weight clipping and subsequent adaptation can
all prevent the intended cancellation. Subtracting one spatial reference does
not mathematically remove the factorial shared-response component measured
across four different audiovisual episodes.

Native return paths remain active. The added reference cells can therefore
alter upstream terminals even without direct somatic inhibition. The wired
control has exactly the same extra paths. Equality of instantaneous forward
output at fixed weights is not equality of future coupled trajectories.

## Verification completed before interpretation

Forty-seven focused tests pass, including seven new reference-pathway tests.
They establish exact inherited trajectories with contrast disabled, opposite
weight changes for above/below-reference inputs under a common error, and
unchanged current somatic output for the matched fixed-weight intervention.
Host-driven reference receptors, direct somatic throughput and invalid bank
construction are rejected.

The embodied recorder reproduces the unobserved run exactly. A separate
auditor reconstructs reference traces and selected weight updates from actual
arrivals, rather than trusting reported eligibility. It also checks reference
arrivals against source-cell release and terminal values. Deliberate corruption
of weights, rates, reference arrivals, reference traces and eligibility is
detected. An executable checkpoint reproduces continuation with reference
traces and signals in flight exactly.

These tests establish the implemented mechanism and observation path, not
learned regulation, retained association, sensory semantics or consciousness.

## Bounded embodied screen now running

`eligibility_reference_acquisition.py` uses the previous magnitude-feedback
birth graphs and the same real audiovisual feature recordings. Every seed must
first reproduce a full old 364-tick body episode exactly with the extension
disabled. Each of three conditions then runs a continuous brain/body history:

| Condition | Reference contrast strength | Basal selected rate |
| --- | --- | --- |
| Wired control | 0 | 1e-5 |
| Slow control | 0 | 2.5e-6 |
| Neural contrast | 1 | 1e-5 |

The quarter-rate control is fixed in advance, not fitted to match observed
learning exposure. It tests one slower-learning alternative, not every possible
rate. All conditions retain error-dependent amplification and positive basal
adaptation. Nonselected and outgoing learning remain active.

The initial screen has one balanced four-pair acquisition block, then one
balanced block with the physical load relation reversed. Reversal changes the
world's force, not neural configuration or sensory labels. Body state, delayed
afferents, neural queues and learning traces continue across the change.
Weight-transfer resting probes and matched acquired-state learned/reset probes
run in isolated branches after each block; none writes back to acquisition.

Seeds 11, 23, 44 and 77 run in
`.live/research/20260909_eligibility_reference_screen_seed{seed}`. Each complete
worker will execute 16,012 ticks, including its unchanged replay. All four
seeds' replays and first wired acquisition blocks have completed; the full
screen is still running. Do not interpret checkpoint existence as completion.

Full cell states, selected weights, actual reference arrivals, every reference
cell's incoming weights, terminal releases, body integration state and local
learning variables are retained. `eligibility_reference_analysis.py` is the
independent whole-course auditor; its full-course invocation awaits completed
workers. Reference-cell intracellular equations are observed but not separately
equation-reconstructed by this auditor.

One block per relation is deliberately a structural preflight. It cannot prove
adequate acquisition or retention. If the reference is active, local learning
remains valid and acquired weights have a useful causal contribution, the next
comparison must extend the original acquisition/retention course and include
relearning after reversal. If it fails, inspect the reference lag, signed
eligibility, clipping and the actual physical/neural error before changing gain
or adding populations. The separate action-dependent body/context experiment
remains necessary; this exogenous-load task is not a full action model.
