# Signed residual learning under delayed bodily feedback

## Question and prior failure

The previous 590-cell experiment demonstrated contextual access to learned
motor responses, but one recording elicited poor recalled action. Ongoing
neural teaching repaired that action during the probe. Both acquisition blocks
ended with recording 0, leaving recency confounded with recording identity.
The immediately available force receptors also allowed feedback to substitute
for a useful stored expectation.

This experiment asks whether the learning circuit should compare two signed
relationships jointly, rather than train two nonnegative quantities separately.
The original motor output already subtracts two predictions, so this is not an
increase in the set of signed motor responses the network can express. An
earlier verbal suggestion that the original whole circuit lacked negative
evidence was too broad. The change is in neural credit assignment.

## Declared intervention before examining results

Keep the real audiovisual receptor streams, two context-gated mixed banks,
590 neurons, all selected initial weights, existing neuron classes and their
positive adaptation rates. Add eight ordinary signed connections. Positive
comparison for branch 0 receives force 0 minus force 1 minus prediction 0 plus
prediction 1; the other comparison polarities follow the corresponding signs.
The actual delayed releases, membrane integration, rectification and native
return signals still determine what reaches each local learning receptor.
This expression describes the wiring, not instantaneous algebra performed by
the host. No neuron equation or shared model source changes.

Both predictor branches can now learn from the same signed residual. Effective
differential learning gain may change. The comparison does not isolate gain
from credit assignment and cannot establish that inhibitory plasticity is
necessary. It tests this circuit as an organization, not a fitted optimum.

Acquisition has the same 16 episodes, 364 ticks each. Reverse which recording
comes first in every pair for order 1. This also reverses which recording ends
each block. Probes branch the entire acquired network and body. Each of the
four context/recording combinations is tested with selected weights intact or
reset to their initial zero values, and bodily afferents delayed by 0 or 64
ticks. Both force and joint feedback are delayed, while audiovisual/context
signals remain current. The delay queue is initialized with the actual final
64 samples of acquisition. It does not receive labels or a probe-phase flag.

At 4 ms per tick, the intervention creates a 256 ms interval before new bodily
feedback reaches the receptors, plus existing neural propagation. This is an
experimental delay, not a universal mammalian latency. The current local rule
treats absent load evidence as absence, not uncertainty. Consequently, recall
can weaken before the delayed load evidence arrives. That possibility is part
of the test, not corrected by freezing weights or suppressing teaching.

The primary evidence is every recorded tick of actual motor command, net
physical torque, angle, supplied versus raw afferents, neural state snapshots,
selected weights, learning rates, context arrivals and neural error arrivals.
Inspect fixed intervals 0–31, 32–63, 64–127 and 128–191 separately. The first
64 ticks specifically concern action before new bodily feedback; the later
intervals concern adaptation and correction. Compare intact against matched
reset within each architecture, then both orders. Graph replication needs at
least four seeds before a robust effect can be claimed.

No arbitrary perfect-position threshold defines success. A favorable endpoint
cannot hide an initially wrong command. The preparation does not yet define
organism survival, multimodal necessity, semantic recognition, learned context
inference, a global workspace or consciousness.

## Literature relation

[Agnes and Vogels, 2024](https://www.nature.com/articles/s41593-024-01597-4)
model plasticity that depends on neighboring excitatory and inhibitory currents.
Their model couples local synaptic changes and studies stable network dynamics.
The abstract, introduction and equations 1–2 motivate examining interactions
between learning pathways rather than treating their plasticity independently.
This experiment does not implement their current variables, calcium/chloride
interpretation, STDP rules or inhibitory gating equation. It uses PAULA's
existing experimental receptor rule and additional neural comparison wiring.
That paper is background for the hypothesis, not validation of this circuit.

## Completed pilot and causal follow-up

Four seed-11 courses are complete: both architectures, both presentation orders.
Each contains 8,896 executed ticks. Two subsequent feedback interventions each
execute 3,072 ticks, including exact unmodified replay before every intervention.
The total is 41,728 ticks. This is one graph seed, not four-seed validation.
No neural or embodied capability is promoted to accepted status.

Every retained tick passes the independent selected-learning recurrence,
full MuJoCo integration-state replay, physical afferent reconstruction and
motor-output/command equality check. All residuals are zero. The new baseline
order-0 recorder reproduces all shared arrays in all 16 original acquisition
episodes and all eight original probe prefixes exactly. The 21 focused tests
include checkpoint continuation with delayed sensor queues and deliberate
corruption of weights, supplied signals, raw physical signals and motor traces.
Those checks establish recording fidelity, not useful cognition.

### Recency matters

Reversing the order reverses which recording has the clearer learned benefit
before new bodily feedback. In baseline context 0 with a 64-tick delay, order 0
gives recording 0 a substantial advantage over selected-weight reset; recording
1 is worse. Under order 1, recording 1 benefits, while recording 0 is approximately
equal to reset. The same direction of preference occurs in context 1. This
resolves the earlier order confound for this graph, but does not establish an
exclusive recency mechanism or robust repertoire learning. The audiovisual
recordings still differ in intensity and feature content.

### Coupling the errors destabilizes correction

The signed-residual architecture is not a repair at its original gains. In
context-0, recording-1 acquisition, order 0, net torque is +0.492 Nm at tick 32,
-0.513 at tick 64, +0.622 at tick 96, -0.645 at tick 128, and +0.984 at tick 299.
The imposed load is only -0.2 Nm. Actual teaching arrivals alternate with the
opposite signs at the two predictors; at tick 272 they are approximately
-4.159 and +4.159. Both effective rates remain near 0.005. The baseline's
corresponding transient decays toward small residual torque. These are actual
neural/physical time series, not fitted oscillators.

One declared intervention halves all incoming information weights of the
four comparator cells, preserving their signs and delays. Every other acquired
state, including selected memory weights, is unchanged at branching. Native
adaptation continues. This is a diagnostic of the acquired correction loop,
not training a normalized circuit from birth. Two branches now respond to the
same signed residual, motivating the one-half control; nonlinear rate dependence,
rectification, weight bounds and residual state prevent exact gain matching.

For recording 1, the intervention lowers late absolute residual torque in all
eight context/order/delay combinations. The following values average ticks
128–191 only; full trajectories are retained and must be inspected with them.

| Order | Context | Somatic delay, ticks | Original mean absolute torque, Nm | Half-feedback torque, Nm |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0.419757 | 0.034835 |
| 0 | 1 | 0 | 0.355542 | 0.028948 |
| 0 | 0 | 64 | 0.279288 | 0.037130 |
| 0 | 1 | 64 | 0.240058 | 0.039176 |
| 1 | 0 | 0 | 0.526882 | 0.074881 |
| 1 | 1 | 0 | 0.457953 | 0.045845 |
| 1 | 0 | 64 | 0.624423 | 0.122242 |
| 1 | 1 | 64 | 0.554290 | 0.088174 |

This does not mean better behavior throughout. For context 0, order 0, delayed
recording 1, final angle worsens from -0.509562 to -0.573697 rad. For context 0,
order 1, delayed recording 0, it worsens from +0.860092 to +1.076322 rad.
Several recording-0 residual-torque intervals worsen too. Damping oscillation
does not undo displacement already accumulated or establish a correct memory.
There is no metabolic cost model in this preparation, so neural common-mode
activity must not be described as measured energetic expenditure.

### Feedback needs a temporal reference

The delay queue contains actual prior bodily measurements. Force channels are
zero for the first 64 probe ticks because acquisition ended with withdrawal;
joint channels retain real prior motion. Current audiovisual input can evoke
a current prediction, but its comparator receives old load evidence. Before
fresh load evidence reaches the receptors, the baseline's selected weights can
already change by roughly 0.079 in order 0 and 0.099 in order 1. The system is
actively modifying recalled predictions, not merely waiting for feedback.

This motivates distinguishing a prediction used for immediate action from the
same prediction aligned with later sensory verification. It is not sufficient
to call all discrepancies novelty or to make learning arbitrarily weak. A
candidate circuit must align predicted and observed consequences in time and
credit the contributing earlier activity, while preserving positive adaptation.
Ordinary delayed pathways and existing eligibility/history dynamics should be
tested before introducing another neuron equation. The current evidence does
not prove a specific alignment architecture will solve repertoire learning.

The next coupled test should combine normalized feedback participation with
explicit temporal alignment, retain both presentation orders and expand to
four graph seeds. Evaluate usable memory before corrective evidence, subsequent
adaptation, and accumulated body motion together. Do not spend a separate
campaign optimizing an exact hinge endpoint. This remains a component-level
research step toward the unchanged embodied brain objective.

## Reproduction and retention

Producer: `opponent_context.py`. Library delta:
`components/learning/opponent_prediction.py`. Causal follow-up:
`opponent_feedback_probe.py`. Auditors and measured phase portraits:
`opponent_context_analysis.py` and `opponent_context_plot.py`.

Raw roots under `.live/research/`:
`20260909_opponent_context_{baseline,opponent}_order{0,1}_seed11` and
`20260909_opponent_feedback_half_order{0,1}_seed11`. These are six explicit
completed runs; braces describe names, not a destructive-operation pattern.
The two analysis roots end in `opponent_context_analysis_seed11` and
`opponent_feedback_analysis_seed11`. The phase portrait is
`20260909_opponent_context_analysis_seed11/feedback-dynamics.png`.

All six runs exited normally. Raw NPZ data total 444,745,020 bytes, including
full tick records and physical state files. Raw data and runtime checkpoints
remain local. Source, tests and this account may be published. Previous recorded
producers and shared neuron sources were left unchanged.

Run from `active-inference/`, using a fresh output path:

```sh
uv run --offline --no-sync --with cloudpickle==3.1.2 python -m simulations.active_inference.experiments.opponent_context --output .live/research/FRESH_RUN --seed 11 --order 0
```

Add `--baseline` for separate errors. Change `--order` to counterbalance
presentation, not to change the world association or neural birth weights.
The media fixture dependencies remain the same local real-media transductions
as the prior context experiment. A public checkout alone does not include them.
