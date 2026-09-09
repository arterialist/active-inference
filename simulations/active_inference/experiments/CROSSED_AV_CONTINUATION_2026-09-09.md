# Does balanced experience acquire a relation or repeatedly overwrite it?

The four-exposure crossed audiovisual experiment failed in all graph seeds,
despite a conditional readout-capacity result on its actual incoming histories.
That does not establish an architectural impossibility or a biological deadline
of four exposures. This continuation tests additional experience without
changing a cell, weight initialization, gain, learning rate or body parameter.

Four courses resume the normal-assignment final checkpoints for seeds 11, 23,
44 and 77, including the displaced body and the 64-tick somatic queue. They extend
the same balanced schedule from four to sixteen blocks. Each block contains
all four sensory pairings. State remains continuous through acquisition.
The reversed assignment already produced nearly symmetric failures; this
time-course experiment does not repeat that assignment factor.

Each completed block gets an executable neural checkpoint, a physical/afferent
snapshot and four 96-tick stored-content probes. The probes transfer only
selected weights into the original birth brain and resting body. Their data
are diagnostic branches and never replace the acquiring state. Blocks 8, 12
and 16 also get acquired-state probes with selected weights intact or reset.
Before continuation, all four parent stored-content probes must replay exactly.

Each course adds 17,472 acquisition ticks, 6,912 diagnostic ticks and 384 exact
parent-replay ticks, totaling 24,768 executed ticks. All four total 99,072.
Keep full trajectories and selected updates. Record whether a response becomes
correct within each pre-feedback interval, whether earlier pairings survive the
next episode, and how that relates to actual teaching and local credit traces.
Do not accept late feedback correction as stored recall. Different intermediate
blocks can have different final pairings, so inspect recency alongside duration.

The producer is `crossed_av_continuation.py PARENT OUTPUT`. Completed-block JSON
records preserve partial evidence if a later block fails. They are not process
liveness signals or an automatic resume implementation. Independent tests check
that diagnostic probes do not change subsequent acquisition, including in-flight
neural and somatic signals. No global fitted readout is used.

## Intermediate evidence, through twelve exposures

The intermediate audit of completed-block records through block 12 checked
65,024 newly recorded ticks. All four courses subsequently completed; see the
final results below. Source, recording, local learning,
physical replay, afferent delay, muscle output and between-episode continuity
checks pass. Diagnostic branches also preserve the acquiring process's Python
and NumPy random streams; an uninterrupted-versus-probed continuation test
matches every recorded array exactly.

The failure contains two concurrent effects. The common load-direction bias
can reverse with the last experienced pairing, while the weaker joint
audiovisual contrast grows. In the actual resting learned-weight probes,
define four per-tick responses in order 00,01,10,11. Their sum divided by four
is the common component; `(00 - 01 - 10 + 11)/4` is the joint interaction.
Visual and auditory main effects complete an exactly invertible decomposition.
These are offline observations, not a decoder or information supplied to cells.

At probe tick 63, the joint component grows from 0.0221–0.0294 at four exposures
to 0.0641–0.0919 at twelve. At eight exposures all four pairings have a negative
prediction; at nine and ten they all have a positive prediction. The last
pairing changes from 01 at eight exposures to 11 at nine. Full factorial
trajectories preserve both the increasing interaction and the larger bias.
The order/duration confound prevents attributing those reversals to recency
alone. A growing interaction does not establish correct behavior.

`crossed_av_credit.py` also applies every recorded acquisition weight vector
to the same four factual input histories. It first reproduces the native
reference predictor outputs exactly. It saves the complete input/weight factors,
per-tick conditional bounds, both opponent update magnitudes and an additive
credit decomposition. This permits reconstruction of every acquisition-tick
by probe-tick comparison without storing the full expanded tensor.

Through ten exposures, 96 episodes and 34,944 acquisition ticks were examined
this way. The context trace present at an episode's start accounts for
2.62–7.47 percent of the absolute proposed update mass, depending on episode
and seed. This does not measure all previous-episode contamination: newly
arriving release can retain earlier neural and queue state, and teaching error
has its own history. Clipping and subtraction roundoff are retained separately.

The four-exposure reference interface has a common-component norm of
4.8297–4.9527 and joint-component norm of 0.3490–0.4029 over probe ticks 16–63.
Per-tick Gram matrices retain the actual overlaps. These are interface geometry,
not a learning Jacobian or an estimate of whole-network convergence time.
The relative scale motivates testing selective competition, but does not prove
it will improve learning or that uniform rate reduction cannot help.

## Direct verification of transient learned content

The conditional audit located brief acquired weight states with positive
prediction margins for all four pairings. A separate experiment tests whether
those states work in the actual coupled neural model, rather than accepting
the fixed-interface calculation as evidence of a functioning memory.

Selection is explicitly retrospective. For each seed, choose the first
recorded acquisition tick whose conditional minimum exceeds 0.001 throughout
probe ticks 16–63 for all four pairings. Then select the factual weights
32 acquisition ticks later. Transfer each selected weight vector into the
original birth neural state and resting body. Probe all pairings for 96 ticks,
with positive learning and native return paths active. No weight is fitted,
optimized or edited. Four parent expression probes must first replay exactly.

All four courses completed. They execute 4,608 ticks, including 1,536 exact
parent replay ticks, and retain 3,072 new ticks. The independent analyzer
rechecks every new tick, the exact selected-weight transfer, actual sensory
inputs, absence of early bodily evidence, positive rates and neural motor use.

| Seed | Acquired state selected, zero-based block/tick | All four signs correct at probe ticks 16–63 | Same property with weights 32 acquisition ticks later |
| --- | --- | --- | --- |
| 11 | 5 / 196 | Yes | No; 01 and 10 become wrong |
| 23 | 7 / 308 | Yes | Yes |
| 44 | 5 / 194 | Yes | No; 01 and 10 become wrong |
| 77 | 5 / 186 | Yes | No; 01 and 10 become wrong |

The effect reaches the physical body. At probe tick 63, all sixteen candidate
pairings reduce displacement relative to the recorded birth-weight baseline,
by 0.275–6.300 percent. These are small contributions, not adequate disturbance
compensation. Some onset ticks still predict or move in the wrong direction;
seed 44's 00 response becomes wrong again at ticks 73–78 after feedback begins.
All unfavorable ticks remain in the analysis, not excluded from the recording.

This establishes a narrow but stronger result than the earlier conditional
capacity calculation: local acquisition actually visits weight arrangements
that support the joint relation in real PAULA/body expression probes. It does
not show reliable ordinary recall in the acquiring activity/body state, useful
response magnitude, long-term retention, novel-stimulus generalization or a
mechanism that autonomously selects these favorable moments. Three seeds lose
the expression property within the specified 32-tick comparison; the fourth
must not be described as doing so.

## Architectural consequence

The next question is how ongoing adaptation preserves and uses a relation it
can already acquire. Success need not mean fixed weights. A changing network
could preserve a useful response while its internal parameters keep moving.
Here, a common response changes enough to overwhelm the joint component.
That gives a concrete target for a regulatory circuit rather than a demand
for more neurons without a specified role.

The initial next-intervention hypothesis was to compare neuron-mediated
suppression of shared activity against a uniform learning-rate control. The
upstream gate failure found below takes priority over that addition. Any later
added population must reduce cross-pair interference
without silencing useful joint responses. Keep the delay/body conditions,
positive adaptation, conductance accounting and four-seed comparisons. The
regulator may observe neural signals only; the offline pair identities and
factorial contrasts must never become its inputs.

[Földiák, 1990](https://www.rctn.org/vs265/foldiak90.pdf) provides a relevant
computational precedent: feedforward Hebbian learning, lateral anti-Hebbian
inhibition and adaptive thresholds form less redundant sparse representations.
Its model settles each input before updating and rounds continuous activity to
binary output. That differs from this streaming, continuously plastic PAULA
preparation; its stability argument cannot simply be transferred here.
The paper motivates competition, not a ready-made embodied solver.

ALERM's prospective coupling claim makes regulation of useful learned structure
relevant, but this preparation does not implement its complete dual-modulator
or metabolic law. Error-amplitude-dependent plasticity is not that law. Neither
the transient association nor an inhibitory extension would establish a
self-sustaining hierarchy, artificial life or consciousness. The full brain
goal remains unchanged.

## Reproduction and retained evidence

The continuation roots are
`.live/research/20260909_crossed_av_continued_seed{11,23,44,77}`.
Run `crossed_av_continuation_analysis --blocks N --output OUTPUT ROOT...` only
for the explicitly completed block count. Block-12 analysis is retained at
`.live/research/20260909_crossed_av_continuation_b12_analysis`.

`crossed_av_credit ANALYSIS_ROOT OUTPUT` generated
`.live/research/20260909_crossed_av_credit_b10` from the block-10 audit.
`crossed_av_transient CREDIT_ROOT --seed SEED --output OUTPUT` generated the
four completed `20260909_crossed_av_transient_seed*` courses.
`crossed_av_transient_analysis --output OUTPUT ROOT...` generated
`.live/research/20260909_crossed_av_transient_analysis`.
Raw media, checkpoints and arrays remain local. No live demo or V1–V4 suite
was started by these experiments.

The pre-gate focused suite passed 43 tests, including executable-state continuation,
unchanged acquisition after diagnostic branching, exact source-ID weight
transfer, credit accounting with clipped updates, corruption rejection and
invertible factorial contrasts. These are instrument checks, not behavioral
acceptance. The four acquiring workers stayed near 200 MB RSS each at block 15.

## Final course result and an upstream failure

All four sixteen-exposure courses completed in 1,935–1,977 wall seconds each.
They executed 99,072 ticks; the final analyzer independently checked 97,536
newly recorded ticks and retained 320 probe cases including parent references.
No course is still running. The full output is
`.live/research/20260909_crossed_av_continuation_b16_analysis`.

At sixteen exposures all seeds still predict the negative load for all pairings
through ticks 16–63, in both acquired-state and resting content probes. The
joint component at probe tick 63 has grown to 0.0827–0.1189, but the common
component is still -0.4459 to -0.4015. More experience is learning part of the
relation; it has not produced reliable joint control. Sixteen exposures do not
establish an asymptotic limit.

The acquired and resting states also differ in a previously unaccounted way.
At the final checkpoint, both mixed banks are active in the acquired brain.
The context lamp's neuron is active, and its 192 inhibitory postsynaptic weights
are still negative, each -3.979200714. But their shared presynaptic terminal's
information coefficient is -0.000128883 in every seed. The gate is wired and
its source soma is active, yet it no longer delivers effective inhibition.

Seed 11's checkpoints locate the change. The coefficient is 0.46913 after four
blocks, 0.22713 after six, 0.11075 after seven and negative after eight. These
snapshots bound the crossing; they do not identify its exact tick.

The mechanism follows the executed PAULA path. Each graded postsynaptic target
never produces a somatic spike, so its inherited native timing direction stays
negative. At an inhibitory input the native information error compares positive
arrival amplitude with the negative information weight. The resulting positive
error, multiplied by the negative timing direction, decreases the shared source
terminal. That terminal receives returns from all 192 targets. Once its release
becomes negative, the base neuron's `input_buffer[:,0] > 0` condition ignores the
signal. This also stops those targets generating further input-triggered return
events, leaving the terminal below zero. The bounded incoming-weight extension
does not bound this outgoing coefficient to positive values.

This is a failure of this graded/signed-synapse/native-retrograde composition,
not evidence that all PAULA configurations suffer the same failure. Splitting
the shared terminal would reduce the per-terminal accumulated feedback but
would not, by itself, correct the underlying sign mismatch. Reducing a learning
rate or deleting the spare bank would conceal rather than explain that mismatch.

## Single-terminal causal test

`crossed_av_gate_probe.py` branches each complete final neural/body/delay state.
It exactly replays all four original acquired-state probes with a read-only
terminal observer. It then restores only the context terminal's coefficient to
its birth value, leaving associative weights, intracellular state, physical
state and all adaptation otherwise unchanged. The observer reconstructs the
native terminal updates event by event and retains before/after release,
return-event count and arriving release. Individual return payloads are not
saved; the executable source/checkpoint supports replay.

All four tests complete, executing and recording 3,072 ticks. Restoration
suppresses every member of the spare bank throughout probe ticks 16–63 for all
sixteen pairings. It restores 192 return events per active tick and the terminal
ends near 0.991394, still decreasing. The acquired-state predictor gain falls
toward the resting-state value, but the 00 and 11 associations remain wrong.
This isolates the upstream gate loss without claiming it caused every earlier
learning failure. The pre-eight-block interference and the failed restored-gate
probes are counterexamples to that broader claim.

The gate analysis retains each bank member at every tick, the terminal traces,
the physical trajectories and matched-state checks. Its output is
`.live/research/20260909_crossed_av_gate_analysis`; raw courses use
`20260909_crossed_av_gate_seed{11,23,44,77}`. A reduced-network regression confirms
that the observer leaves every original recorded field unchanged.

The next repair must make the non-spiking, signed-input and outgoing-plasticity
semantics consistent under composition, then repeat the gate and association
tests. Use an explicit opt-in extension and a baseline-preserving regression.
Do not permanently reset the terminal from Python or disable adaptation. This
is now prior to adding a new inhibitory competition population, which would
otherwise inherit the same failure mode. The full brain goal remains open.

Final verification: 44 focused tests pass in 27.85 seconds. The continuation,
transient-content and terminal-intervention courses together execute 106,752
ticks, including their declared exact replays. All twelve experimental workers
finished; no research simulation from this unit remains running. The shared
neuron-model source was not modified. These results establish usable diagnostic
evidence and transient content, not a repaired autonomous agent.
