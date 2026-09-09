# APL input physics, 10 September 2026

A better spatial fit need not mean better intracellular propagation. The
registered APL experiment now demonstrates that confound directly. Strong local
conductance can amplify the weak, spatially distributed stimulus tail relative
to its peak. Most of the resulting response in a tested vertical region remains
when axial coupling is removed.

## Matched comparison

The preceding [registered-tree test](REGISTERED_APL_FINDINGS_2026-09-10.md) used
current density proportional to dye. [Amin et al.](https://doi.org/10.7554/eLife.56954)
instead stimulated ectopically expressed P2X2 channels with ATP and used dye to
estimate stimulus spread. Their spatial model omitted branch loading. Its
inferred resistance ratio is therefore not an independently measured electrical
constant for the conservative cable. The paper also identifies registration,
stimulus measurement and omitted conductances as limitations.

An exploratory stationary comparison kept all 181,741 nodes, region assignments,
dye fits and readout locations fixed. Eight shared `Rm/Ra` values ranged from
6,250 to 2,000,000 micrometres, including ratios discussed in the paper. Each
was tested with membrane-area current, total-dose-matched length-weighted
current, and membrane conductance at four shared gains. All 48 cases cover
horizontal, vertical and calyx stimulation. No branch, region or individual
stimulation site receives a fitted parameter.

The conductance equation is `(C + L + G)v = I + G*E`, with `G = C*g*dye` and
`E = 1` in arbitrary voltage units. The current conditions use `G = 0`.
Conductance gain, ATP-to-channel occupancy and calcium-to-voltage conversion
are not calibrated. Every solved node state and actual input is retained.

Changing current allocation from area to length makes only modest differences
in this grid. Increasing coupling under current drive improves aggregate fit,
but region 17 under vertical stimulation remains at most 0.185 of peak voltage,
against observed normalized calcium 0.701. The best descriptive aggregate case,
`Rm/Ra = 200000`, conductance gain 1, still predicts only 0.209 there. Selecting
parameters on two stimulation sites does not establish generalization: the
case selected without calyx data has calyx RMSE 0.200. These are diagnostics
on already inspected observations, not unseen biological validation.

## Full-tick causal test

The next experiment holds `Rm/Ra = 25000` and gain 10 fixed. All conditions
start at rest and use 200 ticks of drive followed by 200 ticks without drive.
This is a diagnostic pulse, not a physical-time model of the ATP puff.
The tip-only intervention activates channels only in regions 23 and 25, the
two regions with highest fitted stimulus intensity. It does not select regions
using calcium responses or prediction errors.

| Condition | Region 17 raw voltage at tick 200 | Peak regional voltage | Region 17 / peak |
| --- | ---: | ---: | ---: |
| Full stimulus, coupled tree | 0.42851 | 0.90044 | 0.47590 |
| Same stimulus, zero axial coupling | 0.42035 | 0.90883 | 0.46252 |
| Tip-only channels, coupled tree | 0.11301 | 0.89829 | 0.12581 |

The effect develops during input. At ticks 1, 20 and 200 the intact normalized
region-17 response is 0.12550, 0.41519 and 0.47590. Without axial coupling it is
0.10530, 0.38147 and 0.46252. Thus the similarity is not just one matching final
sample. After input removal, raw voltages decay in all conditions. At tick 400,
the intact peak is only `2.54e-5`, despite a region-17/peak ratio of 0.38037.
Normalizing each frame would hide that near-silence.

These nonlinear conditions are not an additive source decomposition. Removing
channels changes both drive and local loading. Nevertheless, zero axial
coupling is sufficient to retain most of the tested regional response, and
removing the distributed channel drive markedly reduces it. This refutes
attributing the entire improved profile to stronger propagation in this model.
It does not determine the corresponding mechanism in the animal.

An independent scalar calculation explains the effect. The fitted stimulus in
region 17 is 0.07313 of the peak. With no axial exchange, local steady voltage
is `g*dye / (1 + g*dye)`. At gain 10 this gives a region-17/peak voltage ratio
of 0.46464 without any spatial readout interpolation, close to the recorded
0.46252. The receptor-current assumption alone compresses the input contrast.
Two of the 107 region-17 sample edges have a parent in another region, so the
common interpolating readout makes a small further difference. Neither effect
requires signal transmission through the neurite tree.

## What is reusable

The opt-in `ConductanceCable` extends PAULA's intracellular cable operator while
preserving ordinary stepping and zero-conductance trajectories exactly. It is
not installed in the running FlyWire neuron or any agent. A separate auditor
rebuilds membrane area and the axial operator using an edge-incidence matrix,
then checks every recorded state and reconstructs each regional readout.
All 218,634,423 node-state values passed. Maximum row-wise relative backward
error was `3.43e-16`; maximum mass-balance discrepancy was `1.31e-15`.
Corrupted-state and wrong-input tests verify that the checker detects those
failures. These checks establish the numerical experiment, not fly physiology.

The methodological consequence is to control distributed input and its
transduction before attributing a response profile to intracellular or network
communication. The next physiological constraint must discriminate those
causes, for example matched input/output interventions or independently
constrained channel and observation laws. More fitting of this one normalized
profile cannot supply that missing evidence. The connected KC/APL/PN work
must retain this uncertainty rather than promote the lowest-RMSE candidate.
The next connected experiment should use the published APL-to-KC and KC-to-APL
interventions to constrain feedback across a declared electrical-parameter
range. A unique fit to this spatial profile is not a prerequisite for testing
those additional physiological relationships.

## Reproduce

Use the source and registered-geometry directories listed in the preceding
report. Every output path must be new.
The sibling `neuron-model` needs the opt-in conductance operator added in
commit `2ef0bea`; the recording also pins its exact source hash.

```sh
.venv/bin/python -m simulations.drosophila.amin_input_physics REGISTERED_DIR SOURCE_DIR CELLS_JSON NEW_SWEEP_DIR
.venv/bin/python -m simulations.drosophila.amin_conductance_probe run REGISTERED_DIR SOURCE_DIR CELLS_JSON NEW_TRACE_DIR
.venv/bin/python -m simulations.drosophila.amin_conductance_probe audit NEW_TRACE_DIR
.venv/bin/python -m pytest -q tests/test_drosophila*.py
```

Final local artifacts are `.live/research/flywire783/amin-input-physics-20260910/`
and `.live/research/flywire783/amin-conductance-causal-20260910/`. They occupy
251 MiB and about 1.0 GiB. The stationary comparison took 14.39 seconds with
282 MB maximum resident memory; the full traces took 37.02 seconds and 498 MB.
Independent replay inspection took 6.20 seconds and 183 MB. All 124 fly tests
and all 51 tests under `neuron-model/tests/` pass. No spiking/plastic network,
learning task, live server or embodied suite ran in this experiment.
